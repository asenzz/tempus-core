#include <cuda_runtime_api.h>
#include "common/cuda_util.cuh"
#include "common/gpu_handler.tpp"
#include "recombine_parameters.cuh"

namespace svr {

template<const unsigned k_block_size> __device__ void
warp_reduce(
        const uint8_t colct,
        volatile double *_sh_best_score,
        volatile t_params_vec *_sh_best_params_ix,
        const uint32_t thr_ix)
{
#define WARP_NOSYNC(this_block_size) {                                                                                              \
        constexpr unsigned thr_ix_off = this_block_size / 2;                                                                        \
        if (k_block_size >= this_block_size) {                                                                                      \
            if (_sh_best_score[thr_ix + thr_ix_off] < _sh_best_score[thr_ix]) {                                                     \
                _sh_best_score[thr_ix] = _sh_best_score[thr_ix + thr_ix_off];                                                       \
_Pragma("unroll")                                                                                                                   \
                for (size_t i = 0; i < colct; ++i)                                                                                  \
                    _sh_best_params_ix[thr_ix][i] = _sh_best_params_ix[thr_ix + thr_ix_off][i];                                     \
            }                                                                                                                       \
        }                                                                                                                           \
    }

    WARP_NOSYNC(64);
    WARP_NOSYNC(32);
    WARP_NOSYNC(16);
    WARP_NOSYNC(8);
    WARP_NOSYNC(4);
    WARP_NOSYNC(2);
}


template<const unsigned k_block_size> __global__ void
cu_recombine_parameters(
        const uint8_t colct, const uint32_t rowct,
        const uint8_t *__restrict__ p_combinations, // Len of prediction_ct * col_ct
        const t_param_preds_cu *__restrict__ p_params_preds, // Len of col_ct * C_tune_keep_preds
        uint8_t *__restrict__ best_param_ixs, // Len of col_ct
        double *__restrict__ p_best_score,
        const double *__restrict__ recon_signs,
        const double *__restrict__ recon_last_knowns
)
{
    const auto thr_ix = threadIdx.x;
    const auto g_thr_ix = thr_ix + blockIdx.x * k_block_size;
    if (g_thr_ix >= rowct) return;
    const uint32_t grid_size = k_block_size * gridDim.x;
    __shared__ double _sh_best_score[k_block_size];
    __shared__ t_params_vec _sh_best_params_ix[k_block_size];
    _sh_best_score[thr_ix] = DBL_MAX;
    for (uint32_t rowix = g_thr_ix; rowix < rowct; rowix += grid_size) {
        double score = 0;
        {
            double recon_preds[C_emo_test_len];
            for (uint8_t j = 0; j < C_emo_max_j; ++j) {
                memset(recon_preds, 0, C_emo_test_len * sizeof(double));
                const auto elto = C_emo_tune_min_validation_window + (C_emo_max_j - j - 1) * C_emo_slide_skip;
                const auto j_EMO_TEST_LEN = j * C_emo_test_len;
                for (uint16_t colix = 0; colix < colct; ++colix)
                    for (uint16_t el = 0; el < elto; ++el)
                        recon_preds[el] += p_params_preds[colct * p_combinations[colix * rowct + rowix] + colix].predictions[j][el];
                for (uint16_t el = 0; el < elto; ++el)
                    score += abs(_CUSIGN(_CUSIGN(recon_preds[el] - recon_last_knowns[j_EMO_TEST_LEN + el]) - recon_signs[j_EMO_TEST_LEN + el]));
            }
        }
        if (score < _sh_best_score[thr_ix]) {
            _sh_best_score[thr_ix] = score;
            for (uint16_t colix = 0; colix < colct; ++colix)
                _sh_best_params_ix[thr_ix][colix] = p_params_preds[colct * p_combinations[colix * rowct + rowix] + colix].params_ix;
        }
    }
    __syncthreads();

    /*
    for (uint32_t stride = blockDim.x / 2; stride > 32; stride >>= 1) { // /= 2) { // uniform
        if (thr_ix < stride) {
            if (_sh_best_score[thr_ix + stride] < _sh_best_score[thr_ix]) {
                _sh_best_score[thr_ix] = _sh_best_score[thr_ix + stride];
                for (uint16_t colix = 0; colix < colct; ++colix)
                    _sh_best_params_ix[thr_ix][colix] = _sh_best_params_ix[thr_ix + stride][colix];
            }
        }
        __syncthreads();
    }
    */

#define WARP_SYNC(this_block_size) { \
        constexpr unsigned __stride = this_block_size / 2;                                                          \
        if (k_block_size >= this_block_size) {                                                                      \
            if (thr_ix < __stride) {                                                                                \
                if (_sh_best_score[thr_ix + __stride] < _sh_best_score[thr_ix]) {                                   \
                    _sh_best_score[thr_ix] = _sh_best_score[thr_ix + __stride];                                     \
                    memcpy(_sh_best_params_ix[thr_ix], _sh_best_params_ix[thr_ix + __stride], colct);               \
                }                                                                                                   \
            }                                                                                                       \
            __syncthreads();                                                                                        \
        }                                                                                                           \
    }

    WARP_SYNC(1024);
    WARP_SYNC(512);
    WARP_SYNC(256);
    WARP_SYNC(128);

    if (thr_ix < 32) warp_reduce<k_block_size>(colct, _sh_best_score, _sh_best_params_ix, thr_ix);

    if (!thr_ix) {
        if (_sh_best_score[0] < *p_best_score) {
            memcpy(best_param_ixs, _sh_best_params_ix[thr_ix], colct);
            *p_best_score = _sh_best_score[thr_ix];
            // printf("cu_recombine_parameters: Final score %f = %f\n", *p_best_score, _sh_best_score[0]);
        }
    }
}


void
recombine_parameters(
        const uint32_t rows_ct,
        const uint32_t colct,
        const uint8_t *combos,
        const t_param_preds_cu *params_preds,
        double *p_best_score,
        uint8_t *best_params_ixs)
{
    constexpr uint32_t recon_test_len = C_emo_max_j * C_emo_test_len;
    constexpr uint32_t recon_test_size = recon_test_len * sizeof(double);
    const dtype(rows_ct) adj_rows_ct = rows_ct - rows_ct % CUDA_BLOCK_SIZE;
    if (rows_ct != adj_rows_ct) LOG4_ERROR("Input rows count " << rows_ct << ", differs from adjusted " << adj_rows_ct);
    double recon_last_knowns[recon_test_len], recon_signs[recon_test_len];
    memset(recon_last_knowns, 0, recon_test_size);
    memset(recon_signs, 0, recon_test_size);
#pragma omp unroll
    for (uint8_t j = 0; j < uint8_t(C_emo_max_j); ++j) {
#pragma omp unroll
        for (uint16_t el = 0; el < uint16_t(C_emo_tune_min_validation_window + (C_emo_max_j - j - 1) * C_emo_slide_skip); ++el) {
            double recon_labels = 0;
#pragma omp unroll
            for (uint16_t colix = 0; colix < colct; ++colix) {
                recon_labels += params_preds[colix].labels[j][el];
                recon_last_knowns[C_emo_test_len * j + el] += params_preds[colix].last_knowns[j][el];
            }
            recon_signs[C_emo_test_len * j + el] = _CUSIGN(recon_labels - recon_last_knowns[C_emo_test_len * j + el]);
        }
    }

    const common::fat_gpu_context ctx;
    cudaSetDevice(ctx.phy_id());
    cudaStream_t strm;
    cu_errchk(cudaStreamCreate(&strm));
    double *p_best_score_gpu;
    cu_errchk(cudaMallocAsync((void **) &p_best_score_gpu, sizeof(double), strm));
    uint8_t *combos_gpu;
    cu_errchk(cudaMallocAsync((void **) &combos_gpu, adj_rows_ct * colct, strm));
    cu_errchk(cudaMemcpyAsync(combos_gpu, combos, adj_rows_ct * colct, cudaMemcpyHostToDevice, strm));
    t_param_preds_cu *params_preds_gpu;
    const uint32_t param_preds_size = sizeof(t_param_preds_cu) * colct * common::C_tune_keep_preds;
    cu_errchk(cudaMallocAsync((void **) &params_preds_gpu, param_preds_size, strm));
    cu_errchk(cudaMemcpyAsync(params_preds_gpu, params_preds, param_preds_size, cudaMemcpyHostToDevice, strm));
    uint8_t *best_param_ixs_gpu;
    cu_errchk(cudaMallocAsync((void **) &best_param_ixs_gpu, colct, strm));
    *p_best_score = DBL_MAX;
    cu_errchk(cudaMemcpyAsync(p_best_score_gpu, p_best_score, sizeof(double), cudaMemcpyHostToDevice, strm));

    double *recon_last_knowns_gpu;
    double *recon_signs_gpu;
    cu_errchk(cudaMallocAsync((void **) &recon_last_knowns_gpu, recon_test_size, strm));
    cu_errchk(cudaMallocAsync((void **) &recon_signs_gpu, recon_test_size, strm));
    cu_errchk(cudaMemcpyAsync(recon_last_knowns_gpu, recon_last_knowns, recon_test_size, cudaMemcpyHostToDevice, strm));
    cu_errchk(cudaMemcpyAsync(recon_signs_gpu, recon_signs, recon_test_size, cudaMemcpyHostToDevice, strm));

    std::string log_str(0xFF, '\0');
    snprintf(log_str.data(), 0xFF, "blocks %u, threads per block %u\n", CUDA_THREADS_BLOCKS(rows_ct));
    LOG4_DEBUG("Calling cu_recombine_parameters() with " << log_str);
    cu_recombine_parameters<CUDA_BLOCK_SIZE><<<CUDA_THREADS_BLOCKS(adj_rows_ct), 0, strm>>>
            (colct, rows_ct, combos_gpu, params_preds_gpu, best_param_ixs_gpu, p_best_score_gpu, recon_signs_gpu, recon_last_knowns_gpu);
    log_str.clear();

    cu_errchk(cudaMemcpyAsync(best_params_ixs, best_param_ixs_gpu, colct, cudaMemcpyDeviceToHost, strm));
    cu_errchk(cudaMemcpyAsync(p_best_score, p_best_score_gpu, sizeof(double), cudaMemcpyDeviceToHost, strm));
    cu_errchk(cudaFreeAsync(best_param_ixs_gpu, strm));
    cu_errchk(cudaFreeAsync(params_preds_gpu, strm));
    cu_errchk(cudaFreeAsync(combos_gpu, strm));
    cu_errchk(cudaFreeAsync(p_best_score_gpu, strm));
    cu_errchk(cudaFreeAsync(recon_last_knowns_gpu, strm));
    cu_errchk(cudaFreeAsync(recon_signs_gpu, strm));
    cu_errchk(cudaDeviceSynchronize());
    cu_errchk(cudaStreamDestroy(strm));
}

}