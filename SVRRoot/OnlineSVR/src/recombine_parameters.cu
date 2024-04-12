#include <cuda_runtime_api.h>
#include "common/cuda_util.cuh"
#include "common/gpu_handler.hpp"
#include "recombine_parameters.cuh"

namespace svr {


template<unsigned k_block_size> __device__ void
warp_reduce(
        volatile double *_sh_best_score,
        volatile t_params_vec *_sh_best_params_ix,
        const uint32_t thr_ix,
        const uint8_t colct)
{
#define WARP_NOSYNC(this_block_size, thr_ix_off)                                                                                    \
        if (k_block_size >= this_block_size) {                                                                                      \
            if (_sh_best_score[thr_ix + thr_ix_off] < _sh_best_score[thr_ix]) {                                                     \
                _sh_best_score[thr_ix] = _sh_best_score[thr_ix + thr_ix_off];                                                       \
                for (uint16_t colix = 0; colix < colct; ++colix)                                                                    \
                    _sh_best_params_ix[thr_ix][colix] = _sh_best_params_ix[thr_ix + thr_ix_off][colix];                             \
            }                                                                                                                       \
        }

    WARP_NOSYNC(64, 32);
    WARP_NOSYNC(32, 16);
    WARP_NOSYNC(16, 8);
    WARP_NOSYNC(8, 4);
    WARP_NOSYNC(4, 2);
    WARP_NOSYNC(2, 1);
}


template<unsigned k_block_size> __global__ void
cu_recombine_parameters(
        const uint8_t colct,
        const uint32_t rowct,
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
                memset(&(recon_preds[0]), 0, C_emo_test_len * sizeof(double));
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

#define WARP_SYNC(this_block_size, stride)                                                                      \
    if (k_block_size >= this_block_size) {                                                                      \
        if (thr_ix < stride) {                                                                                  \
            if (_sh_best_score[thr_ix + stride] < _sh_best_score[thr_ix]) {                                     \
                _sh_best_score[thr_ix] = _sh_best_score[thr_ix + stride];                                       \
                memcpy(&(_sh_best_params_ix[thr_ix][0]), &(_sh_best_params_ix[thr_ix + stride][0]), colct);     \
            }                                                                                                   \
        }                                                                                                       \
        __syncthreads();                                                                                        \
    }

    //WARP_SYNC(512, 256);
    WARP_SYNC(256, 128);
    WARP_SYNC(128, 64);

    if (thr_ix < 32)
        warp_reduce<k_block_size>(&(_sh_best_score[0]), (t_params_vec *) &(_sh_best_params_ix[0][0]), thr_ix, colct);

    if (!thr_ix) {
        if (_sh_best_score[0] < *p_best_score) {
            memcpy((void *)&(best_param_ixs[0]), &(_sh_best_params_ix[thr_ix][0]), colct);
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
    constexpr unsigned recon_test_size = sizeof(double) * C_emo_max_j * C_emo_test_len;
    const dtype(rows_ct) adj_rows_ct = rows_ct - rows_ct % CUDA_BLOCK_SIZE;
    if (rows_ct != adj_rows_ct) LOG4_ERROR("Input rows count " << rows_ct << ", differs from adjusted " << adj_rows_ct);
    double recon_last_knowns[C_emo_max_j * C_emo_test_len], recon_signs[C_emo_max_j * C_emo_test_len];
    memset(recon_last_knowns, 0, recon_test_size);
    memset(recon_signs, 0, recon_test_size);
#pragma omp parallel for simd // collapse(2)
    for (uint8_t j = 0; j < uint8_t(C_emo_max_j); ++j) {
        for (uint16_t el = 0; el < uint16_t(C_emo_tune_min_validation_window + (C_emo_max_j - j - 1) * C_emo_slide_skip); ++el) {
            double recon_labels = 0;
            for (uint16_t colix = 0; colix < colct; ++colix) {
                recon_labels += params_preds[colix].labels[j][el];
                recon_last_knowns[C_emo_test_len * j + el] += params_preds[colix].last_knowns[j][el];
            }
            recon_signs[C_emo_test_len * j + el] = _CUSIGN(recon_labels - recon_last_knowns[C_emo_test_len * j + el]);
        }
    }

    const common::gpu_context ctx;
    cudaSetDevice(ctx.phy_id());
    double *p_best_score_gpu = nullptr;
    cu_errchk(cudaMalloc(&p_best_score_gpu, sizeof(double)));
    uint8_t *combos_gpu;
    cu_errchk(cudaMalloc(&combos_gpu, adj_rows_ct * colct));
    cu_errchk(cudaMemcpy(combos_gpu, combos, adj_rows_ct * colct, cudaMemcpyHostToDevice));
    t_param_preds_cu *params_preds_gpu;
    cu_errchk(cudaMalloc(&params_preds_gpu, sizeof(t_param_preds_cu) * colct * TUNE_KEEP_PREDS));
    cu_errchk(cudaMemcpy(params_preds_gpu, params_preds, sizeof(t_param_preds_cu) * colct * TUNE_KEEP_PREDS, cudaMemcpyHostToDevice));
    uint8_t *best_param_ixs_gpu;
    cu_errchk(cudaMalloc(&best_param_ixs_gpu, colct));
    *p_best_score = DBL_MAX;
    cu_errchk(cudaMemcpy(p_best_score_gpu, p_best_score, sizeof(double), cudaMemcpyHostToDevice));

    double *recon_last_knowns_gpu;
    double *recon_signs_gpu;
    cu_errchk(cudaMalloc(&recon_last_knowns_gpu, recon_test_size));
    cu_errchk(cudaMalloc(&recon_signs_gpu, recon_test_size));
    cu_errchk(cudaMemcpy(recon_last_knowns_gpu, recon_last_knowns, recon_test_size, cudaMemcpyHostToDevice));
    cu_errchk(cudaMemcpy(recon_signs_gpu, recon_signs, recon_test_size, cudaMemcpyHostToDevice));

    printf("cu_recombine_parameters: Blocks %u, threads per block %u\n", CUDA_THREADS_BLOCKS(rows_ct));
    cu_recombine_parameters<CUDA_BLOCK_SIZE><<<CUDA_THREADS_BLOCKS(adj_rows_ct)>>>
            (colct, adj_rows_ct, combos_gpu, params_preds_gpu, best_param_ixs_gpu, p_best_score_gpu, recon_signs_gpu, recon_last_knowns_gpu);
    cu_errchk(cudaMemcpy(best_params_ixs, best_param_ixs_gpu, colct, cudaMemcpyDeviceToHost));
    cu_errchk(cudaMemcpy(p_best_score, p_best_score_gpu, sizeof(double), cudaMemcpyDeviceToHost));
    cu_errchk(cudaFree(best_param_ixs_gpu));
    cu_errchk(cudaFree(params_preds_gpu));
    cu_errchk(cudaFree(combos_gpu));
    cu_errchk(cudaFree(p_best_score_gpu));
    cu_errchk(cudaFree(recon_last_knowns_gpu));
    cu_errchk(cudaFree(recon_signs_gpu));
    cu_errchk(cudaDeviceSynchronize());
}


}