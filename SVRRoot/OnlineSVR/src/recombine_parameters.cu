#include <cuda_runtime_api.h>
#include <cfloat>
#include "common/cuda_util.cuh"
#include "common/gpu_handler.tpp"
#include "util/math_utils.hpp"
#include "recombine_parameters.cuh"

namespace svr {

template<const unsigned k_block_size> __device__ inline void
warp_reduce(
        const uint8_t colct,
        volatile double *_sh_best_score,
        volatile t_params_vec *_sh_best_params_ix,
        const uint32_t thr_ix)
{
#define _WARP_NOSYNC1(this_block_size, block_size) {                                                                                  \
        constexpr unsigned thr_ix_off = this_block_size / 2;                                                                        \
        if (block_size >= this_block_size) {                                                         \
            if (_sh_best_score[thr_ix + thr_ix_off] < _sh_best_score[thr_ix]) {                                                     \
                _sh_best_score[thr_ix] = _sh_best_score[thr_ix + thr_ix_off];                                                       \
_Pragma("unroll")                                                                                                                   \
                for (size_t i = 0; i < colct; ++i)                                                                                  \
                    _sh_best_params_ix[thr_ix][i] = _sh_best_params_ix[thr_ix + thr_ix_off][i];                                     \
            }                                                                                                                       \
        }                                                                                                                           \
    }

    _WARP_NOSYNC1(64, k_block_size);
    _WARP_NOSYNC1(32, k_block_size);
    _WARP_NOSYNC1(16, k_block_size);
    _WARP_NOSYNC1(8, k_block_size);
    _WARP_NOSYNC1(4, k_block_size);
    _WARP_NOSYNC1(2, k_block_size);
}

template<const unsigned k_block_size> __global__ void cu_recombine_parameters(
        const uint8_t colct, const uint32_t rowct,
        const uint8_t *__restrict__ p_combinations, // Len of rowct * col_ct
        const t_param_preds_cu *__restrict__ p_params_preds, // Len of col_ct * C_tune_keep_preds
        uint8_t *__restrict__ best_param_ixs, // Len of gridDim.x * col_ct
        double *__restrict__ p_best_score,
        const double *__restrict__ recon_signs,
        const double *__restrict__ recon_last_knowns
)
{
    const auto g_tid = tid + blockIdx.x * k_block_size;
    if (g_tid >= rowct) return;
    const auto grid_size = k_block_size * gridDim.x;
    __shared__ double _sh_best_score[k_block_size];
    __shared__ t_params_vec _sh_best_params_ix[k_block_size];
    _sh_best_score[tid] = std::numeric_limits<double>::max();
    uint32_t best_rowix = 0;
// #pragma unroll
    for (auto rowix = g_tid; rowix < rowct; rowix += grid_size) {
        double score = 0;
        double recon_preds[C_test_len];
#pragma unroll C_max_j
        for (uint8_t j = 0; j < C_max_j; ++j) {
            memset(recon_preds, 0, C_test_len * sizeof(double));
            const auto elto = C_tune_min_validation_window + (C_max_j - j - 1) * C_slide_skip;
            const auto j_emo_test_len = j * C_test_len;
#ifdef PRODUCTION_BUILD
#pragma unroll
#endif
            for (uint16_t colix = 0; colix < colct; ++colix)
                for (uint16_t el = 0; el < elto; ++el)
                    recon_preds[el] += p_params_preds[colct * p_combinations[colix * rowct + rowix] + colix].predictions[j][el];
#ifdef PRODUCTION_BUILD
#pragma unroll
#endif
            for (uint16_t el = 0; el < elto; ++el)
                score += fabs(_SIGN(_SIGN(recon_preds[el] - recon_last_knowns[j_emo_test_len + el]) - recon_signs[j_emo_test_len + el]));
        }
        if (_sh_best_score[tid] > score) {
            _sh_best_score[tid] = score;
            best_rowix = rowix;
        }
    }
#ifdef PRODUCTION_BUILD
#pragma unroll
#endif
    for (uint16_t colix = 0; colix < colct; ++colix)
        _sh_best_params_ix[tid][colix] = p_params_preds[colct * p_combinations[colix * rowct + best_rowix] + colix].params_ix;
    __syncthreads();

#define WARP_SYNC(this_block_size, block_size) { \
        constexpr unsigned __stride = this_block_size / 2;                                                          \
        if (block_size >= this_block_size) {     \
            const auto tid_stride = tid + __stride;                                     \
            if (tid < __stride) {                                                                                \
                if (_sh_best_score[tid_stride] < _sh_best_score[tid]) {                                   \
                    _sh_best_score[tid] = _sh_best_score[tid_stride];                                     \
                    memcpy(_sh_best_params_ix[tid], _sh_best_params_ix[tid_stride], colct);               \
                }                                                                                                   \
            }                                                                                                       \
            __syncthreads();                                                                                        \
        }                                                                                                           \
    }

    WARP_SYNC(1024, k_block_size);
    WARP_SYNC(512, k_block_size);
    WARP_SYNC(256, k_block_size);
    WARP_SYNC(128, k_block_size);

    if (tid >= 32) return;
    warp_reduce<k_block_size>(colct, _sh_best_score, _sh_best_params_ix, tid);
    if (tid) return;
    memcpy(best_param_ixs + blockIdx.x * colct, _sh_best_params_ix[tid], colct);
    p_best_score[blockIdx.x] = _sh_best_score[tid];
}


__device__ inline void warp_reduce(
        const uint32_t caller_block_size,
        const uint32_t rowct,
        const uint8_t colct,
        volatile double *_sh_best_score,
        volatile t_params_vec *_sh_best_params_ix,
        const uint32_t thr_ix)
{
#define _WARP_NOSYNC2(this_block_size, block_size) {                                                                                  \
        constexpr unsigned thr_ix_off = this_block_size / 2;                                                                        \
        if (block_size >= this_block_size && thr_ix + thr_ix_off < rowct) {                                                         \
            if (_sh_best_score[thr_ix + thr_ix_off] < _sh_best_score[thr_ix]) {                                                     \
                _sh_best_score[thr_ix] = _sh_best_score[thr_ix + thr_ix_off];                                                       \
_Pragma("unroll")                                                                                                                   \
                for (size_t i = 0; i < colct; ++i)                                                                                  \
                    _sh_best_params_ix[thr_ix][i] = _sh_best_params_ix[thr_ix + thr_ix_off][i];                                     \
            }                                                                                                                       \
        }                                                                                                                           \
    }

    _WARP_NOSYNC2(64, caller_block_size);
    _WARP_NOSYNC2(32, caller_block_size);
    _WARP_NOSYNC2(16, caller_block_size);
    _WARP_NOSYNC2(8, caller_block_size);
    _WARP_NOSYNC2(4, caller_block_size);
    _WARP_NOSYNC2(2, caller_block_size);
}

// When rowct < blocksize
__global__ void cu_recombine_parameters(
        const uint8_t colct, const uint32_t rowct,
        const t_param_preds_cu *__restrict__ p_params_preds, // Len of col_ct * C_tune_keep_preds
        uint8_t *__restrict__ best_param_ixs, // Len of rowct * col_ct
        double *__restrict__ p_best_score,
        const double *__restrict__ recon_signs,
        const double *__restrict__ recon_last_knowns
)
{
    if (tid >= rowct) return;
    extern __shared__ double _sh_best_score[];
    extern __shared__ t_params_vec _sh_best_params_ix[];
    _sh_best_score[tid] = std::numeric_limits<double>::max();
    double score = 0;
    double recon_preds[C_test_len];
#pragma unroll C_max_j
    for (uint8_t j = 0; j < C_max_j; ++j) {
        memset(recon_preds, 0, C_test_len * sizeof(double));
        const auto elto = C_tune_min_validation_window + (C_max_j - j - 1) * C_slide_skip;
        const auto j_emo_test_len = j * C_test_len;
#ifdef PRODUCTION_BUILD
#pragma unroll
#endif
        for (uint16_t colix = 0; colix < colct; ++colix)
            for (uint16_t el = 0; el < elto; ++el)
                recon_preds[el] += p_params_preds[colct * best_param_ixs[colix * rowct + tid] + colix].predictions[j][el];
#ifdef PRODUCTION_BUILD
#pragma unroll
#endif
        for (uint16_t el = 0; el < elto; ++el)
            score += fabs(_SIGN(_SIGN(recon_preds[el] - recon_last_knowns[j_emo_test_len + el]) - recon_signs[j_emo_test_len + el]));
    }
    if (_sh_best_score[tid] > score) {
        _sh_best_score[tid] = score;
#ifdef PRODUCTION_BUILD
#pragma unroll
#endif
        for (uint16_t colix = 0; colix < colct; ++colix)
            _sh_best_params_ix[tid][colix] = p_params_preds[colct * best_param_ixs[colix * rowct + tid] + colix].params_ix;
    }

#define WARP_SYNC2(this_block_size, block_size) { \
        constexpr unsigned __stride = this_block_size / 2;                                                \
        if (block_size >= this_block_size) {                                                              \
            const auto tid_stride = tid + __stride;                                                       \
            if (tid < __stride && tid_stride < rowct) {                                                   \
                if (_sh_best_score[tid_stride] < _sh_best_score[tid]) {                                   \
                    _sh_best_score[tid] = _sh_best_score[tid_stride];                                     \
                    memcpy(_sh_best_params_ix[tid], _sh_best_params_ix[tid_stride], colct);               \
                }                                                                                         \
            }                                                                                             \
            __syncthreads();                                                                              \
        }                                                                                                 \
    }

    __syncthreads();
    WARP_SYNC2(1024, blockDim.x);
    WARP_SYNC2(512, blockDim.x);
    WARP_SYNC2(256, blockDim.x);
    WARP_SYNC2(128, blockDim.x);

    if (tid >= 32) return;
    warp_reduce(blockDim.x, rowct, colct, _sh_best_score, _sh_best_params_ix, tid);
    if (tid) return;
    if (*p_best_score > _sh_best_score[0]) {
        *p_best_score = *_sh_best_score;
        memcpy(best_param_ixs, *_sh_best_params_ix, colct);
    }
}


void
recombine_parameters(
        const uint32_t rowct,
        const uint32_t colct,
        const uint8_t *combos,
        const t_param_preds_cu *params_preds,
        double *p_best_score,
        uint8_t *best_params_ixs)
{
    constexpr uint32_t recon_test_len = C_max_j * C_test_len;
    constexpr uint32_t recon_test_size = recon_test_len * sizeof(double);
    double recon_last_knowns[recon_test_len], recon_signs[recon_test_len];
    memset(recon_last_knowns, 0, recon_test_size);
    memset(recon_signs, 0, recon_test_size);
#pragma unroll C_max_j
    for (uint8_t j = 0; j < uint8_t(C_max_j); ++j) {
#pragma omp unroll
        for (uint16_t el = 0; el < uint16_t(C_tune_min_validation_window + (C_max_j - j - 1) * C_slide_skip); ++el) {
            double recon_labels = 0;
#pragma omp unroll
            for (uint16_t colix = 0; colix < colct; ++colix) {
                recon_labels += params_preds[colix].labels[j][el];
                recon_last_knowns[C_test_len * j + el] += params_preds[colix].last_knowns[j][el];
            }
            recon_signs[C_test_len * j + el] = _SIGN(recon_labels - recon_last_knowns[C_test_len * j + el]);
        }
    }

    const auto clamped_n = clamp_n(rowct);
    const auto blocks = CUDA_BLOCKS(clamped_n);
    const auto threads = CUDA_THREADS(clamped_n);

    const common::gpu_context ctx;
    cudaSetDevice(ctx.phy_id());
    cudaStream_t stm;
    cu_errchk(cudaStreamCreateWithFlags(&stm, cudaStreamNonBlocking));
    double *d_best_scores, *d_recon_last_knowns, *d_recon_signs;
    cu_errchk(cudaMallocAsync((void **) &d_best_scores, sizeof(double), stm));
    uint8_t *combos_gpu, *d_best_param_ixs;
    cu_errchk(cudaMallocAsync((void **) &combos_gpu, rowct * colct, stm));
    cu_errchk(cudaMemcpyAsync(combos_gpu, combos, rowct * colct, cudaMemcpyHostToDevice, stm));
    t_param_preds_cu *d_params_preds;
    const uint32_t param_preds_size = sizeof(t_param_preds_cu) * colct * common::C_tune_keep_preds;
    cu_errchk(cudaMallocAsync((void **) &d_params_preds, param_preds_size, stm));
    cu_errchk(cudaMemcpyAsync(d_params_preds, params_preds, param_preds_size, cudaMemcpyHostToDevice, stm));
    cu_errchk(cudaMallocAsync((void **) &d_best_param_ixs, colct * blocks, stm));
    cu_errchk(cudaMallocAsync((void **) &d_recon_last_knowns, recon_test_size, stm));
    cu_errchk(cudaMallocAsync((void **) &d_recon_signs, recon_test_size, stm));
    cu_errchk(cudaMemcpyAsync(d_recon_last_knowns, recon_last_knowns, recon_test_size, cudaMemcpyHostToDevice, stm));
    cu_errchk(cudaMemcpyAsync(d_recon_signs, recon_signs, recon_test_size, cudaMemcpyHostToDevice, stm));

    if (threads == common::C_cu_block_size) {
        cu_recombine_parameters<common::C_cu_block_size><<<blocks, common::C_cu_block_size, 0, stm>>>(
                colct, rowct, combos_gpu, d_params_preds, d_best_param_ixs, d_best_scores, d_recon_signs, d_recon_last_knowns);
        cu_recombine_parameters<<<1, blocks, blocks * sizeof(double) + blocks * sizeof(t_params_vec), stm>>>(
                colct, blocks, d_params_preds, d_best_param_ixs, d_best_scores, d_recon_signs, d_recon_last_knowns);
    } else
        cu_recombine_parameters<<<1, threads, threads * sizeof(double) + threads * sizeof(t_params_vec), stm>>>(
                colct, rowct, d_params_preds, d_best_param_ixs, d_best_scores, d_recon_signs, d_recon_last_knowns);

    cu_errchk(cudaMemcpyAsync(best_params_ixs, d_best_param_ixs, colct, cudaMemcpyDeviceToHost, stm));
    cu_errchk(cudaMemcpyAsync(p_best_score, d_best_scores, sizeof(double), cudaMemcpyDeviceToHost, stm));
    cu_errchk(cudaFreeAsync(d_best_param_ixs, stm));
    cu_errchk(cudaFreeAsync(d_params_preds, stm));
    cu_errchk(cudaFreeAsync(combos_gpu, stm));
    cu_errchk(cudaFreeAsync(d_best_scores, stm));
    cu_errchk(cudaFreeAsync(d_recon_last_knowns, stm));
    cu_errchk(cudaFreeAsync(d_recon_signs, stm));
    cu_errchk(cudaStreamSynchronize(stm));
    cu_errchk(cudaStreamDestroy(stm));
}

}