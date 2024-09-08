//
// Created by zarko on 7/4/24.
//
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <cublas.h>
#include "common/cuda_util.cuh"
#include "align_features.cuh"
#include "cuda_path.hpp"

namespace svr {

#ifdef INTEGRATION_TEST
constexpr unsigned C_integration_test_validation_window = 460;
#else
constexpr unsigned C_integration_test_validation_window = 0;
#endif

constexpr unsigned C_tuning_window = common::C_default_kernel_max_chunk_len + C_test_len;
constexpr unsigned C_active_window = C_tuning_window + C_integration_test_validation_window;
constexpr unsigned C_validate_cutoff = 1000; // C_tuning_window;

__device__ inline double vec_dist(CRPTR(double) mean_L, CRPTR(double) features, const unsigned n_rows, const float st, const unsigned sh)
{
    double res = 0;
    const auto validate_rows = n_rows - sh - C_integration_test_validation_window;
UNROLL()
    for (unsigned r = 0; r < validate_rows; ++r) res += abs(mean_L[r] - stretch_ix(features, r, n_rows, st, 1));
    return res / validate_rows;
}

// Use only when stretch coefficient below or eq 1
__device__ inline double vec_dist_stretch(CRPTR(double) mean_L, CRPTR(double) features, const unsigned n_rows, const float st, const unsigned sh, const float sk)
{
    double res = 0;
    const auto validate_rows = n_rows - sh - C_integration_test_validation_window;
UNROLL()
    for (unsigned r = validate_rows - C_validate_cutoff; r < validate_rows; ++r)
        res += fabs(mean_L[r] - features[STRETCHSKIP_(r)]);
    return res / validate_rows;
}

__global__ void cu_align_features(
        CRPTR(double) features, CRPTR(double) mean_L,
        double *__restrict__ const scores, float *__restrict__ const stretches, unsigned *__restrict__ const shifts, float *__restrict__ const skips,
        const unsigned n_rows, const unsigned n_cols)
{
    // const unsigned shift_limit = n_rows * C_shift_limit;
    const unsigned shift_limit = n_rows - C_active_window;
    CU_STRIDED_FOR_i(n_cols) {
        scores[i] = std::numeric_limits<dtype(*scores) >::infinity();
        CPTR(double) features_col = features + n_rows * i;
UNROLL()
        for (unsigned sh = 0; sh < shift_limit; sh += umax(1, C_shift_inc_mul * sh)) { // TODO Unroll loop into an array supplied at kernel launch
            const unsigned shift_limit_sh = shift_limit - sh;
            // 3040 - 1 = 3039
            CPTR(double) mean_L_sh = mean_L + shift_limit_sh;
UNROLL()
            for (float st = 1; st > C_stretch_limit; st *= C_stretch_multiplier) {
UNROLL()
                for (float sk = 1; sk < C_skip_limit; sk *= C_skip_multiplier) {
                    const auto dist = vec_dist_stretch(mean_L_sh, features_col, n_rows, st, shift_limit_sh, sk);
                    if (dist >= scores[i]) continue;
                    scores[i] = dist;
                    shifts[i] = shift_limit_sh;
                    stretches[i] = st;
                    skips[i] = sk;
                }
            }
        }
    }
}

void align_features(
        CPTR(double) p_features, CPTR(double) p_mean_L, double *const p_scores, float *const p_stretches,
        unsigned *const p_shifts, float *const p_skips, const unsigned n_rows, const unsigned n_cols)
{
    common::gpu_context ctx;
    cu_errchk(cudaSetDevice(ctx.phy_id()));
    cudaStream_t custream;
    cu_errchk(cudaStreamCreateWithFlags(&custream, cudaStreamNonBlocking));
    const auto d_features = cumallocopy(p_features, n_rows * n_cols, cudaMemcpyHostToDevice, custream);
    const auto d_mean_L = cumallocopy(p_mean_L, n_rows, cudaMemcpyHostToDevice, custream);
    double *d_scores;
    cu_errchk(cudaMallocAsync((void **) &d_scores, n_cols * sizeof(double), custream));
    float *d_stretches, *d_skips;
    const auto cols_size_float = n_cols * sizeof(float);
    cu_errchk(cudaMallocAsync((void **) &d_stretches, cols_size_float, custream));
    cu_errchk(cudaMallocAsync((void **) &d_skips, cols_size_float, custream));
    unsigned *d_shifts;
    cu_errchk(cudaMallocAsync((void **) &d_shifts, n_cols * sizeof(unsigned), custream));
    cu_align_features<<<CU_BLOCKS_THREADS(n_cols), 0, custream>>>(d_features, d_mean_L, d_scores, d_stretches, d_shifts, d_skips, n_rows, n_cols);
    cu_errchk(cudaFreeAsync(d_features, custream));
    cu_errchk(cudaFreeAsync(d_mean_L, custream));
    cufreecopy(p_scores, d_scores, custream, n_cols);
    cufreecopy(p_stretches, d_stretches, custream, n_cols);
    cufreecopy(p_skips, d_skips, custream, n_cols);
    cufreecopy(p_shifts, d_shifts, custream, n_cols);
    cu_errchk(cudaStreamSynchronize(custream));
    cu_errchk(cudaStreamDestroy(custream));
}


template<typename T> __device__ inline unsigned before_bound(CRPTR(T) cbegin, CRPTR(T) cend, const T value)
{
    auto res = thrust::lower_bound(thrust::seq, cbegin, cend, value); // TODO Test with thrust::device
    while (*res > value && res > cbegin) --res;
    return res - cbegin;
}

__global__ void G_quantise_features(
        CRPTR(double) d_decon_F, CRPTR(unsigned) d_times_F, CRPTR(unsigned) d_times_last_F, CRPTR(unsigned) d_ixs_last_F,
        const unsigned n_rows, const unsigned quantise, const unsigned start_row, const unsigned n_cols_quantise_, const unsigned n_cols_,
        double *__restrict__ const features /* zeroed out before */)
{
    CU_STRIDED_FOR_i(n_rows) {
        const auto in_row = start_row + i;
        const auto end_time = d_times_last_F[in_row];
        const auto start_time = end_time - n_cols_quantise_;
        const auto iter_end = d_ixs_last_F[in_row];
        auto iter_F = iter_end > n_cols_quantise_ ? iter_end - n_cols_quantise_ : 0;
        iter_F += before_bound(d_times_F + iter_F, d_times_F + iter_end, start_time);
        auto last_price = d_decon_F[iter_F];
UNROLL()
        for (auto time_iter = start_time; time_iter < end_time; ++time_iter) {
            while (iter_F < iter_end && d_times_F[iter_F] <= time_iter) {
                last_price = d_decon_F[iter_F];
                ++iter_F;
            }
            features[i + ((time_iter - start_time) / quantise) * n_rows] += last_price;
        }
#ifdef EMO_DIFF
UNROLL()
        for (auto j = n_cols_ - 1; j > 0; --j) {
            const auto out_i = i + j * n_rows;
            features[out_i] = (features[out_i] - features[out_i - n_rows]) / double(quantise);
        }
#else
        for (auto j = n_cols_ - 1; j >= 0; --j) features[i + j * n_rows] /= double(quantise);
#endif
    }
}

void quantise_features(CPTR(double) decon, CPTR(unsigned) times_F, CPTR(unsigned) times_last_F, CPTR(unsigned) ixs_last_F,
                       const unsigned start_row, const unsigned n_rows_chunk, const unsigned n_rows, const unsigned n_feat_rows,
                       const unsigned level, const unsigned n_cols_coef_, const unsigned n_cols_coef, const unsigned quantise, double *const p_features)
{
    common::gpu_context ctx;
    cu_errchk(cudaSetDevice(ctx.phy_id()));
    cudaStream_t custream;
    cu_errchk(cudaStreamCreate(&custream));
    const auto end_row = start_row + n_rows_chunk - 1;
    auto d_features = cucalloc<double>(custream, n_rows_chunk * n_cols_coef_);
    const auto d_decon_F = cumallocopy(decon + n_feat_rows * level, ixs_last_F[end_row], cudaMemcpyHostToDevice, custream);
    const auto d_times_F = cumallocopy(times_F, ixs_last_F[end_row], cudaMemcpyHostToDevice, custream);
    const auto d_times_last_F = cumallocopy(times_last_F, end_row + 1, cudaMemcpyHostToDevice, custream);
    const auto d_ixs_last_F = cumallocopy(ixs_last_F, end_row + 1, cudaMemcpyHostToDevice, custream);
    G_quantise_features<<<CU_BLOCKS_THREADS(clamp_n(n_rows_chunk)), 0, custream>>>(
            d_decon_F, d_times_F, d_times_last_F, d_ixs_last_F, n_rows_chunk, quantise, start_row, n_cols_coef_ * quantise, n_cols_coef_, d_features);
#ifndef NDEBUG
    cu_errchk(cudaPeekAtLastError());
#endif
    cu_errchk(cudaFreeAsync(d_decon_F, custream));
    cu_errchk(cudaFreeAsync(d_times_F, custream));
    cu_errchk(cudaFreeAsync(d_times_last_F, custream));
    cu_errchk(cudaFreeAsync(d_ixs_last_F, custream));
#ifdef EMO_DIFF
    cb_errchk(cublasGetMatrixAsync(n_rows_chunk, n_cols_coef, sizeof(double), d_features + n_rows_chunk, n_rows_chunk, p_features + start_row, n_rows, custream));
#else
    cb_errchk(cublasGetMatrixAsync(n_rows_chunk, n_cols_coef, sizeof(double), d_features, n_rows_chunk, p_features + start_row, n_rows, custream));
#endif
    cu_errchk(cudaFreeAsync(d_features, custream));
    cu_errchk(cudaStreamSynchronize(custream));
    cu_errchk(cudaStreamDestroy(custream));
}

}
