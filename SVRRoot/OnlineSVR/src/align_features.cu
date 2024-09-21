//
// Created by zarko on 7/4/24.
//
#include <thrust/binary_search.h>
#include <thrust/async/for_each.h>
#include <cublas.h>
#include "common/cuda_util.cuh"
#include "align_features.cuh"

namespace svr {


__device__ inline double vec_dist(CRPTR(double) mean_L, CRPTR(double) features, const unsigned n_rows, const float st, const unsigned sh)
{
    double res = 0;
    const auto validate_rows = n_rows - sh - C_integration_test_validation_window;
    UNROLL()
    for (unsigned r = 0; r < validate_rows; ++r) res += abs(mean_L[r] - stretch_ix(features, r, n_rows, st, 1));
    return res / validate_rows;
}


__device__ inline double vec_dist_stretch(CRPTR(double) labels, CRPTR(double) features, const unsigned n_rows, const float st /* < 1 */, const unsigned sh, const float sk)
{
    double res = 0;
    const auto validate_rows = n_rows - sh - C_integration_test_validation_window;
    UNROLL(C_validate_cutoff)
    for (unsigned r = validate_rows - C_validate_cutoff; r < validate_rows; ++r)
        res += fabs(labels[r] - features[STRETCHSKIP_(r)]);
    return res / validate_rows;
}

__global__ void cu_align_features(
        CRPTR(double) features, CRPTR(double) labels,
        RPTR(double) scores, float *__restrict__ const stretches, unsigned *__restrict__ const shifts, float *__restrict__ const skips,
        const unsigned n_rows, const unsigned n_cols)
{
    // const unsigned shift_limit = n_rows * C_shift_limit;
    const unsigned shift_limit = n_rows - C_active_window;
    CU_STRIDED_FOR_i(n_cols) {
        scores[i] = common::C_bad_validation;
        CPTR(double) features_col = features + n_rows * i;
        UNROLL()
        for (unsigned sh = 0; sh < shift_limit; sh += umax(1, C_shift_inc_mul * sh)) { // TODO Unroll loop into an array supplied at kernel launch
            const unsigned shift_limit_sh = shift_limit - sh;
            CPTR(double) mean_L_sh = labels + shift_limit_sh;
            UNROLL()
            for (float st = 1; st > C_stretch_limit; st *= C_stretch_multiplier) {
// UNROLL()
//                for (float sk = 1; sk < C_skip_limit; sk *= C_skip_multiplier) {
                const auto dist = vec_dist_stretch(mean_L_sh, features_col, n_rows, st, shift_limit_sh, 1);
                if (dist >= scores[i]) continue;
                scores[i] = dist;
                if (shifts) shifts[i] = shift_limit_sh;
                if (stretches) stretches[i] = st;
                // skips[i] = sk; // Skips degrade precision, retest and enable if necessary
//                }
            }
        }
    }
}


void align_features(
        CPTR(double) p_features, CPTR(double) labels, double *const p_scores, float *const p_stretches,
        unsigned *const p_shifts, float *const p_skips, const unsigned n_rows, const unsigned n_cols)
{
    CTX_CUSTREAM;
    const auto d_features = cumallocopy(p_features, n_rows * n_cols, cudaMemcpyHostToDevice, custream);
    const auto d_labels = cumallocopy(labels, n_rows, cudaMemcpyHostToDevice, custream);
    double *d_scores;
    cu_errchk(cudaMallocAsync((void **) &d_scores, n_cols * sizeof(double), custream));
    float *d_stretches, *d_skips;
    const auto cols_size_float = n_cols * sizeof(float);
    cu_errchk(cudaMallocAsync((void **) &d_stretches, cols_size_float, custream));
    cu_errchk(cudaMallocAsync((void **) &d_skips, cols_size_float, custream));
    unsigned *d_shifts;
    cu_errchk(cudaMallocAsync((void **) &d_shifts, n_cols * sizeof(unsigned), custream));
    cu_align_features<<<CU_BLOCKS_THREADS(n_cols), 0, custream>>>(d_features, d_labels, d_scores, d_stretches, d_shifts, d_skips, n_rows, n_cols);
    cu_errchk(cudaFreeAsync(d_features, custream));
    cu_errchk(cudaFreeAsync(d_labels, custream));
    cufreecopy(p_scores, d_scores, custream, n_cols);
    cufreecopy(p_stretches, d_stretches, custream, n_cols);
    cufreecopy(p_skips, d_skips, custream, n_cols);
    cufreecopy(p_shifts, d_shifts, custream, n_cols);
    cu_sync_destroy(custream);
}


template<typename T> __device__ inline int before_bound(CRPTR(T) cbegin, CRPTR(T) cend, const T value)
{
    auto res = thrust::lower_bound(thrust::seq, cbegin, cend, value); // TODO Test with thrust::device
    while (*res > value && res > cbegin) --res;
    return res - cbegin;
}


__global__ void G_quantise_features(
        RPTR(double) features /* zeroed out before */, CRPTR(double) d_decon_F, CRPTR(unsigned) d_times_F, CRPTR(t_feat_params) d_feat_params,
        const unsigned rows, const unsigned cols, const unsigned quantise, const unsigned interleave_quantise, const unsigned interleave_quantise_skip)
{
    CU_STRIDED_FOR_i(rows) {
        auto const d_feat_params_i = d_feat_params + i;
        const auto ix_end = d_feat_params_i->ix_end;
        auto ix_F = d_feat_params_i->ix_start;
        UNROLL(16)
        for (unsigned j = 0; j < cols; ++j) {
            auto last_price = d_decon_F[ix_F];
            const auto j_time_start = d_feat_params_i->time_start + j * interleave_quantise;
            auto const feat_i_j_rows = features + i + j * rows;
#ifdef EMO_DIFF
            auto prev_ix_F = ix_F - quantise;
            auto last_prev_price = d_decon_F[prev_ix_F];
            double prev_price = 0;
#endif
            for (auto time_iter = j_time_start; time_iter < j_time_start + quantise; ++time_iter) {
                for (; ix_F < ix_end && d_times_F[ix_F] <= time_iter; ++ix_F) last_price = d_decon_F[ix_F];
                *feat_i_j_rows += last_price;
#ifdef EMO_DIFF
                const auto prev_time_iter = time_iter - quantise;
                for (; prev_ix_F < ix_end && d_times_F[prev_ix_F] <= prev_time_iter; ++prev_ix_F) last_prev_price = d_decon_F[prev_ix_F];
                prev_price += last_prev_price;
#endif
            }
#ifdef EMO_DIFF
            *feat_i_j_rows = (*feat_i_j_rows - prev_price) / quantise;
            prev_ix_F += interleave_quantise_skip;
#else
            *feat_i_j_rows /= quantise;
#endif
            ix_F += interleave_quantise_skip;
        }
    }
}

__global__ void G_quantise_features(
        CRPTR(double) d_decon_F, CRPTR(unsigned) d_times_F, CRPTR(t_feat_params) d_feat_params,
        const unsigned n_rows, const unsigned quantise, const unsigned start_row, const unsigned n_cols_,
        RPTR(double) features /* zeroed out before */)
{
    CU_STRIDED_FOR_i(n_rows) {
        const auto d_feat_params_i = d_feat_params + start_row + i;
        auto ix_F = d_feat_params_i->ix_start;
        auto last_price = d_decon_F[ix_F];
        UNROLL(16)
        for (auto time_iter = d_feat_params_i->time_start; time_iter < d_feat_params_i->end_time; ++time_iter) {
            for (; ix_F < d_feat_params_i->ix_end && d_times_F[ix_F] <= time_iter; ++ix_F) last_price = d_decon_F[ix_F];
            features[i + ((time_iter - d_feat_params_i->time_start) / quantise) * n_rows] += last_price;
        }
        UNROLL(16)
#ifdef EMO_DIFF
        for (auto j = n_cols_ - 1; j > 0; --j) {
            const auto out_i = i + j * n_rows;
            features[out_i] = (features[out_i] - features[out_i - n_rows]) / quantise;
        }
#else
        for (auto j = n_cols_ - 1; j >= 0; --j) features[i + j * n_rows] /= quantise;
#endif
    }
}

void quantise_features(CPTR(double) decon, CPTR(unsigned) times_F, CPTR(t_feat_params) feat_params,
                       const unsigned start_row, const unsigned n_rows_chunk, const unsigned n_rows, const unsigned n_feat_rows,
                       const unsigned level, const unsigned n_cols_coef_, const unsigned n_cols_coef, const unsigned quantise, double *const p_features)
{
    CTX_CUSTREAM;
    const auto end_row = start_row + n_rows_chunk - 1;
    auto d_features = cucalloc<double>(custream, n_rows_chunk * n_cols_coef_);
    const auto d_decon_F = cumallocopy(decon + n_feat_rows * level, feat_params[end_row].ix_end, cudaMemcpyHostToDevice, custream);
    const auto d_times_F = cumallocopy(times_F, feat_params[end_row].ix_end, cudaMemcpyHostToDevice, custream);
    const auto d_feat_params = cumallocopy(feat_params, end_row + 1, cudaMemcpyHostToDevice, custream);
    G_quantise_features<<<CU_BLOCKS_THREADS(clamp_n(n_rows_chunk)), 0, custream>>>(
            d_decon_F, d_times_F, d_feat_params, n_rows_chunk, quantise, start_row, n_cols_coef_, d_features);
    cu_errchk(cudaFreeAsync(d_decon_F, custream));
    cu_errchk(cudaFreeAsync(d_times_F, custream));
    cu_errchk(cudaFreeAsync(d_feat_params, custream));
#ifdef EMO_DIFF
    cb_errchk(cublasGetMatrixAsync(n_rows_chunk, n_cols_coef, sizeof(double), d_features + n_rows_chunk, n_rows_chunk, p_features + start_row, n_rows, custream));
#else
    cb_errchk(cublasGetMatrixAsync(n_rows_chunk, n_cols_coef, sizeof(double), d_features, n_rows_chunk, p_features + start_row, n_rows, custream));
#endif
    cu_errchk(cudaFreeAsync(d_features, custream));
    cu_sync_destroy(custream);
}


}
