//
// Created by zarko on 7/4/24.
//
#include <thrust/binary_search.h>
#include <thrust/async/for_each.h>
#include <cublas.h>
#include "common/cuda_util.cuh"
#include "align_features.cuh"

namespace svr {


// Not used
__device__ __host__ __forceinline__ double stretch_ix(CRPTRd features, const uint32_t i, const uint32_t n_rows, const float st, const float sk)
{
    const auto n_rows_1 = n_rows - 1;
    auto j = umin(STRETCHSKIP_(i), n_rows_1);
    if (j == n_rows_1) return features[j];

    auto res = features[j];
    const auto to_j = umin(STRETCHSKIP_(i + 1), n_rows_1);
    if (j + 1 >= to_j) return res;

    const auto ct = to_j - j;
    UNROLL()
    while (j < to_j) {
        res += features[j];
        ++j;
    }
    if (ct > 1) res /= double(ct);
    return res;
}

__device__ __forceinline__ double vec_dist(CRPTRd mean_L, CRPTRd features, const uint32_t n_rows, const float st, const uint32_t sh, const uint32_t integration_test)
{
    double res = 0;
    const auto validate_rows = n_rows - sh - integration_test;
    UNROLL(16)
    for (uint32_t r = 0; r < validate_rows; ++r) res += abs(mean_L[r] - stretch_ix(features, r, n_rows, st, 1));
    return res / validate_rows;
}


__device__ __forceinline__ double vec_dist_stretch(CRPTRd labels, CRPTRd features, const uint32_t validate_rows, const float st /* < 1 */, const float sk)
{
    double res = 0;
    UNROLL(C_align_validate)
    for (uint32_t r = validate_rows - C_align_validate; r < validate_rows; ++r)
        res += abs(labels[r] - features[STRETCHSKIP_(r)]);
    return res;
}

__global__ void G_align_features(
        CRPTRd features, CRPTRd labels,
        RPTR(double) scores, RPTR(float) stretches, RPTR(uint32_t) shifts, RPTR(float) skips,
        const uint32_t n_rows, const uint32_t n_cols, const float shift_inc_mul, const double stretch_limit)
{
    CU_STRIDED_FOR_i(n_cols) {
        scores[i] = common::C_bad_validation;
        CPTRd features_col = features + n_rows * i;
        UNROLL(C_shift_lim / 10)
        for (uint32_t sh = 0; sh < C_shift_lim; sh += max(1, uint32_t(shift_inc_mul * sh))) { // TODO Unroll loop into an array supplied at kernel launch
            CPTRd labels_sh = labels + sh;
            const auto validate_rows = n_rows - sh;
            UNROLL()
            for (float st = 1; st > stretch_limit; st *= C_stretch_multiplier) {
// UNROLL()
//                for (float sk = 1; sk < C_skip_limit; sk *= C_skip_multiplier) {
                const auto score = vec_dist_stretch(labels_sh, features_col, validate_rows, st, 1);
                if (score >= scores[i]) continue;
                scores[i] = score;
                if (shifts) shifts[i] = sh;
                if (stretches) stretches[i] = st;
                // skips[i] = sk; // Skips degrade precision, retest and enable if necessary
//                }
            }
        }
    }
}


void align_features(
        CPTRd p_features, CPTRd labels, double *const p_scores, float *const p_stretches,
        RPTR(uint32_t) p_shifts, float *const p_skips, const uint32_t n_rows, const uint32_t n_cols)
{
    const auto n_rows_integration = n_rows - common::C_integration_test_validation_window;
    assert(n_rows_integration - C_shift_lim >= C_align_validate);

    LOG4_TRACE("Aligning features test offset " << common::C_integration_test_validation_window << ", rows " << n_rows << ", cols " << n_cols);
#ifndef REMOVE_OUTLIERS
    if (n_rows < C_active_window) LOG4_THROW("Rows " << n_rows << " less than active window " << C_active_window);
#endif
    CTX4_CUSTREAM;
    double *d_features;
    cu_errchk(cudaMallocAsync((void **) &d_features, n_rows_integration * n_cols * sizeof(double), custream));
    copy_submat(p_features, d_features, n_rows, 0, 0, n_rows_integration, n_cols, n_rows_integration, cudaMemcpyHostToDevice, custream);
    const auto d_labels = cumallocopy(labels, custream, n_rows);
    double *d_scores;
    cu_errchk(cudaMallocAsync((void **) &d_scores, n_cols * sizeof(double), custream));
    float *d_stretches, *d_skips;
    const auto cols_size_float = n_cols * sizeof(float);
    cu_errchk(cudaMallocAsync((void **) &d_stretches, cols_size_float, custream));
    cu_errchk(cudaMallocAsync((void **) &d_skips, cols_size_float, custream));
    uint32_t *d_shifts;
    cu_errchk(cudaMallocAsync((void **) &d_shifts, n_cols * sizeof(uint32_t), custream));
    G_align_features<<<CU_BLOCKS_THREADS(n_cols), 0, custream>>>(d_features, d_labels, d_scores, d_stretches, d_shifts, d_skips, n_rows_integration, n_cols, 0, C_stretch_limit);
    cu_errchk(cudaFreeAsync(d_features, custream));
    cu_errchk(cudaFreeAsync(d_labels, custream));
    cufreecopy(p_scores, d_scores, custream, n_cols);
    cufreecopy(p_stretches, d_stretches, custream, n_cols);
    cufreecopy(p_skips, d_skips, custream, n_cols);
    cufreecopy(p_shifts, d_shifts, custream, n_cols);
    cusyndestroy(custream);
}


template<typename T> __device__ __forceinline__ int before_bound(CRPTR(T) cbegin, CRPTR(T) cend, const T value)
{
    auto res = thrust::lower_bound(thrust::seq, cbegin, cend, value);
    while (*res > value && res > cbegin) --res;
    return res - cbegin;
}


__global__ void G_quantise_features(
        RPTR(double) features /* zeroed out before */, CRPTRd d_decon_F, CRPTR(t_feat_params) d_feat_params,
        const uint32_t rows, const uint32_t cols, const uint16_t quantise, const uint32_t interleave_quantise)
{
    CU_STRIDED_FOR_i(rows) {
        auto const d_feat_params_i = d_feat_params + i;
        const auto ix_end = d_feat_params_i->ix_end;
        UNROLL(32)
        for (uint32_t j = 0; j < cols; ++j) {
            auto const feat_i_j_rows = features + i + j * rows;
#ifdef EMO_DIFF
            double prev_price = 0;
            const auto j_interleave_quantise = d_feat_params_i->ix_start + j * interleave_quantise;
#endif
            uint32_t ix_F = j_interleave_quantise;
            for (; ix_F < j_interleave_quantise + quantise && ix_F <= ix_end; ++ix_F) {
                *feat_i_j_rows += d_decon_F[ix_F];
#ifdef EMO_DIFF
                prev_price += d_decon_F[ix_F - quantise];
#endif
            }

#ifdef EMO_DIFF
            *feat_i_j_rows = (*feat_i_j_rows - prev_price) / (ix_F - j_interleave_quantise);
#else
            *feat_i_j_rows /= (ix_F - j_interleave_quantise);
#endif
        }
    }
}

__global__ void G_quantise_features(
        CRPTRd d_decon_F, CRPTR(t_feat_params) d_feat_params,
        const uint32_t n_rows, const uint16_t quantise, const uint32_t start_row, const uint32_t n_cols_, RPTR(double) features /* zeroed out before */)
{
    CU_STRIDED_FOR_i(n_rows) {
        const auto d_feat_params_i = d_feat_params + start_row + i;
        const auto ix_start = d_feat_params_i->ix_start;
        UNROLL(16) // fill one row of features
        for (auto ix_F = ix_start; ix_F <= d_feat_params_i->ix_end; ++ix_F)
            features[i + ((ix_F - ix_start) / quantise) * n_rows] += d_decon_F[ix_F];

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

// TODO Align quantisation per column
void quantise_features(CPTRd decon, CPTR(t_feat_params) feat_params,
                       const uint32_t start_row, const uint32_t n_rows_chunk, const uint32_t n_rows, const uint32_t n_feat_rows,
                       const uint16_t level, const uint32_t n_cols_coef_, const uint32_t n_cols_coef, const uint16_t quantise, RPTR(double) p_features)
{
    CTX4_CUSTREAM;
    const auto end_row = start_row + n_rows_chunk - 1;
    auto d_features = cucalloc<double>(custream, n_rows_chunk * n_cols_coef_);
    const auto d_decon_F = cumallocopy(decon + n_feat_rows * level, custream, feat_params[end_row].ix_end + 1);
    const auto d_feat_params = cumallocopy(feat_params, custream, end_row + 1);
    G_quantise_features<<<CU_BLOCKS_THREADS(clamp_n(n_rows_chunk)), 0, custream>>>(
            d_decon_F, d_feat_params, n_rows_chunk, quantise, start_row, n_cols_coef_, d_features);
    cu_errchk(cudaFreeAsync(d_decon_F, custream));
    cu_errchk(cudaFreeAsync(d_feat_params, custream));
#ifdef EMO_DIFF
    cb_errchk(cublasGetMatrixAsync(n_rows_chunk, n_cols_coef, sizeof(double), d_features + n_rows_chunk, n_rows_chunk, p_features + start_row, n_rows, custream));
#else
    cb_errchk(cublasGetMatrixAsync(n_rows_chunk, n_cols_coef, sizeof(double), d_features, n_rows_chunk, p_features + start_row, n_rows, custream));
#endif
    cu_errchk(cudaFreeAsync(d_features, custream));
    cusyndestroy(custream);
}


void quantise_labels(const uint32_t label_len, const std::vector<double> &in, const std::vector<t_label_ix> &label_ixs, const std::vector<uint32_t> &ix_end_F, RPTR(double) p_labels, const uint16_t multistep)
{
    CTX4_CUSTREAM;
    const auto rows = label_ixs.size();
    auto d_labels = cucalloc<double>(custream, rows * multistep);
    const auto d_ix_end_F = cumallocopy(ix_end_F, custream);
    const auto d_label_ixs = cumallocopy(label_ixs, custream);
    const auto d_in = cumallocopy(in, custream);
    constexpr bool do_label_bias = C_label_bias > 0;
    G_quantise_labels<do_label_bias><<<CU_BLOCKS_THREADS(rows), 0, custream>>>
            (d_in, d_labels, rows, label_len, d_label_ixs, d_ix_end_F, multistep, label_ixs.front().n_ixs / multistep);
    cu_errchk(cudaFreeAsync(d_in, custream));
    cu_errchk(cudaFreeAsync(d_label_ixs, custream));
    cu_errchk(cudaFreeAsync(d_ix_end_F, custream));
    cufreecopy(p_labels, d_labels, custream, rows * multistep);
    cusyndestroy(custream);
}

}
