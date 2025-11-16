//
// Created by zarko on 7/4/24.
//
#include <thrust/binary_search.h>
#include <thrust/transform_reduce.h>
#include <thrust/transform.h>
#include <cublas_v2.h>
#include "align_features.cuh"
#include "common/cuda_util.cuh"
#include "appcontext.hpp"
#include "ScalingFactorService.hpp"
#include "cuqrsolve.cuh"

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


__device__ __forceinline__ double vec_dist_stretch(CRPTRd labels, CRPTRd features, const uint32_t validate_rows, const float st, const float sk, const uint32_t align_validate)
{
#if 1 // Normalised cross correlation
    double sum_X = 0, sum_Y = 0, sum_XY = 0;
    double squareSum_X = 0, squareSum_Y = 0;
    for (uint32_t r = validate_rows - align_validate; r < validate_rows; ++r) {
	const auto X_r = features[STRETCHSKIP_(r)];
        sum_XY += X_r * labels[r];
        sum_X += X_r;
        squareSum_X += X_r * X_r;
        sum_Y += labels[r]; // TODO Optimize by moving labels related calculation out of this kernel, test thoroughly since validate_rows varies
        squareSum_Y += labels[r] * labels[r]; // this too
    }
    return 2 - ((validate_rows * sum_XY - sum_X * sum_Y) / sqrt((validate_rows * squareSum_X - sum_X * sum_X) * (validate_rows * squareSum_Y - sum_Y * sum_Y)) + 1);
#else // Absolute Euclidean distance
    double res = 0;
    for (auto r = validate_rows - align_validate; r < validate_rows; ++r) res += abs(labels[r] - features[STRETCHSKIP_(r)]);
    return res;
#endif
}

__global__ void G_align_features(
    CRPTRd features, CRPTRd labels,
    RPTR(double) scores, RPTR(float) stretches, RPTR(uint32_t) shifts,
        const uint32_t n_rows, const uint32_t n_cols, const float shift_inc_mul, const float stretch_limit, const uint32_t align_validate,
    const uint32_t shift_limit, const float stretch_multiplier)
{
    CU_STRIDED_FOR_i(n_cols) {
        scores[i] = common::C_bad_validation;
        CPTRd features_col = features + n_rows * i;
        for (DTYPE(shift_limit) sh = 0; sh < shift_limit; sh += max(1, uint32_t(shift_inc_mul * sh))) { // TODO Unroll loop into an array supplied at kernel launch
            CPTRd labels_sh = labels + sh;
            const auto validate_rows = n_rows - sh;
            UNROLL()
            for (DTYPE(stretch_limit) st = 1; st > stretch_limit; st *= stretch_multiplier) {
                const auto score = vec_dist_stretch(labels_sh, features_col, validate_rows, st, 1, align_validate);
                if (score >= scores[i]) continue;
                scores[i] = score;
                if (shifts) shifts[i] = sh;
                if (stretches) stretches[i] = st;
            }
        }
    }
}


void align_features(CPTRd p_features, CPTRd labels, double *const p_scores, float *const p_stretches, RPTR(uint32_t) p_shifts, const uint32_t n_rows, const uint32_t n_cols)
{
    const auto n_rows_integration = n_rows - common::C_integration_test_validation_window;
    const uint32_t align_window = (n_rows_integration - PROPS.get_shift_limit() - PROPS.get_outlier_slack()) * PROPS.get_stretch_limit();
    assert(n_rows_integration - PROPS.get_shift_limit() >= align_window);
#ifdef INTEGRATION_TEST
    LOG4_DEBUG("Aligning features test offset " << common::C_integration_test_validation_window << ", rows " << n_rows << ", cols " << n_cols << ", align window " << align_window);
#endif
    CTX_CUSTREAM_(2);
    double *d_features;
    CU_ERRCHK(cudaMallocAsync((void **) &d_features, n_rows_integration * n_cols * sizeof(double), custream));
    copy_submat(p_features, d_features, n_rows, 0, 0, n_rows_integration, n_cols, n_rows_integration, cudaMemcpyHostToDevice, custream);
    const auto d_labels = cumallocopy(labels, custream, n_rows_integration);
    double *d_scores;
    CU_ERRCHK(cudaMallocAsync((void **) &d_scores, n_cols * sizeof(double), custream));
    float *d_stretches;
    if (p_stretches) {
        CU_ERRCHK(cudaMallocAsync((void **) &d_stretches, n_cols * sizeof(float), custream));
    } else d_stretches = nullptr;
    uint32_t *d_shifts;
    if (p_shifts) {
    CU_ERRCHK(cudaMallocAsync((void **) &d_shifts, n_cols * sizeof(uint32_t), custream));
    } else d_shifts = nullptr;
    G_align_features<<<CU_BLOCKS_THREADS(n_cols), 0, custream>>>(
            d_features, d_labels, d_scores, d_stretches, d_shifts, n_rows_integration, n_cols, PROPS.get_shift_multi(), PROPS.get_stretch_limit(), align_window, PROPS.get_shift_limit(),
        PROPS.get_stretch_coef());
    CU_ERRCHK(cudaFreeAsync(d_features, custream));
    CU_ERRCHK(cudaFreeAsync(d_labels, custream));
    cufreecopy(p_scores, d_scores, custream, n_cols);
    if (p_stretches) cufreecopy(p_stretches, d_stretches, custream, n_cols);
    if (p_shifts) cufreecopy(p_shifts, d_shifts, custream, n_cols);
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
#endif
            const auto j_interleave_quantise_start = d_feat_params_i->ix_start + j * interleave_quantise;
            auto ix_F = j_interleave_quantise_start;
            for (; ix_F < j_interleave_quantise_start + quantise && ix_F <= ix_end; ++ix_F) {
                *feat_i_j_rows += d_decon_F[ix_F];
#ifdef EMO_DIFF
                prev_price += d_decon_F[ix_F - quantise];
#endif
            }

#ifdef EMO_DIFF
            *feat_i_j_rows = (*feat_i_j_rows - prev_price) / (ix_F - j_interleave_quantise_start);
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
        for (auto j = 0; j < n_cols_; ++j) features[i + j * n_rows] /= quantise;
#endif
        // features[i + ((d_feat_params_i->ix_end - ix_start) / quantise) * n_rows] = d_decon_F[d_feat_params_i->ix_end];
    }
}

// TODO Align quantisation per column
void quantise_features(
    CPTRd decon, CPTR(t_feat_params) feat_params, const uint32_t start_row, const uint32_t n_rows_chunk, const uint32_t n_rows, const uint32_t n_feat_rows, const uint16_t level,
    const uint32_t n_cols_coef_, const uint32_t n_cols_coef, const uint16_t quantise, RPTR(double) p_features)
{
    CTX_CUSTREAM_(2);
    const auto end_row = start_row + n_rows_chunk - 1;
    auto d_features = cucalloc<double>(custream, n_rows_chunk * n_cols_coef_);
    const auto d_decon_F = cumallocopy(decon + n_feat_rows * level, custream, feat_params[end_row].ix_end + 1);
    const auto d_feat_params = cumallocopy(feat_params, custream, end_row + 1);
    G_quantise_features<<<CU_BLOCKS_THREADS(clamp_n(n_rows_chunk)), 0, custream>>>(
            d_decon_F, d_feat_params, n_rows_chunk, quantise, start_row, n_cols_coef_, d_features);
//     double stub_sf, stub_dc;
//    business::ScalingFactorService::cu_scale_calc_I(d_features, n_rows_chunk * n_cols_coef_, stub_sf, stub_dc, custream); // TODO Check if really needed
    CU_ERRCHK(cudaFreeAsync(d_decon_F, custream));
    CU_ERRCHK(cudaFreeAsync(d_feat_params, custream));
#ifdef EMO_DIFF
    CB_ERRCHK(cublasGetMatrixAsync(n_rows_chunk, n_cols_coef, sizeof(double), d_features + n_rows_chunk, n_rows_chunk, p_features + start_row, n_rows, custream));
#else
    CB_ERRCHK(cublasGetMatrixAsync(n_rows_chunk, n_cols_coef, sizeof(double), d_features, n_rows_chunk, p_features + start_row, n_rows, custream));
#endif
    CU_ERRCHK(cudaFreeAsync(d_features, custream));
    cusyndestroy(custream);
}

void quantise_labels(const uint32_t label_len, const std::vector<double> &in, const std::vector<t_label_ix> &label_ixs, const std::vector<uint32_t> &ix_end_F, RPTR(double) p_labels,
                     const uint16_t steps, CRPTR(float) points, CRPTR(uint32_t) steps_ixs)
{
    CTX_CUSTREAM_(2);
    assert(steps);
    const auto rows = label_ixs.size();
    auto d_labels = cucalloc<double>(custream, rows * steps);
    const auto d_ix_end_F = cumallocopy(ix_end_F, custream);
    const auto d_label_ixs = cumallocopy(label_ixs, custream);
    const auto d_in = cumallocopy(in, custream);
    const auto d_points = steps > 1 ? cumallocopy(points, custream, steps - 1) : nullptr;
    const auto d_steps_ixs = steps > 1 ? cumallocopy(steps_ixs, custream, steps) : nullptr;
    constexpr bool do_label_bias = C_label_bias > 0;
    if (steps == 1)
        G_quantise_labels<do_label_bias><<<CU_BLOCKS_THREADS(rows), 0, custream>>>(d_in, d_labels, rows, d_label_ixs, d_ix_end_F);
    else
        G_quantise_labels<do_label_bias><<<CU_BLOCKS_THREADS(rows), 0, custream>>>(d_in, d_labels, rows, d_label_ixs, d_ix_end_F, steps - 1, d_steps_ixs, d_points, steps);
#ifndef NDEBUG
    CU_ERRCHK(cudaDeviceSynchronize());
    CU_ERRCHK(cudaPeekAtLastError());
#endif
    CU_ERRCHK(cudaFreeAsync(d_in, custream));
    CU_ERRCHK(cudaFreeAsync(d_label_ixs, custream));
    CU_ERRCHK(cudaFreeAsync(d_ix_end_F, custream));
    if (steps > 1) {
        CU_ERRCHK(cudaFreeAsync(d_points, custream));
        CU_ERRCHK(cudaFreeAsync(d_steps_ixs, custream));
    }
    double stub_sf, stub_dc;
    business::ScalingFactorService::cu_scale_calc_I(d_labels, rows * steps, stub_sf, stub_dc, custream); // TODO Check if really needed
    cufreecopy(p_labels, d_labels, custream, rows);
    cusyndestroy(custream);
}

}
