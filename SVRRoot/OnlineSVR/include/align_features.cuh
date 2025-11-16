//
// Created by zarko on 7/4/24.
//

#ifndef SVR_ALIGN_FEATURES_CUH
#define SVR_ALIGN_FEATURES_CUH

#include "common/compatibility.hpp"
#include "common/cuda_util.cuh"
#include "util/math_utils.hpp"
#include "model_features.hpp"

namespace svr {

// constexpr float C_skip_multiplier = 1. + 1e-3;
// constexpr float C_skip_limit = C_skip_multiplier;

#ifndef __CUDACC__
__device__ __host__ inline unsigned umin(const unsigned a, const unsigned b) { return _MIN(a, b); }
#endif

#define STRETCH_(IX) uint32_t((IX) * st)
#define SKIP_(IX) unsigned(unsigned((IX) / sk) * sk)
#define STRETCHSKIP_(IX) STRETCH_(IX) // SKIP_(STRETCH_(IX))

__global__ void G_align_features(
    CRPTRd features, CRPTRd labels,
        RPTR(double) scores, RPTR(float) stretches, RPTR(unsigned) shifts, uint32_t n_rows, uint32_t n_cols,
        float shift_inc_mul, float stretch_limit, uint32_t align_validate, uint32_t shift_limit, float stretch_multiplier);

__global__ void G_quantise_features(
        RPTR(double) features /* zeroed out before */, CRPTRd d_decon_F, CRPTR(t_feat_params) d_feat_params, uint32_t rows, uint32_t cols, uint16_t quantise, uint32_t interleave_quantise);

void align_features(CPTRd p_features, CPTRd p_labels, double *p_scores, float *p_stretches, uint32_t *p_shifts, uint32_t n_rows, uint32_t n_cols);

void quantise_features(
        CPTRd decon, CPTR(t_feat_params) feat_params, uint32_t start_row, uint32_t n_rows_chunk,
        uint32_t n_rows, uint32_t n_feat_rows, uint16_t level, uint32_t n_cols_coef_, uint32_t n_cols_coef, uint16_t quantise,
        RPTR(double) p_features);

#ifdef __CUDACC__

template<const bool do_label_bias = false> __global__ void G_quantise_labels(CRPTRd d_in, RPTR(double) d_labels, const uint32_t rows, CRPTR(t_label_ix) d_label_ixs, CRPTR(uint32_t) ix_end_F)
{
    CU_STRIDED_FOR_i(rows) {
        const auto len = d_label_ixs[i].n_ixs;
        UNROLL(36)
        for (DTYPE(len) j = 0; j < len; ++j) d_labels[i] += d_in[d_label_ixs[i].label_ixs[j]];
        d_labels[i] /= len;
        if constexpr (do_label_bias) {
            constexpr auto label_bias_1 = C_label_bias + 1;
            d_labels[i] = (C_label_bias * d_labels[i] + d_in[d_label_ixs[i].special_x]) / label_bias_1;
        }
#ifdef EMO_DIFF
        d_labels[i] -= d_in[ix_end_F[i]];
#endif
    }
}

template<const bool do_label_bias = false> __global__ void G_quantise_labels(
        CRPTRd d_in, RPTR(double) d_labels, const uint32_t rows, CRPTR(t_label_ix) d_label_ixs, CRPTR(uint32_t) ix_end_F, const uint16_t steps_1,
        CRPTR(uint32_t) d_step_ixs, CRPTR(float) points, const uint16_t steps)
{
    CU_STRIDED_FOR_i(rows) {
        const auto len = d_label_ixs[i].n_ixs;
        UNROLL(36)
        for (DTYPE(len) j = 0; j < len; ++j) {
            auto j_step = float(j) / float(len);
            DTYPE(steps_1) p;
            for (p = 0; j_step >= 0; ++p) j_step -= points[p];
            --p;
            d_labels[i + p * rows] += d_in[d_label_ixs[i].label_ixs[j]];
        }
        for  (DTYPE(steps) p = 0; p < steps; ++p) {
            const auto lix = i + p * rows;
            d_labels[lix] /= d_step_ixs[p];
            if constexpr(do_label_bias) {
                constexpr auto label_bias_1 = C_label_bias + 1;
                d_labels[lix] = (C_label_bias * d_labels[lix] + d_in[d_label_ixs[i].special_x]) / label_bias_1;
            }
#ifdef EMO_DIFF
            d_labels[lix] -= d_in[ix_end_F[i]];
#endif
        }
    }
}

#endif

void quantise_labels(uint32_t label_len, const std::vector<double> &in, const std::vector<t_label_ix> &label_ixs, const std::vector<uint32_t> &ix_end_F, RPTR(double) p_labels,
                     uint16_t steps, CRPTR(float) points, CRPTR(uint32_t) steps_ixs);

}

#endif // SVR_ALIGN_FEATURES_CUH
