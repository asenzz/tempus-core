//
// Created by zarko on 7/4/24.
//

#ifndef SVR_ALIGN_FEATURES_CUH
#define SVR_ALIGN_FEATURES_CUH

#include "common/compatibility.hpp"
#include "common/cuda_util.cuh"
#include "model_features.hpp"

namespace svr {

constexpr float C_skip_multiplier = 1. + 1e-3;
constexpr float C_skip_limit = C_skip_multiplier;

constexpr uint32_t C_max_label_ixs = 14'400 + 1; // TODO Deduce from main input queue resolution and remove this constant
constexpr double C_label_bias = 0; // Bias toward TWAP, set to zero to disable bias

typedef struct _label_ix {
    uint32_t n_ixs, label_ixs[C_max_label_ixs], special_x;
} t_label_ix, *t_label_ix_ptr;

#ifndef __CUDACC__
__device__ __host__ inline unsigned umin(const unsigned a, const unsigned b) { return _MIN(a, b); }
#endif

#define STRETCH_(IX) uint32_t((IX) * st)
#define SKIP_(IX) unsigned(unsigned((IX) / sk) * sk)
#define STRETCHSKIP_(IX) STRETCH_(IX) // SKIP_(STRETCH_(IX))

__global__ void G_align_features(
        CRPTRd features, CRPTRd labels,
        RPTR(double) scores, RPTR(float) stretches, RPTR(unsigned) shifts, RPTR(float) skips, const uint32_t n_rows, const uint32_t n_cols,
        const float shift_inc_mul, const double stretch_limit, const uint32_t align_validate, const uint32_t shift_limit, const float stretch_multiplier);

__global__ void G_quantise_features(
        RPTR(double) features /* zeroed out before */, CRPTRd d_decon_F, CRPTR(t_feat_params) d_feat_params,
        const uint32_t rows, const uint32_t cols, const uint16_t quantise, const uint32_t interleave_quantise);

void align_features(CPTRd p_features, CPTRd p_labels, double *const p_scores, float *const p_stretches, uint32_t *const p_shifts, float *const p_skips,
                    const uint32_t n_rows, const uint32_t n_cols);

void quantise_features(
        CPTRd decon, CPTR(t_feat_params) feat_params, const uint32_t start_row, const uint32_t n_rows_chunk,
        const uint32_t n_rows, const uint32_t n_feat_rows, const uint16_t level, const uint32_t n_cols_coef_, const uint32_t n_cols_coef, const uint16_t quantise,
        RPTR(double) p_features);

#ifdef __CUDACC__

template<const bool do_label_bias = false> __global__ void G_quantise_labels(
        CRPTRd d_in, RPTR(double) d_labels, const uint32_t rows, const uint32_t label_len,
        CRPTR(t_label_ix) d_label_ixs, CRPTR(uint32_t) ix_end_F, const uint16_t multistep, const uint32_t step_ixs)
{
    CU_STRIDED_FOR_i(rows) {
        UNROLL(36)
        for (uint32_t j = 0; j < d_label_ixs[i].n_ixs; ++j) d_labels[rows * (j / step_ixs) + i] += d_in[d_label_ixs[i].label_ixs[j]];
#ifdef EMO_DIFF
        const auto lk = d_in[ix_end_F[i]];
#endif
        for (uint16_t j = 0; j < multistep; ++j) {
            const auto lix = j * rows + i;
            d_labels[lix] /= step_ixs;
            if constexpr(do_label_bias) {
                constexpr auto label_bias_1 = C_label_bias + 1;
                d_labels[lix] = (C_label_bias * d_labels[lix] + d_in[d_label_ixs[i].special_x]) / label_bias_1;
            }
#ifdef EMO_DIFF
            d_labels[lix] -= lk;
#endif
        }
    }
}

#endif

void
quantise_labels(const uint32_t label_len, const std::vector<double> &in, const std::vector<t_label_ix> &label_ixs, const std::vector<uint32_t> &feat_params,
                RPTR(double) p_labels, const uint16_t multistep);

}

#endif // SVR_ALIGN_FEATURES_CUH
