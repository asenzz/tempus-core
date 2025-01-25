//
// Created by zarko on 7/4/24.
//

#ifndef SVR_ALIGN_FEATURES_CUH
#define SVR_ALIGN_FEATURES_CUH

#include "common/compatibility.hpp"
#include "common/cuda_util.cuh"
#include "model_features.hpp"

namespace svr {

constexpr float C_stretch_multiplier = 1 - 1e-4; // (1, 0) Closer to 1, higher precision, slower tuning
constexpr float C_stretch_limit = .5; // (0..1] Lower is higher precision slower tuning. // Seems like it should be chunk len / decrement offset
constexpr float C_skip_multiplier = 1. + 1e-3;
constexpr float C_skip_limit = C_skip_multiplier;
constexpr float C_shift_inc_mul = 1 + 1e4; // (1, +inf] The closer to 1, higher precision, slower tuning

#ifndef REMOVE_OUTLIERS
constexpr unsigned C_tuning_window = common::C_default_kernel_max_chunk_len + C_test_len;
const unsigned C_active_window = C_tuning_window + common::C_integration_test_validation_window;
#endif
constexpr uint32_t C_align_validate = 1000; // common::C_default_kernel_max_chunk_len;
constexpr uint32_t C_align_window_oemd = common::C_best_decrement; // C_shift_lim + C_align_validate;

constexpr uint32_t C_max_label_ixs = 3600 + 1; // TODO Deduce from main input queue resolution and remove this constant

typedef struct _label_ix {
    uint32_t n_ixs, label_ixs[C_max_label_ixs];
} t_label_ix, *t_label_ix_ptr;

#ifndef __CUDACC__
__device__ __host__ inline unsigned umin(const unsigned a, const unsigned b) { return _MIN(a, b); }
#endif

#define STRETCH_(IX) uint32_t((IX) * st)
#define SKIP_(IX) unsigned(unsigned((IX) / sk) * sk)
#define STRETCHSKIP_(IX) STRETCH_(IX) // SKIP_(STRETCH_(IX))

__global__ void G_align_features(
        CRPTRd features, CRPTRd labels,
        RPTR(double) scores, RPTR(float) stretches, RPTR(unsigned) shifts, RPTR(float) skips, const uint32_t n_rows, const uint32_t n_cols);

__global__ void G_quantise_features(
        RPTR(double) features /* zeroed out before */, CRPTRd d_decon_F, CRPTR(uint32_t) d_times_F, CRPTR(t_feat_params) d_feat_params,
        const uint32_t rows, const uint32_t cols, const uint16_t quantise, const uint32_t interleave_quantise, const uint32_t interleave_quantise_skip);

void score_features(CPTRd features, CPTRd labels, double *const scores, const uint32_t rows, const uint32_t cols);

void align_features(CPTRd p_features, CPTRd p_labels, double *const p_scores, float *const p_stretches, uint32_t *const p_shifts, float *const p_skips,
                    const uint32_t n_rows, const uint32_t n_cols);

void quantise_features(
        CPTRd decon, CPTR(uint32_t) times_F, CPTR(t_feat_params) feat_params, const uint32_t start_row, const uint32_t n_rows_chunk,
        const uint32_t n_rows, const uint32_t n_feat_rows, const uint16_t level, const uint32_t n_cols_coef_, const uint32_t n_cols_coef, const uint16_t quantise,
        RPTR(double) p_features);

__global__ void G_quantise_labels_quick(
        CRPTRd d_in, RPTR(double) d_labels, const uint32_t rows, const uint32_t label_len,
        CRPTR(t_label_ix) d_label_ixs, CRPTR(uint32_t) ix_end_F, const uint16_t multistep, const uint32_t step_ixs);

void
quantise_labels(const uint32_t label_len, const std::vector<double> &in, const std::vector<t_label_ix> &label_ixs, const std::vector<uint32_t> &feat_params,
                RPTR(double) p_labels, const uint16_t multistep);

}

#endif // SVR_ALIGN_FEATURES_CUH
