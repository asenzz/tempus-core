//
// Created by zarko on 7/4/24.
//

#ifndef SVR_ALIGN_FEATURES_CUH
#define SVR_ALIGN_FEATURES_CUH

#include "common/compatibility.hpp"
#include "common/cuda_util.cuh"

namespace svr {

constexpr float C_stretch_multiplier = 1 - 1e-4; // (1, 0) Closer to 1, higher precision, slower tuning
constexpr float C_stretch_limit = .5; // (0..1] Lower is higher precision slower tuning.
constexpr float C_skip_multiplier = 1.001;
constexpr float C_skip_limit = C_skip_multiplier;
constexpr float C_shift_inc_mul = 1 + 1e4; // (1, +inf] The closer to 1, higher precision, slower tuning

#ifdef INTEGRATION_TEST
constexpr unsigned C_integration_test_validation_window = 460;
#else
constexpr unsigned C_integration_test_validation_window = 0;
#endif

constexpr unsigned C_tuning_window = common::C_default_kernel_max_chunk_len + C_test_len;
constexpr unsigned C_active_window = C_tuning_window + C_integration_test_validation_window;
constexpr unsigned C_validate_cutoff = _MIN(1000, common::C_default_kernel_max_chunk_len);
constexpr unsigned C_align_window = C_active_window + common::C_default_kernel_max_chunk_len;

typedef struct _feat_params {
    unsigned end_time, time_start, ix_end, ix_start;
} t_feat_params, *t_feat_params_ptr;

#ifndef __CUDACC__
__device__ __host__ inline unsigned umin(const unsigned a, const unsigned b) { return _MIN(a, b); }
#endif

#define STRETCH_(IX) unsigned((IX) * st)
#define SKIP_(IX) unsigned(unsigned((IX) / sk) * sk)
#define STRETCHSKIP_(IX) STRETCH_(IX) // SKIP_(STRETCH_(IX))

// Not used
__device__ __host__ inline double stretch_ix(CRPTR(double) features, const unsigned i, const unsigned n_rows, const float st, const float sk)
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

__global__ void cu_align_features(
        CRPTR(double) features, CRPTR(double) labels,
         RPTR(double) scores, float *__restrict__ const stretches, unsigned *__restrict__ const shifts, float *__restrict__ const skips,
        const unsigned n_rows, const unsigned n_cols);

__global__ void G_quantise_features(
        RPTR(double) features /* zeroed out before */, CRPTR(double) d_decon_F, CRPTR(unsigned) d_times_F, CRPTR(t_feat_params) d_feat_params,
        const unsigned rows, const unsigned cols, const unsigned quantise, const unsigned interleave_quantise, const unsigned interleave_quantise_skip
        );

void score_features(CPTR(double) features, CPTR(double) labels, double *const scores, const unsigned rows, const unsigned cols);

void align_features(CPTR(double) p_features, CPTR(double) p_labels, double *const p_scores, float *const p_stretches, unsigned *const p_shifts, float *const p_skips,
                    const unsigned n_rows, const unsigned n_cols);

void quantise_features(
        CPTR(double) decon, CPTR(unsigned) times_F, CPTR(t_feat_params) feat_params, const unsigned start_row, const unsigned n_rows_chunk,
        const unsigned n_rows, const unsigned n_feat_rows, const unsigned level, const unsigned n_cols_coef_, const unsigned n_cols_coef, const unsigned quantise,
        double *const p_features);

}

#endif //SVR_ALIGN_FEATURES_CUH
