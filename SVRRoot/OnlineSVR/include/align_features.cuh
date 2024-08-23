//
// Created by zarko on 7/4/24.
//

#ifndef SVR_ALIGN_FEATURES_CUH
#define SVR_ALIGN_FEATURES_CUH

#include "common/compatibility.hpp"
#include "common/cuda_util.cuh"

namespace svr {

constexpr float C_stretch_multiplier = 1 - 1e-3; // (1, 0) Closer to 1, higher precision, slower tuning
constexpr float C_stretch_limit = .5; // (0..1] Lower is higher precision slower tuning.
constexpr float C_skip_multiplier = 1.001;
constexpr float C_skip_limit = C_skip_multiplier;
constexpr float C_shift_inc_mul = 1 + 1e3; // (1, +inf] The closer to 1, higher precision, slower tuning
constexpr float C_shift_limit = .5;
constexpr unsigned C_validate_cutoff = 1000;

#ifndef __CUDACC__
__device__ __host__ inline unsigned umin(const unsigned a, const unsigned b) { return _MIN(a, b); }
#endif

#define STRETCH_(IX) unsigned((IX) * st)
#define SKIP_(IX) unsigned(unsigned((IX) / sk) * sk)
#define STRETCHSKIP_(IX) SKIP_(STRETCH_(IX))

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


void align_features(CPTR(double) p_features, CPTR(double) p_mean_L, double *const p_scores, float *const p_stretches, unsigned *const p_shifts, float *const p_skips,
                    const unsigned n_rows, const unsigned n_cols);

void quantise_features(
        CPTR(double) decon, CPTR(unsigned) times_F, CPTR(unsigned) times_last_F, CPTR(unsigned) ixs_last_F, const unsigned start_row, const unsigned n_rows_chunk,
        const unsigned n_rows, const unsigned n_feat_rows, const unsigned level, const unsigned n_cols_coef_, const unsigned n_cols_coef, const unsigned quantise,
        double *const p_features);

}

#endif //SVR_ALIGN_FEATURES_CUH
