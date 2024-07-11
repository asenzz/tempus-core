//
// Created by zarko on 7/4/24.
//

#ifndef SVR_ALIGN_FEATURES_CUH
#define SVR_ALIGN_FEATURES_CUH

namespace svr {

__device__ __host__ inline double stretch_ix(const double *const __restrict__ shifted_features, const unsigned i, const unsigned n_rows, const double stretch_factor)
{
    unsigned ct = 1;
    unsigned j = unsigned(i * stretch_factor) % n_rows;
    double res = shifted_features[j];
    const unsigned to_j = unsigned((i + 1) * stretch_factor) % n_rows;
    while (j < to_j) {
        res += shifted_features[j];
        ++ct;
        ++j;
    }
    if (ct > 1) res /= double(ct);
    return res;
}

void align_features(const double *const p_double_features, const double *const p_mean_L, double *const p_scores, double *const p_stretches, const unsigned n_rows,
                    const unsigned n_cols_superset);

}

#endif //SVR_ALIGN_FEATURES_CUH
