//
// Created by zarko on 31/12/2024.
//

#ifndef SVR_MODEL_FEATURES_HPP
#define SVR_MODEL_FEATURES_HPP

#include <cstdint>

namespace svr {
typedef struct _feat_params { uint32_t ix_end, ix_start;} t_feat_params, *t_feat_params_ptr;

constexpr uint32_t C_max_label_ixs = 24 * 3'600 + 1; // TODO Deduce from main input queue resolution and remove this constant
constexpr double C_label_bias = 0; // Bias toward TWAP, set to zero to disable bias

typedef struct _label_ix {
    uint32_t n_ixs, label_ixs[C_max_label_ixs], special_x;
} t_label_ix, *t_label_ix_ptr;

}
#endif //SVR_MODEL_FEATURES_HPP
