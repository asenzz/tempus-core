//
// Created by zarko on 31/12/2024.
//

#ifndef SVR_MODEL_FEATURES_HPP
#define SVR_MODEL_FEATURES_HPP

#include <cstdint>

typedef struct _feat_params {
    uint32_t end_time, time_start, ix_end, ix_start;
} t_feat_params, *t_feat_params_ptr;


#endif //SVR_MODEL_FEATURES_HPP
