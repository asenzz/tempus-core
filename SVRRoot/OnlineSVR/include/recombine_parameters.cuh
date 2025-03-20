#include <cstdint>
#include "common/defines.h"
#include "common/constants.hpp"

#define MAX_COL_CT 31
// #define DEBUG_VALUES

typedef uint8_t t_params_vec[MAX_COL_CT];

namespace svr {

struct t_param_preds_cu
{
    double predictions[1][1];
    double labels[1][1];
    double last_knowns[1][1];
    uint8_t params_ix;
};
typedef t_param_preds_cu *t_param_preds_cu_ptr;

void
recombine_parameters(
        const uint32_t rowct,
        const uint32_t colct,
        const uint8_t *combos,
        const t_param_preds_cu *params_preds,
        double *p_best_score,
        uint8_t *best_params_ixs);


}