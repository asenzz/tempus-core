//
// Created by zarko on 2/24/22.
//

#ifndef SVR_DQ_SCALING_FACTORS_SERVICE_IMPL_CUH
#define SVR_DQ_SCALING_FACTORS_SERVICE_IMPL_CUH

#include <vector>

namespace svr {
namespace business {

void cu_get_nth_max(
        const std::vector<double> &flat_row_matrix,
        const size_t alpha_n,
        const size_t levels,
        const size_t rows,
        std::vector<double> &means,
        std::vector<double> &scaling_factor,
        const int id);

}
}

#endif //SVR_DQ_SCALING_FACTORS_SERVICE_IMPL_CUH
