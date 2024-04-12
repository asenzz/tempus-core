//
// Created by zarko on 3/30/21.
//

#ifndef SVR_NM_HPP
#define SVR_NM_HPP

#include <vector>
#include <functional>
#include "model/SVRParameters.hpp"
#include "optimizer.hpp"

namespace svr {
namespace optimizer {

std::pair<double, std::vector<double>>
nm(const loss_callback_t &loss_fun, const std::vector<double>& initial_values, const NM_parameters& nm_parameters);

}
} //svr


#endif //SVR_NM_HPP
