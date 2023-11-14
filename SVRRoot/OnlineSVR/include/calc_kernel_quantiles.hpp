//
// Created by zarko on 5/29/22.
//

#ifndef SVR_CALC_KERNEL_QUANTILES_HPP
#define SVR_CALC_KERNEL_QUANTILES_HPP

#include <limits>
#include <cstdlib>
#include <armadillo>

#define NUM_CLASSES_QUANTILES (16)

namespace svr {

/* Original code by Emanouil Atanasov, used for both kernel gamma and lambda
 */
class calc_kernel_quantiles
{
    double _weight = std::numeric_limits<double>::quiet_NaN();
public:
    operator double() { return _weight; }

    calc_kernel_quantiles(const arma::mat &K, const arma::mat &Y, const size_t num_classes = NUM_CLASSES_QUANTILES);
};

}

#endif //SVR_CALC_KERNEL_QUANTILES_HPP
