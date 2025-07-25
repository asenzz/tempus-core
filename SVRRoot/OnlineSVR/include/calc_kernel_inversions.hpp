//
// Created by zarko on 5/29/22.
//

#ifndef SVR_CALC_KERNEL_INVERSIONS_HPP
#define SVR_CALC_KERNEL_INVERSIONS_HPP


#include <limits>
#include <cstdlib>
#include <armadillo>
#include "common/compatibility.hpp"

namespace svr {

/* Original code by Emanouil Atanasov
 * used to find kernel lambda parameter
 */
class calc_kernel_inversions
{
    double _weight = std::numeric_limits<double>::quiet_NaN();
public:
    operator double() { return _weight; }

    calc_kernel_inversions(CPTRd Z, const arma::mat &Y);
};

double score_distance_kernel(const size_t sizeX, double *Z_distances, double *Y);

}

#endif //SVR_CALC_KERNEL_INVERSIONS_HPP
