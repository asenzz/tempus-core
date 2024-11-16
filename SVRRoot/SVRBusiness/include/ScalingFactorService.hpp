//
// Created by zarko on 10/1/24.
//

#ifndef SVR_SCALINGFACTORSERVICE_HPP
#define SVR_SCALINGFACTORSERVICE_HPP

#include <armadillo>
#include "util/math_utils.hpp"

namespace svr {
namespace business {

class ScalingFactorService {
public:
    static double calc_scaling_factor(const arma::mat &v);
    static double calc_dc_offset(const arma::mat &v);
    static std::pair<double, double> calc(const arma::mat &v);

    template<typename T> inline static T scale(const T &v)
    {
        const auto [dc, sf] = calc(v);
        return common::scale(v, sf, dc);
    }

    template<typename T> inline static T &scale_I(T &v)
    {
        const auto [dc, sf] = calc(v);
        return common::scale_I(v, sf, dc);
    }

    template<typename T> inline static T &scale_I(T &v, const double sf, const double dc)
    {
        return common::scale_I(v, sf, dc);
    }
};

}
}

#endif //SVR_SCALINGFACTORSERVICE_HPP
