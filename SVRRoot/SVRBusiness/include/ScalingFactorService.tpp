//
// Created by zarko on 08/07/2025.
//

#ifndef SCALINGFACTORSERVICE_TPP
#define SCALINGFACTORSERVICE_TPP

#include "ScalingFactorService.hpp"

namespace svr {
namespace business {

template<typename T> inline T ScalingFactorService::scale(const T &v)
{
    double sf, dc;
    return scale_calc(v, sf, dc);
}

template<typename T> inline T &ScalingFactorService::scale_I(T &v)
{
    double sf, dc;
    return scale_calc_I(v, sf, dc);
}

template<typename T> inline T &ScalingFactorService::scale_calc_I(T &v, double &sf, double &dc, const double obseg)
{
    dc = calc_dc_offset(v);
    v -= dc; // Remove DC offset
    sf = calc_scaling_factor(v, obseg);
    LOG4_TRACE("DC offset " << dc << ", scaling factor " << sf << ", values " << common::present(v));
    return v /= sf; // Scale the values
}

template<typename T> inline T ScalingFactorService::scale_calc(T v, double &sf, double &dc, const double obseg)
{
    dc = calc_dc_offset(v);
    v -= dc; // Remove DC offset
    sf = calc_scaling_factor(v, obseg);
    LOG4_TRACE("DC offset " << dc << ", scaling factor " << sf << ", values " << common::present(v));
    return v /= sf;
}

template<typename T> inline T ScalingFactorService::scale(const T &v, const double sf, const double dc)
{
    return common::scale<T>(v, sf, dc);
}

template<typename T> inline T &ScalingFactorService::scale_I(T &v, const double sf, const double dc)
{
    return common::scale_I(v, sf, dc);
}

template<typename T> std::pair<double, double> ScalingFactorService::calc(const arma::Mat<T> &v, const double obseg)
{
    const auto dc = calc_dc_offset(v);
    auto sf = calc_scaling_factor<T>(v - dc, obseg);
    if (sf == 0) {
        sf = 1;
        LOG4_WARN("Scaling factor is zero, setting to 1");
    }
    LOG4_TRACE("DC offset " << dc << ", scaling factor " << sf << ", values " << common::present(v));
    return {dc, sf};
}

template<typename T> double ScalingFactorService::calc_dc_offset(const arma::Mat<T> &v)
{
    return common::mean(v);
}

template<typename T> double ScalingFactorService::calc_scaling_factor(const arma::Mat<T> &v, const double obseg)
{
#if 0
    return arma::median(arma::abs(arma::vectorise(v))) / obseg;
#else
    return common::meanabs(v) / obseg;
#endif
}

}
}

#endif //SCALINGFACTORSERVICE_TPP
