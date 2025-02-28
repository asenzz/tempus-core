//
// Created by zarko on 10/1/24.
//

#include <ipp.h>
#include "common/constants.hpp"
#include "ScalingFactorService.hpp"


namespace svr {
namespace business {


double ScalingFactorService::calc_dc_offset(const arma::mat &v)
{
    return common::mean(v);
}


double ScalingFactorService::calc_scaling_factor(const arma::mat &v)
{
#if 0
    return arma::median(arma::abs(arma::vectorise(v))) / common::C_input_obseg_labels;
#else
    return common::meanabs(v) / common::C_input_obseg_labels;
#endif
}


std::pair<double, double> ScalingFactorService::calc(const arma::mat &v)
{
    const auto dc = calc_dc_offset(v);
    const auto sf = calc_scaling_factor(v - dc);
    LOG4_TRACE("DC offset " << dc << ", scaling factor " << sf << ", values " << common::present(v));
    return {dc, sf};
}


}
}