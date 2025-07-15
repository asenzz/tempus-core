//
// Created by zarko on 10/1/24.
//

#ifndef SVR_SCALINGFACTORSERVICE_HPP
#define SVR_SCALINGFACTORSERVICE_HPP

#include <cublas_v2.h>
#include <armadillo>
#include "util/math_utils.hpp"

namespace svr {
namespace business {

class ScalingFactorService {
public:
    static double calc_scaling_factor(const arma::mat &v, const double obseg = common::C_input_obseg_labels);
    static double calc_dc_offset(const arma::mat &v);
    static std::pair<double, double> calc(const arma::mat &v, const double obseg);

    template<typename T> inline static T scale(const T &v)
    {
        const auto [dc, sf] = calc(v, common::C_input_obseg_labels);
        return common::scale(v, sf, dc);
    }

    template<typename T> inline static T &scale_I(T &v)
    {
        double sf, dc;
        return scale_calc_I(v, sf, dc);
    }

    template<typename T> inline static T &scale_calc_I(T &v, double &sf, double &dc, const double obseg = common::C_input_obseg_labels)
    {
        std::tie(dc, sf) = calc(v, obseg);
        return common::scale_I(v, sf, dc);
    }

    static void cu_scale_calc_I(RPTR(double) v, const size_t n, double &sf, double &dc, const cudaStream_t custream, const cublasHandle_t cublas_H);

    template<typename T> inline static T scale(const T &v, const double sf, const double dc)
    {
        return common::scale<T>(v, sf, dc);
    }

    template<typename T> inline static T &scale_I(T &v, const double sf, const double dc)
    {
        return common::scale_I(v, sf, dc);
    }
};

}
}

#endif //SVR_SCALINGFACTORSERVICE_HPP
