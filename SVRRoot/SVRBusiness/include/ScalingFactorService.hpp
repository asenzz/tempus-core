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
    template<typename T> static double calc_scaling_factor(const arma::Mat<T> &v, const double obseg = common::C_input_obseg_labels);
    template<typename T> static double calc_dc_offset(const arma::Mat<T> &v);
    template<typename T> static std::pair<double, double> calc(const arma::Mat<T> &v, const double obseg);

    template<typename T> inline static T scale(const T &v);
    template<typename T> inline static T &scale_I(T &v);
    template<typename T> inline static T &scale_calc_I(T &v, double &sf, double &dc, const double obseg = common::C_input_obseg_labels);
    template<typename T> inline static T scale_calc(T v, double &sf, double &dc, const double obseg = common::C_input_obseg_labels);
    static void cu_scale_calc_I(RPTR(double) v, const size_t n, double &sf, double &dc, const cudaStream_t custream);

    template<typename T> inline static T scale(const T &v, const double sf, const double dc);
    template<typename T> inline static T &scale_I(T &v, const double sf, const double dc);
};

}
}

#include "ScalingFactorService.tpp"

#endif //SVR_SCALINGFACTORSERVICE_HPP
