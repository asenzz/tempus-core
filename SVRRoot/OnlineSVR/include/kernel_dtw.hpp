//
// Created by zarko on 18/03/2025.
//

#ifndef SVR_KERNEL_DTW_HPP
#define SVR_KERNEL_DTW_HPP

#include <vector>
#include "common/gpu_handler.hpp"
#include "kernel_base.hpp"
// #include "new_path_kernel.cuh"

namespace svr {
namespace kernel {

template<typename T>
class kernel_dtw : public kernel_base<T> {
public:
    explicit kernel_dtw(const datamodel::SVRParameters &p) : kernel_base<T>(p)
    {}

    virtual arma::Mat<T> kernel(const arma::Mat<T> &X, const arma::Mat<T> &Xy) const
    { return {}; }

    virtual arma::Mat<T> distances(const arma::Mat<T> &X, const arma::Mat<T> &Xy) const
    { return {}; }

    virtual void d_kernel(CRPTR(T) d_Z, const uint32_t m, RPTR(T) d_K, const cudaStream_t custream) const
    {}

    virtual void d_distances(CRPTR(T) d_X, CRPTR(T) &d_Xy, const uint32_t m, const uint32_t n_X, const uint32_t n_Xy, RPTR(T) d_Z, const cudaStream_t custream) const
    {}
};

// TODO Integrate the below
// auto const d_K_train = kernel::cu_compute_path_distances(dx_.d_train_features_t, dx_.d_train_features_t, train_F_rows, train_F_rows, train_F_cols, train_F_cols, lag, lambda, 0, custream);
// *p_Zy = kernel::path_distances_t(features_t, predict_features_t, params.get_lag_count(), params.get_svr_kernel_param2(), params.get_kernel_param3());
// *p_Z = kernel::path_distances_t(features_t, features_t, params.get_lag_count(), params.get_svr_kernel_param2(), params.get_kernel_param3())

}
}

#endif //SVR_KERNEL_DTW_HPP
