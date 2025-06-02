//
// Created by zarko on 19/03/2025.
//

#include <oneapi/mkl/types.hpp>
#include <armadillo>
#include "kernel_path.hpp"
#include "appcontext.hpp"
#include "util/math_utils.hpp"
#include "common/parallelism.hpp"

namespace svr {
namespace kernel {
// Specializations
#define T double

template<> kernel_path<T>::kernel_path(const datamodel::SVRParameters &p) : kernel_base<T>(p)
{
}

template<> kernel_path<T>::kernel_path(const kernel_base<T> &k) : kernel_base<T>(k.get_parameters())
{
}

template<> arma::Mat<T> kernel_path<T>::kernel(const arma::Mat<T> &X, const arma::Mat<T> &Xy) const
{
    arma::Mat<T> Ky(X.n_cols, Xy.n_cols, ARMA_DEFAULT_FILL);
    kernel_xy(X.n_cols, Xy.n_cols, X.n_rows, parameters.get_lag_count(), parameters.get_svr_kernel_param(), parameters.get_min_Z(),
              parameters.get_svr_kernel_param2(), parameters.get_kernel_param3(), parameters.get_H_feedback(), parameters.get_D_feedback(), parameters.get_V_feedback(),
              X.mem, Xy.mem, Ky.memptr());
    LOG4_TRACE("Prepared kernel " << common::present(Ky) << " with parameters " << parameters << ", from X " << common::present(X) << " and Xy " << common::present(Xy));
    return Ky;
}

template<> arma::Mat<T> kernel_path<T>::distances(const arma::Mat<T> &X, const arma::Mat<T> &Xy) const
{
    arma::Mat<T> Zy(X.n_cols, Xy.n_cols, ARMA_DEFAULT_FILL);
    distances_xy(X.n_cols, Xy.n_cols, X.n_rows, parameters.get_lag_count(), parameters.get_kernel_param3(), parameters.get_H_feedback(),
                 parameters.get_D_feedback(), parameters.get_V_feedback(), X.mem, Xy.mem, Zy.memptr());
    LOG4_TRACE("Prepared distances " << common::present(Zy) << " with parameters " << parameters << ", from X " << common::present(X) << " and Xy " << common::present(Xy));
    return Zy;
}


template<> void kernel_path<T>::d_kernel(CRPTR(T) d_Z, const uint32_t m, RPTR(T) d_K, const cudaStream_t custream) const
{
    kernel::d_kernel_from_distances(d_K, d_Z, m, m, parameters.get_svr_kernel_param(), parameters.get_min_Z(), parameters.get_svr_kernel_param2(), custream);
}

template<> void kernel_path<T>::d_distances(CRPTR(T) d_X, CRPTR(T) &d_Xy, const uint32_t m, const uint32_t n_X, const uint32_t n_Xy, RPTR(T) d_Z, const cudaStream_t custream) const
{
    cu_distances_xy(n_X, n_Xy, parameters.get_lag_count(), m / parameters.get_lag_count(), parameters.get_kernel_param3(), parameters.get_H_feedback(),
                    parameters.get_D_feedback(), parameters.get_V_feedback(), d_X, d_Xy, d_Z, custream);
}

}
}
