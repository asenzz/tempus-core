//
// Created by zarko on 19/03/2025.
//

#ifndef SVR_KERNEL_BASE_TPP
#define SVR_KERNEL_BASE_TPP

#include "kernel_base.hpp"
#include "common/compatibility.hpp"
#include "calc_cache.hpp"
#include "util/math_utils.hpp"

namespace svr {
namespace kernel {

template<typename T> arma::Mat<T> get_reference_Z(const arma::Mat<T> &y)
{
    const uint32_t n = y.n_rows;
    const arma::Mat<T> y_t = y.t();
    arma::Mat<T> r(n, n, ARMA_DEFAULT_FILL);
    OMP_FOR_i(n) r.row(i) = y(i, 0) - y_t;
    LOG4_TRACE("Prepared reference kernel matrix " << common::present(r) << " from labels " << common::present(y));
    return r;
}

template<typename T> datamodel::SVRParameters &kernel_base<T>::get_parameters()
{
    return parameters;
}

template<typename T> datamodel::SVRParameters kernel_base<T>::get_parameters() const
{
    return parameters;
}

template<typename T> kernel_base<T>::kernel_base(datamodel::SVRParameters &p) : parameters(p)
{}

template<typename T> kernel_base<T>::~kernel_base() = default;

template<typename T> void kernel_base<T>::d_distances(CRPTR(T) d_X, const uint32_t m, const uint32_t n, RPTR(T) d_Z, const cudaStream_t custream) const
{
    d_distances(d_X, d_X, m, n, n, d_Z, custream);
}

template<typename T> void kernel_base<T>::d_kernel_from_distances(CRPTR(T) d_X, const uint32_t m, const uint32_t n, RPTR(T) d_Z, const cudaStream_t custream) const
{
    d_kernel_from_distances(d_X, d_Z, m, n, parameters.get_svr_kernel_param(), parameters.get_min_Z(), parameters.get_svr_kernel_param2(), custream);
}

template<typename T> arma::Mat<T> kernel_base<T>::kernel(const arma::Mat<T> &X) const
{
    return kernel(X, X);
}

template<typename T> arma::Mat<T> kernel_base<T>::distances(const arma::Mat<T> &X) const
{
    return distances(X, X);
}

template<typename T> void kernel_base<T>::kernel_from_distances_I(arma::Mat<T> &Kz) const
{
    kernel::kernel_from_distances(Kz.memptr(), Kz.n_rows, Kz.n_cols, parameters.get_svr_kernel_param(), parameters.get_min_Z(), parameters.get_svr_kernel_param2());
    LOG4_TRACE("Prepared K " << common::present(Kz) << " with parameters " << parameters);
}

template<typename T> arma::Mat<T> kernel_base<T>::kernel_from_distances(const arma::Mat<T> &Z) const
{
    arma::Mat<T> K(arma::size(Z), ARMA_DEFAULT_FILL);
    kernel::kernel_from_distances(K.memptr(), Z.mem, Z.n_rows, Z.n_cols, parameters.get_svr_kernel_param(), parameters.get_min_Z(), parameters.get_svr_kernel_param2());
    LOG4_TRACE("Prepared K " << common::present(K) << " with parameters " << parameters << ", from Z " << common::present(Z));
    return K;
}

template<typename T> arma::Mat<T> kernel_base<T>::kernel(business::calc_cache &cc, const arma::Mat<T> &X, const bpt::ptime &X_time) const
{
    LOG4_BEGIN();
    return cc.get_Ky(*this, X, X, X_time, X_time);
}

template<typename T> arma::Mat<T> kernel_base<T>::kernel(
        business::calc_cache &cc, const arma::Mat<T> &X, const arma::Mat<T> &Xy, const bpt::ptime &X_time, const bpt::ptime &Xy_time) const
{
    return cc.get_Ky(*this, X, Xy, X_time, Xy_time);
}

template<typename T> arma::Mat<T> kernel_base<T>::distances(business::calc_cache &cc, const arma::Mat<T> &X, const bpt::ptime &X_time) const
{
    return cc.get_Zy(*this, X, X, X_time, X_time);
}

template<typename T> arma::Mat<T> kernel_base<T>::distances(
        business::calc_cache &cc, const arma::Mat<T> &X, const arma::Mat<T> &Xy, const bpt::ptime &X_time, const bpt::ptime &Xy_time) const
{
    return cc.get_Zy(*this, X, Xy, X_time, Xy_time);
}

}
}

#endif //SVR_KERNEL_BASE_TPP
