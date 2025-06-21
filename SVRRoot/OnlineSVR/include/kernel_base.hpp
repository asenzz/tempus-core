//
// Created by jarko on 11.09.17.
//

#ifndef SVR_KERNEL_BASE_HPP
#define SVR_KERNEL_BASE_HPP

#include <armadillo>
#include "model/SVRParameters.hpp"
#include "calc_cache.hpp"

namespace svr {
namespace kernel {

template<typename T> arma::Mat<T> get_reference_Z(const arma::Mat<T> &y);

template<typename T> void kernel_from_distances(RPTR(T) Kz, const uint32_t m, const uint32_t n, const T gamma, const T mean, const T degree);

template<typename T> void kernel_from_distances(RPTR(T) K, CRPTR(T) Z, const uint32_t m, const uint32_t n, const T gamma, const T mean, const T degree);

template<typename T> void d_kernel_from_distances(RPTR(T) d_K, CRPTR(T) d_Z, const uint32_t m, const uint32_t n, const T gamma, const T mean, const T degree, const cudaStream_t custream);

template<typename T>
class kernel_base {
protected:
    datamodel::SVRParameters &parameters;

public:
    datamodel::SVRParameters &get_parameters();

    datamodel::SVRParameters get_parameters() const;

    kernel_base() = default;

    explicit kernel_base(datamodel::SVRParameters &p);

    virtual ~kernel_base();

    void d_distances(CRPTR(T) d_X, const uint32_t m, const uint32_t n, RPTR(T) d_Z, const cudaStream_t custream) const;

    void d_kernel_from_distances(CRPTR(T) d_X, const uint32_t m, const uint32_t n, RPTR(T) d_Z, const cudaStream_t custream) const;

    arma::Mat<T> kernel(const arma::Mat<T> &X) const;

    arma::Mat<T> distances(const arma::Mat<T> &X) const;

    void kernel_from_distances_I(arma::Mat<T> &Kz) const;

    arma::Mat<T> kernel_from_distances(const arma::Mat<T> &Z) const;

    arma::Mat<T> kernel(business::calc_cache &cc, const arma::Mat<T> &X, const bpt::ptime &X_time) const;

    arma::Mat<T> kernel(business::calc_cache &cc, const arma::Mat<T> &X, const arma::Mat<T> &Xy, const bpt::ptime &X_time, const bpt::ptime &Xy_time) const;

    arma::Mat<T> distances(business::calc_cache &cc, const arma::Mat<T> &X, const bpt::ptime &X_time) const;

    arma::Mat<T> distances(business::calc_cache &cc, const arma::Mat<T> &X, const arma::Mat<T> &Xy, const bpt::ptime &X_time, const bpt::ptime &Xy_time) const;

    virtual arma::Mat<T> kernel(const arma::Mat<T> &X, const arma::Mat<T> &Xy) const = 0; // K is a kernel matrix

    virtual arma::Mat<T> distances(const arma::Mat<T> &X, const arma::Mat<T> &Xy) const = 0; // Z is a distance matrix

    virtual void d_kernel(CRPTR(T) d_X, const uint32_t m, RPTR(T) d_K, const cudaStream_t custream) const = 0;

    // m is common rows count (features length), n is the number of columns or samples
    virtual void d_distances(CRPTR(T) d_X, CRPTR(T) &d_Xy, const uint32_t m, const uint32_t n_X, const uint32_t n_Xy, RPTR(T) d_Z, const cudaStream_t custream) const = 0;
};

}
}

#include "kernel_base.tpp"

#endif //SVR_KERNEL_BASE_HPP
