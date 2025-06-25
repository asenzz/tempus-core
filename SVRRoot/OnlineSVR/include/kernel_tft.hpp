//
// Created by zarko on 16/06/2025.
//

#ifndef KERNEL_TFT_HPP
#define KERNEL_TFT_HPP

#include "kernel_base.hpp"

namespace svr {
namespace kernel {


template<typename T> class kernel_tft final : public kernel_base<T>
{
public:
    void init(const arma::Mat<T> &X, const arma::Mat<T> &Y);

    explicit kernel_tft(datamodel::SVRParameters &p);

    arma::Mat<T> kernel(const arma::Mat<T> &X, const arma::Mat<T> &Xy) const override;

    arma::Mat<T> distances(const arma::Mat<T> &X, const arma::Mat<T> &Xy) const override;

    virtual void d_kernel(CRPTR (T) d_Z, const uint32_t m, RPTR (T) d_K, const cudaStream_t custream) const override;

    virtual void d_distances(CRPTR (T) d_X, CRPTR (T) &d_Xy, const uint32_t m, const uint32_t n_X, const uint32_t n_Xy, RPTR (T) d_Z, const cudaStream_t custream) const override;
};

} // kernel
} // svr

#endif //KERNEL_TFT_HPP
