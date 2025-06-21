//
// Created by jarko on 13.09.17.
//

#ifndef SVR_KERNEL_POLYNOMIAL_HPP
#define SVR_KERNEL_POLYNOMIAL_HPP

#include "kernel_base.hpp"

namespace svr {
namespace kernel {

template<typename T>
class kernel_polynomial : public kernel_base<T> {
    using kernel_base<T>::parameters;
public:
    explicit kernel_polynomial(datamodel::SVRParameters &p) : kernel_base<T>(p)
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

#ifdef DEPRECATED_KERNEL_API

// K = (V1*V2' + 1) ^ KernelParam

template<typename T>
class kernel_polynomial : public kernel_base<T>
{
    using kernel_base<T>::parameters;
public:
    explicit kernel_polynomial(const datamodel::SVRParameters &p): kernel_base<T> (p) {}

    void operator()(
            const viennacl::matrix<T> &features,
            viennacl::matrix<T> &kernel_matrix)
    {
        kernel_matrix = viennacl::linalg::prod(features, viennacl::trans(features));
        viennacl::matrix<T> I = viennacl::identity_matrix<T>(kernel_matrix.size1());
        kernel_matrix *= parameters.get_svr_kernel_param();
        kernel_matrix += I * parameters.get_svr_kernel_param2();
    }

#ifdef ENABLE_OPENCL
    using kernel_base<T>::operator();

    void operator()(
            viennacl::ocl::context &ctx,
            const viennacl::matrix<T> &features,
            viennacl::matrix<T> &kernel_matrix)
    {
        kernel_matrix = viennacl::linalg::prod(features, viennacl::trans(features));
        viennacl::matrix<T> I = viennacl::identity_matrix<T>(kernel_matrix.size1(), ctx);
        kernel_matrix *= this->parameters.get_svr_kernel_param();
        kernel_matrix += I * this->parameters.get_svr_kernel_param2();
        // dotMatrix = viennacl::linalg::element_pow(dotMatrix, 1);
    }
#endif /* #ifdef ENABLE_OPENCL */

};

#endif

}
}

#endif //SVR_KERNEL_POLYNOMIAL_HPP