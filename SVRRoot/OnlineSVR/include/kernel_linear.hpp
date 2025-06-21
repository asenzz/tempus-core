#ifndef SVR_KERNEL_LINEAR_HPP
#define SVR_KERNEL_LINEAR_HPP

#include "kernel_base.hpp"

namespace svr {
namespace kernel {
// K = V1 * V2'

template<typename T>
class kernel_linear : public kernel_base<T> {
public:
    explicit kernel_linear(datamodel::SVRParameters &p) : kernel_base<T>(p)
    {}

    virtual arma::Mat<T> kernel(const arma::Mat<T> &X, const arma::Mat<T> &Xy) const
    { return {}; }

    virtual arma::Mat<T> distances(const arma::Mat<T> &X, const arma::Mat<T> &Xy) const
    { return {}; }

    virtual void d_kernel(CRPTR(T) d_Z, const uint32_t m, RPTR(T) d_K, const cudaStream_t custream) const
    {}

    virtual void d_distances(CRPTR(T) d_X, CRPTR(T) &d_Xy, const uint32_t m, const uint32_t n_X, const uint32_t n_Xy, RPTR(T) d_Z, const cudaStream_t custream) const
    {}

#ifdef DEPRECATED_KERNEL_API
    void operator() (const viennacl::matrix<T> &features, viennacl::matrix<T> &kernel_matrix)
    {
        kernel_matrix = viennacl::linalg::prod(features, viennacl::trans(features));
    }

    void operator() (viennacl::ocl::context &ctx,const viennacl::matrix<T> &features, viennacl::matrix<T> &kernel_matrix)
    {
        kernel_matrix = viennacl::linalg::prod(features, viennacl::trans(features));
    }
#endif

};

}
}

#endif //SVR_KERNEL_LINEAR_HPP