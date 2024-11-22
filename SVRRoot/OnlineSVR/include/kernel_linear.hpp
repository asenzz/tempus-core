#ifndef SVR_KERNEL_LINEAR_HPP
#define SVR_KERNEL_LINEAR_HPP

#include "kernel_base.hpp"

namespace svr {

// K = V1 * V2'

template<typename scalar_type>
class kernel_linear: public kernel_base<scalar_type> {
public:
    explicit kernel_linear(const datamodel::SVRParameters &p): kernel_base<scalar_type> (p) {}

    void operator() (
            const viennacl::matrix<scalar_type> &features,
            viennacl::matrix<scalar_type> &kernel_matrix)
    {
        kernel_matrix = viennacl::linalg::prod(features, viennacl::trans(features));
    }

#ifdef ENABLE_OPENCL
    using kernel_base<scalar_type>::operator();

    void operator() (
            viennacl::ocl::context &ctx,
            const viennacl::matrix<scalar_type> &features,
            viennacl::matrix<scalar_type> &kernel_matrix)
    {
        kernel_matrix = viennacl::linalg::prod(features, viennacl::trans(features));
    }
#endif /* #ifdef ENABLE_OPENCL */

};

}

#endif //SVR_KERNEL_LINEAR_HPP
