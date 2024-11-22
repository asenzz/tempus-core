//
// Created by jarko on 13.09.17.
//

#ifndef SVR_KERNEL_POLYNOMIAL_HPP
#define SVR_KERNEL_POLYNOMIAL_HPP

#include "kernel_base.hpp"

namespace svr {

// K = (V1*V2' + 1) ^ KernelParam

template<typename scalar_type>
class kernel_polynomial : public kernel_base<scalar_type>
{
private:
public:
    explicit kernel_polynomial(const datamodel::SVRParameters &p): kernel_base<scalar_type> (p) {}

#if 0
    scalar_type operator()(const vektor<scalar_type> &a, const vektor<scalar_type> &b)
    {
        const auto K = vektor<scalar_type>::product_vector_scalar(a, b);
        return std::pow(K + 1, this->parameters.get_svr_kernel_param());
    }
#endif

    void operator()(
            const viennacl::matrix<scalar_type> &features,
            viennacl::matrix<scalar_type> &kernel_matrix)
    {
        kernel_matrix = viennacl::linalg::prod(features, viennacl::trans(features));
        viennacl::matrix<scalar_type> I = viennacl::identity_matrix<scalar_type>(kernel_matrix.size1());
        kernel_matrix *= this->parameters.get_svr_kernel_param();
        kernel_matrix += I * this->parameters.get_svr_kernel_param2();
        // dotMatrix = viennacl::linalg::element_pow(dotMatrix, 1);
    }

#ifdef ENABLE_OPENCL
    using kernel_base<scalar_type>::operator();

    void operator()(
            viennacl::ocl::context &ctx,
            const viennacl::matrix<scalar_type> &features,
            viennacl::matrix<scalar_type> &kernel_matrix)
    {
        kernel_matrix = viennacl::linalg::prod(features, viennacl::trans(features));
        viennacl::matrix<scalar_type> I = viennacl::identity_matrix<scalar_type>(kernel_matrix.size1(), ctx);
        kernel_matrix *= this->parameters.get_svr_kernel_param();
        kernel_matrix += I * this->parameters.get_svr_kernel_param2();
        // dotMatrix = viennacl::linalg::element_pow(dotMatrix, 1);
    }
#endif /* #ifdef ENABLE_OPENCL */

};

}

#endif //SVR_KERNEL_POLYNOMIAL_HPP
