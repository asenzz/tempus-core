//
// Created by jarko on 13.09.17.
//

#ifndef SVR_KERNEL_SIGMOIDAL_HPP
#define SVR_KERNEL_SIGMOIDAL_HPP

#include "kernel_base.hpp"

namespace svr {

// K = tanh((V1*V2')*KernelParam + KernelParam2)

template<typename scalar_type>
class kernel_sigmoidal : public kernel_base<scalar_type>
{
private:
public:
    explicit kernel_sigmoidal(const SVRParameters &p): kernel_base<scalar_type> (p) {}
#if 0 // TODO port to Armadillo
    scalar_type operator()(
            const svr::datamodel::vektor<scalar_type> &a,
            const vektor<scalar_type> &b)
    {
        // K = tanh((V1*V2')*KernelParam + KernelParam2)
        auto K = vektor<double>::product_vector_scalar(a, b);
        K = std::tanh(K * this->parameters.get_svr_kernel_param() + this->parameters.get_svr_kernel_param2());
        return K;
    }
#endif

    void operator()(
            const viennacl::matrix<scalar_type> &features,
            viennacl::matrix<scalar_type> &kernel_matrix)
    {
        THROW_EX_FS(std::invalid_argument,  "This kernel was not implemented in this file properly!");
        kernel_matrix = viennacl::linalg::prod(features, viennacl::trans(features));
        viennacl::vector<double> diagonal = viennacl::diag(kernel_matrix, 0);
        viennacl::vector<double> i = viennacl::scalar_vector<double>(diagonal.size(), 1.);
        viennacl::matrix<double> temp = viennacl::linalg::outer_prod(i, diagonal);
        kernel_matrix = (temp + viennacl::trans(temp) - 2. * kernel_matrix) /
                (-2. * std::pow(this->parameters.get_svr_kernel_param(), 2));
        kernel_matrix = viennacl::linalg::element_exp(kernel_matrix);
    }


#ifdef VIENNACL_WITH_OPENCL

    void operator()(
            viennacl::ocl::context &ctx,
            const viennacl::matrix<scalar_type> &features,
            viennacl::matrix<scalar_type> &kernel_matrix)
    {
    	THROW_EX_FS(std::invalid_argument,  "This kernel was not implemented in this file properly!");
    }

#endif /* #ifdef VIENNACL_WITH_OPENCL */

};

}



#endif //SVR_KERNEL_SIGMOIDAL_HPP
