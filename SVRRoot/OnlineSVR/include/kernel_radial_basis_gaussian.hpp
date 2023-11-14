#ifndef SVR_KERNEL_RADIAL_BASIS_GAUSSIAN_HPP
#define SVR_KERNEL_RADIAL_BASIS_GAUSSIAN_HPP

#include "kernel_base.hpp"

namespace svr {

// K = exp (-sum(dist(V1,V2) / 2*(KernelParam^2))

template<typename scalar_type>
class kernel_radial_basis_gaussian: public kernel_base<scalar_type>
{
private:
public:
    explicit kernel_radial_basis_gaussian(const SVRParameters &p): kernel_base<scalar_type> (p) {}

#if 0
    scalar_type operator()
            (const svr::datamodel::vektor<scalar_type> &a,
             const vektor<scalar_type> &b)
    {
       // K = exp (-sum(dist(V1,V2)^2 / 2*(KernelParam^2))
        auto V = vektor<double>::subtract_vector(a, b);
        V.square_scalar();
        auto K = V.sum();
        if (this->parameters.get_svr_kernel_param() != 0)
            K /= -(2. * std::pow(this->parameters.get_svr_kernel_param(), 2));
        else
            K /= -2.;
        return std::exp(K);
    }
#endif

    void operator()(
            const viennacl::matrix<scalar_type> &features,
            viennacl::matrix<scalar_type> &kernel_matrix)
    {
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
        kernel_matrix = viennacl::linalg::prod(features, viennacl::trans(features));
        viennacl::vector<double> diagonal = viennacl::diag(kernel_matrix, 0);
        viennacl::vector<double> i = viennacl::scalar_vector<double>(diagonal.size(), 1, ctx);
        viennacl::matrix<double> temp = viennacl::linalg::outer_prod(i, diagonal);
        kernel_matrix = (temp + viennacl::trans(temp) - 2. * kernel_matrix) /
                        (-2. * std::pow(this->parameters.get_svr_kernel_param(), 2));
        kernel_matrix = viennacl::linalg::element_exp(kernel_matrix);
    }

#endif /* #ifdef VIENNACL_WITH_OPENCL */

};

}


#endif //SVR_KERNEL_RADIAL_BASIS_GAUSSIAN_HPP
