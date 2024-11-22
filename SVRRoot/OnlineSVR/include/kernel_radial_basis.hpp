#ifndef SVR_KERNEL_RADIAL_BASIS_HPP
#define SVR_KERNEL_RADIAL_BASIS_HPP

#include "kernel_base.hpp"

namespace svr
{

// K = exp (-KernelParam * sum(dist(V1,V2)^2))

template<typename scalar_type>
class kernel_radial_basis: public kernel_base<scalar_type> {
private:
public:
    explicit kernel_radial_basis(const datamodel::SVRParameters &p): kernel_base<scalar_type> (p) {}

#if 0
    scalar_type operator()(const svr::datamodel::vektor<scalar_type> &a, const vektor<scalar_type> &b)
    {
        auto V = svr::datamodel::vektor<scalar_type>::subtract_vector(a, b);
        V.square_scalar();
        auto K = V.sum();
        K *= -this->parameters.get_svr_kernel_param();
        return std::exp(K);
    }
#endif

    viennacl::matrix<scalar_type> distances(const viennacl::matrix<scalar_type> &features) override
    {
        viennacl::matrix<scalar_type> kernel_matrix = viennacl::linalg::prod(features, viennacl::trans(features));
        viennacl::vector<scalar_type> diagonal = viennacl::diag(kernel_matrix, 0);
        viennacl::vector<scalar_type> i = viennacl::scalar_vector<scalar_type>(diagonal.size(), 1.);
        viennacl::matrix<scalar_type> temp = viennacl::linalg::outer_prod(i, diagonal);

        kernel_matrix = (temp + viennacl::trans(temp) - 2. * kernel_matrix);
        return kernel_matrix;
    }

    void operator() (
            const viennacl::matrix<scalar_type> &features,
            viennacl::matrix<scalar_type> &kernel_matrix) override
    {
        kernel_matrix = viennacl::linalg::element_exp(distances(features) * -this->parameters.get_svr_kernel_param());
    }

#ifdef ENABLE_OPENCL
    using kernel_base<scalar_type>::operator();

    void operator() (
            viennacl::ocl::context &ctx,
            const viennacl::matrix<scalar_type> &features,
            viennacl::matrix<scalar_type> &kernel_matrix)
    {
        kernel_matrix = viennacl::linalg::prod(features, viennacl::trans(features));
        viennacl::vector<scalar_type> diagonal = viennacl::diag(kernel_matrix, 0);
        viennacl::vector<scalar_type> i = viennacl::scalar_vector<scalar_type>(diagonal.size(), 1, ctx);
        viennacl::matrix<scalar_type> temp = viennacl::linalg::outer_prod(i, diagonal);

        kernel_matrix = (temp + viennacl::trans(temp) - 2. * kernel_matrix) * -this->parameters.get_svr_kernel_param();
        kernel_matrix = viennacl::linalg::element_exp(kernel_matrix);
    }
#endif /* #ifdef ENABLE_OPENCL */

};


}

#endif //SVR_KERNEL_RADIAL_BASIS_HPP
