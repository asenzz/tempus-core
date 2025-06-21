#ifndef SVR_KERNEL_RADIAL_BASIS_HPP
#define SVR_KERNEL_RADIAL_BASIS_HPP

#include "kernel_base.hpp"

namespace svr {
namespace kernel {

// K = exp (-KernelParam * sum(dist(V1,V2)^2))

template<typename T>
class kernel_radial_basis : public kernel_base<T> {
public:
    explicit kernel_radial_basis(datamodel::SVRParameters &p) : kernel_base<T>(p)
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

#if 0
    T operator()(const svr::datamodel::vektor<T> &a, const vektor<T> &b)
    {
        auto V = svr::datamodel::vektor<T>::subtract_vector(a, b);
        V.square_scalar();
        auto K = V.sum();
        K *= -this->parameters.get_svr_kernel_param();
        return std::exp(K);
    }
#endif

    viennacl::matrix<T> distances(const viennacl::matrix<T> &features) override
    {
        viennacl::matrix<T> kernel_matrix = viennacl::linalg::prod(features, viennacl::trans(features));
        viennacl::vector<T> diagonal = viennacl::diag(kernel_matrix, 0);
        viennacl::vector<T> i = viennacl::scalar_vector<T>(diagonal.size(), 1.);
        viennacl::matrix<T> temp = viennacl::linalg::outer_prod(i, diagonal);

        kernel_matrix = (temp + viennacl::trans(temp) - 2. * kernel_matrix);
        return kernel_matrix;
    }

    void operator() (
            const viennacl::matrix<T> &features,
            viennacl::matrix<T> &kernel_matrix) override
    {
        kernel_matrix = viennacl::linalg::element_exp(distances(features) * -this->parameters.get_svr_kernel_param());
    }

#ifdef ENABLE_OPENCL
    using kernel_base<T>::operator();

    void operator() (
            viennacl::ocl::context &ctx,
            const viennacl::matrix<T> &features,
            viennacl::matrix<T> &kernel_matrix)
    {
        kernel_matrix = viennacl::linalg::prod(features, viennacl::trans(features));
        viennacl::vector<T> diagonal = viennacl::diag(kernel_matrix, 0);
        viennacl::vector<T> i = viennacl::scalar_vector<T>(diagonal.size(), 1, ctx);
        viennacl::matrix<T> temp = viennacl::linalg::outer_prod(i, diagonal);

        kernel_matrix = (temp + viennacl::trans(temp) - 2. * kernel_matrix) * -this->parameters.get_svr_kernel_param();
        kernel_matrix = viennacl::linalg::element_exp(kernel_matrix);
    }
#endif /* #ifdef ENABLE_OPENCL */

#endif

};

}
}

#endif //SVR_KERNEL_RADIAL_BASIS_HPP
