//
// Created by jarko on 13.09.17.
//

#ifndef SVR_KERNEL_SIGMOIDAL_HPP
#define SVR_KERNEL_SIGMOIDAL_HPP

#include "kernel_base.hpp"

namespace svr {
namespace kernel {

// K = tanh((V1*V2')*KernelParam + KernelParam2)

template<typename T> class kernel_sigmoidal : public kernel_base<T> {
public:
    explicit kernel_sigmoidal(const datamodel::SVRParameters &p) : kernel_base<T>(p)
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

#if 0 // TODO port to Armadillo
    T operator()(
            const svr::datamodel::vektor<T> &a,
            const vektor<T> &b)
    {
        // K = tanh((V1*V2')*KernelParam + KernelParam2)
        auto K = vektor<double>::product_vector_scalar(a, b);
        K = std::tanh(K * this->parameters.get_svr_kernel_param() + this->parameters.get_svr_kernel_param2());
        return K;
    }
#endif

    void operator()(
            const viennacl::matrix<T> &features,
            viennacl::matrix<T> &kernel_matrix)
    {
        THROW_EX_FS(std::invalid_argument, "This kernel was not implemented in this file properly!");
        kernel_matrix = viennacl::linalg::prod(features, viennacl::trans(features));
        viennacl::vector<double> diagonal = viennacl::diag(kernel_matrix, 0);
        viennacl::vector<double> i = viennacl::scalar_vector<double>(diagonal.size(), 1.);
        viennacl::matrix<double> temp = viennacl::linalg::outer_prod(i, diagonal);
        kernel_matrix = (temp + viennacl::trans(temp) - 2. * kernel_matrix) /
                        (-2. * std::pow(this->parameters.get_svr_kernel_param(), 2));
        kernel_matrix = viennacl::linalg::element_exp(kernel_matrix);
    }


    using kernel_base<T>::operator();

    void operator()(
            viennacl::ocl::context &ctx,
            const viennacl::matrix<T> &features,
            viennacl::matrix<T> &kernel_matrix)
    {
        THROW_EX_FS(std::invalid_argument, "This kernel was not implemented in this file properly!");
    }

#endif // #ifdef DEPRECATED_KERNEL_API

};

}
}


#endif // SVR_KERNEL_SIGMOIDAL_HPP
