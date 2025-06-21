#ifndef SVR_KERNEL_RADIAL_BASIS_EXPONENTIAL_HPP
#define SVR_KERNEL_RADIAL_BASIS_EXPONENTIAL_HPP

#include "kernel_base.hpp"
#include "util/math_utils.hpp"
#include "common/compatibility.hpp"

namespace svr {
namespace kernel {

// K = exp (-sum(dist(V1,V2) / 2*(KernelParam^2))

template<typename T>
class kernel_radial_basis_exponential : public kernel_base<T> {
public:
    explicit kernel_radial_basis_exponential(datamodel::SVRParameters &p) : kernel_base<T>(p)
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
    T operator()
            (const svr::datamodel::vektor<T> &a,
             const vektor<T> &b)
    {
        LOG4_DEBUG("In kernel_radial_basis_exponential");
        auto V = vektor<double>::subtract_vector(a, b);
        V.square_scalar();
        auto K = V.sum();
        if (this->parameters.get_svr_kernel_param() != 0.)
            K /= -(2. * std::pow(this->parameters.get_svr_kernel_param(), 2.));
        else
            K /= -2.;
        return std::exp(K);
    }

    void operator() (
            const viennacl::matrix<T> &features,
            viennacl::matrix<T> &kernel_matrix)
    {
        kernel_matrix = viennacl::linalg::prod(features, viennacl::trans(features));
        viennacl::vector<double> diagonal = viennacl::diag(kernel_matrix, 0);
        viennacl::vector<double> i = viennacl::scalar_vector<double>(diagonal.size(), 1.);
        viennacl::matrix<double> temp = viennacl::linalg::outer_prod(i, diagonal);
        kernel_matrix = temp + viennacl::trans(temp) - 2. * kernel_matrix;
        kernel_matrix = viennacl::linalg::element_sqrt(kernel_matrix);
        kernel_matrix = kernel_matrix / (-2. * std::pow(this->parameters.get_svr_kernel_param(), 2.));
        kernel_matrix = viennacl::linalg::element_exp(kernel_matrix);
    }
#endif


#ifdef ENABLE_OPENCL
    virtual viennacl::matrix<T> distances(const viennacl::matrix<T> &d_features) override
    {
        viennacl::matrix<T> d_kernel_matrix = viennacl::linalg::prod(d_features, viennacl::trans(d_features));
        viennacl::vector<T> dia = viennacl::diag(d_kernel_matrix, 0);
        viennacl::vector<T> i = viennacl::scalar_vector<T>(dia.size(), 1., d_features.handle().opencl_handle().context()); // d_features.memory_domain() == viennacl::memory_types::OPENCL_MEMORY ? d_features.handle().opencl_handle() : d_features.handle().ram_handle() );
        viennacl::matrix<T> tmp = viennacl::linalg::outer_prod(i, dia);
        d_kernel_matrix = viennacl::linalg::element_sqrt(tmp + viennacl::trans(tmp) - 2. * d_kernel_matrix);
        return d_kernel_matrix;
    }

    void operator() (
            const viennacl::matrix<T> &d_features,
            viennacl::matrix<T> &d_kernel_matrix) override
    {
        d_kernel_matrix = viennacl::linalg::element_exp(distances(d_features) / (-2. * std::pow<double>(this->parameters.get_svr_kernel_param(), 2.)));
    }

    virtual void operator()(const arma::mat &features, arma::mat &kernel_matrix) override
    {
        const svr::common::gpu_context c;
        viennacl::matrix<T> d_kernel_matrix(kernel_matrix.n_rows, kernel_matrix.n_cols, c.ctx());
        operator()(common::tovcl(features, c.ctx()), d_kernel_matrix);
        kernel_matrix = common::toarma(d_kernel_matrix);
#if 0
        kernel_matrix = features * features.t();
        arma::vec diagonal = diagvec(kernel_matrix);
        arma::vec i = ones(size(diagonal));
        arma::mat temp = diagonal * i.t();
        kernel_matrix = temp + temp.t() - kernel_matrix * 2.;
        kernel_matrix = sqrt(kernel_matrix);
        kernel_matrix = kernel_matrix / (-2. * std::pow(this->parameters.get_svr_kernel_param(), 2));
        kernel_matrix = exp(kernel_matrix);
#endif
    }


    virtual void operator()(const arma::mat &x_train, const arma::mat &x_test, arma::mat &kernel_matrix) override
    {
        kernel_matrix.resize(x_train.n_rows, x_test.n_rows);
        OMP_FOR_(x_train.n_rows * x_test.n_rows, collapse(2))
        for (unsigned i = 0; i < x_train.n_rows; ++i)
            for (unsigned j = 0; j < x_test.n_rows; ++j)
                kernel_matrix(i, j) = this->operator()(x_train.row(i), x_test.row(j));
    }

    inline virtual double distance(const arma::rowvec &a, const arma::rowvec &b)
    {
        return arma::norm(a - b, 2);
    }

    inline virtual double operator()(const arma::rowvec &a, const arma::rowvec &b) override
    {
        return distance(a, b) / (this->parameters.get_svr_kernel_param() ? -(2. * std::pow(this->parameters.get_svr_kernel_param(), 2.)) : -2.);
    }

#endif /* #ifdef ENABLE_OPENCL */

#endif

};

}
}

#endif //SVR_KERNEL_RADIAL_BASIS_EXPONENTIAL_HPP