#ifndef SVR_KERNEL_RADIAL_BASIS_EXPONENTIAL_HPP
#define SVR_KERNEL_RADIAL_BASIS_EXPONENTIAL_HPP

#include "kernel_base.hpp"
#include "util/math_utils.hpp"

namespace svr
{

// K = exp (-sum(dist(V1,V2) / 2*(KernelParam^2))

template<typename scalar_type>
class kernel_radial_basis_exponential: public kernel_base<scalar_type> {
private:
public:
    explicit kernel_radial_basis_exponential(const SVRParameters &p): kernel_base<scalar_type> (p) {}

#if 0
    scalar_type operator()
            (const svr::datamodel::vektor<scalar_type> &a,
             const vektor<scalar_type> &b)
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
#endif

    void operator() (
            const viennacl::matrix<scalar_type> &features,
            viennacl::matrix<scalar_type> &kernel_matrix)
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


#ifdef VIENNACL_WITH_OPENCL
    void operator() (
            viennacl::ocl::context &ctx,
            const viennacl::matrix<scalar_type> &features,
            viennacl::matrix<scalar_type> &kernel_matrix)
    {
        kernel_matrix = viennacl::linalg::prod(features, viennacl::trans(features));
        viennacl::vector<double> diagonal = viennacl::diag(kernel_matrix, 0);
        viennacl::vector<double> i = viennacl::scalar_vector<double>(diagonal.size(), 1, ctx);
        viennacl::matrix<double> temp = viennacl::linalg::outer_prod(i, diagonal);
        kernel_matrix = temp + viennacl::trans(temp) - 2. * kernel_matrix;
        kernel_matrix = viennacl::linalg::element_sqrt(kernel_matrix);
        kernel_matrix = kernel_matrix / (-2. * std::pow(this->parameters.get_svr_kernel_param(), 2));
        kernel_matrix = viennacl::linalg::element_exp(kernel_matrix);
    }

    virtual void operator()(const arma::mat & features, arma::mat & kernel_matrix)
    {
        kernel_matrix = features * features.t();
        arma::vec diagonal = diagvec(kernel_matrix);
        arma::vec i = ones(size(diagonal));
        arma::mat temp = diagonal * i.t();
        kernel_matrix = temp + temp.t() - kernel_matrix * 2.;
        kernel_matrix = sqrt(kernel_matrix);
        kernel_matrix = kernel_matrix / (-2. * std::pow(this->parameters.get_svr_kernel_param(), 2));
        kernel_matrix = exp(kernel_matrix);
    }

#if 0 // TODO Port to Armadillo
    void operator()(const arma::mat &features,  const arma::rowvec &learning, vektor<double> & kernel_values){
        for(size_t i = 0; i < features.n_rows; ++i){
                kernel_values[i] = (*this)(features.row(i), learning);
        }
    }
#endif

    void operator()(const arma::mat &x_train, const arma::mat &x_test, arma::mat &kernel_matrix)
    {
        kernel_matrix.resize(x_train.n_rows, x_test.n_rows);

        svr::common::armd::print_arma_sizes(x_train, "x_train");
        svr::common::armd::print_arma_sizes(x_test, "x_test");
        svr::common::armd::print_arma_sizes(kernel_matrix, "kernel_matrix");

#pragma omp parallel for collapse(2)
        for (size_t i = 0; i < x_train.n_rows; ++i)
            for (size_t j = 0; j < x_test.n_rows; ++j)
                kernel_matrix(i, j) = (*this)(x_train.row(i), x_test.row(j));
    }

    double arma_distance(const arma::rowvec &a, const arma::rowvec &b)
    {
        double ret = 0.0;
        for(size_t i = 0; i < a.n_elem; ++i){

            double dist = a(i) - b(i);
            ret += dist * dist;
        }
        return ret > 0.0 ? std::sqrt(ret) : 0.0;

    }

    virtual double operator()(const arma::rowvec &a,  const arma::rowvec &b)
    {
        arma::rowvec V = a - b;
        double K = norm(V, 2);
        //double K = arma_distance(a, b);
        if (this->parameters.get_svr_kernel_param() != 0.)
            K /= -(2. * std::pow(this->parameters.get_svr_kernel_param(), 2.));
        else
            K /= -2.;
        return std::exp(K);
    }



#endif /* #ifdef VIENNACL_WITH_OPENCL */

};


}

#endif //SVR_KERNEL_RADIAL_BASIS_EXPONENTIAL_HPP
