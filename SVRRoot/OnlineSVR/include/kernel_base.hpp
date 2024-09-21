//
// Created by jarko on 11.09.17.
//

#ifndef SVR_KERNEL_BASE_HPP
#define SVR_KERNEL_BASE_HPP

#include <armadillo>
#include <viennacl/scalar.hpp>
#include <viennacl/vector.hpp>
#include <viennacl/matrix.hpp>
#include <viennacl/linalg/inner_prod.hpp>
#include <viennacl/linalg/norm_1.hpp>
#include <viennacl/linalg/prod.hpp>

#include "model/SVRParameters.hpp"
#include "common/gpu_handler.hpp"

namespace svr
{

datamodel::e_kernel_type get_kernel_type_from_string(const std::string &kernel_type_str);
std::string kernel_type_to_string(const datamodel::e_kernel_type kernel_type);

template<typename scalar_type>
class kernel_base
{
protected:
    const datamodel::SVRParameters parameters;

public:
    const datamodel::SVRParameters &get_parameters() const
    { return parameters; }

    kernel_base() = default;

    explicit kernel_base(const datamodel::SVRParameters &p) : parameters(p)
    {}

    virtual ~kernel_base()
    {}


    // virtual void operator()(const vmatrix<scalar_type> &features, vmatrix<scalar_type> &p_kernel_matrices) = 0; - not needed at this point, maybe we will put it back
    virtual void operator()(const viennacl::matrix<scalar_type> &features, viennacl::matrix<scalar_type> &kernel_matrix) = 0;  // CPU based - to be avoided

    virtual viennacl::matrix<scalar_type> operator()(const viennacl::matrix<scalar_type> &features)
    {
        viennacl::matrix<scalar_type> kernel_matrix;
        (*this)(features, kernel_matrix);
        return kernel_matrix;
    }


#ifdef VIENNACL_WITH_OPENCL

    //virtual vmatrix<scalar_type> operator()(
        //viennacl::ocl::context &ctx, const vmatrix<scalar_type> &features, vmatrix<scalar_type> &p_kernel_matrices) = 0; // not needed at this point?
        //if implemented, it should be done in this place and will remove code from build_kernel_matrix
    // virtual void operator()(viennacl::ocl::context &ctx, const viennacl::matrix<scalar_type> &features, viennacl::matrix<scalar_type> &kernel_matrix) {};

    virtual void operator()(const arma::mat & features, arma::mat & kernel_matrix)
    { LOG4_THROW("Not implemented"); }

    virtual double operator()(const arma::rowvec &a,  const arma::rowvec &b)
    { LOG4_THROW("Not implemented"); return std::numeric_limits<double>::signaling_NaN(); }

    virtual void operator()(const arma::mat &x_train,  const arma::mat &x_test, arma::mat &kernel_matrix)
    { LOG4_THROW("Not implemented"); }

    virtual void operator()(
            viennacl::ocl::context &ctx,
            const viennacl::matrix<scalar_type> &x,
            const viennacl::matrix<scalar_type> &y,
            viennacl::matrix<scalar_type> &kernel_matrix)
    { LOG4_THROW("Not implemented"); }

#endif
};


}

#endif //SVR_KERNEL_BASE_HPP
