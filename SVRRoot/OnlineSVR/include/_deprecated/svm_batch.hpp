#ifndef _SVM_VCL_H
#define _SVM_VCL_H


#define DEFAULT_WORKING_SET 1024

#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/matrix_proxy.hpp"
#include "viennacl/linalg/inner_prod.hpp"
#include "viennacl/linalg/norm_1.hpp"
#include "viennacl/linalg/prod.hpp"

#include <algorithm>


#include "OnlineSMOSVR.hpp"
#include "model/SVRParameters.hpp"
#include "r_matrix.h"
#include <boost/interprocess/sync/named_semaphore.hpp>

namespace svr {
namespace batch {


class svm_batch
{
private:
    const bool update_r_matrix_;
    const size_t max_smo_iterations_;
    const double smo_epsilon_divisor_;

public:
    explicit svm_batch(
            const bool update_r_matrix = !common::__dont_update_r_matrix,
            const size_t max_smo_iterations = common::__max_iter,
            const double smo_epsilon_divisor = common::__smo_epsilon_divisor) :
        update_r_matrix_(update_r_matrix),
        max_smo_iterations_(max_smo_iterations),
        smo_epsilon_divisor_(smo_epsilon_divisor) {}

    bool get_update_r_matrix() const { return update_r_matrix_; }

    void train(
            const arma::mat &learning_data,
            const arma::colvec &reference_data,
            ::svr::OnlineSVR &model,
            const bool only_sv = false);

    void merge_models(
        const datamodel::vmatrix<double> &learning_data,
        const datamodel::vektor<double> &reference_data,
        const std::vector<::svr::OnlineSVR> &models,
        svr::OnlineSVR &model); // TODO Test if warm start SMO can do merging of models and implement

    void
    solve_epsilon_svr_warm_start(
            const viennacl::matrix<double> &q_matrix,
            const datamodel::vektor<double> &y,
            datamodel::vektor<double> &alpha,
            double &in_out_rho,
            svr::OnlineSVR &model);

private:
    size_t
    solve_epsilon_svr(
            const viennacl::matrix<double> &q_matrix,
            const viennacl::vector<double> &y,
            const svr::datamodel::SVRParameters &param,
            viennacl::vector<double> &alpha,
            double &out_rho,
            int loo);
};


} //batch
} //svr


#endif /* _SVM_VCL_H */
