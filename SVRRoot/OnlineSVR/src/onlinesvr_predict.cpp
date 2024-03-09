//
// Created by zarko on 11/11/21.
//

//#define BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS 1 // Otherwise Valgrind crashes on init
#include <armadillo>
#include <boost/math/distributions/students_t.hpp>
#include <mutex>
#include <cmath>
#include <tuple>
#include <unordered_set>
#include <vector>
#include "common/parallelism.hpp"
#include "common/gpu_handler.hpp"
#include "util/math_utils.hpp"
#include "onlinesvr.hpp"
#include "cuqrsolve.hpp"

namespace svr {
namespace datamodel {

arma::mat OnlineMIMOSVR::predict(const arma::mat &x_predict)
{
    if (is_manifold()) return manifold_predict(x_predict);

    arma::mat prediction;
    OMP_LOCK(predict_l);
#pragma omp parallel for schedule(static, 1) num_threads(adj_threads(ixs.size()))
    for (size_t i = 0; i < ixs.size(); ++i) {
        const auto &params = *get_params_ptr(i);
        const arma::mat chunk_predict_K = prepare_Ky(params, p_features->rows(ixs.at(i)), x_predict);
        arma::mat multiplicated(chunk_predict_K.n_rows, chunk_weights[i].n_cols);
#pragma omp parallel for collapse(2) num_threads(adj_threads(chunk_predict_K.n_rows * chunk_weights[i].n_cols))
        for (size_t r = 0; r < chunk_predict_K.n_rows; ++r)
            for (size_t c = 0; c < chunk_weights[i].n_cols; ++c)
                multiplicated(r, c) =
                        arma::as_scalar(chunk_predict_K.row(r) * chunk_weights[i].col(c)) * chunks_weight[i] - chunk_bias[i] - params.get_svr_epsilon();
        omp_set_lock(&predict_l);
        if (prediction.empty())
            prediction = multiplicated;
        else
            prediction += multiplicated;
        omp_unset_lock(&predict_l);
    }
    return labels_scaling_factor * prediction / double(ixs.size());
}


t_gradient_data OnlineMIMOSVR::produce_residuals()
{
    if (ixs.size() < 2) {
        LOG4_ERROR("At least two chunks are needed to produce residuals.");
        return {};
    }

    arma::mat divisors(arma::size(*p_labels), arma::fill::ones);
    arma::mat residuals(arma::size(*p_labels));
    const auto all_ixs = arma::regspace<arma::uvec>(0, p_labels->n_rows - 1);
    OMP_LOCK(residuals_l) OMP_LOCK(divisors_l)
#pragma omp parallel for num_threads(adj_threads(ixs.size())) schedule(static, 1)
    for (size_t chunk_ix = 0; chunk_ix < ixs.size(); ++chunk_ix) {
        auto excluded_chunk_ixs = all_ixs;
        excluded_chunk_ixs.shed_rows(ixs.at(chunk_ix));
        omp_set_lock(&divisors_l);
        divisors.rows(excluded_chunk_ixs) += 1;
        omp_unset_lock(&divisors_l);
        const auto p_params = get_params_ptr(chunk_ix);
        const auto this_residuals = p_labels->rows(excluded_chunk_ixs) + p_params->get_svr_epsilon() + chunk_bias[chunk_ix]
                - prepare_Ky(*p_params, p_features->rows(ixs.at(chunk_ix)), p_features->rows(excluded_chunk_ixs))
                * chunk_weights[chunk_ix] * chunks_weight[chunk_ix];
        omp_set_lock(&residuals_l);
        residuals.rows(excluded_chunk_ixs) += this_residuals;
        omp_unset_lock(&residuals_l);
    }
    residuals /= divisors;
    const auto res_rows = arma::find(residuals != 0);
    return {ptr<arma::mat>(p_features->rows(res_rows)), ptr<arma::mat>(residuals.rows(res_rows)), ptr<arma::vec>(p_last_knowns->rows(res_rows))};
}

} // namespace datamodel
} // namespace svr