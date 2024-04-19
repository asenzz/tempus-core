//
// Created by zarko on 11/11/21.
//

//#define BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS 1 // Otherwise Valgrind crashes on init
#include <algorithm>
#include <armadillo>
#include <boost/math/distributions/students_t.hpp>
#include <execution>
#include <mutex>
#include <cmath>
#include <numeric>
#include <tuple>
#include <unordered_set>
#include <vector>
#include "common/compatibility.hpp"
#include "common/parallelism.hpp"
#include "common/gpu_handler.hpp"
#include "util/math_utils.hpp"
#include "onlinesvr.hpp"
#include "cuqrsolve.hpp"
#include "DQScalingFactorService.hpp"

namespace svr {
namespace datamodel {


std::deque<size_t> OnlineMIMOSVR::get_predict_chunks() const
{
    if (chunks_score.empty()) LOG4_THROW("Chunk score board empty!");
    std::deque<size_t> score_idx(chunks_score.size());
    std::iota(score_idx.begin(), score_idx.end(), 0);
    std::stable_sort(std::execution::par_unseq, score_idx.begin(), score_idx.end(), [&](const size_t i1, const size_t i2) { return chunks_score[i1] < chunks_score[i2]; });
    LOG4_DEBUG("Predicting using " << C_predict_chunks * score_idx.size() << " chunks from " << score_idx << ", scores " << chunks_score);
    score_idx.erase(score_idx.begin() + std::max<size_t>(1, C_predict_chunks * score_idx.size()), score_idx.end());
    return score_idx;
}


arma::mat OnlineMIMOSVR::predict(const arma::mat &x_predict)
{
    if (is_manifold()) return manifold_predict(x_predict);
    const auto score_idx = get_predict_chunks();
    const arma::mat x_predict_t = x_predict.t();
    arma::mat prediction;
    OMP_LOCK(predict_l);
#pragma omp parallel for schedule(static, 1) num_threads(adj_threads(score_idx.size()))
    for (const auto chunk_ix: score_idx) {
        LOG4_TRACE("Predicting " << arma::size(x_predict) << " with chunk " << chunk_ix << ", level " << decon_level << ", gradient " << gradient);
        const auto p_params = get_params_ptr(chunk_ix);
        arma::mat chunk_x_predict_t = x_predict_t;
        const auto chunk_sf = business::DQScalingFactorService::slice(scaling_factors, chunk_ix, gradient);
        business::DQScalingFactorService::scale_features(chunk_ix, gradient, p_params->get_lag_count(), chunk_sf, chunk_x_predict_t);
        const auto chunk_predict_K = prepare_Ky(*p_params, train_feature_chunks_t[chunk_ix], chunk_x_predict_t);
        arma::mat multiplicated(chunk_predict_K.n_rows, weight_chunks[chunk_ix].n_cols);
#pragma omp parallel for collapse(2) num_threads(adj_threads(chunk_predict_K.n_rows * weight_chunks[chunk_ix].n_cols))
        for (size_t r = 0; r < chunk_predict_K.n_rows; ++r)
            for (size_t c = 0; c < weight_chunks[chunk_ix].n_cols; ++c)
                multiplicated(r, c) = arma::as_scalar(chunk_predict_K.row(r) * weight_chunks[chunk_ix].col(c)) - p_params->get_svr_epsilon();
        business::DQScalingFactorService::unscale_labels(
                *business::DQScalingFactorService::find(chunk_sf, model_id, chunk_ix, gradient, decon_level, false, true), multiplicated);
        omp_set_lock(&predict_l);
        if (prediction.empty())
            prediction = multiplicated;
        else
            prediction += multiplicated;
        omp_unset_lock(&predict_l);
    }
    return prediction / double(score_idx.size());
}


t_gradient_data OnlineMIMOSVR::produce_residuals()
{
    if (ixs.size() < 2) LOG4_THROW("At least two chunks are needed to produce residuals.");

    arma::vec row_divisors(p_labels->n_rows);
    arma::mat residuals(arma::size(*p_labels));
    const arma::uvec all_rows = arma::regspace<arma::uvec>(0, p_labels->n_rows - 1);
    arma::uvec res_rows;
    const auto score_idx = get_predict_chunks();
    OMP_LOCK(residuals_l)
#pragma omp parallel for num_threads(adj_threads(score_idx.size())) schedule(static, 1)
    for (size_t chunk_ix: score_idx) {
        arma::uvec chunk_excluded_rows = all_rows;
        chunk_excluded_rows.shed_rows(ixs[chunk_ix]);
        const auto p_params = get_params_ptr(chunk_ix);
        arma::mat excluded_features_t = p_features->rows(chunk_excluded_rows).t();
        arma::mat excluded_labels = p_labels->rows(chunk_excluded_rows);
        const auto chunk_sf = business::DQScalingFactorService::slice(scaling_factors, chunk_ix, gradient);
        business::DQScalingFactorService::scale_features(chunk_ix, gradient, p_params->get_lag_count(), chunk_sf, excluded_features_t);
        const auto p_sf = business::DQScalingFactorService::find(chunk_sf, model_id, chunk_ix, gradient, decon_level, false, true);
        business::DQScalingFactorService::scale_labels(*p_sf, excluded_labels);
        arma::mat this_residuals = excluded_labels + p_params->get_svr_epsilon()
                - prepare_Ky(*p_params, train_feature_chunks_t[chunk_ix], excluded_features_t) * weight_chunks[chunk_ix];
        business::DQScalingFactorService::unscale_labels(*p_sf, this_residuals);
        omp_set_lock(&residuals_l);
        residuals.rows(chunk_excluded_rows) += this_residuals;
        row_divisors.rows(chunk_excluded_rows) += 1;
        res_rows.insert_rows(res_rows.n_rows, chunk_excluded_rows);
        omp_unset_lock(&residuals_l);
    }
    row_divisors.rows(arma::find(row_divisors == 0)).ones();
    res_rows = arma::sort(arma::unique(res_rows));
    LOG4_TRACE("Resultant rows " << common::present(res_rows) << ", residuals " << common::present(residuals));
    return {ptr<arma::mat>(p_features->rows(res_rows)),
                ptr<arma::mat>(residuals.rows(res_rows) / row_divisors.rows(res_rows)),
                    ptr<arma::vec>(p_last_knowns->rows(res_rows))};
}

} // namespace datamodel
} // namespace svr