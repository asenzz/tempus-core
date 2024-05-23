//
// Created by zarko on 9/29/22.
//

#include <execution>
#include <cublas_v2.h>
#include <armadillo>
#include <exception>
#include <iterator>
#include <limits>
#include <cmath>
#include <complex>
#include <deque>
#include <magma_auxiliary.h>
#include <string>
#include <tuple>
#include <omp.h>
#include <magma.h>
#include <prima/prima.h>
#include <boost/date_time/posix_time/ptime.hpp>
#include "DQScalingFactorService.hpp"
#include "common/compatibility.hpp"
#include "model/dbcache.tpp"
#include "pprima.hpp"
#include "firefly.hpp"
#include "onlinesvr.hpp"
#include "common/logging.hpp"
#include "common/constants.hpp"
#include "firefly.hpp"
#include "appcontext.hpp"
#include "cuqrsolve.cuh"
#include "util/math_utils.hpp"
#include "util/string_utils.hpp"
#include "recombine_parameters.cuh"
#include "common/cuda_util.cuh"


#define TUNE_THREADS(MAX_THREADS) num_threads(adj_threads(std::min<size_t>((MAX_THREADS), 4)))


namespace svr {
namespace datamodel {

void OnlineMIMOSVR::tune_fast()
{
    if (is_manifold()) {
        LOG4_DEBUG("Skipping tuning of manifold kernel!");
        return;
    }

    const auto num_chunks = ixs.size();
    if (chunks_score.size() != num_chunks) chunks_score.resize(num_chunks);
    LOG4_TRACE("Tuning labels " << common::present(*p_labels) << ", features " << common::present(*p_features) << ", last-knowns " << common::present(*p_last_knowns) <<
                                ", slide skip " << C_emo_slide_skip << ", max j " << C_emo_max_j << ", tune min validation window "
                                << C_emo_tune_min_validation_window << ", test len " << C_emo_test_len << ", level " << decon_level << ", num chunks " << num_chunks <<
                                ", first chunk " << common::present_chunk(ixs.front(), C_chunk_tail) <<
                                ", pre-last chunk " << common::present_chunk(*(ixs.rbegin() + 1), C_chunk_tail) <<
                                ", last chunk " << common::present_chunk(ixs.back(), C_chunk_tail));
    OMP_LOCK(params_l);
    p_kernel_matrices->resize(num_chunks);
    weight_chunks.resize(num_chunks);
#pragma omp parallel for schedule(static, 1) TUNE_THREADS(num_chunks)
    for (size_t chunk_ix = 0; chunk_ix < num_chunks; ++chunk_ix) {
        auto p_template_chunk_params = get_params_ptr(chunk_ix);
        if (!p_template_chunk_params) {
            for (const auto &p: param_set)
                if (!p->is_manifold()) {
                    LOG4_WARN("Parameters for chunk " << chunk_ix << " not found, using template from " << *p);
                    p_template_chunk_params = otr(*p);
                    p_template_chunk_params->set_chunk_index(chunk_ix);
                    p_template_chunk_params->set_grad_level(gradient);
                    p_template_chunk_params->set_decon_level(decon_level);
                    break;
                }
            if (!p_template_chunk_params) LOG4_THROW("Template parameters for chunk " << chunk_ix << " not found");
        }

        arma::uvec chunk_ixs_tune = ixs[chunk_ix] - C_emo_test_len;
        const size_t train_len = chunk_ixs_tune.n_rows;
        const size_t K_train_size = train_len * train_len * sizeof(double);
        arma::mat tune_features_t = arma::join_cols(
                p_features->rows(chunk_ixs_tune), p_features->rows(p_features->n_rows - C_emo_test_len, p_features->n_rows - 1)).t();
        arma::mat tune_labels = arma::join_cols(
                p_labels->rows(chunk_ixs_tune), p_labels->rows(p_labels->n_rows - C_emo_test_len, p_labels->n_rows - 1));
        auto p_labels_sf = business::DQScalingFactorService::find(scaling_factors, model_id, chunk_ix, gradient, decon_level, false, true);
        auto features_sf = business::DQScalingFactorService::slice(scaling_factors, chunk_ix, gradient);
        bool calculated_sf = false;
        const auto lag = p_template_chunk_params->get_lag_count();
        if (!p_labels_sf || features_sf.size() != train_feature_chunks_t[chunk_ix].n_rows / lag) {
            features_sf = business::DQScalingFactorService::calculate(chunk_ix, *this, tune_features_t, tune_labels);
            p_labels_sf = business::DQScalingFactorService::find(features_sf, model_id, chunk_ix, gradient, decon_level, false, true);
            calculated_sf = true;
        }
        business::DQScalingFactorService::scale_features(chunk_ix, gradient, lag, features_sf, train_feature_chunks_t[chunk_ix]);
        business::DQScalingFactorService::scale_features(chunk_ix, gradient, lag, features_sf, tune_features_t);
        business::DQScalingFactorService::scale_labels(*p_labels_sf, train_label_chunks[chunk_ix]);
        business::DQScalingFactorService::scale_labels(*p_labels_sf, tune_labels);
        auto all_labels = *p_labels;
        business::DQScalingFactorService::scale_labels(*p_labels_sf, all_labels);
        const auto all_labels_meanabs = common::meanabs(all_labels);
        release_cont(all_labels);
        const double labels_mean = arma::mean(arma::vectorise(train_label_chunks[chunk_ix]));
        auto train_cuml = all_cumulatives(*p_template_chunk_params, train_feature_chunks_t[chunk_ix]);
        auto tune_cuml = all_cumulatives(*p_template_chunk_params, tune_features_t);
        release_cont(tune_features_t);

        OMP_LOCK(chunk_preds_l)
        constexpr size_t n_particles = 35;
        constexpr size_t iters = 22;
        constexpr size_t D = 2;
        constexpr double lambda_multiplier = C_tune_range_max_lambda;
        constexpr double lambda_scaler = 1e8;
        const auto gamma_variance = get_gamma_range_variance(train_len);
        const double equiexp = std::log(std::sqrt(lambda_multiplier)) / M_LN2;
        arma::mat bounds(D, 2);
        bounds(0, 0) = 0;
        bounds(0, 1) = 1;
        bounds(1, 0) = C_tune_range_min_lambda;
        bounds(1, 1) = 1. / lambda_scaler;
        const auto tune_iters = PROPS.get_online_learn_iter_limit();
        const auto svr_epsilon = p_template_chunk_params->get_svr_epsilon();
        double test_train_mape_ratio;
        p_kernel_matrices->at(chunk_ix).set_size(train_len, train_len);
        bool mape_ratio_unset = true;
        auto p_mape_l = new omp_lock_t;
        omp_init_lock(p_mape_l);
        auto &tune_results = ccache().checkin_tuner(*this, chunk_ix);

        const auto prima_cb = [&](const double x[], double *const f) {
            const auto xx = optimizer::pprima::ensure_bounds(x, bounds);
            const auto lambda = lambda_multiplier * std::pow(xx[1] * lambda_scaler, equiexp);
            const common::gpu_context ctx;
            const auto [score, gamma, epsco, p_predictions, d_K_train] = cuvalidate(
                    gamma_variance, labels_mean, lambda, xx[0], svr_epsilon, all_labels_meanabs, train_len, lag, *tune_cuml, *train_cuml, tune_labels, tune_iters,
                    test_train_mape_ratio, mape_ratio_unset, p_mape_l, ctx.phy_id());
            if ((*f = score) == C_bad_validation) goto __bail;

            omp_set_lock(&chunk_preds_l);
            if (score < tune_results.param_pred.front().score) {
                p_template_chunk_params->set_svr_kernel_param(gamma);
                p_template_chunk_params->set_svr_kernel_param2(lambda);
                p_template_chunk_params->set_svr_C(1. / (2. * epsco));
                std::rotate(
                        tune_results.param_pred.begin(),
                        tune_results.param_pred.begin() + 1,
                        tune_results.param_pred.end());
                tune_results.param_pred.front().free();
                tune_results.param_pred.front().score = score;
                tune_results.param_pred.front().p_predictions = p_predictions;
                tune_results.param_pred.front().params = *p_template_chunk_params;

                cu_errchk(cudaMemcpy(p_kernel_matrices->at(chunk_ix).memptr(), d_K_train, K_train_size, cudaMemcpyDeviceToHost));
                omp_unset_lock(&chunk_preds_l);
                cu_errchk(cudaFree(d_K_train));

                LOG4_TRACE("New best score " << score << ", parameters " << tune_results.param_pred.front().params);
                return;
            }
#if 0
            if (!chunk_ix) LOG4_FILE("/tmp/tune_score_gamma_lambda_level_0.csv", score << ',' << gamma << ',' << lambda);
#endif
            omp_unset_lock(&chunk_preds_l);
            __bail:
            for (auto &p: *p_predictions) delete p;
            delete p_predictions;
            cu_errchk(cudaFree(d_K_train));
        };

        const optimizer::t_pprima_res res = optimizer::pprima(prima_algorithm_t::PRIMA_LINCOA, n_particles, bounds, prima_cb, iters, .25, 1e-11); // default rhobeg=range/4, default rhoend 5e-10
        delete p_mape_l;
        arma::mat tune_lastknowns = arma::join_cols(
                p_last_knowns->rows(chunk_ixs_tune), p_last_knowns->rows(p_last_knowns->n_rows - C_emo_test_len, p_last_knowns->n_rows - 1));
        assert(tune_lastknowns.n_rows == tune_labels.n_rows);
        release_cont(chunk_ixs_tune);
        release_cont(bounds);
        tune_cuml.reset();
        train_cuml.reset();
        chunks_score[chunk_ix].first = tune_results.param_pred.front().score;
        business::DQScalingFactorService::scale_labels(*p_labels_sf, tune_lastknowns);
        const ssize_t start_point = tune_lastknowns.n_rows - C_emo_test_len - train_len;
        assert(start_point >= 0);
        if (start_point != 0) LOG4_WARN("Start point is " << start_point << ", size of tune lastknowns " << arma::size(tune_lastknowns));
        for (size_t j = 0; j < C_emo_max_j; ++j) {
            const auto test_start = start_point + j * C_emo_slide_skip + train_len;
            tune_results.labels[j] = tune_labels.rows(test_start, tune_lastknowns.n_rows - 1);
            tune_results.last_knowns[j] = tune_lastknowns.rows(test_start, tune_lastknowns.n_rows - 1);
        }
        release_cont(tune_lastknowns);
        release_cont(tune_labels);
        ccache().checkout_tuner(*this, chunk_ix);
        LOG4_INFO("Tune best score " << tune_results.param_pred.front().score << ", final parameters " << *p_template_chunk_params);

        if (calculated_sf) set_scaling_factors(features_sf);
        calc_weights(chunk_ix, PROPS.get_stabilize_iterations_count());
        recombine_params(chunk_ix);
    }
    clean_chunks();
}

std::deque<size_t> OnlineMIMOSVR::get_predict_chunks(const std::deque<std::pair<double, double>> &chunks_score)
{
    if (chunks_score.empty()) LOG4_THROW("Chunk score board empty!");
    std::deque<size_t> res(chunks_score.size());
    double mean_first = 0, mean_second = 0;
    for (const auto &p: chunks_score) {
        mean_first += p.first;
        mean_second += p.second;
    }
    mean_first /= double(chunks_score.size());
    mean_second /= double(chunks_score.size());
    std::iota(res.begin(), res.end(), 0);
    std::stable_sort(std::execution::par_unseq, res.begin(), res.end(), [&](const size_t i1, const size_t i2) {
        double score_1 = 0, score_2 = 0;
        if (mean_first) {
            score_1 += chunks_score[i1].first / mean_first;
            score_2 += chunks_score[i2].first / mean_first;
        }
        if (mean_second) {
            score_1 += chunks_score[i1].second / mean_second;
            score_2 += chunks_score[i2].second / mean_second;
        }
        return score_1 < score_2;
    });
    res.erase(res.begin() + svr::datamodel::C_predict_chunks, res.end());
    LOG4_DEBUG("Using " << svr::datamodel::C_predict_chunks << " chunks " << res << ", of all chunks with scores " << chunks_score);
    std::sort(std::execution::par_unseq, res.begin(), res.end());
    return res;
}

void OnlineMIMOSVR::clean_chunks()
{
    const auto used_chunks = get_predict_chunks(chunks_score);

    for (auto iter = param_set.begin(); iter != param_set.end();) {
        bool chunk_ix_set = false;
        for (size_t i = 0; i < used_chunks.size(); ++i)
            if ((**iter).get_chunk_index() == used_chunks[i]) {
                (**iter).set_chunk_index(i);
                chunk_ix_set = true;
                ++iter;
                break;
            }
        if (!chunk_ix_set) iter = param_set.erase(iter);
    }

    for (auto iter = scaling_factors.begin(); iter != scaling_factors.end();) {
        bool chunk_ix_set = false;
        for (size_t i = 0; i < used_chunks.size(); ++i)
            if ((**iter).get_chunk_index() == used_chunks[i]) {
                (**iter).set_chunk_index(i);
                chunk_ix_set = true;
                ++iter;
                break;
            }
        if (!chunk_ix_set) iter = scaling_factors.unsafe_erase(iter);
    }

    common::keep_indices(ixs, used_chunks);
    common::keep_indices(train_feature_chunks_t, used_chunks);
    common::keep_indices(train_label_chunks, used_chunks);
    common::keep_indices(chunks_score, used_chunks);
    common::keep_indices(*p_kernel_matrices, used_chunks);
    common::keep_indices(weight_chunks, used_chunks);
}


}// datamodel
} // svr
