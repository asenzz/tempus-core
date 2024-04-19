//
// Created by zarko on 9/29/22.
//

#include "DQScalingFactorService.hpp"
#include <boost/date_time/posix_time/ptime.hpp>
#include <execution>
#include <vector>

#define TUNE_THREADS(MAX_THREADS) num_threads(adj_threads(std::min<size_t>((MAX_THREADS), 1 + common::gpu_handler::get().get_gpu_devices_count())))

// #define TUNE_FIREFLY
#define TUNE_PRIMA
// #undef TUNE_ADAPTIVE_GRID

#ifdef TUNE_CERES
#define GLOG_STATIC_DEFINE
#include </usr/local/include/glog/export.h>
#include <ceres/ceres.h>
#elif defined TUNE_PRIMA

#include <prima/prima.h>
#include "pprima.hpp"

#endif

#include <algorithm>
#include <armadillo>
#include <exception>
#include <iterator>
#include <limits>
#include <cmath>
#include <complex>
#include <deque>
#include <omp.h>
#include <string>
#include <tuple>
#include "common/compatibility.hpp"
#include "common/defines.h"
#include "firefly.hpp"
#include "model/SVRParameters.hpp"
#include "onlinesvr.hpp"
#include "common/logging.hpp"
#include "common/constants.hpp"
#include "firefly.hpp"
#include "appcontext.hpp"
#include "cuqrsolve.hpp"
#include "util/math_utils.hpp"
#include "util/string_utils.hpp"
#include "ModelService.hpp"
#include "recombine_parameters.cuh"

namespace svr {
namespace datamodel {

double OnlineMIMOSVR::get_gamma_range_variance(const size_t train_len)
{
    return 1. / double(train_len);
}

auto
eval_score(const datamodel::SVRParameters &params, const arma::mat &K, const arma::mat &labels, const arma::mat &last_knowns, const size_t train_len,
           const double meanabs_labels)
{
    const ssize_t start_point_K = K.n_rows - C_emo_test_len - train_len;
    const ssize_t start_point_labels = labels.n_rows - C_emo_test_len - train_len;
    if (start_point_K < 0 || start_point_labels < 0 || labels.n_rows != K.n_rows || labels.n_rows != last_knowns.n_rows)
        LOG4_THROW("Shorter K " << start_point_K << " or labels " << start_point_labels << " for K " << arma::size(K) << ", labels " << arma::size(labels)
                                << ", last-knowns " << arma::size(last_knowns));
    auto p_out_preds = ptr<std::deque<arma::mat>>(C_emo_max_j);
    auto p_out_labels = ptr<std::deque<arma::mat>>(C_emo_max_j);
    auto p_out_last_knowns = ptr<std::deque<arma::mat>>(C_emo_max_j);
    arma::mat K_epsco = K;
    K_epsco.diag() += 1. / (2. * params.get_svr_C());
    double score = 0;
#pragma omp parallel for reduction(+:score) schedule(static, 1) TUNE_THREADS(C_emo_max_j)
    for (size_t j = 0; j < C_emo_max_j; ++j) {
        const size_t x_train_start = j * C_emo_slide_skip;
        const size_t x_train_final = x_train_start + train_len - 1;
        const size_t x_test_start = x_train_final + 1;
        LOG4_TRACE("Try " << j << ", K " << arma::size(K) << ", start point labels " << start_point_labels << ", start point K " << start_point_K << ", train start " <<
                          x_train_start << ", train final " << x_train_final << ", test start " << x_test_start << ", test final is mat end, train len " << train_len
                          << ", labels " << arma::size(labels) << ", current score " << score);
        try {
#if 0
            const size_t x_test_final = x_test_start + EMO_TEST_LEN - j * EMO_SLIDE_SKIP - 1;
            double this_score = arma::mean(arma::abs(arma::vectorise(K.submat(K.n_rows - 1 - start_point_K - x_test_final, K.n_rows - 1 - start_point_K - x_train_final,
                                                                        K.n_rows - 1 - start_point_K - x_test_start, K.n_rows - 1 - start_point_K - x_train_start)
                                                               * OnlineMIMOSVR::solve_dispatch(
                                                                                               K.submat(K.n_rows - 1 - start_point_K - x_train_final,
                                                                                                        K.n_rows - 1 - start_point_K - x_train_final,
                                                                                                        K.n_rows - 1 - start_point_K - x_train_start,
                                                                                                        K.n_rows - 1 - start_point_K - x_train_start),
                                                                                               labels.rows(labels.n_rows - 1 - start_point_labels - x_train_final,
                                                                                                           labels.n_rows - 1 - start_point_labels - x_train_start),
                                                                                               PROPS.get_online_learn_iter_limit(), false)
                                                               - labels.rows(labels.n_rows - 1 - start_point_labels - x_test_final,
                                                                             labels.n_rows - 1 - start_point_labels - x_test_start)))) / meanabs_labels;
#endif
            p_out_labels->at(j) = labels.rows(start_point_labels + x_test_start, labels.n_rows - 1);
            p_out_last_knowns->at(j) = last_knowns.rows(start_point_labels + x_test_start, last_knowns.n_rows - 1);
            p_out_preds->at(j) =
                    K.submat(start_point_K + x_test_start, start_point_K + x_train_start, K.n_rows - 1, start_point_K + x_train_final)
                    * OnlineMIMOSVR::solve_dispatch(
                            K_epsco.submat(start_point_K + x_train_start, start_point_K + x_train_start, start_point_K + x_train_final, start_point_K + x_train_final),
                            K.submat(start_point_K + x_train_start, start_point_K + x_train_start, start_point_K + x_train_final, start_point_K + x_train_final),
                            labels.rows(start_point_labels + x_train_start, start_point_labels + x_train_final),
                            PROPS.get_online_learn_iter_limit(), false) - params.get_svr_epsilon();
            const auto this_score = common::meanabs<double>(p_out_labels->at(j) - p_out_preds->at(j)) / meanabs_labels;
            if (!std::isnormal(this_score)) LOG4_THROW("Score not normal for " << params);
            score += this_score;
        } catch (const std::exception &ex) {
            LOG4_ERROR(
                    "Error solving matrix, try " << j << ", K " << arma::size(K) << ", " << x_train_start << ", " << x_train_final << ", " << x_test_start << ", error "
                                                 << ex.what());
            p_out_preds->at(j) = arma::mat(p_out_preds->at(j).n_rows - x_test_start, labels.n_cols, arma::fill::value(BAD_VALIDATION));
            p_out_labels->at(j) = p_out_preds->at(j);
            p_out_last_knowns->at(j) = p_out_preds->at(j);
            score += BAD_VALIDATION;
        }
    }
    return std::tuple(p_out_preds, p_out_labels, p_out_last_knowns, score);
}

double validate_gammas(
        const double mingamma, const arma::mat &Z, const arma::mat &Ztrain, const std::deque<double> &gamma_multis, const SVRParameters &score_params,
        const size_t chunk_ix, t_parameter_predictions_set &chunk_preds, omp_lock_t *p_chunk_preds_l, const double all_labels_meanabs, const size_t train_len,
        const arma::mat &chunk_labels, const arma::mat &chunk_lastknowns)
{
    LOG4_DEBUG("Validating " << gamma_multis.size() << " gamma multipliers, starting from " << gamma_multis.front() << " to " << gamma_multis.back() <<
                             ", min gamma " << mingamma << ", Z " << arma::size(Z) << ", template parameters " << score_params);
    double call_score = std::numeric_limits<double>::infinity();
#pragma omp parallel for schedule(static, 1) TUNE_THREADS(gamma_multis.size())
    for (const double gamma_mult: gamma_multis) {
        auto p_gamma_params = otr(score_params);
        p_gamma_params->set_svr_kernel_param(gamma_mult * mingamma);
        double epsco;
        arma::mat K(arma::size(Z));
#pragma omp parallel num_threads(adj_threads(2))
        {
#pragma omp task
            {
                arma::mat Ktrain(arma::size(Ztrain));
                PROFILE_EXEC_TIME(solvers::kernel_from_distances(Ktrain.memptr(), Ztrain.mem, Ztrain.n_rows, Ztrain.n_cols, p_gamma_params->get_svr_kernel_param()),
                                  "Kernel from distances " << arma::size(Ztrain));
                epsco = OnlineMIMOSVR::calc_epsco(Ktrain);
                p_gamma_params->set_svr_C(1. / (2. * epsco));
            }
#pragma omp task
            PROFILE_EXEC_TIME(solvers::kernel_from_distances(K.memptr(), Z.mem, Z.n_rows, Z.n_cols, p_gamma_params->get_svr_kernel_param()),
                              "Kernel from distances " << arma::size(Z));
        }
        const auto [p_out_preds, p_out_labels, p_out_last_knowns, score] = eval_score(
                *p_gamma_params, K, chunk_labels, chunk_lastknowns, train_len, all_labels_meanabs);
        K.clear();
        if (!chunk_ix) LOG4_FILE("/tmp/tune_score_gamma_lambda.csv",
                                 score << ',' << p_gamma_params->get_svr_kernel_param() << ',' << p_gamma_params->get_svr_kernel_param2());
        omp_set_lock(p_chunk_preds_l);
        if (chunk_preds.size() < size_t(common::C_tune_keep_preds) || score < (**chunk_preds.cbegin()).score) {
            p_gamma_params->set_svr_C(1. / (2. * epsco));
            chunk_preds.emplace(otr<t_param_preds>(score, p_gamma_params, p_out_preds, p_out_labels, p_out_last_knowns));
            LOG4_DEBUG("Lambda, gamma tune best score " << score << ", MAPE " << 100. * score / double(C_emo_max_j) << " pct, for gamma multi " << gamma_mult <<
                                                        ", mingamma " << mingamma << ", parameters " << *p_gamma_params);
            if (chunk_preds.size() > size_t(common::C_tune_keep_preds))
                chunk_preds.erase(std::next(chunk_preds.begin(), common::C_tune_keep_preds), chunk_preds.end());
        }
        if (score < call_score) call_score = score;
        omp_unset_lock(p_chunk_preds_l);
    }
    return call_score;
};

#ifdef SYSTEMIC_TUNE

void OnlineMIMOSVR::tune()
{
    if (is_manifold()) {
        LOG4_DEBUG("Skipping tuning of manifold kernel!");
        return;
    }

    auto p_predictions = ccache().checkin_tuner(*this);

    const auto num_chunks = ixs_tune.size();
    const auto train_len = ixs_tune.front().n_rows;
    const auto meanabs_all_labels = common::meanabs(*p_labels);

    std::deque<arma::uvec> ixs_train(num_chunks);
    std::deque<arma::mat> train_feature_chunks_t(num_chunks), feature_chunks_t(num_chunks), train_label_chunks(num_chunks), label_chunks(num_chunks), lastknown_chunks(num_chunks);

    const auto validate_gammas = [&](
            const double mingamma, const arma::mat &Z, const arma::mat &Ztrain, const auto &gamma_multis, const auto &score_params, const size_t chunk_ix,
            auto &chunk_preds, omp_lock_t *p_predictions_l)
    {
        LOG4_DEBUG("Validating " << gamma_multis.size() << " gamma multipliers, starting from " << gamma_multis.front() << " to " << gamma_multis.back() <<
                                 ", min gamma " << mingamma << ", Z " << arma::size(Z) << ", template_parameters " << score_params);

#pragma omp parallel for schedule(static, 1) num_threads(adj_threads(std::min<size_t>(gamma_multis.size(), TUNE_THREADS)))
        for (const double gamma_mult: gamma_multis) {
            auto p_gamma_params = otr(score_params);
            p_gamma_params->set_svr_kernel_param(gamma_mult * mingamma);
            double epsco;
            arma::mat K(arma::size(Z));
// #pragma omp parallel num_threads(adj_threads(2))
            {
/*
#pragma omp task
                {
                    arma::mat Ktrain(arma::size(Ztrain));
                    solvers::kernel_from_distances(Ktrain.memptr(), Ztrain.mem, Ztrain.n_rows, Ztrain.n_cols, p_gamma_params->get_svr_kernel_param());
                    epsco = calc_epsco(Ktrain);
                    p_gamma_params->set_svr_C(1. / (2. * epsco));
                }
*/
                solvers::kernel_from_distances(K.memptr(), Z.mem, Z.n_rows, Z.n_cols, p_gamma_params->get_svr_kernel_param());
                epsco = calc_epsco(K);
                p_gamma_params->set_svr_C(1. / (2. * epsco));

            }
            if (epsco <= 0) LOG4_WARN("Auto epsco is negative indefinite " << epsco << ",  K " << common::present(K) << ", gamma template_parameters " << *p_gamma_params);
            const auto [p_out_preds, p_out_labels, p_out_last_knowns, score] = eval_score(
                    *p_gamma_params, K, label_chunks[chunk_ix], lastknown_chunks[chunk_ix], train_len, meanabs_all_labels);

            omp_set_lock(p_predictions_l);
            if (chunk_preds.size() < size_t(common::C_tune_keep_preds) || score < (**chunk_preds.begin()).score) {
                p_gamma_params->set_svr_C(1. / (2. * epsco));
                chunk_preds.emplace(otr<t_param_preds>(score, p_gamma_params, p_out_preds, p_out_labels, p_out_last_knowns));
                LOG4_DEBUG("Lambda, gamma tune best score " << score / double(C_emo_max_j) << ", for " << *p_gamma_params);
                if (chunk_preds.size() > size_t(common::C_tune_keep_preds))
                    chunk_preds.erase(std::next(chunk_preds.begin(), common::C_tune_keep_preds), chunk_preds.end());
            }
            omp_unset_lock(p_predictions_l);
        }
    };

    OMP_LOCK(ins_chunk_results_l)

#pragma omp parallel for num_threads(adj_threads(num_chunks)) schedule(static, 1)
    for (size_t i = 0; i < num_chunks; ++i) {
        ixs_train[i] = ixs_tune[i] + EMO_TEST_LEN;
        feature_chunks_t[i] = arma::join_cols(
                p_features->rows(ixs_tune[i]), p_features->rows(p_features->n_rows - EMO_TEST_LEN, p_features->n_rows - 1)).t();
        label_chunks[i] = arma::join_cols(
                p_labels->rows(ixs_tune[i]), p_labels->rows(p_labels->n_rows - EMO_TEST_LEN, p_labels->n_rows - 1));
        lastknown_chunks[i] = arma::join_cols(
                p_last_knowns->rows(ixs_tune[i]), p_last_knowns->rows(p_last_knowns->n_rows - EMO_TEST_LEN, p_last_knowns->n_rows - 1));
        train_feature_chunks_t[i] = p_features->rows(ixs_train[i]).t();
        train_label_chunks[i] = p_labels->rows(ixs_train[i]);
        auto p_template_chunk_params = get_params_ptr(i);
        if (!p_template_chunk_params) {
            for (const auto &p: param_set)
                if (!p->is_manifold()) {
                    p_template_chunk_params = otr(*p);
                    LOG4_WARN("Parameters for chunk " << i << " not found, using template from " << *p);
                    p_template_chunk_params->set_chunk_ix(i);
                    break;
                }
            if (!p_template_chunk_params) LOG4_THROW("Template parameters for chunk " << i << " not found");
        }
        const auto original_input_queue_column_name = p_template_chunk_params->get_input_queue_column_name();

        auto chunk_params = *p_template_chunk_params;
        chunk_params.set_input_queue_column_name("TUNE_" + original_input_queue_column_name);
        chunk_params.set_chunk_ix(i);
        auto best_chunk_params = chunk_params;
        auto best_score = std::numeric_limits<double>::max();
        arma::mat best_Z;
        auto range_min_lambda = C_tune_range_min_lambda, range_max_lambda = C_tune_range_max_lambda;
        for (size_t grid_level_lambda = 0; grid_level_lambda < C_grid_depth; ++grid_level_lambda) {
            const double range_lambda = range_max_lambda - range_min_lambda;
            std::deque<double> lambdas;
            for (double r = range_min_lambda; r < range_max_lambda; r += range_lambda / C_grid_range_div) lambdas.emplace_back(r);
#pragma omp parallel for schedule(static, 1) num_threads(adj_threads(std::min<size_t>(lambdas.size(), TUNE_THREADS)))
            for (const double lambda: lambdas) {
                chunk_params.set_svr_kernel_param2(lambda);
//                const auto p_Z = prepare_Z(chunk_params, train_feature_chunks_t[i]);
                const double score = calc_kernel_inversions(p_Z->mem, train_label_chunks[i]);
                if (score < best_score) {
                    best_Z = *p_Z;
                    best_score = score;
                    best_chunk_params = chunk_params;
                }
            }
            const auto best_lambda = best_chunk_params.get_svr_kernel_param2();
            const double range_half_lambda = range_lambda * .5;
            const double lambda_range_tightening = std::abs(best_lambda - range_half_lambda) / range_half_lambda;
            range_min_lambda = std::max(0., best_lambda - range_half_lambda * lambda_range_tightening);
            range_max_lambda = range_min_lambda + 2. * range_half_lambda * lambda_range_tightening;
        }
        train_feature_chunks_t[i].clear();

        auto p_chunk_preds = otr<t_parameter_predictions_set>();
        OMP_LOCK(predictions_l)
        const auto mingamma = calc_gamma(best_Z, arma::mean(arma::vectorise(train_label_chunks[i])));
        train_label_chunks[i].clear();
        const auto p_Z = prepare_Z(best_chunk_params, feature_chunks_t[i]);
        double range_min_gamma = C_tune_range_min_gammamulti, range_max_gamma = C_tune_range_max_gammamulti;
        for (size_t grid_level_gamma = 0; grid_level_gamma < C_grid_depth; ++grid_level_gamma) {
            std::deque<double> gamma_multis;
            const double range_gamma = range_max_gamma - range_min_gamma;
            const double range_half_gamma = range_gamma * .5;
            for (double r = range_min_gamma; r < range_max_gamma; r += range_gamma / C_grid_range_div) gamma_multis.emplace_back(r);
            PROFILE_EXEC_TIME(validate_gammas(mingamma, *p_Z, best_Z, gamma_multis, best_chunk_params, i, *p_chunk_preds, &predictions_l),
                              "Validate " << gamma_multis.size() << " gammas " << best_chunk_params);
            const auto best_gamma_multi = (**p_chunk_preds->cbegin()).p_params->get_svr_kernel_param() / mingamma;
            const double gamma_range_tightening = std::abs(best_gamma_multi - range_half_gamma) / range_half_gamma;
            range_min_gamma = std::max(0., best_gamma_multi - range_half_gamma * gamma_range_tightening);
            range_max_gamma = range_min_gamma + 2. * range_half_gamma * gamma_range_tightening;
        }
        feature_chunks_t[i].clear();
        label_chunks[i].clear();
        best_Z.clear();
        p_Z->clear();

        if (p_chunk_preds->size() > common::C_tune_keep_preds) p_chunk_preds->erase(std::next(p_chunk_preds->begin(), common::C_tune_keep_preds), p_chunk_preds->end());
        std::for_each(std::execution::par_unseq, p_chunk_preds->begin(), p_chunk_preds->end(), [&original_input_queue_column_name](auto &pp) {
            pp->p_params->set_input_queue_column_name(original_input_queue_column_name);
            LOG4_INFO("Final best score " << pp->score << ", final parameters " << *pp->p_params);
        });

        omp_set_lock(&ins_chunk_results_l);
        p_predictions->emplace(std::tuple{p_template_chunk_params->get_decon_level(), p_template_chunk_params->get_grad_level(), i}, p_chunk_preds);
        omp_unset_lock(&ins_chunk_results_l);
    }

    ccache().checkout_tuner(*this);
}

#else


void OnlineMIMOSVR::tune()
{
    if (is_manifold()) {
        LOG4_DEBUG("Skipping tuning of manifold kernel!");
        return;
    }

    auto p_predictions = ccache().checkin_tuner(*this);
    const auto num_chunks = ixs.size();
    const auto train_len = ixs.front().n_rows;
    if (chunks_score.size() != num_chunks) chunks_score.resize(num_chunks);

    LOG4_TRACE("Tuning labels " << common::present(*p_labels) << ", features " << common::present(*p_features) << ", last-knowns " << common::present(*p_last_knowns) <<
                                ", slide skip " << C_emo_slide_skip << ", max j " << C_emo_max_j << ", tune min validation window "
                                << C_emo_tune_min_validation_window <<
                                ", test len " << C_emo_test_len << ", level " << decon_level << ", train len " << train_len << ", num chunks " << num_chunks);

    OMP_LOCK(ins_chunk_results_l)
#pragma omp parallel for schedule(static, 1) TUNE_THREADS(num_chunks)
    for (size_t chunk_ix = 0; chunk_ix < num_chunks; ++chunk_ix) {
        auto p_template_chunk_params = get_params_ptr(chunk_ix);
        if (!p_template_chunk_params) {
            for (const auto &p: param_set)
                if (!p->is_manifold()) {
                    p_template_chunk_params = otr(*p);
                    LOG4_WARN("Parameters for chunk " << chunk_ix << " not found, using template from " << *p);
                    p_template_chunk_params->set_chunk_ix(chunk_ix);
                    break;
                }
            if (!p_template_chunk_params) LOG4_THROW("Template parameters for chunk " << chunk_ix << " not found");
        }

        arma::uvec chunk_ixs_tune = ixs[chunk_ix] - C_emo_test_len;
        arma::mat tune_features_t = arma::join_cols(
                p_features->rows(chunk_ixs_tune), p_features->rows(p_features->n_rows - C_emo_test_len, p_features->n_rows - 1)).t();
        arma::mat tune_labels = arma::join_cols(
                p_labels->rows(chunk_ixs_tune), p_labels->rows(p_labels->n_rows - C_emo_test_len, p_labels->n_rows - 1));
        arma::mat tune_lastknowns = arma::join_cols(
                p_last_knowns->rows(chunk_ixs_tune), p_last_knowns->rows(p_last_knowns->n_rows - C_emo_test_len, p_last_knowns->n_rows - 1));
        chunk_ixs_tune.clear();
        auto p_sf = business::DQScalingFactorService::find(scaling_factors, model_id, chunk_ix, gradient, decon_level, false, true);
        auto chunk_sf = business::DQScalingFactorService::slice(scaling_factors, chunk_ix, gradient);
        bool calculated_sf = false;
        if (!p_sf || chunk_sf.size() != train_feature_chunks_t[chunk_ix].n_rows / p_template_chunk_params->get_lag_count()) {
            chunk_sf = business::DQScalingFactorService::calculate(chunk_ix, *this, tune_features_t, tune_labels);
            p_sf = business::DQScalingFactorService::find(chunk_sf, model_id, chunk_ix, gradient, decon_level, false, true);
            calculated_sf = true;
        }
        business::DQScalingFactorService::scale_features(chunk_ix, gradient, p_template_chunk_params->get_lag_count(), chunk_sf, train_feature_chunks_t[chunk_ix]);
        business::DQScalingFactorService::scale_features(chunk_ix, gradient, p_template_chunk_params->get_lag_count(), chunk_sf, tune_features_t);
        business::DQScalingFactorService::scale_labels(*p_sf, train_label_chunks[chunk_ix]);
        business::DQScalingFactorService::scale_labels(*p_sf, tune_labels);
        business::DQScalingFactorService::scale_labels(*p_sf, tune_lastknowns);
        auto all_labels = *p_labels;
        business::DQScalingFactorService::scale_labels(*p_sf, all_labels);
        const auto all_labels_meanabs = common::meanabs(all_labels);
        all_labels.clear();
        const double labels_mean = arma::mean(arma::vectorise(train_label_chunks[chunk_ix]));

        OMP_LOCK(chunk_preds_l)
        auto p_chunk_preds = otr<t_parameter_predictions_set>();

#if defined(TUNE_FIREFLY) || defined(TUNE_PRIMA)

        constexpr unsigned D = 2;
        constexpr double lambda_multiplier = C_tune_range_max_lambda;
        constexpr double lambda_scaler = 1e8;
        const auto gamma_variance = get_gamma_range_variance(train_len);
        const double equiexp = std::log(std::sqrt(lambda_multiplier)) / M_LN2;
        arma::mat bounds(D, 2);
        bounds(0, 0) = 0;
        bounds(0, 1) = 1;
        bounds(1, 0) = C_tune_range_min_lambda;
        bounds(1, 1) = 1. / lambda_scaler;

#endif

#ifdef TUNE_FIREFLY
        const auto cost_f = [&](const std::vector<double> &xx)
        {
            auto score_params = *p_template_chunk_params;
            const auto gamma = xx[0] * gamma_variance + 1 - gamma_variance;
            const auto lambda = lambda_multiplier * std::pow(xx[1] * lambda_scaler, equiexp);
            score_params.set_svr_kernel_param2(lambda);
            const auto p_Ztune = prepare_Z(ccache(), score_params, tune_feature_chunks_t[chunk_ix], bpt::not_a_date_time);
            const auto p_Ztrain = prepare_Z(ccache(), score_params, train_feature_chunks_t[chunk_ix], bpt::not_a_date_time);
            const auto mingamma = calc_gamma(*p_Ztrain, tune_labels_mean);
            double res;
            PROFILE_EXEC_TIME(res = validate_gammas(mingamma, *p_Ztune, *p_Ztrain, {gamma}, score_params, chunk_ix, *p_chunk_preds, &predictions_l),
                              "Validation, score " << res << ", gamma " << gamma << ", mingamma " << mingamma << ", lambda " << lambda << " with parameters " << score_params);
            return res;
        };

        const std::pair<double, std::vector<double>> res = optimizer::firefly(
                    bounds.n_rows, 20, 20, FFA_ALPHA, FFA_BETAMIN, FFA_GAMMA, bounds.col(0), bounds.col(1),
                    arma::vec(bounds.n_rows, arma::fill::ones), cost_f);
#elif defined(TUNE_PRIMA)

        constexpr unsigned n = 35;
        constexpr unsigned iters = 22;
        const optimizer::t_pprima_res res = optimizer::pprima(prima_algorithm_t::PRIMA_LINCOA, n, bounds, [&](const double *x, double *const f) {
            const auto xx = optimizer::pprima::ensure_bounds(x, bounds);
            auto score_params = *p_template_chunk_params;
            const auto lambda = lambda_multiplier * std::pow(xx[1] * lambda_scaler, equiexp);
            score_params.set_svr_kernel_param2(lambda);
            const auto p_Ztune = prepare_Z(ccache(), score_params, tune_features_t, bpt::not_a_date_time);
            const auto p_Ztrain = prepare_Z(ccache(), score_params, train_feature_chunks_t[chunk_ix], bpt::not_a_date_time);
            const auto mingamma = calc_gamma(*p_Ztrain, labels_mean);
            const std::deque<double> gamma_m{xx[0] * gamma_variance + 1. - gamma_variance};
            PROFILE_EXEC_TIME(
                    *f = validate_gammas(mingamma, *p_Ztune, *p_Ztrain, gamma_m, score_params, chunk_ix, *p_chunk_preds, &chunk_preds_l, all_labels_meanabs, train_len,
                                         tune_labels, tune_lastknowns),
                    "Validation, score " << *f << ", gamma base parameter " << xx[0] << ", mingamma " << mingamma << ", lambda " << lambda << " with parameters "
                                         << score_params);
        }, iters, 2. / n, 1e-11); // default rhobeg=range/4, default rhoend 5e-10
        tune_features_t.clear();
        tune_labels.clear();
        tune_lastknowns.clear();
        bounds.clear();

#else // Adaptive grid
        constexpr unsigned C_grid_depth = 6; // Tune
        constexpr double C_grid_range_div = 10;
        double range_min_lambda = C_tune_range_min_lambda, range_max_lambda = C_tune_range_max_lambda;
        for (size_t grid_level_lambda = 0; grid_level_lambda < C_grid_depth; ++grid_level_lambda) {
            const double range_lambda = range_max_lambda - range_min_lambda;
            std::deque<double> lambdas;
            for (double r = range_min_lambda; r < range_max_lambda; r += range_lambda / C_grid_range_div) lambdas.emplace_back(r);
#pragma omp parallel for schedule(static, 1) num_threads(adj_threads(std::min<size_t>(lambdas.size(), TUNE_THREADS)))
            for (const double lambda: lambdas) {
                auto lambda_params = *p_template_chunk_params;
                lambda_params.set_svr_kernel_param2(lambda);
                const auto p_Ztune = prepare_Z(lambda_params, tune_feature_chunks_t[chunk_ix]);
                const auto p_Ztrain = prepare_Z(lambda_params, train_feature_chunks_t[chunk_ix]);
                const auto mingamma = calc_gamma(*p_Ztrain, tune_labels_mean);
                double range_min_gamma = C_tune_range_min_gammamulti, range_max_gamma = C_tune_range_max_gammamulti;
                for (size_t grid_level_gamma = 0; grid_level_gamma < C_grid_depth; ++grid_level_gamma) {
                    const double range_gamma = range_max_gamma - range_min_gamma;
                    const double range_half_gamma = range_gamma * .5;
                    std::deque<double> gamma_multis;
                    for (double r = range_min_gamma; r < range_max_gamma; r += range_gamma / C_grid_range_div) gamma_multis.emplace_back(r);
                    PROFILE_EXEC_TIME(validate_gammas(mingamma, *p_Ztune, *p_Ztrain, gamma_multis, lambda_params, chunk_ix, *p_chunk_preds, &predictions_l),
                                      "Validate " << gamma_multis << " gammas " << lambda_params);
                    const auto best_gamma_multi = (**p_chunk_preds->cbegin()).p_params->get_svr_kernel_param() / mingamma;
                    const double gamma_range_tightening = std::abs(best_gamma_multi - range_half_gamma) / range_half_gamma;
                    range_min_gamma = std::max(C_tune_range_min_gammamulti, best_gamma_multi - range_half_gamma * gamma_range_tightening);
                    range_max_gamma = range_min_gamma + 2. * range_half_gamma * gamma_range_tightening;
                }
                LOG4_DEBUG("Crass gamma pass score " << (**p_chunk_preds->cbegin()).score << ", mingamma " << mingamma << ", best parameters " << *(**p_chunk_preds->cbegin()).p_params);
            }
            const auto best_lambda = (**p_chunk_preds->cbegin()).p_params->get_svr_kernel_param2();
            const double range_half_lambda = range_lambda * .5;
            const double lambda_range_tightening = std::abs(best_lambda - range_half_lambda) / range_half_lambda;
            range_min_lambda = std::max(C_tune_range_min_lambda, best_lambda - range_half_lambda * lambda_range_tightening);
            range_max_lambda = range_min_lambda + 2. * range_half_lambda * lambda_range_tightening;
        }
#endif
        if (p_chunk_preds->size() > common::C_tune_keep_preds) p_chunk_preds->erase(std::next(p_chunk_preds->begin(), common::C_tune_keep_preds), p_chunk_preds->end());
        std::for_each(std::execution::par_unseq, p_chunk_preds->cbegin(), p_chunk_preds->cend(), [](const auto &pp) {
            LOG4_INFO("Final best score " << pp->score << ", final parameters " << *pp->p_params);
        });
        chunks_score[chunk_ix] = (**p_chunk_preds->cbegin()).score;
        omp_set_lock(&ins_chunk_results_l);
        p_predictions->emplace(std::tuple{p_template_chunk_params->get_decon_level(), p_template_chunk_params->get_grad_level(), chunk_ix}, p_chunk_preds);
        if (calculated_sf) set_scaling_factors(chunk_sf);
        omp_unset_lock(&ins_chunk_results_l);
    }

    ccache().checkout_tuner(*this);
}

#endif

void
OnlineMIMOSVR::recombine_params(const size_t chunk_ix)
{
    if (not ccache().recombine_go(*this)) return;

    const auto &tune_predictions = ccache().get_tuner_state(column_name);
    const auto colct = p_dataset->get_model_count();
    const auto levct = p_dataset->get_transformation_levels();
    constexpr size_t grad_level = 0; // Only gradient level 0 supported
    const auto max_num_combos = std::pow<double>(double(common::C_tune_keep_preds), double(colct));
    const uint64_t num_combos = (uint64_t) std::min<double>(common::C_num_combos, max_num_combos);
    if (num_combos < CUDA_BLOCK_SIZE) return;
    const double filter_combos = double(max_num_combos) / double(num_combos);
    std::vector<t_param_preds_cu> params_preds(colct * common::C_tune_keep_preds);
#pragma omp parallel for collapse(2) num_threads(adj_threads(colct))
    for (uint32_t colix = 0; colix < colct; ++colix) {
        for (uint32_t rowix = 0; rowix < common::C_tune_keep_preds; ++rowix) {
            params_preds[rowix * colct + colix].params_ix = rowix;
            const auto levix = business::ModelService::to_level_ix(colix, levct);
            if (tune_predictions.at({levix, grad_level, chunk_ix})->size() < common::C_tune_keep_preds)
                LOG4_THROW("Not enough tune results " << tune_predictions.at({levix, grad_level, chunk_ix})->size() << " to recombine!");
            const auto p_sf = business::DQScalingFactorService::find(scaling_factors, 0, levix, grad_level, chunk_ix, false, true);
            const auto &tp = *tune_predictions.at({levix, grad_level, chunk_ix}) ^ rowix;
#pragma omp parallel for num_threads(adj_threads(C_emo_max_j))
            for (uint32_t j = 0; j < C_emo_max_j; ++j) {
                const uint32_t elto = C_emo_tune_min_validation_window + (C_emo_max_j - j - 1) * C_emo_slide_skip;
                for (uint32_t el = 0; el < elto; ++el) {
                    auto &preds = params_preds[rowix * colct + colix].predictions[j][el] = arma::mean(tp->p_predictions->at(j).row(el));
                    business::DQScalingFactorService::unscale_labels(*p_sf, preds);
                    auto &labels = params_preds[rowix * colct + colix].labels[j][el] = arma::mean(tp->p_labels->at(j).row(el));
                    business::DQScalingFactorService::unscale_labels(*p_sf, labels);
                    auto &lastknowns = params_preds[rowix * colct + colix].last_knowns[j][el] = arma::mean(tp->p_last_knowns->at(j).row(el));
                    business::DQScalingFactorService::unscale_labels(*p_sf, lastknowns);
                    LOG4_TRACE("Row " << rowix << ", J " << j << ", col " << colix << ", prediction " << params_preds[rowix * colct + colix].predictions[j][el]
                                      << ", label " << params_preds[rowix * colct + colix].labels[j][el] << ", last known "
                                      << params_preds[rowix * colct + colix].last_knowns[j][el]);
                }
                for (uint32_t el = elto; el < C_emo_test_len; ++el) {
                    params_preds[rowix * colct + colix].predictions[j][el] = 0;
                    params_preds[rowix * colct + colix].labels[j][el] = 0;
                    params_preds[rowix * colct + colix].last_knowns[j][el] = 0;
                }
            }
        }
    }

    const uint32_t rows_gpu = common::gpu_handler::get().get_max_gpu_data_chunk_size() / 2 / colct / (2 * CUDA_BLOCK_SIZE) * (2 * CUDA_BLOCK_SIZE);
    auto best_score = std::numeric_limits<double>::max();
    std::vector<uint8_t> best_params_ixs(colct, uint8_t(0));
    LOG4_DEBUG("Predictions filtered out " << filter_combos << ", total prediction rows " << num_combos << ", rows per GPU " << rows_gpu << ", column count " << colct
                                           << ", limit num combos " << common::C_num_combos);

    // const auto start_time = std::chrono::steady_clock::now();
    OMP_LOCK(best_score_l)
#pragma omp parallel for schedule(static, 1) num_threads(common::gpu_handler::get().get_max_running_gpu_threads_number())
    for (uint64_t start_row_ix = 0; start_row_ix < num_combos; start_row_ix += rows_gpu) {
        // if (best_score != std::numeric_limits<double>::max()) continue;
        // if (std::chrono::steady_clock::now() - start_time > std::chrono::minutes(45)) continue;
        const uint64_t end_row_ix = std::min<uint64_t>(start_row_ix + rows_gpu, num_combos);
        const uint64_t chunk_rows_ct = end_row_ix - start_row_ix;
        const arma::colvec colixs = arma::linspace<arma::colvec>(double(start_row_ix), double(end_row_ix), chunk_rows_ct);
        arma::uchar_mat combos(chunk_rows_ct, colct);
#pragma omp parallel for schedule(static, 1) num_threads(adj_threads(colct))
        for (uint32_t colix = 0; colix < colct; ++colix)
            combos.col(colix) = arma::conv_to<arma::uchar_colvec>::from(common::mod<double>(
                    colixs / std::pow<double>(common::C_tune_keep_preds, colct - colix) / filter_combos, double(common::C_tune_keep_preds)));

        double chunk_best_score;
        std::vector<uint8_t> chunk_best_params_ixs(colct, 0);
        PROFILE_EXEC_TIME(recombine_parameters(chunk_rows_ct, colct, combos.mem, params_preds.data(), &chunk_best_score, chunk_best_params_ixs.data()),
                          "Recombine chunk " << chunk_rows_ct << "x" << colct << ", added set of size " << unsigned(common::C_tune_keep_preds)
                                             << ", filter out " << filter_combos - 1 << " combinations, start row " << start_row_ix << ", end row " << end_row_ix
                                             << ", score " << chunk_best_score);
        dtype(combos){}.swap(combos);
        omp_set_lock(&best_score_l);
        if (chunk_best_score < best_score) {
            best_score = chunk_best_score;
            best_params_ixs = chunk_best_params_ixs;
            LOG4_DEBUG("Found best score " << best_score << ", " << 100. * best_score / C_emo_slides_len << " pct direction error, indexes "
                                           << common::to_string(chunk_best_params_ixs));
        }
        omp_unset_lock(&best_score_l);
    }
    dtype(params_preds){}.swap(params_preds);

    auto p_ensemble = p_dataset->get_ensemble(column_name);
#pragma omp parallel for schedule(static, 1) num_threads(adj_threads(colct))
    for (uint32_t colix = 0; colix < colct; ++colix) {
        const uint32_t levix = business::ModelService::to_level_ix(colix, p_dataset->get_transformation_levels());
        const auto p_params = (**tune_predictions.at(std::forward_as_tuple(levix, grad_level, chunk_ix))->cbegin()).p_params;
        auto m_grad = p_ensemble->get_model(levix)->get_gradient(grad_level);
        if (m_grad->is_manifold()) m_grad->get_manifold()->set_params(p_params, chunk_ix);
        else m_grad->set_params(p_params, chunk_ix);
    }
}


}// datamodel
} // svr
