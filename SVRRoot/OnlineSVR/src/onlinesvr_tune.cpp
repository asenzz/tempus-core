//
// Created by zarko on 9/29/22.
//

#include <armadillo>
#include <exception>
#include <iterator>
#include <limits>
#include <cmath>
#include <complex>
#include <deque>
#include <memory>
#include <omp.h>
#include <string>
#include <tuple>
#include "common/compatibility.hpp"
#include "common/defines.h"
#include "firefly.hpp"
#include "model/SVRParameters.hpp"
#include "onlinesvr.hpp"
#include "common/Logging.hpp"
#include "common/constants.hpp"
#include "firefly.hpp"
#include "appcontext.hpp"
#include "cuqrsolve.hpp"
#include "util/math_utils.hpp"
#include "util/string_utils.hpp"
#include "ModelService.hpp"
#include "recombine_parameters.cuh"


#define TUNE_THREADS (1 + common::gpu_handler::get().get_gpu_devices_count())


namespace svr {
namespace datamodel {

auto
eval_score(const datamodel::SVRParameters &params, const arma::mat &K, const arma::mat &labels, const arma::mat &last_knowns, const size_t train_len, const double meanabs_labels)
{
    const off_t start_point_K = K.n_rows - EMO_TEST_LEN - train_len;
    const off_t start_point_labels = labels.n_rows - EMO_TEST_LEN - train_len;
    if (start_point_K < 0 || start_point_labels < 0 || labels.n_rows != K.n_rows)
        LOG4_THROW("Shorter K " << start_point_K << " or labels " << start_point_labels << " for K " << arma::size(K) << ", labels " << arma::size(labels));
    auto p_out_preds = ptr<std::deque<arma::mat>>(EMO_MAX_J);
    auto p_out_labels = ptr<std::deque<arma::mat>>(EMO_MAX_J);
    auto p_out_last_knowns = ptr<std::deque<arma::mat>>(EMO_MAX_J);
    arma::mat K_epsco = K;
    K_epsco.diag() += 1. / (2. * params.get_svr_C());
    double score = 0;
#pragma omp parallel for reduction(+:score) schedule(static, 1) num_threads(adj_threads(std::min<size_t>(EMO_MAX_J, TUNE_THREADS)))
    for (size_t j = 0; j < EMO_MAX_J; ++j) {
        const size_t x_train_start = j * EMO_SLIDE_SKIP;
        const size_t x_train_final = x_train_start + train_len - 1;
        const size_t x_test_start = x_train_final + 1;
        LOG4_TRACE("Try " << j << ", K " << arma::size(K) << ", start point labels " << start_point_labels << ", start point K " << start_point_K << ", train start " <<
                    x_train_start << ", train final " << x_train_final << ", test start " << x_test_start << ", test final is mat end.");
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
            score += this_score;
        } catch (const std::exception &ex) {
            LOG4_ERROR("Error solving matrix, try "  << j << ", K " << arma::size(K) << ", " << x_train_start << ", " << x_train_final << ", " << x_test_start << ", error " << ex.what());
            p_out_preds->at(j) = arma::mat(p_out_preds->at(j).n_rows - x_test_start, labels.n_cols, arma::fill::value(BAD_VALIDATION));
            p_out_labels->at(j) = p_out_preds->at(j);
            p_out_last_knowns->at(j) = p_out_preds->at(j);
            score += BAD_VALIDATION;
        }
    }
    return std::tuple(p_out_preds, p_out_labels, p_out_last_knowns, score);
}


void OnlineMIMOSVR::tune()
{
    if (is_manifold()) {
        LOG4_DEBUG("Skipping tuning of manifold kernel!");
        return;
    }

    auto p_predictions = ccache().checkin_tuner(*this);

    LOG4_DEBUG("Tuning labels " << common::present(*p_labels) << ", features " << common::present(*p_features) << ", last-knowns "
                                << common::present(*p_last_knowns) << ", EMO_SLIDE_SKIP " << EMO_SLIDE_SKIP << ", EMO_MAX_J " << EMO_MAX_J << ", EMO_TUNE_MIN_VALIDATION_WINDOW " <<
                                EMO_TUNE_MIN_VALIDATION_WINDOW << ", EMO_TEST_LEN " << EMO_TEST_LEN << ", decon level " << decon_level);
    const auto meanabs_all_labels = common::meanabs(*p_labels);
    const auto ixs_tune = get_indexes();
    const auto num_chunks = ixs_tune.size();
    const auto train_len = ixs_tune.front().n_rows;
    std::deque<arma::uvec> ixs_train(num_chunks);
    std::deque<arma::mat> feature_chunks_t(num_chunks), label_chunks(num_chunks), lastknown_chunks(num_chunks);
#pragma omp parallel for num_threads(adj_threads(num_chunks)) schedule(static, 1)
    for (size_t i = 0; i < num_chunks; ++i) {
        ixs_train[i] = ixs_tune[i] + EMO_TEST_LEN;
        feature_chunks_t[i] = arma::join_cols(
                p_features->rows(ixs_tune[i]), p_features->rows(p_features->n_rows - EMO_TEST_LEN, p_features->n_rows - 1)).t();
        label_chunks[i] = arma::join_cols(
                p_labels->rows(ixs_tune[i]), p_labels->rows(p_labels->n_rows - EMO_TEST_LEN, p_labels->n_rows - 1));
        lastknown_chunks[i] = arma::join_cols(
                p_last_knowns->rows(ixs_tune[i]), p_last_knowns->rows(p_last_knowns->n_rows - EMO_TEST_LEN, p_last_knowns->n_rows - 1));
    }
    const auto validate_gammas = [&](
            const double mingamma, const arma::mat &Z, const matrix_ptr p_Ztrain, const auto &gamma_multis, const auto &score_params, const size_t chunk_ix,
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
#pragma omp parallel num_threads(adj_threads(2))
            {
#pragma omp task
                {
                    arma::mat K_train(arma::size(*p_Ztrain));
                    solvers::kernel_from_distances(K_train.memptr(), p_Ztrain->mem, p_Ztrain->n_rows, p_Ztrain->n_cols, p_gamma_params->get_svr_kernel_param());
                    epsco = calc_epsco(K_train);
                    p_gamma_params->set_svr_C(1. / (2. * epsco));
                }
#pragma omp task
                solvers::kernel_from_distances(K.memptr(), Z.mem, Z.n_rows, Z.n_cols, p_gamma_params->get_svr_kernel_param());
            }
            if (epsco <= 0) LOG4_WARN("Auto epsco is negative indefinite " << epsco << ",  K " << common::present(K) << ", gamma template_parameters " << *p_gamma_params);
            const auto [p_out_preds, p_out_labels, p_out_last_knowns, score] = eval_score(
                    *p_gamma_params, K, label_chunks[chunk_ix], lastknown_chunks[chunk_ix], train_len, meanabs_all_labels);

            omp_set_lock(p_predictions_l);
            if (chunk_preds.size() < size_t(common::C_tune_keep_preds) || score < (**chunk_preds.begin()).score) {
                p_gamma_params->set_svr_C(1. / (2. * epsco));
                chunk_preds.emplace(otr<t_param_preds>(score, p_gamma_params, p_out_preds, p_out_labels, p_out_last_knowns));
                LOG4_DEBUG("Lambda, gamma tune best score " << score / double(EMO_MAX_J) << ", for " << *p_gamma_params);
                if (chunk_preds.size() > size_t(common::C_tune_keep_preds))
                    chunk_preds.erase(std::next(chunk_preds.begin(), common::C_tune_keep_preds), chunk_preds.end());
            }
            omp_unset_lock(p_predictions_l);
        }
    };

    OMP_LOCK(ins_chunk_results_l)
#pragma omp parallel for schedule(static, 1) num_threads(adj_threads(std::min<size_t>(num_chunks, TUNE_THREADS)))
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
        const auto original_input_queue_column_name = p_template_chunk_params->get_input_queue_column_name();

        OMP_LOCK(predictions_l)
        auto p_chunk_preds = otr<t_parameter_predictions_set>();
#ifdef TUNE_FIREFLY
        optimizer::firefly(1, C_grid_range_div, C_grid_depth, FFA_ALPHA, FFA_BETAMIN, FFA_GAMMA, arma::vec(1), arma::vec(1, arma::fill::value(3.)), arma::vec(1, arma::fill_ones), [&](const auto &la) {
            auto lambda_params = *chunk_params;
            lambda_params.set_svr_kernel_param2(la[0]);
            const arma::mat &Z = get_cached_Z(lambda_params, feature_chunks_t[chunk_ix], bpt::not_a_date_time);
            const double mean_train_labels = arma::mean(arma::vectorise(labels.rows(ixs_train[chunk_ix])));
            auto tmp_params = lambda_params;
            tmp_params.set_input_queue_column_name(tmp_params.get_input_queue_column_name() + "_TMP");
            const arma::mat *p_Ztrain = prepare_Z(tmp_params, features.rows(ixs_train[chunk_ix]).t(), train_len);
            const auto mingamma = calc_gamma(*p_Ztrain, train_len, mean_train_labels);
            const auto res = optimizer::firefly(1, C_grid_range_div, C_grid_depth, FFA_ALPHA, FFA_BETAMIN, FFA_GAMMA, std::vector<double>{1.}, std::vector<double>{1e3}, std::vector<double>{1.}, [&](const auto &ga) {
                PROFILE_EXEC_TIME(validate_gammas(mingamma, Z, p_Ztrain, {ga[0]}, lambda_params, chunk_ix), "Validate gammas " << lambda_params);
                return (**predictions[chunk_ix].begin()).score;
            }).operator std::pair<double, std::vector<double>>().first;
            delete p_Ztrain;
            return res;
        });
#else // Adaptive grid
        double range_min_lambda = C_tune_range_min_lambda, range_max_lambda = C_tune_range_max_lambda;
        for (size_t grid_level_lambda = 0; grid_level_lambda < C_grid_depth; ++grid_level_lambda) {
            const double range_lambda = range_max_lambda - range_min_lambda;
            std::deque<double> lambdas;
            if (!grid_level_lambda) lambdas = C_tune_extra_lambdas;
            for (double r = range_min_lambda; r < range_max_lambda; r += range_lambda / C_grid_range_div) lambdas.emplace_back(r);
#pragma omp parallel for schedule(static, 1) num_threads(adj_threads(std::min<size_t>(lambdas.size(), TUNE_THREADS)))
            for (const double lambda: lambdas) {
                auto lambda_params = *p_template_chunk_params;
                lambda_params.set_input_queue_column_name("TUNE_" + original_input_queue_column_name);
                lambda_params.set_chunk_ix(chunk_ix);
                lambda_params.set_svr_kernel_param2(lambda);
                const arma::mat &Z = ccache().get_cached_Z(lambda_params, feature_chunks_t[chunk_ix], bpt::not_a_date_time);
                const auto p_Ztrain = prepare_Z(lambda_params, p_features->rows(ixs_train[chunk_ix]).t());
                constexpr auto mingamma = 1.; //calc_gamma(*p_Ztrain, arma::mean(arma::vectorise(p_labels->rows(ixs_train[chunk_ix]))));
                double range_min_gamma = C_tune_range_min_gammamulti, range_max_gamma = C_tune_range_max_gammamulti;
                for (size_t grid_level_gamma = 0; grid_level_gamma < C_grid_depth; ++grid_level_gamma) {
                    std::deque<double> gamma_multis;
                    const double range_gamma = range_max_gamma - range_min_gamma;
                    const double range_half_gamma = range_gamma * .5;
                    gamma_multis.clear();
                    for (double r = range_min_gamma; r < range_max_gamma; r += range_gamma / C_grid_range_div) gamma_multis.emplace_back(r);
                    PROFILE_EXEC_TIME(validate_gammas(mingamma, Z, p_Ztrain, gamma_multis, lambda_params, chunk_ix, *p_chunk_preds, &predictions_l),
                                      "Validate " << gamma_multis.size() << " gammas " << lambda_params);
                    const auto best_gamma_multi = (**p_chunk_preds->cbegin()).p_params->get_svr_kernel_param() / mingamma;
                    const double gamma_range_tightening = std::abs(best_gamma_multi - range_half_gamma) / range_half_gamma;
                    range_min_gamma = std::max(0., best_gamma_multi - range_half_gamma * gamma_range_tightening);
                    range_max_gamma = range_min_gamma + 2. * range_half_gamma * gamma_range_tightening;
                }
                LOG4_DEBUG("Crass gamma pass score " << (**p_chunk_preds->cbegin()).score << ", mingamma " << mingamma << ", best parameters " << *(**p_chunk_preds->cbegin()).p_params);
            }
            const auto best_lambda = (**p_chunk_preds->cbegin()).p_params->get_svr_kernel_param2();
            const double range_half_lambda = range_lambda * .5;
            const double lambda_range_tightening = std::abs(best_lambda - range_half_lambda) / range_half_lambda;
            range_min_lambda = std::max(0., best_lambda - range_half_lambda * lambda_range_tightening);
            range_max_lambda = range_min_lambda + 2. * range_half_lambda * lambda_range_tightening;
        }
#endif
        if (p_chunk_preds->size() > common::C_tune_keep_preds) p_chunk_preds->erase(std::next(p_chunk_preds->begin(), common::C_tune_keep_preds), p_chunk_preds->end());
        std::for_each(std::execution::par_unseq, p_chunk_preds->begin(), p_chunk_preds->end(), [&original_input_queue_column_name](auto &pp) {
            pp->p_params->set_input_queue_column_name(original_input_queue_column_name);
            LOG4_INFO("Final best score " << pp->score << ", final parameters " << *pp->p_params);
        });

        omp_set_lock(&ins_chunk_results_l);
        p_predictions->emplace(std::tuple{p_template_chunk_params->get_decon_level(), p_template_chunk_params->get_grad_level(), chunk_ix}, p_chunk_preds);
        omp_unset_lock(&ins_chunk_results_l);
    }

    ccache().checkout_tuner(*this);
}

void
OnlineMIMOSVR::recombine_params(const size_t chunk_ix)
{
    if (not ccache().recombine_go(*this)) return;

    const auto &tune_predictions = ccache().get_tuner_state(column_name);
    const auto colct = p_dataset->get_model_count();
    const auto levct = p_dataset->get_transformation_levels();
    constexpr size_t grad_level = 0; // Only gradient level 0 supported
    const auto &scaling_factors = p_dataset->get_dq_scaling_factors();
    const auto max_num_combos = std::pow<double>(double(common::C_tune_keep_preds), double(colct));
    const uint64_t num_combos = std::min<double>(common::C_num_combos, max_num_combos);
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
            const auto &tp = *tune_predictions.at({levix, grad_level, chunk_ix}) ^ rowix;
#pragma omp parallel for num_threads(adj_threads(EMO_MAX_J))
            for (uint32_t j = 0; j < EMO_MAX_J; ++j) {
                const uint32_t elto = EMO_TUNE_MIN_VALIDATION_WINDOW + (EMO_MAX_J - j - 1) * EMO_SLIDE_SKIP;
                for (uint32_t el = 0; el < elto; ++el) {
                    params_preds[rowix * colct + colix].predictions[j][el] = business::DQScalingFactorService::unscale(
                            arma::mean(tp->p_predictions->at(j).row(el)), levix, tp->p_params->get_input_queue_column_name(), scaling_factors);
                    params_preds[rowix * colct + colix].labels[j][el] = business::DQScalingFactorService::unscale(
                            arma::mean(tp->p_labels->at(j).row(el)), levix, tp->p_params->get_input_queue_column_name(), scaling_factors);
                    params_preds[rowix * colct + colix].last_knowns[j][el] = business::DQScalingFactorService::unscale(
                            arma::mean(tp->p_last_knowns->at(j).row(el)), levix, tp->p_params->get_input_queue_column_name(), scaling_factors);
                    LOG4_TRACE("Row " << rowix << ", J " << j << ", col " << colix << ", prediction " << params_preds[rowix * colct + colix].predictions[j][el]
                                      << ", label " << params_preds[rowix * colct + colix].labels[j][el] << ", last known " << params_preds[rowix * colct + colix].last_knowns[j][el]);
                }
                for (uint32_t el = elto; el < EMO_TEST_LEN; ++el) {
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
            combos.col(colix) = arma::conv_to<arma::uchar_colvec>::from(common::mod<arma::colvec>(
                    colixs / std::pow<double>(common::C_tune_keep_preds, colct - colix) / filter_combos, double(common::C_tune_keep_preds)));

        double chunk_best_score;
        std::vector<uint8_t> chunk_best_params_ixs(colct, 0);
        PROFILE_EXEC_TIME(recombine_parameters(chunk_rows_ct, colct, combos.mem, params_preds.data(), &chunk_best_score, chunk_best_params_ixs.data()),
                          "Recombine chunk " << chunk_rows_ct << "x" << colct << ", added set of size " << unsigned(common::C_tune_keep_preds)
                                             << ", filter out " << filter_combos - 1 << " combinations, start row " << start_row_ix << ", end row " << end_row_ix << ", score " << chunk_best_score);
        decltype(combos){}.swap(combos);
        omp_set_lock(&best_score_l);
        if (chunk_best_score < best_score) {
            best_score = chunk_best_score;
            best_params_ixs = chunk_best_params_ixs;
            LOG4_DEBUG("Found best score " << best_score << ", " << 100. * best_score / double(EMO_SLIDES_LEN) << " pct direction error, indexes "
                                           << common::to_string(chunk_best_params_ixs));
        }
        omp_unset_lock(&best_score_l);
    }
    decltype(params_preds){}.swap(params_preds);

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
