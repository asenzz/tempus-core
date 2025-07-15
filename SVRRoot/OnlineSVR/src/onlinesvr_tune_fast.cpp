//
// Created by zarko on 9/29/22.
//

#include <jemalloc/jemalloc.h>
#include <ipp/ipp.h>
#include <execution>
#include <cublas_v2.h>
#include <armadillo>
#include <exception>
#include <iterator>
#include <limits>
#include <cmath>
#include <complex>
#include <deque>
#include <tuple>
#include <mkl_vsl.h>
#include <magma_auxiliary.h>
#include "pprune.hpp"
#include "DQScalingFactorService.hpp"
#include "onlinesvr.hpp"
#include "appcontext.hpp"
#include "recombine_parameters.cuh"
#include "common/logging.hpp"
#include "cuqrsolve.cuh"
#include "kernel_factory.hpp"

namespace svr {
namespace datamodel {

// TODO Port outlier detection to CUDA
arma::vec score_dataset(const arma::mat &labels, const arma::mat &features_t, const float dual_lag_ratio)
{
    assert(dual_lag_ratio < .5);
    const uint32_t lag = (labels.n_rows - 1) * dual_lag_ratio;
    arma::vec score(labels.n_rows, arma::fill::none);
    const auto lag_2 = 2 * lag;
    OMP_FOR_i(labels.n_rows) {
        uint32_t start_i, end_i;
        if (i < lag) {
            start_i = 0;
            end_i = lag_2;
        } else if (i >= labels.n_rows - lag) {
            start_i = labels.n_rows - lag_2;
            end_i = labels.n_rows - 1;
        } else {
            start_i = i - lag;
            end_i = i + lag;
        }
        score[i] = 1. / std::abs(
                arma::mean(arma::vectorise(labels.row(i))) - arma::mean(arma::vectorise(labels.rows(start_i, end_i))) *
                                                             arma::mean(arma::vectorise(features_t.col(i)) - arma::mean(arma::vectorise(features_t.cols(start_i, end_i)))));
    }
    return score;
}

void OnlineMIMOSVR::tune_sys()
{
    if (is_manifold()) {
        LOG4_DEBUG("Skipping tuning of manifold kernel!");
        return;
    }

    const uint16_t num_chunks = ixs.size();
    if (chunks_score.size() != num_chunks) chunks_score.resize(num_chunks);
    p_kernel_matrices->resize(num_chunks);
    weight_chunks.resize(num_chunks);

    const auto opt_particles = PROPS.get_tune_particles();
    const auto opt_iters = PROPS.get_tune_iterations();
    constexpr uint8_t D = 2;
    // static const auto equiexp = std::log(std::sqrt(PROPS.get_tune_max_lambda())) / M_LN2;
    static const auto bounds = [] {
        arma::mat r(D, 2);
        r.col(0).zeros();
        r.col(1).ones();
        return r;
    }();
    static const auto x0 = [&]() -> arma::mat {
        arma::mat r(D, opt_particles, arma::fill::none);
        common::equispaced(r, bounds.tail_rows(D), {});
        return r;
    }();

    LOG4_TRACE("Systuning level " << level << ", step " << step << ", num chunks " << num_chunks << ", first chunk " << common::present_chunk(ixs.front(), C_chunk_header) <<
              ", last chunk " << common::present_chunk(ixs.back(), C_chunk_header) << " labels " << common::present(*p_labels) << ", features " << common::present(*p_features) <<
              ", last-knowns " << common::present(*p_last_knowns) << ", max lambda " << PROPS.get_tune_max_lambda() << ", gamma variance " << solvers::C_gamma_variance << ", particles " << opt_particles <<
                                  ", iterations " << opt_iters);

#pragma omp parallel for schedule(static, 1) ADJ_THREADS(std::min<unsigned>(num_chunks, 1)) default(shared) firstprivate(num_chunks, opt_particles, opt_iters)
    for (DTYPE(num_chunks) chunk_ix = 0; chunk_ix < num_chunks; ++chunk_ix) {
        auto p_chunk_params = get_params_ptr(chunk_ix);
        if (!p_chunk_params) {
            for (const auto &p: param_set)
                if (!p->is_manifold()) {
                    LOG4_WARN("Parameters for chunk " << chunk_ix << " not found, using template from " << *p);
                    p_chunk_params = otr(*p);
                    p_chunk_params->set_chunk_index(chunk_ix);
                    p_chunk_params->set_grad_level(gradient);
                    p_chunk_params->set_decon_level(level);
                    break;
                }
            if (!p_chunk_params) LOG4_THROW("Template parameters for chunk " << chunk_ix << " not found");
        }
        bool samples_selected = PROPS.get_outlier_slack() < 1;
        __prepare_tune_data:
#ifndef NDEBUG
        LOG4_TRACE("Before scaling chunk " << chunk_ix << ", train labels " << common::present(train_label_chunks[chunk_ix]) << ", train features " <<
                                           common::present(train_feature_chunks_t[chunk_ix]));
#endif
        // Scaling factors are calculated on tune chunks
        auto p_labels_sf = business::DQScalingFactorService::find(scaling_factors, model_id, chunk_ix, gradient, step, level, false, true);
        auto features_sf = business::DQScalingFactorService::slice(scaling_factors, chunk_ix, gradient, step);
        bool calculated_sf = false;
        const auto lag = p_chunk_params->get_lag_count();
        if (!p_labels_sf || features_sf.size() != train_feature_chunks_t[chunk_ix].n_rows / lag) {
            features_sf = business::DQScalingFactorService::calculate(chunk_ix, *this, train_feature_chunks_t[chunk_ix], train_label_chunks[chunk_ix]);
            p_labels_sf = business::DQScalingFactorService::find(features_sf, model_id, chunk_ix, gradient, step, level, false, true);
            business::DQScalingFactorService::scale_features(chunk_ix, gradient, step, lag, features_sf, train_feature_chunks_t[chunk_ix]);
            business::DQScalingFactorService::scale_labels(*p_labels_sf, train_label_chunks[chunk_ix]);
            calculated_sf = true;
        }
        LOG4_TRACE("After scaling chunk " << chunk_ix << ", train labels " << common::present(train_label_chunks[chunk_ix]) << ", train features " <<
                                common::present(train_feature_chunks_t[chunk_ix]) << ", labels scaling factor " << *p_labels_sf << ", features scaling factors " << features_sf);
        const arma::mat L_off = train_label_chunks[chunk_ix].tail_rows(train_label_chunks[chunk_ix].n_rows - PROPS.get_tune_skip());
        const double L_sum = arma::sum(arma::vectorise(L_off * C_diff_coef));
        const solvers::mmm_t L3m{L_sum / L_off.n_elem, L_sum, L_off.max() * C_diff_coef, L_off.min() * C_diff_coef};
        if (!samples_selected) { // TODO Rewrite
            p_chunk_params->set_svr_kernel_param(0);
            p_chunk_params->set_svr_kernel_param2(datamodel::C_default_svrparam_kernel_param2);
            p_chunk_params->set_kernel_param3(datamodel::C_default_svrparam_kernel_param_tau);
            const auto K = kernel::IKernel<double>::get(*p_chunk_params)->kernel(train_feature_chunks_t[chunk_ix]); // calc_gamma(*Kz, train_label_chunks[chunk_ix])
            if (K.has_nonfinite()) LOG4_THROW("Kernel matrix has non-finites!");
            arma::uvec sorted_ixs;
            if (true) {
                const arma::mat w_bias = train_label_chunks[chunk_ix] % arma::linspace(1., PROPS.get_weights_slope(), train_label_chunks[chunk_ix].n_rows);
                const auto epsco = calc_epsco(K, w_bias);
#ifdef INSTANCE_WEIGHTS
                const auto &w = calc_weights(K, w_bias, instance_weight_matrix(chunk_ixs_tune, *p_input_weights), epsco, PROPS.get_stabilize_iterations_count(),
                                                                                                                chunk_ixs_tune.n_rows * PROPS.get_solve_iterations_coefficient());
#else
                const auto w = calc_weights(K, w_bias, epsco, PROPS.get_stabilize_iterations_count(),
                                                                                ixs[chunk_ix].n_rows * PROPS.get_solve_iterations_coefficient(), weight_scaling_factors, chunk_ix);
#endif
                constexpr double C_weight_exp = 1;
                sorted_ixs =
                        arma::stable_sort_index(arma::pow(arma::abs(self_predict(K, w, w_bias)), PROPS.get_weights_exp()) % score_dataset(w_bias, train_feature_chunks_t[chunk_ix], 1e-2))
                            + PROPS.get_shift_limit();
            } else
                sorted_ixs = ixs[chunk_ix](arma::stable_sort_index(score_dataset(train_label_chunks[chunk_ix], train_feature_chunks_t[chunk_ix], 1e-2)));

            const auto max_chunk_rows = std::min<uint32_t>(sorted_ixs.n_rows - PROPS.get_outlier_slack(), max_chunk_size);
            ixs[chunk_ix] = arma::sort(sorted_ixs.tail(max_chunk_rows));
            train_feature_chunks_t[chunk_ix] = feature_chunk_t(ixs[chunk_ix]);
            train_label_chunks[chunk_ix] = p_labels->rows(ixs[chunk_ix]);
            LOG4_TRACE("Train chunk " << chunk_ix << " ixs " << common::present(ixs[chunk_ix]) << ", chunk tune ixs " << common::present(ixs[chunk_ix]) <<
                                      ", sorted ixs " << common::present(sorted_ixs) << ", labels rows " << p_labels->n_rows << ", max chunk rows " << max_chunk_rows);
#if 0 // TODO Test BACON outlier detection
            const arma::mat joint_dataset = common::normalize_cols<double>(arma::join_rows(train_label_chunks[chunk_ix], train_feature_chunks_t[chunk_ix].t()));
            LOG4_DEBUG("joint_dataset " << common::present(joint_dataset));
            const MKL_INT DIM = joint_dataset.n_cols; /* dimension of the task */
            const MKL_INT N = joint_dataset.n_rows; /* number of observations */
            double BaconParams[VSL_SS_BACON_PARAMS_N];
            arma::vec BaconWeights(N);
            /* Task and Initialization Parameters */
            constexpr MKL_INT xstorage = VSL_SS_MATRIX_STORAGE_ROWS;
            /* Parameters of the BACON algorithm */
            constexpr MKL_INT NParams = VSL_SS_BACON_PARAMS_N;
            BaconParams[0] = VSL_SS_METHOD_BACON_MEDIAN_INIT;
            BaconParams[1] = 1e-2; /* alpha */
            BaconParams[2] = 1e-2; /* beta */
            /* Create a task */
            VSLSSTaskPtr task;
            vs_errchk(vsldSSNewTask( &task, &DIM, &N, &xstorage, joint_dataset.mem, nullptr, nullptr ));
            /* Initialize the task parameters */
            vs_errchk(vsldSSEditOutliersDetection( task, &NParams, BaconParams, BaconWeights.memptr() ));
            /* Detect the outliers in the observations */
            vs_errchk(vsldSSCompute( task, VSL_SS_OUTLIERS, VSL_SS_METHOD_BACON ));
            /* BaconWeights will hold zeros or/and ones */
            const arma::uvec to_shed = arma::find(BaconWeights == 1);
            LOG4_ERROR("BACON found " << to_shed << " indexes to shed from chunk with size " << arma::size(joint_dataset) << ", BACON weights " << BaconWeights);
            /* Deallocate the task resources */
            vs_errchk(vslSSDeleteTask( &task ));

            ixs[chunk_ix].shed_rows(to_shed);
            train_label_chunks[chunk_ix] = p_labels->rows(ixs[chunk_ix]);
            train_feature_chunks_t[chunk_ix] = p_features->rows(ixs[chunk_ix]).t();
#endif
            samples_selected = true;
            goto __prepare_tune_data;
        }

        const auto &train_cuml = p_chunk_params->get_kernel_type() == e_kernel_type::PATH ?
                train_feature_chunks_t[chunk_ix] :
                ccache().get_cumulatives(*p_chunk_params, train_feature_chunks_t[chunk_ix], last_trained_time);
        train_label_chunks[chunk_ix] *= C_diff_coef;

        t_omp_lock chunk_preds_l;
        auto prev_score = std::numeric_limits<double>::max();
        std::atomic<uint32_t> call_ct = 0;
        arma::mat W_tune, W_train;
#ifdef INSTANCE_WEIGHTS
        if (p_input_weights && p_input_weights->n_elem) {
            W_tune = weight_matrix(chunk_ixs_tune, *p_input_weights);
            W_train = instance_weight_matrix(ixs[chunk_ix], *p_input_weights);
        }
#endif
        {
            const auto max_lambda = PROPS.get_tune_max_lambda();
            const auto max_tau = PROPS.get_tune_max_tau();
            const cusys cv(lag, train_cuml, train_label_chunks[chunk_ix], W_train, L3m, *p_chunk_params);
            auto costF = [&, max_lambda, max_tau](const double x[], double *const f) {
                ++call_ct;
                const auto xx = optimizer::pprune::ensure_bounds(x, bounds);
                const auto lambda = max_lambda * xx[0];
#ifdef TUNE_TAU
                const auto tau = max_tau * xx[1];
#else
                constexpr double tau = C_default_svrparam_kernel_param_tau;
#endif
                const auto [score, gamma, min] = cv(lambda, tau);
                *f = score;
                if (score < prev_score) {
                    chunk_preds_l.set();
                    if (score < prev_score) {
                        p_chunk_params->set_svr_kernel_param(gamma);
                        p_chunk_params->set_svr_kernel_param2(lambda);
                        p_chunk_params->set_min_Z(min);
#ifdef TUNE_TAU
                        p_chunk_params->set_kernel_param3(tau);
#endif
                        LOG4_TRACE("New best score " << score << ", previous best " << prev_score << ", improvement " << common::imprv(score, prev_score) << "pc, parameters " <<
                                                     *p_chunk_params << ", prima callback count " << call_ct << ", opt arg " << arma::conv_to<arma::rowvec>::from(xx));
                        prev_score = score;
                    }
                    chunk_preds_l.unset();
                }
            };
            const optimizer::t_pprune_res res = optimizer::pprune(optimizer::pprune::C_default_algo, opt_particles, bounds, costF,
                                                                  chunk_ix >= start_predict_chunk ? opt_iters : 1, 0, 0, x0);
        }
        if (calculated_sf) set_scaling_factors(features_sf);
        p_kernel_matrices->at(chunk_ix) = kernel::IKernel<double>::get(*p_chunk_params)->kernel(ccache(), train_feature_chunks_t[chunk_ix], last_trained_time);
        p_chunk_params->set_svr_C(1 / calc_epsco(p_kernel_matrices->at(chunk_ix), train_label_chunks[chunk_ix]));
        set_params(p_chunk_params, chunk_ix);
        calc_weights(chunk_ix, PROPS.get_stabilize_iterations_count(), ixs[chunk_ix].n_rows * PROPS.get_solve_iterations_coefficient());
        LOG4_INFO("Tune best score " << chunks_score[chunk_ix] << ", final parameters " << *p_chunk_params);

        if (!model_id) continue;
        if (APP.svr_parameters_service.exists(p_chunk_params)) APP.svr_parameters_service.remove(p_chunk_params);
        APP.svr_parameters_service.save(p_chunk_params);
        if (!calculated_sf) continue;
        for (const auto &sf: features_sf) {
            if (APP.dq_scaling_factor_service.exists(sf)) APP.dq_scaling_factor_service.remove(sf);
            APP.dq_scaling_factor_service.save(sf);
        }
    }
    clean_chunks();
}

std::deque<size_t> OnlineMIMOSVR::get_predict_chunks(const std::deque<std::pair<double, double>> &chunks_score)
{
    if (chunks_score.empty()) LOG4_THROW("Chunk scores empty!");
    std::deque<size_t> res(chunks_score.size());
    std::iota(res.begin(), res.end(), 0);
    double mean_first = 0;
#ifdef SECOND_SCORE
    double mean_second = 0;
#endif
    if (chunks_score.size() <= PROPS.get_predict_chunks()) goto __bail;
#define GOOD_SCORE(x) std::isnormal(x) && x != common::C_bad_validation
#define ADD_NORMAL(x, y) if (GOOD_SCORE(y)) x += y
    for (const auto &p: chunks_score) {
        ADD_NORMAL(mean_first, p.first);
#ifdef SECOND_SCORE
        ADD_NORMAL(mean_second, p.second);
#endif
    }
    mean_first /= double(chunks_score.size());
#ifdef SECOND_SCORE
    mean_second /= double(chunks_score.size());
#endif
    std::stable_sort(C_default_exec_policy, res.begin(), res.end(), [&](const size_t i1, const size_t i2) {
        double score_1 = 0, score_2 = 0;
#define ADD_PART(x, y, z) \
        if (GOOD_SCORE(y) && GOOD_SCORE(z)) x += y / z; \
            else x = common::C_bad_validation;

        ADD_PART(score_1, chunks_score[i1].first, mean_first);
        ADD_PART(score_2, chunks_score[i2].first, mean_first);
#ifdef SECOND_SCORE
        ADD_PART(score_1, chunks_score[i1].second, mean_second);
        ADD_PART(score_2, chunks_score[i2].second, mean_second);
#endif
        return score_1 < score_2;
    });
    res.erase(res.begin() + PROPS.get_predict_chunks(), res.end());
    std::sort(C_default_exec_policy, res.begin(), res.end());
    __bail:
    LOG4_DEBUG("Using upto " << PROPS.get_predict_chunks() << " chunks " << res << ", of all chunks with scores " << chunks_score);
    return res;
}

void OnlineMIMOSVR::clean_chunks()
{
    return;

    if (chunks_score.size() <= PROPS.get_predict_chunks()) return;

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
        if (!chunk_ix_set) iter = scaling_factors.erase(iter);
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
