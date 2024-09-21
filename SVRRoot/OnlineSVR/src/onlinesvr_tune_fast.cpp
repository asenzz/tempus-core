//
// Created by zarko on 9/29/22.
//

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

#define TUNE_THREADS(MAX_THREADS) num_threads(adj_threads(std::min<unsigned>((MAX_THREADS), 4)))

namespace svr {
namespace datamodel {

arma::mat get_reference_Z(const arma::mat &y, const size_t train_len)
{
    assert(y.n_rows - C_test_len - train_len == 0);
    const size_t n = y.n_rows;
    const arma::mat y_t = y.t();
    arma::mat r(n, n);
    OMP_FOR(n)
    for (size_t i = 0; i < n; ++i) r.row(i) = arma::abs(y_t - arma::mean(y.row(i)));
    return r;
}

void OnlineMIMOSVR::tune_fast()
{
    if (is_manifold()) {
        LOG4_DEBUG("Skipping tuning of manifold kernel!");
        return;
    }
    ixs = generate_indexes();
    const auto num_chunks = ixs.size();
    train_feature_chunks_t.resize(num_chunks);
    train_label_chunks.resize(num_chunks);
    t_omp_lock param_set_l;
    OMP_FOR(num_chunks)
    for (size_t i = 0; i < num_chunks; ++i) {
        train_feature_chunks_t[i] = feature_chunk_t(ixs[i]);
        train_label_chunks[i] = p_labels->rows(ixs[i]);
        SVRParameters_ptr p;
        if (!(p = get_params_ptr(i))) {
            p = otr(get_params());
            p->set_chunk_index(i);
            param_set_l.set();
            param_set.emplace(p);
            param_set_l.unset();
        }
    }

    if (chunks_score.size() != num_chunks) chunks_score.resize(num_chunks);
    t_omp_lock params_l;
    p_kernel_matrices->resize(num_chunks);
    weight_chunks.resize(num_chunks);

    constexpr unsigned opt_particles = 100;
    constexpr unsigned opt_iters = 40;
    constexpr unsigned D = 2;
    // static const auto equiexp = std::log(std::sqrt(C_tune_range_max_lambda)) / M_LN2;
    static const arma::mat bounds = []() {
        arma::mat r(D, 2);
        r.col(0).zeros();
        r.col(1).ones();
        return r;
    }();
    LOG4_TRACE("Tuning slide skip " << C_slide_skip << ", max j " << C_max_j << ", tune min validation window " << C_tune_min_validation_window << ", test len " << C_test_len
                                    << ", level " << level << ", step " << step << ", num chunks " << num_chunks << ", first chunk "
                                    << common::present_chunk(ixs.front(), C_chunk_header) << ", last chunk "
                                    << common::present_chunk(ixs.back(), C_chunk_header) << " labels " << common::present(*p_labels) << ", features "
                                    << common::present(*p_features) << ", last-knowns " << common::present(*p_last_knowns) << ", max lambda " << C_tune_range_max_lambda
                                    << ", gamma variance " << C_gamma_variance << ", particles " << opt_particles << ", iterations " << opt_iters);

    const arma::uvec test_ixs = arma::regspace<arma::uvec>(p_labels->n_rows - C_test_len, p_labels->n_rows - 1);
    std::array<arma::mat, C_max_j> test_labels;
    std::array<arma::mat, C_max_j> test_last_knowns;
    UNROLL(C_max_j)
    for (size_t j = 0; j < C_max_j; ++j) {
        const size_t test_tail = test_ixs.n_rows - j * C_slide_skip;
        test_labels[j] = p_labels->tail_rows(test_tail);
        test_last_knowns[j] = p_last_knowns->tail_rows(test_tail);
    }

#pragma omp parallel for simd schedule(static, 1) TUNE_THREADS(num_chunks) default(shared) firstprivate(num_chunks, opt_particles, opt_iters, D)
    for (size_t chunk_ix = 0; chunk_ix < num_chunks; ++chunk_ix) {
        auto p_template_chunk_params = get_params_ptr(chunk_ix);
        if (!p_template_chunk_params) {
            for (const auto &p: param_set)
                if (!p->is_manifold()) {
                    LOG4_WARN("Parameters for chunk " << chunk_ix << " not found, using template from " << *p);
                    p_template_chunk_params = otr(*p);
                    p_template_chunk_params->set_chunk_index(chunk_ix);
                    p_template_chunk_params->set_grad_level(gradient);
                    p_template_chunk_params->set_decon_level(level);
                    break;
                }
            if (!p_template_chunk_params) LOG4_THROW("Template parameters for chunk " << chunk_ix << " not found");
        }
        arma::uvec chunk_ixs_tune = arma::join_cols(ixs[chunk_ix] - C_test_len, test_ixs);
#ifdef BACON_OUTLIER // TODO Test BACON outlier detection
        bool samples_selected = false;
        __prepare_tune_data:
#endif
        arma::mat tune_features_t = feature_chunk_t(chunk_ixs_tune);

        for (unsigned j = 0; j < ixs[chunk_ix].size(); ++j) {
            const arma::vec diff_col = tune_features_t.col(j + C_test_len) - train_feature_chunks_t[chunk_ix].col(j);
            if (arma::accu(diff_col) > 0) LOG4_WARN("Col " << j << " differs " << common::present(diff_col));
        }

        arma::mat tune_labels = p_labels->rows(chunk_ixs_tune);
        auto p_labels_sf = business::DQScalingFactorService::find(scaling_factors, model_id, chunk_ix, gradient, step, level, false, true);
        auto features_sf = business::DQScalingFactorService::slice(scaling_factors, chunk_ix, gradient, step);
        bool calculated_sf = false;
        const auto lag = p_template_chunk_params->get_lag_count();
        if (!p_labels_sf || features_sf.size() != train_feature_chunks_t[chunk_ix].n_rows / lag) {
            features_sf = business::DQScalingFactorService::calculate(chunk_ix, *this, tune_features_t, tune_labels);
            p_labels_sf = business::DQScalingFactorService::find(features_sf, model_id, chunk_ix, gradient, step, level, false, true);
            calculated_sf = true;
        }
        business::DQScalingFactorService::scale_features(chunk_ix, gradient, step, lag, features_sf, train_feature_chunks_t[chunk_ix]);
        business::DQScalingFactorService::scale_labels(*p_labels_sf, train_label_chunks[chunk_ix]);
        const mmm_t train_L_m{common::mean(train_label_chunks[chunk_ix] * C_diff_coef),
                              train_label_chunks[chunk_ix].max() * C_diff_coef,
                              train_label_chunks[chunk_ix].min() * C_diff_coef};
#ifdef BACON_OUTLIER // TODO Test BACON outlier detection
        if (!samples_selected) {
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
            chunk_ixs_tune = arma::join_cols(ixs[chunk_ix] - C_test_len, arma::regspace<arma::uvec>(p_labels->n_rows - C_test_len, p_labels->n_rows - 1));
            samples_selected = true;
            goto __prepare_tune_data;
        }
#endif
        business::DQScalingFactorService::scale_features(chunk_ix, gradient, step, lag, features_sf, tune_features_t);
        business::DQScalingFactorService::scale_labels(*p_labels_sf, tune_labels);

        auto &train_cuml = ccache().get_cached_cumulatives(*p_template_chunk_params, train_feature_chunks_t[chunk_ix], last_trained_time);
        auto &tune_cuml = ccache().get_cached_cumulatives(*p_template_chunk_params, tune_features_t, last_trained_time);
        assert(tune_labels.n_rows - C_test_len - ixs[chunk_ix].n_elem == 0);
        tune_labels *= C_diff_coef;
        t_omp_lock chunk_preds_l;
        std::atomic<size_t> call_ct = 0;
        const auto labels_f = p_labels_sf->get_labels_factor();
        auto &tune_results = ccache().checkin_tuner(*this, chunk_ix);
        const auto prima_cb = [&, lag, chunk_ix](const double x[], double *const f) {
            ++call_ct;
            const auto xx = optimizer::pprune::ensure_bounds(x, bounds);
            const auto lambda = C_tune_range_max_lambda * xx[1];
            const common::gpu_context ctx;
            cu_errchk(cudaSetDevice(ctx.phy_id()));
            magma_queue_t ma_queue;
            magma_queue_create(ctx.phy_id(), &ma_queue);
            auto [score, gamma, epsco, p_predictions] =
                    cuvalidate(lambda, xx[0], lag, tune_cuml, train_cuml, tune_labels, train_L_m, labels_f, ma_queue);
            if ((*f = score) == common::C_bad_validation)
                LOG4_THROW("Bad validation for chunk " << chunk_ix << ", tune indexes " << common::present_chunk(ixs[chunk_ix], C_chunk_header) << ", gamma " << gamma <<
                                   ", epsco " << epsco << ", lambda " << lambda << ", tune features " << common::present(tune_features_t) << ", tune labels " << tune_labels);
            chunk_preds_l.set();
            const auto prev_score = tune_results.param_pred.front().score;
            if (score < prev_score) {
                p_template_chunk_params->set_svr_kernel_param(gamma);
                p_template_chunk_params->set_svr_kernel_param2(lambda);
                p_template_chunk_params->set_svr_C(1 / epsco);
                if (PROPS.get_recombine_parameters()) {
                    std::rotate(tune_results.param_pred.begin(), tune_results.param_pred.begin() + 1, tune_results.param_pred.end());
                    tune_results.param_pred.front().free();
                    tune_results.param_pred.front().p_predictions = p_predictions;
                    tune_results.param_pred.front().params = *p_template_chunk_params;
                }
                tune_results.param_pred.front().score = score;
                LOG4_TRACE("New best score " << score << ", previous best " << prev_score << ", improvement " << common::imprv(score, prev_score) << "pc, parameters " <<
                                             *p_template_chunk_params << ", prima callback count " << call_ct << ", parameters " << arma::conv_to<arma::rowvec>::from(xx));
                chunk_preds_l.unset();
                goto __bail;
            }
            chunk_preds_l.unset();
            if (PROPS.get_recombine_parameters()) {
                t_param_preds::free_predictions(p_predictions);
                delete p_predictions;
            }
            __bail:
            cu_errchk(cudaStreamSynchronize(magma_queue_get_cuda_stream(ma_queue)));
            magma_queue_destroy(ma_queue);
        };
        const optimizer::t_pprune_res res = optimizer::pprune(0 /* prima_algorithm_t::PRIMA_LINCOA */, opt_particles, bounds, prima_cb, opt_iters, .25, 1e-11);
        release_cont(chunk_ixs_tune);
        chunks_score[chunk_ix].first = tune_results.param_pred.front().score;
        tune_results.labels = test_labels;
        tune_results.last_knowns = test_last_knowns;
        release_cont(tune_labels);
        release_cont(tune_features_t);

        ccache().checkout_tuner(*this, chunk_ix);
        if (PROPS.get_recombine_parameters())
            recombine_params(chunk_ix, step);
        else
            set_params(p_template_chunk_params, chunk_ix);
        if (calculated_sf) set_scaling_factors(features_sf);
        p_kernel_matrices->at(chunk_ix) = *prepare_K(get_params(chunk_ix), train_feature_chunks_t[chunk_ix]);
        calc_weights(chunk_ix, PROPS.get_stabilize_iterations_count());
        LOG4_INFO("Tune best score " << chunks_score[chunk_ix] << ", final parameters " << *p_template_chunk_params);
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
    if (chunks_score.size() <= svr::datamodel::C_predict_chunks) goto __bail;
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
    res.erase(res.begin() + svr::datamodel::C_predict_chunks, res.end());
    std::sort(C_default_exec_policy, res.begin(), res.end());
    __bail:
    LOG4_DEBUG("Using " << svr::datamodel::C_predict_chunks << " chunks " << res << ", of all chunks with scores " << chunks_score);
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
