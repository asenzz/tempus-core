//
// Created by zarko on 9/29/22.
//
#include <jemalloc/jemalloc.h>
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
#include <hdbscan/hdbscan.h>
#include <hdbscan/logger.h>
#include "pprune.hpp"
#include "DQScalingFactorService.hpp"
#include "onlinesvr.hpp"
#include "appcontext.hpp"
#include "recombine_parameters.cuh"
#include "common/logging.hpp"
#include "kernel_factory.hpp"
#include "kernel_gbm.hpp"

namespace svr {
namespace datamodel {

// Returns indexes of outliers
arma::uvec outlier_hdbscan(const arma::mat &features_t)
{
    const auto min_points = std::min<uint32_t>(PROPS.get_hdbs_points(), features_t.n_cols);
    clustering::hdbscan *scan = clustering::hdbscan_init(nullptr, min_points);
    auto err = clustering::hdbscan_run(scan, (distance_t *) features_t.mem, features_t.n_cols, features_t.n_rows, TRUE, H_DOUBLE); // Transposed and colmajor
    if (err == HDBSCAN_ERROR)
        LOG4_THROW("Could not run hdbscan. Error code: " << err);
#ifndef NDEBUG
    clustering::hdbscan_print_outlier_scores(scan->outlierScores, scan->numPoints);
#endif
    auto ures = arma::regspace<arma::uvec>(0, scan->numPoints - 1);
    std::ranges::sort(ures, [&scan](const auto i1, const auto i2) { return scan->outlierScores[i1].score < scan->outlierScores[i2].score; });
    LOG4_TRACE("Returning " << common::present(ures));
    return ures.tail_rows(PROPS.get_outlier_slack());
}

arma::uvec outlier_bacon(const arma::mat &features_t)
{
    VSLSSTaskPtr task;
    const arma::mat x = features_t.t();
    const MKL_INT N = x.n_rows;
    const MKL_INT DIM = x.n_cols;
    const MKL_INT xstorage = VSL_SS_MATRIX_STORAGE_ROWS;
    const MKL_INT NParams = VSL_SS_BACON_PARAMS_N;
    constexpr double BaconParams[VSL_SS_BACON_PARAMS_N] = {VSL_SS_METHOD_BACON_MEDIAN_INIT, .01, .01};
    arma::vec BaconWeights(N, ARMA_DEFAULT_FILL);
    /* Create a task */
    vs_errchk(vsldSSNewTask(&task, &DIM, &N, &xstorage, x.mem, nullptr, nullptr));

    /* Initialize the task parameters */
    vs_errchk(vsldSSEditOutliersDetection(task, &NParams, BaconParams, BaconWeights.memptr()));

    /* Detect the outliers in the observations */
    vs_errchk(vsldSSCompute(task, VSL_SS_OUTLIERS, VSL_SS_METHOD_BACON));

    /* BaconWeights will hold zeros or/and ones */ /* Deallocate the task resources */
    vs_errchk(vslSSDeleteTask(&task));

    return arma::find(BaconWeights == 0);
}

// TODO Port outlier detection to CUDA
arma::vec score_dataset(const arma::mat &labels, const arma::mat &features_t, const float dual_lag_ratio)
{
    assert(dual_lag_ratio < .5);
    const uint32_t lag = (labels.n_rows - 1) * dual_lag_ratio;
    arma::vec score(labels.n_rows, ARMA_DEFAULT_FILL);
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

void save_chunk_params(const SVRParameters_ptr &p_params)
{
    if (APP.svr_parameters_service.exists(p_params))
        APP.svr_parameters_service.remove(p_params);
    APP.svr_parameters_service.save(p_params);
}

void OnlineSVR::tune()
{
    if (const auto p_params = is_manifold()) {
        assert(ixs.size() == 1);
        assert(train_feature_chunks_t.size() == 1);
        assert(train_label_chunks.size() == 1);
        PROFILE_(kernel::IKernel<double>::get<kernel::kernel_deep_path<double>>(*p_params)->init(
            projection, p_dataset, train_feature_chunks_t.front(), train_label_chunks.front(), last_trained_time));
        p_params->set_svr_kernel_param(1); // Setting SVR Kernel param to 1 to indicate that the kernel parameters are initialized
        return;
    }

    constexpr uint8_t D = 1;
    // static const auto equiexp = std::log(std::sqrt(PROPS.get_tune_max_lambda())) / M_LN2;
    static const auto bounds1 = [] {
        arma::mat r(4, 2, ARMA_DEFAULT_FILL);
        r.col(0).zeros();
        r.col(1).fill(PROPS.get_tune_max_fback());
        r(0, 1) = PROPS.get_tune_max_tau();
        return r;
    }();
    static const auto bounds2 = [] {
        arma::mat r(D, 2, ARMA_DEFAULT_FILL);
        r(0, 0) = 0;
        r(0, 1) = PROPS.get_tune_max_lambda();
        return r;
    }();
    const auto num_chunks = ixs.size();
    LOG4_TRACE("Systuning level " << level << ", step " << step << ", num chunks " << num_chunks << ", first chunk " << common::present_chunk(ixs.front(), .1) <<
        ", last chunk " << common::present_chunk(ixs.back(), .1) << " labels " << common::present(*p_labels) << ", features " << common::present(*p_features) <<
        ", max lambda " << PROPS.get_tune_max_lambda() << ", tau particles " << PROPS.get_tune_particles1() << ", iterations " << PROPS.get_tune_iteration1() << ", lambda particles " <<
        PROPS.get_tune_particles2() << ", iterations " << PROPS.get_tune_iteration2() << ", opt depth " << PROPS.get_opt_depth());

#pragma omp parallel for schedule(static, 1) ADJ_THREADS(std::min<uint32_t>(num_chunks, PROPS.get_parallel_chunks())) default(shared) firstprivate(num_chunks)
    for (DTYPE(num_chunks) chunk_ix = 0; chunk_ix < num_chunks; ++chunk_ix) {
        auto p_chunk_params = get_params_ptr(chunk_ix);
        if (!p_chunk_params)
            LOG4_THROW("Template parameters for chunk " << chunk_ix << " not found");
        if (PROPS.get_outlier_slack()) {
            ixs[chunk_ix].shed_rows(outlier_hdbscan(train_feature_chunks_t[chunk_ix]));
            prepare_chunk(chunk_ix);
            LOG4_TRACE("Trimmed chunk " << chunk_ix << " ixs " << common::present(ixs[chunk_ix]) << ", chunk ixs " << common::present(ixs[chunk_ix]) << ", labels rows " << p_labels->n_rows);
        }
        if (p_chunk_params->get_kernel_type() == kernel_type::TFT) {
            PROFILE_(kernel::IKernel<double>::get<kernel::kernel_tft<double>>(*p_chunk_params)->init(train_feature_chunks_t[chunk_ix], train_label_chunks[chunk_ix]));
            p_chunk_params->set_svr_kernel_param(1); // Setting SVR Kernel param to 1 to indicate that the kernel parameters are initialized
            if (model_id) save_chunk_params(p_chunk_params);
            continue; // No tuning for TFT
        } else if (p_chunk_params->get_kernel_type() == kernel_type::GBM) {
            PROFILE_(kernel::IKernel<double>::get<kernel::kernel_gbm<double>>(*p_chunk_params)->init(train_feature_chunks_t[chunk_ix], train_label_chunks[chunk_ix]));
            p_chunk_params->set_svr_kernel_param(1);
            if (model_id) save_chunk_params(p_chunk_params);
            continue; // No tuning for GBM
        }
        tbb::mutex chunk_preds_l;
        auto best_score = std::numeric_limits<double>::max();
        arma::mat W_tune, W_train;
#ifdef INSTANCE_WEIGHTS
        if (p_input_weights && p_input_weights->n_elem) {
            W_tune = weight_matrix(chunk_ixs_tune, *p_input_weights);
            W_train = instance_weight_matrix(ixs[chunk_ix], *p_input_weights);
        }
#endif
        cutuner cv(train_feature_chunks_t[chunk_ix], train_label_chunks[chunk_ix], W_train, *p_chunk_params);
        auto costF = [&](const double x[], double *const f) {
            const auto [score, gamma, min] = cv.phase1(x[0], x[1], x[2], x[3]);
            *f = score;
            const tbb::mutex::scoped_lock lk(chunk_preds_l);
            if (score < best_score) {
                p_chunk_params->set_svr_kernel_param(gamma);
                p_chunk_params->set_kernel_param3(*x);
                p_chunk_params->set_H_feedback(x[1]);
                p_chunk_params->set_D_feedback(x[2]);
                p_chunk_params->set_V_feedback(x[3]);
                p_chunk_params->set_min_Z(min);
                LOG4_TRACE("New best score distances " << score << ", previous best " << best_score << ", improvement " << common::imprv(score, best_score) << "pc, parameters " <<
                    *p_chunk_params << ", opt arg " << common::to_string(x, 4));
                best_score = score;
            }
        };
        (void) optimizer::pprune(optimizer::pprune::C_default_algo, PROPS.get_tune_particles1(), bounds1, costF, PROPS.get_tune_iteration1(), 0, 0, {}, {}, std::min<uint32_t>(PROPS.get_tune_iteration1(), PROPS.get_opt_depth()));
        cv.prepare_second_phase(*p_chunk_params);
        auto costF2 = [&](const double x[], double *const f) {
            const auto [score, gamma, min] = cv.phase2(*x);
            *f = score;
            const tbb::mutex::scoped_lock lk(chunk_preds_l);
            if (score < best_score) {
                p_chunk_params->set_svr_kernel_param(gamma);
                p_chunk_params->set_svr_kernel_param2(*x);
                p_chunk_params->set_min_Z(min);
                LOG4_TRACE("New best score kernel " << score << ", previous best " << best_score << ", improvement " << common::imprv(score, best_score) << "pc, parameters " <<
                    *p_chunk_params << ", opt arg " << *x);
                best_score = score;
            }
        };
        chunks_score[chunk_ix] = best_score;
        (void) optimizer::pprune(optimizer::pprune::C_default_algo, PROPS.get_tune_particles2(), bounds2, costF2, PROPS.get_tune_iteration2(), 0, 0, {}, {}, std::min<uint32_t>(PROPS.get_tune_iteration2(), PROPS.get_opt_depth()));

        assert(p_chunk_params->get_svr_kernel_param() != 0);

        set_params(p_chunk_params, chunk_ix);
        LOG4_INFO("Tuned best score " << chunks_score[chunk_ix] << ", final parameters " << *p_chunk_params);

        if (model_id) save_chunk_params(p_chunk_params);
    }
    clean_chunks();
}

arma::u32_vec OnlineSVR::get_predict_chunks() const
{
    LOG4_BEGIN();
    assert(chunks_score.size());
    auto res = arma::regspace<arma::u32_vec>(0, chunks_score.size() - 1);
    if (PROPS.get_predict_chunks() > chunks_score.size()) goto __bail;
    std::stable_sort(C_default_exec_policy, res.begin(), res.end(), [this](const auto i1, const auto i2) { return chunks_score[i1] < chunks_score[i2]; });
    res.shed_rows(0, res.size() - PROPS.get_predict_chunks() - 1);
__bail:
    LOG4_TRACE("Using up to " << PROPS.get_predict_chunks() << " chunks with scores " << common::present(chunks_score) << ", selected " << common::present(res));
    return res;
}

void OnlineSVR::clean_chunks()
{
    return;

    if (chunks_score.size() <= PROPS.get_predict_chunks()) return;

    const auto used_chunks = get_predict_chunks();

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
} // datamodel
} // svr
