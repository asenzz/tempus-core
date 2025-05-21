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

    const auto opt_particles = PROPS.get_tune_particles();
    const auto opt_iters = PROPS.get_tune_iterations();
    const auto opt_depth = std::max<DTYPE(opt_iters) >(1, opt_iters / 20);
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
    const auto num_chunks = ixs.size();
    LOG4_TRACE("Systuning level " << level << ", step " << step << ", num chunks " << num_chunks << ", first chunk "
                                  << common::present_chunk(ixs.front(), .1) << ", last chunk " << common::present_chunk(ixs.back(), .1) << " labels " << common::present(*p_labels)
                                  << ", features " << common::present(*p_features) << ", last-knowns " << common::present(*p_last_knowns) << ", max lambda "
                                  << PROPS.get_tune_max_lambda() << ", gamma variance " << solvers::C_gamma_variance << ", particles " << opt_particles << ", iterations "
                                  << opt_iters);

#pragma omp parallel for schedule(static, 1) ADJ_THREADS(std::min<uint32_t>(num_chunks, PROPS.get_parallel_chunks())) default(shared) firstprivate(num_chunks, opt_particles, opt_iters)
    for (DTYPE(num_chunks) chunk_ix = 0; chunk_ix < num_chunks; ++chunk_ix) {
        auto p_chunk_params = get_params_ptr(chunk_ix);
        if (!p_chunk_params) LOG4_THROW("Template parameters for chunk " << chunk_ix << " not found");
        const auto &train_cuml = p_chunk_params->get_kernel_type() == e_kernel_type::PATH ?
                                 ccache().get_cumulatives(*p_chunk_params, train_feature_chunks_t[chunk_ix], last_trained_time) : train_feature_chunks_t[chunk_ix];
        if (PROPS.get_outlier_slack()) { // TODO Rewrite outlier detection
            p_chunk_params->set_svr_kernel_param2(datamodel::C_default_svrparam_kernel_param2);
            p_chunk_params->set_kernel_param3(datamodel::C_default_svrparam_kernel_param_tau);
            auto K = kernel::IKernel<double>::get(*p_chunk_params)->distances(train_cuml);
            const auto [gamma, dc] = calc_gamma(K, train_label_chunks[chunk_ix]);
            p_chunk_params->set_svr_kernel_param(gamma);
            p_chunk_params->set_min_Z(dc);
            kernel::IKernel<double>::get(*p_chunk_params)->kernel_from_distances_I(K);
            if (K.has_nonfinite()) LOG4_THROW("Kernel matrix has non-finites!");
            arma::uvec sorted_ixs;
            if (true) {
                const arma::mat L_biased = train_label_chunks[chunk_ix] % arma::linspace(1., PROPS.get_weights_slope(), train_label_chunks[chunk_ix].n_rows);
                arma::mat w;
#ifdef INSTANCE_WEIGHTS
                calc_weights(K * instance_weight_matrix(chunk_ixs_tune, *p_input_weights), L_biased, chunk_ixs_tune.n_rows * PROPS.get_solve_iterations_coefficient(),
                             PROPS.get_stabilize_iterations_count(), w);
#else
                calc_weights(K, L_biased, ixs[chunk_ix].n_rows * PROPS.get_solve_iterations_coefficient(), PROPS.get_stabilize_iterations_count(), w);
#endif
                sorted_ixs = arma::stable_sort_index(arma::pow(arma::abs(self_predict(K, w, L_biased)), PROPS.get_weights_exp()) % score_dataset(
                        L_biased, train_feature_chunks_t[chunk_ix], 1e-2)) + PROPS.get_shift_limit();
            } else
                sorted_ixs = ixs[chunk_ix](arma::stable_sort_index(score_dataset(train_label_chunks[chunk_ix], train_feature_chunks_t[chunk_ix], 1e-2)));

            const auto max_chunk_rows = std::min<uint32_t>(sorted_ixs.n_rows - PROPS.get_outlier_slack(), max_chunk_size);
            ixs[chunk_ix] = arma::sort(sorted_ixs.tail(max_chunk_rows));
            prepare_chunk(chunk_ix);
            LOG4_TRACE("Trimmed chunk " << chunk_ix << " ixs " << common::present(ixs[chunk_ix]) << ", chunk tune ixs " << common::present(ixs[chunk_ix]) <<
                                      ", sorted ixs " << common::present(sorted_ixs) << ", labels rows " << p_labels->n_rows << ", max chunk rows " << max_chunk_rows);
#if 0 // TODO Test BACON outlier detection, consider HDBSCAN
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
        }
        tbb::mutex chunk_preds_l;
        auto prev_score = std::numeric_limits<double>::max();
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
            const cusys cv(train_cuml, train_label_chunks[chunk_ix], W_train, *p_chunk_params);
            auto costF = [&, max_lambda, max_tau](const double x[], double *const f) {
                const auto xx = optimizer::pprune::ensure_bounds(x, bounds);
                const auto lambda = max_lambda * xx[0];
                const auto tau = max_tau * xx[1];
                const auto [score, gamma, min] = cv(lambda, tau);
                *f = score;
                if (score < prev_score) {
                    const tbb::mutex::scoped_lock lk(chunk_preds_l);
                    if (score < prev_score) {
                        p_chunk_params->set_svr_kernel_param(gamma);
                        p_chunk_params->set_svr_kernel_param2(lambda);
                        p_chunk_params->set_min_Z(min);
                        p_chunk_params->set_kernel_param3(tau);
                        LOG4_TRACE("New best score " << score << ", previous best " << prev_score << ", improvement " << common::imprv(score, prev_score) << "pc, parameters " <<
                                                     *p_chunk_params << ", opt arg " << arma::conv_to<arma::rowvec>::from(xx));
                        prev_score = score;
                    }
                }
            };
            const optimizer::t_pprune_res res = optimizer::pprune(optimizer::pprune::C_default_algo, opt_particles, bounds, costF, opt_iters, 0, 0, x0, {}, opt_depth);
        }
        set_params(p_chunk_params, chunk_ix);
        LOG4_INFO("Tuned best score " << chunks_score[chunk_ix] << ", final parameters " << *p_chunk_params);

        if (!model_id) continue;
        if (APP.svr_parameters_service.exists(p_chunk_params)) APP.svr_parameters_service.remove(p_chunk_params);
        APP.svr_parameters_service.save(p_chunk_params);
    }
    clean_chunks();
}

arma::u32_vec OnlineMIMOSVR::get_predict_chunks() const
{
    LOG4_BEGIN();
    assert(chunks_score.size());
    auto res = arma::regspace<arma::u32_vec>(0, chunks_score.size() - 1);
    if (PROPS.get_predict_chunks() > chunks_score.size()) goto __bail;
    std::stable_sort(C_default_exec_policy, res.begin(), res.end(), [this](const auto i1, const auto i2) { return chunks_score[i1] < chunks_score[i2]; });
    res.shed_rows(0, std::min<uint32_t>(res.size(), PROPS.get_predict_chunks()) - 1);
    __bail:
    LOG4_TRACE("Using up to " << PROPS.get_predict_chunks() << " chunks with scores " << common::present(chunks_score) << ", selected " << common::present(res));
    return res;
}

void OnlineMIMOSVR::clean_chunks()
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


}// datamodel
} // svr
