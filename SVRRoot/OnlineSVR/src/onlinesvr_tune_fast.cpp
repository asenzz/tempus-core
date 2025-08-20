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
#include <tuple>
#include <mkl_vsl.h>
#include <magma_auxiliary.h>
#ifdef USE_HDBSCAN
#include <hdbscan/hdbscan.h>
#include <hdbscan/logger.h>
#endif
#include "pprune.hpp"
#include "DQScalingFactorService.hpp"
#include "SVRParametersService.hpp"
#include "onlinesvr.hpp"
#include "appcontext.hpp"
#include "recombine_parameters.cuh"
#include "common/logging.hpp"
#include "kernel_factory.hpp"
#include "kernel_gbm.hpp"

namespace svr {
namespace datamodel {
#ifdef USE_HDBSCAN
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
#endif
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

// TODO Port to CUDA
// Return indexes of best samples to train on
void OnlineSVR::score_indexes(const arma::mat &features_t, const arma::mat &labels, arma::uvec &ixs)
{
    assert(labels.n_rows == features_t.n_cols);
    assert(labels.n_rows == ixs.n_elem);
    const uint32_t n_rows = labels.n_rows;
    arma::vec score(n_rows, arma::fill::zeros);
    const float predict_focus = PROPS.get_predict_focus();
    const uint32_t start_j = n_rows * (1 - predict_focus);
    OMP_FOR_i(n_rows)
        for (uint32_t j = start_j; j < n_rows; ++j)
            score[i] += common::sumabs<double>(labels.row(i) - labels.row(j)); // * common::sumabs<double>(features_t.col(i) - features_t.col(j));
    ixs = ixs.rows(arma::sort(arma::stable_sort_index(score).eval().head_rows(n_rows - PROPS.get_outlier_slack())));
}

void OnlineSVR::tune()
{
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
            // ixs[chunk_ix].shed_rows(outlier_hdbscan(train_feature_chunks_t[chunk_ix]));
            score_indexes(train_feature_chunks_t[chunk_ix], train_label_chunks[chunk_ix], ixs[chunk_ix]);
            // business::DQScalingFactorService::reset(scaling_factors, level, chunk_ix, step, gradient);
            prepare_chunk(chunk_ix);
            LOG4_TRACE("Trimmed chunk " << chunk_ix << " ixs " << common::present(ixs[chunk_ix]) << ", labels rows " << p_labels->n_rows);
        }
        kernel::IKernel<double>::get(*p_chunk_params)->init(*this, chunk_ix);
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
