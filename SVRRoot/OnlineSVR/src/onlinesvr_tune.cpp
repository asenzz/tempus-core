//
// Created by zarko on 9/29/22.
//

#include <armadillo>
#include <iterator>
#include <map>
#include <limits>
#include <cmath>
#include <complex>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include </opt/intel/oneapi/tbb/latest/include/tbb/parallel_reduce.h>

#include "common/defines.h"
#include "firefly.hpp"
#include "kernel_factory.hpp"
#include "model/SVRParameters.hpp"
#include "onlinesvr.hpp"
#include "common/Logging.hpp"
#include "common/constants.hpp"
#include "firefly.hpp"
#include "calc_kernel_inversions.hpp"
#include "cuda_path.hpp"
#include "appcontext.hpp"
#include "cuqrsolve.hpp"
#include "util/math_utils.hpp"
#include "util/string_utils.hpp"
#include "SVRParametersService.hpp"
#include "ModelService.hpp"


#define TUNE_THREADS (common::gpu_handler::get().get_gpu_devices_count())


namespace svr {

/* original features_t before matrix transposition
  = {   label 0 = { Level 0 lag 0, Level 1 lag 1, Level 1 lag 2, ... Level 1 lag 719, Level 2 lag 0, Level 2 lag 1, Level 2 lag 2, ... , Level 2 lag 719, ... Level 31 lag 0, Level 31 lag 1 ... Level 31 lag 31 } ,
        label 1 = { Level 0 lag 0, Level 1 lag 1, Level 1 lag 2, ... Level 1 lag 719, Level 2 lag 0, Level 2 lag 1, Level 2 lag 2, ... , Level 2 lag 719, ... Level 31 lag 0, Level 31 lag 1 ... Level 31 lag 31 } ..
        ...
        label 6000 = { Level 0 lag 0, Level 1 lag 1, Level 1 lag 2, ... Level 1 lag 719, Level 2 lag 0, Level 2 lag 1, Level 2 lag 2, ... , Level 2 lag 719, ... Level 31 lag 0, Level 31 lag 1 ... Level 31 lag 31 }
     }
*/

std::deque<arma::mat> *OnlineMIMOSVR::prepare_cumulatives(const SVRParameters &params, const arma::mat &features_t)
{
    const auto lag = params.get_lag_count();
    const auto levels = features_t.n_rows / lag;
    const auto p_cums = new std::deque<arma::mat>(levels);
#pragma omp parallel for schedule(static, 1) num_threads(adj_threads(levels))
    for (size_t i = 0; i < levels; ++i)
        p_cums->at(i) = arma::cumsum(features_t.rows(i * lag, (i + 1) * lag - 1));
    LOG4_TRACE("Prepared " << levels << " cumulatives with " << lag << " lag, parameters " << params << ", from features_t " << arma::size(features_t));
    return p_cums;
}


/* Faster method
 * arma::mat cum_XX = arma::zeros(features_len, samples);
__omp_pfor_i(0, samples,
      for (size_t l = 0; l < levels; ++l)
          for (size_t k = 0; k < lag; ++k)
              cum_XX(l * lag + k, i) = cums_X[l](k, i) );
LOG4_DEBUG(
        "Features matrix " << arma::size(features) << ", cumulative sums " << cums_X.size() << ", first level cumulative sum dims " << arma::size(cums_X.front()) <<
        ", concatenated cumulative sums matrix dims " << arma::size(cum_XX));
*/


/*
 * arma::mat this_first_Z = arma::zeros(short_size_X, short_size_X);
PROFILE_EXEC_TIME(
        double *d_Z; common::gpu_context ctx;
        cu_distances_xx(ctx.id(), features_len, levels, short_size_X, 0, 0, short_size_X, short_size_X, cum_XX.mem, params.get_svr_kernel_param2(), DEFAULT_TAU_TUNE, w_sum_sym,
                       this_first_Z.memptr(), d_Z), "CUDA kernel XX " << params);
*/


double OnlineMIMOSVR::calc_epsco(const arma::mat &K, const size_t train_len)
{
    return common::mean_asymm(K, train_len);
}


double OnlineMIMOSVR::calc_gamma(const arma::mat &Z, const double train_len, const double mean_L)
{
    const auto mean_Z = common::mean_asymm(Z, train_len);
    const auto res = std::sqrt(train_len * mean_Z / (2. * (train_len - mean_L)));
    LOG4_DEBUG("Mean Z " << mean_Z << ", mean L " << mean_L << ", train len " << train_len << ", gamma " << res);
    return res;
}


arma::mat *OnlineMIMOSVR::prepare_Z(const SVRParameters &params, const arma::mat &features_t, size_t len)
{
    if (!len) len = features_t.n_cols;
    arma::mat *p_Z;
    switch (params.get_kernel_type()) {
        case kernel_type_e::PATH: {
            const auto cums_X = get_cached_cumulatives(params, features_t);
            if (cums_X.empty()) LOG4_THROW("Failed preparing cumulatives matrices.");
            p_Z = new arma::mat(len, len, arma::fill::zeros);
#pragma omp parallel for num_threads(adj_threads(cums_X.size())) schedule(static, 1)
            for (const auto &c_X: cums_X) {
                arma::mat z = arma::mat(arma::size(*p_Z));
                kernel::path::cu_distances_xx(c_X.n_rows, 1, len, 0, 0, len, len, c_X.mem, params.get_svr_kernel_param2(), DEFAULT_TAU_TUNE, 1, z.memptr());
#pragma omp critical
                *p_Z += z;
            }
            LOG4_DEBUG("Returning path Z " << arma::size(*p_Z) << " for " << params << ", cumulative matrices " << cums_X.size() << ", of dimensions " << arma::size(cums_X.front()) << ", features " << arma::size(features_t));
            break;
        }
        default:
            const auto kf = IKernel<double>::get(params);
            p_Z = new arma::mat(len, len, arma::fill::zeros);
            kf->operator()(features_t.t(), *p_Z);
            break;
    }
    return p_Z;
}


arma::mat *OnlineMIMOSVR::prepare_Zy(const SVRParameters &params, const arma::mat &features_t /* transposed*/, const arma::mat &predict_features_t /* transposed*/, const bpt::ptime &pred_time)
{
    arma::mat *p_Zy = nullptr;
    switch (params.get_kernel_type()) {
        case kernel_type_e::PATH: {
            const auto cums_X = get_cached_cumulatives(params, features_t, pred_time);
            if (cums_X.empty()) LOG4_THROW("Failed preparing feature cumulative matrices.");
            auto p_cums_Y_tmp = prepare_cumulatives(params, predict_features_t);
            if (!p_cums_Y_tmp || p_cums_Y_tmp->empty()) LOG4_THROW("Failed preparing predict cumulative matrices.");

            p_Zy = new arma::mat(predict_features_t.n_cols, features_t.n_cols, arma::fill::zeros);
#pragma omp parallel for schedule(static, 1) num_threads(adj_threads(cums_X.size()))
            for (size_t i = 0; i < cums_X.size(); ++i) {
                arma::mat z = arma::mat(arma::size(*p_Zy));
                kernel::path::cu_distances_xy(
                        cums_X[i].n_rows, 1, features_t.n_cols, predict_features_t.n_cols, 0, 0, features_t.n_cols, predict_features_t.n_cols,
                        cums_X[i].memptr(), p_cums_Y_tmp->at(i).memptr(), params.get_svr_kernel_param2(), DEFAULT_TAU_TUNE, 1, z.memptr());
#pragma omp critical
                *p_Zy += z;
                p_cums_Y_tmp->at(i).clear();
            }
            LOG4_DEBUG("Returning Zy " << arma::size(*p_Zy) << ", features t " << arma::size(features_t) << ", predict features t " << arma::size(predict_features_t) <<
                " for parameters " << params << ", cumulative matrices " << cums_X.size() << " and " << p_cums_Y_tmp->size() << ", of dimensions " <<
                arma::size(cums_X.front()));
            delete p_cums_Y_tmp;
            break;
        }
        case kernel_type_e::DEEP_PATH:
        case kernel_type_e::DEEP_PATH2:
        default:
            LOG4_THROW("Kernel type " << int(params.get_kernel_type()) << " not handled!");
    }

    return p_Zy;
}

#if !defined(HYBRID_QUANTILES) && defined(TUNE_HYBRID)

double
OnlineMIMOSVR::produce_kernel_inverse_order(
        const SVRParameters &svr_parameters,
        const arma::mat &x_train,
        const arma::mat &y_train)
{
    const auto indexes = get_indexes(x_train.n_rows, svr_parameters);
    const auto num_chunks = indexes.size();
    const auto overall_score = tbb::parallel_reduce(
        tbb::blocked_range<std::size_t>(0, num_chunks), double(0), [&] (const tbb::blocked_range<size_t> &r, double err) {
            for (std::size_t i = r.begin(); i < r.end(); ++i) {
                arma::mat res_kernel_matrix;
                PROFILE_EXEC_TIME(
                        res_kernel_matrix = init_kernel_matrix(svr_parameters, x_train.rows(indexes[i])),
                        "Init kernel matrix " << arma::size(res_kernel_matrix));
                double cur_err;
                PROFILE_EXEC_TIME(
                        cur_err = calc_kernel_inversions(res_kernel_matrix.mem, y_train.rows(indexes[i])),
                        "Matrix inverse of chunk " << i << " level " << svr_parameters.get_decon_level() << " error " << cur_err << " total error " << err);
                err += cur_err;
            }
            return err;
        }, std::plus<double>());
    LOG4_DEBUG("Inverse matrix overall score " << overall_score << ", gamma " << svr_parameters.get_svr_kernel_param() <<
                    ", lambda " << svr_parameters.get_svr_kernel_param2() << ", level " << svr_parameters.get_decon_level());
    return overall_score; // / double(num_chunks);
}

#elif (defined(HYBRID_QUANTILES) && defined(TUNE_HYBRID))

double
OnlineMIMOSVR::produce_kernel_quantiles(
        const SVRParameters &svr_parameters,
        const arma::mat &x_train,
        const arma::mat &y_train)
{
    const auto p_manifold = init_manifold(x_train, y_train, svr_parameters);
    const auto indexes = get_indexes(x_train.n_rows, svr_parameters);
    const auto num_chunks = std::min<size_t>(indexes.size(), MAX_TUNE_CHUNKS);
    double overall_score = 0;
#pragma omp parallel for reduce(+:overall_score) default(shared) num_threads(adj_threads(num_chunks))
    for (size_t i = 0; i < num_chunks; ++i) {
        arma::mat res_kernel_matrix;
        PROFILE_EXEC_TIME(
                res_kernel_matrix = init_kernel_matrix(svr_parameters, x_train.rows(indexes[i]), y_train.rows(indexes[i])),
                "init_kernel_matrix");
        overall_score += calc_kernel_quantiles(res_kernel_matrix, y_train.rows(indexes[i]));
    }

    LOG4_DEBUG("Quantiles error " << overall_score << ", gamma " << svr_parameters.get_svr_kernel_param() << ", lambda " << svr_parameters.get_svr_kernel_param2());
    return overall_score;
}


double
opt_loss_quantiles_lambda(
        std::deque<double> &params,
        const arma::mat &features,
        const arma::mat &labels,
        const SVRParameters &default_svr_parameters)
{
    auto svr_parameters = default_svr_parameters;
    svr_parameters.set_svr_kernel_param2(params[0]);
    PROFILE_EXEC_TIME(
            return svr::OnlineMIMOSVR::produce_kernel_quantiles(svr_parameters, features, labels),
            "Produce kernel matrices quantiles");
}

#endif

#ifdef TUNE_HYBRID

namespace {

auto
eval_score(const arma::mat &K, const double epsco, const arma::mat &labels, const arma::mat &last_knowns, const size_t train_len, const double meanabs_labels)
{
    const off_t start_point_K = K.n_rows - EMO_TUNE_TEST_SIZE - train_len;
    const off_t start_point_labels = labels.n_rows - EMO_TUNE_TEST_SIZE - train_len;
    if (start_point_K < 0 || start_point_labels < 0 || labels.n_rows != K.n_rows)
        LOG4_THROW("Shorter K " << start_point_K << " or labels " << start_point_labels << " for K " << arma::size(K) << ", labels " << arma::size(labels));
    auto p_out_preds = std::make_shared<std::deque<arma::mat>>(EMO_MAX_J);
    auto p_out_labels = std::make_shared<std::deque<arma::mat>>(EMO_MAX_J);
    auto p_out_last_knowns = std::make_shared<std::deque<arma::mat>>(EMO_MAX_J);
    double score = 0;
#pragma omp parallel for reduction(+:score) schedule(static, 1) num_threads(adj_threads(std::min<size_t>(EMO_MAX_J, TUNE_THREADS)))
    for (size_t j = 0; j < EMO_MAX_J; ++j) {
        const size_t x_train_start = j * EMO_SLIDE_SKIP;
        const size_t x_train_final = x_train_start + train_len - 1;
        const size_t x_test_start = x_train_final + 1;
        const size_t x_test_final = x_test_start + EMO_TUNE_TEST_SIZE - j * EMO_SLIDE_SKIP - 1;
        const size_t x_test_len = x_train_final - x_train_start + 1;
        const arma::mat eye_epsco = arma::eye(x_test_len, x_test_len) * epsco;
        LOG4_TRACE("Try " << j << ", K " << arma::size(K) << ", start point labels " << start_point_labels << ", start point K " << start_point_K << ", train start " <<
                    x_train_start << ", train final " << x_train_final << ", test start " << x_test_start << ", test final " << x_test_final);
        try {
#if 0
            double this_score = arma::mean(arma::abs(arma::vectorise(K.submat(K.n_rows - 1 - start_point_K - x_test_final, K.n_rows - 1 - start_point_K - x_train_final,
                                                                        K.n_rows - 1 - start_point_K - x_test_start, K.n_rows - 1 - start_point_K - x_train_start)
                                                               * OnlineMIMOSVR::solve_dispatch(eye_epsco,
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
            p_out_labels->at(j) = labels.rows(start_point_labels + x_test_start, start_point_labels + x_test_final);
            p_out_preds->at(j) =
                    K.submat(start_point_K + x_test_start, start_point_K + x_train_start, start_point_K + x_test_final, start_point_K + x_train_final)
                    * OnlineMIMOSVR::solve_dispatch(eye_epsco,
                                                    K.submat(start_point_K + x_train_start, start_point_K + x_train_start, start_point_K + x_train_final,
                                                             start_point_K + x_train_final),
                                                    labels.rows(start_point_labels + x_train_start, start_point_labels + x_train_final),
                                                    PROPS.get_online_learn_iter_limit(), false);
            const auto this_score = common::meanabs<double>(p_out_labels->at(j) - p_out_preds->at(j)) / meanabs_labels;
#ifdef DIRECTION_VALIDATION
            p_out_last_knowns->at(j) = common::extrude_rows<double>(last_knowns.rows(start_point_labels + x_test_start, start_point_labels + x_test_final), p_out_preds->at(j).n_cols);
            this_score += this_score * common::sumabs<double>(arma::sign((arma::sign(p_out_labels->at(j) - p_out_last_knowns->at(j)) - arma::sign(p_out_preds->at(j) - p_out_last_knowns->at(j))))) / double(p_out_labels->at(j).n_elem);
#else
            p_out_last_knowns->at(j) = last_knowns.rows(start_point_labels + x_test_start, start_point_labels + x_test_final);
#endif
            score += this_score;
        } catch (const std::exception &ex) {
            LOG4_ERROR("Error solving matrix, try "  << j << ", K " << arma::size(K) << ", " << x_train_start << ", " << x_train_final << ", " << x_test_start << ", " << x_test_final << ", error " << ex.what());
            p_out_preds->at(j) = arma::mat(x_test_len, labels.n_cols, arma::fill::value(BAD_VALIDATION));
            p_out_labels->at(j) = p_out_preds->at(j);
            p_out_last_knowns->at(j) = p_out_preds->at(j);
            score += BAD_VALIDATION;
        }
    }
    return std::tuple(p_out_preds, p_out_labels, p_out_last_knowns, score);
}

std::atomic<size_t> running_tuners {0};
constexpr size_t C_grid_depth = 3;
constexpr double C_grid_range_div = 6;

};

void OnlineMIMOSVR::tune(
        t_gradient_tuned_parameters &predictions,
        const t_param_set &template_parameters,
        const arma::mat &features, const arma::mat &labels, const arma::mat &last_knowns,
        const size_t chunk_size)
{
    const auto p_head_parameters = *template_parameters.begin();
    datamodel::SVRParameters_ptr p_manifold_parameters;
    if (business::SVRParametersService::is_manifold(template_parameters, p_manifold_parameters)) {
        LOG4_DEBUG("Skipping tuning of manifold kernel!");
        return;
    }

    ++running_tuners;

    // TODO Fix tuning for multiple chunks, ixs_tune are not properly validated in loss function
    LOG4_DEBUG("Tuning parameters " << *p_head_parameters << ", labels " << common::present(labels) << ", features " << common::present(features) << ", last-knowns "
            << common::present(last_knowns) << ", EMO_SLIDE_SKIP " << EMO_SLIDE_SKIP << ", EMO_MAX_J " << EMO_MAX_J << ", EMO_TUNE_VALIDATION_WINDOW " << EMO_TUNE_VALIDATION_WINDOW);
    const std::string original_input_queue_column_name = p_head_parameters->get_input_queue_column_name();
    const std::string original_input_queue_table_name = p_head_parameters->get_input_queue_table_name();
    p_head_parameters->set_input_queue_column_name("TUNE_COLUMN_" + p_head_parameters->get_input_queue_column_name());
    p_head_parameters->set_input_queue_table_name("TUNE_TABLE_" + p_head_parameters->get_input_queue_table_name());
    const double meanabs_all_labels = common::meanabs(labels);
    const auto ixs_tune = get_indexes(features.n_rows, *p_head_parameters, chunk_size);
    const size_t num_chunks = ixs_tune.size();
    if (predictions.size() != num_chunks) predictions.resize(num_chunks);
    std::deque<arma::ucolvec> ixs_train;
    for (const auto &ixtu: ixs_tune) ixs_train.emplace_back(ixtu + EMO_TUNE_TEST_SIZE);
    const size_t train_len = ixs_tune.front().n_rows;
    std::deque<arma::mat> feature_chunks_t, label_chunks, lastknown_chunks;
    for (const auto &chunk_ixs: ixs_tune) {
        feature_chunks_t.emplace_back(features.rows(chunk_ixs).t());
        label_chunks.emplace_back(labels.rows(chunk_ixs));
        lastknown_chunks.emplace_back(last_knowns.rows(chunk_ixs));
        feature_chunks_t.back() = arma::join_rows(feature_chunks_t.back(), features.rows(features.n_rows - EMO_TUNE_TEST_SIZE, features.n_rows - 1).t());
        label_chunks.back() = arma::join_cols(label_chunks.back(), labels.rows(labels.n_rows - EMO_TUNE_TEST_SIZE, labels.n_rows - 1));
        lastknown_chunks.back() = arma::join_cols(lastknown_chunks.back(), last_knowns.rows(last_knowns.n_rows - EMO_TUNE_TEST_SIZE, last_knowns.n_rows - 1));
    }
    const arma::mat k_eye = arma::eye(train_len, train_len);
    const auto validate_gammas = [&](const double mingamma, const arma::mat &Z, const arma::mat *p_Ztrain,
            const std::deque<double> &gamma_multis, const SVRParameters &score_params, const size_t chunk_ix)
    {
        LOG4_DEBUG("Validating " << gamma_multis.size() << " gamma multipliers, starting from " << gamma_multis.front() << " to " << gamma_multis.back() <<
                    ", min gamma " << mingamma << ", Z " << arma::size(Z) << ", template_parameters " << score_params);
#pragma omp parallel for schedule(static, 1) num_threads(adj_threads(std::min<size_t>(gamma_multis.size(), TUNE_THREADS)))
        for (const double gamma_mult: gamma_multis) {
            auto p_gamma_params = std::make_shared<SVRParameters>(score_params);
            p_gamma_params->set_svr_kernel_param(gamma_mult * mingamma);
            arma::mat K(arma::size(Z));
            arma::mat *p_Ktrain = new arma::mat(arma::size(*p_Ztrain));
#pragma omp parallel num_threads(adj_threads(2))
            {
#pragma omp task
                solvers::kernel_from_distances(p_Ktrain->memptr(), p_Ztrain->mem, p_Ztrain->n_rows, p_Ztrain->n_cols, p_gamma_params->get_svr_kernel_param());
#pragma omp task
                solvers::kernel_from_distances(K.memptr(), Z.mem, Z.n_rows, Z.n_cols, p_gamma_params->get_svr_kernel_param());
            }
            const auto epsco = calc_epsco(*p_Ktrain, train_len);
            delete p_Ktrain;
            if (epsco <= 0) LOG4_WARN("Auto epsco is negative indefinite " << epsco << ",  K: " << common::present(K) << ", gamma template_parameters " << *p_gamma_params);
            const auto [p_out_preds, p_out_labels, p_out_last_knowns, score] = eval_score(
                    K, epsco, label_chunks[chunk_ix], lastknown_chunks[chunk_ix], train_len, meanabs_all_labels);
#pragma omp critical
            {
                if (predictions[chunk_ix].size() < size_t(common::C_tune_keep_preds)
                    || score < predictions[chunk_ix].begin()->get()->score) {
                    p_gamma_params->set_svr_C(1. / (2. * epsco));
                    p_gamma_params->set_input_queue_table_name(original_input_queue_table_name);
                    p_gamma_params->set_input_queue_column_name(original_input_queue_column_name);
                    predictions[chunk_ix].emplace(std::make_shared<t_param_preds>(score, p_gamma_params, p_out_preds, p_out_labels, p_out_last_knowns));
                    LOG4_DEBUG("Lambda, gamma tune best score " << score / double(EMO_MAX_J) << ", for " << *p_gamma_params);
                    if (predictions[chunk_ix].size() > size_t(common::C_tune_keep_preds))
                        predictions[chunk_ix].erase(std::next(predictions[chunk_ix].begin(), common::C_tune_keep_preds), predictions[chunk_ix].end());
                }
            }
        }
    };

#pragma omp parallel for schedule(static, 1) num_threads(adj_threads(std::min<size_t>(num_chunks, TUNE_THREADS)))
    for (size_t chunk_ix = 0; chunk_ix < num_chunks; ++chunk_ix) {
        auto chunk_params = business::SVRParametersService::find(template_parameters, chunk_ix);
        if (!chunk_params) {
            for (const auto &p: template_parameters)
                if (!p->is_manifold()) {
                    chunk_params = std::make_shared<SVRParameters>(*p);
                    LOG4_WARN("Parameters for chunk " << chunk_ix << " not found, using template from " << *p);
                    chunk_params->set_chunk_ix(chunk_ix);
                    break;
                }
            if (!chunk_params) LOG4_THROW("Parameters for chunk " << chunk_ix << " not found");
        }
#if 0
        optimizer::firefly(1, C_grid_range_div, C_grid_depth, FFA_ALPHA, FFA_BETAMIN, FFA_GAMMA, std::vector<double>{0.}, std::vector<double>{3.}, std::vector<double>{1.}, [&](const auto &la) {
            auto lambda_params = *chunk_params;
            lambda_params.set_svr_kernel_param2(la[0]);
            const arma::mat &Z = get_cached_Z(lambda_params, feature_chunks_t[chunk_ix]);
            const double mean_train_labels = arma::mean(arma::vectorise(labels.rows(ixs_train[chunk_ix])));
            auto tmp_params = lambda_params;
            tmp_params.set_input_queue_table_name("TMP_" + tmp_params.get_input_queue_table_name());
            tmp_params.set_input_queue_column_name("TMP_" + tmp_params.get_input_queue_column_name());
            const arma::mat *p_Ztrain = prepare_Z(tmp_params, features.rows(ixs_train[chunk_ix]).t(), train_len);
            const auto mingamma = calc_gamma(*p_Ztrain, train_len, mean_train_labels);
            const auto res = optimizer::firefly(1, C_grid_range_div, C_grid_depth, FFA_ALPHA, FFA_BETAMIN, FFA_GAMMA, std::vector<double>{1.}, std::vector<double>{1e3}, std::vector<double>{1.}, [&](const auto &ga) {
                PROFILE_EXEC_TIME(validate_gammas(mingamma, Z, p_Ztrain, {ga[0]}, lambda_params, chunk_ix), "Validate gammas " << lambda_params);
                return (**predictions[chunk_ix].begin()).score;
            }).operator std::pair<double, std::vector<double>>().first;
            delete p_Ztrain;
            return res;
        });
#else
        /*
        switch (chunk_params->get_kernel_type()) {
            case kernel_type_e::PATH:
                tuned_lambdas = C_tune_lambdas_path;
                break;
            default:
                tuned_lambdas.emplace_back(0);
        }
        */
        double range_min_lambda = 0, range_max_lambda = 1;
        for (size_t grid_level_lambda = 0; grid_level_lambda < C_grid_depth; ++grid_level_lambda) {
            const double range_lambda = range_max_lambda - range_min_lambda;
            const double range_half_lambda = range_lambda * .5;
            std::deque<double> lambdas;
            if (!grid_level_lambda) lambdas = {1e1, 1e2};
            for (double r = range_min_lambda; r < range_max_lambda; r += range_lambda / C_grid_range_div) lambdas.emplace_back(r);
#pragma omp parallel for schedule(static, 1) num_threads(adj_threads(std::min<size_t>(lambdas.size(), TUNE_THREADS)))
            for (const double lambda: lambdas) {
                auto lambda_params = *chunk_params;
                lambda_params.set_chunk_ix(chunk_ix);
                lambda_params.set_svr_kernel_param2(lambda);
                const arma::mat &Z = get_cached_Z(lambda_params, feature_chunks_t[chunk_ix]);
                const double mean_train_labels = arma::mean(arma::vectorise(labels.rows(ixs_train[chunk_ix])));
                auto tmp_params = lambda_params;
                tmp_params.set_input_queue_table_name("TMP_" + tmp_params.get_input_queue_table_name());
                tmp_params.set_input_queue_column_name("TMP_" + tmp_params.get_input_queue_column_name());
                const arma::mat *p_Ztrain = prepare_Z(tmp_params, features.rows(ixs_train[chunk_ix]).t(), train_len);
                const auto mingamma = calc_gamma(*p_Ztrain, train_len, mean_train_labels);
                // PROFILE_EXEC_TIME(validate_gammas(mingamma, Z, p_Ztrain, C_gamma_multis, lambda_params, chunk_ix), "Validate gammas " << lambda_params);
                double range_min_gamma = 1, range_max_gamma = 1e2;
                for (size_t grid_level_gamma = 0; grid_level_gamma < C_grid_depth; ++grid_level_gamma) {
                    std::deque<double> gamma_multis;
                    const double range_gamma = range_max_gamma - range_min_gamma;
                    const double range_half_gamma = range_gamma * .5;
                    gamma_multis.clear();
                    for (double r = range_min_gamma; r < range_max_gamma; r += range_gamma / C_grid_range_div) gamma_multis.emplace_back(r);
                    PROFILE_EXEC_TIME(validate_gammas(mingamma, Z, p_Ztrain, gamma_multis, lambda_params, chunk_ix), "Validate gammas " << lambda_params);
                    const auto best_gamma_multi = (**predictions[chunk_ix].begin()).p_params->get_svr_kernel_param() / mingamma;
                    const double gamma_range_tightening = std::abs(best_gamma_multi - range_half_gamma) / range_half_gamma;
                    range_min_gamma = std::max(0., best_gamma_multi - range_half_gamma * gamma_range_tightening);
                    range_max_gamma = range_min_gamma + 2. * range_half_gamma * gamma_range_tightening;
                }

                LOG4_DEBUG("Crass gamma pass score " << predictions[chunk_ix].begin()->get()->score << ", mingamma " << mingamma << ", best parameters "
                                                     << *predictions[chunk_ix].begin()->get()->p_params);
                delete p_Ztrain;
            }
            const auto best_lambda = (**predictions[chunk_ix].begin()).p_params->get_svr_kernel_param2();
            const double lambda_range_tightening = std::abs(best_lambda - range_half_lambda) / range_half_lambda;
            range_min_lambda = std::max(0., best_lambda - range_half_lambda * lambda_range_tightening);
            range_max_lambda = range_min_lambda + 2. * range_half_lambda * lambda_range_tightening;
        }
#endif
        if (predictions[chunk_ix].size() > common::C_tune_keep_preds)
            predictions[chunk_ix].erase(std::next(predictions[chunk_ix].begin(), common::C_tune_keep_preds), predictions[chunk_ix].end());

        for (const auto &p: predictions[chunk_ix])
            LOG4_INFO("Final best score " << p->score << ", final parameters " << *p->p_params);
    }

    --running_tuners;
    if (running_tuners == 0) clear_gradient_caches(*p_head_parameters);
}

#elif defined(TUNE_IDEAL)

void
OnlineMIMOSVR::tune_kernel_params(datamodel::SVRParameters_ptr &p_model_parameters, const arma::mat &features, const arma::mat &labels)
{
    auto [ref_kmatrices, norm_ref_kmatrices] = get_reference_matrices(labels, *p_model_parameters);
    const auto indexes = get_indexes(labels.n_rows, *p_model_parameters);
    const auto num_chunks = indexes.size();
    std::deque<arma::mat> feature_chunks_t(num_chunks);
    __tbb_pfor_i(num_chunks - MAX_TUNE_CHUNKS, num_chunks, feature_chunks_t[i] = features.rows(indexes[i]).t())
    const size_t train_len = feature_chunks_t.back().n_cols;
    const double meanabs_labels = common::meanabs(labels);
    const auto f_ideal = [&](std::deque<double> &params) -> double
    {
        SVRParameters svr_parameters = *p_model_parameters;
        svr_parameters.set_svr_kernel_param2(common::C_tune_lambdas_path[std::round(params[1])]);
        svr_parameters.set_svr_kernel_param(params[0] * get_cached_gamma(*p_model_parameters, get_cached_Z(svr_parameters, last_features_chunk, train_len), train_len, meanabs_labels));
        double err = 0;
        for (size_t i = num_chunks - MAX_TUNE_CHUNKS; i < num_chunks; ++i) {
            /* TODO Test CUDA port seems buggy !
             * double err1;
               PROFILE_EXEC_TIME(
                    err1 = svr::solvers::score_kernel(ref_kmatrices[i].mem, norm_ref_kmatrices[i], Z.mem, k_stride, svr_parameters.get_svr_kernel_param()),
                              "Calc GPU kernel matrix error " << err << " chunk " << i << " level " << svr_parameters.get_decon_level());
              */
            double err2;
            PROFILE_EXEC_TIME(
                    const arma::mat res_kernel_matrix = 1. - get_cached_Z(svr_parameters, feature_chunks_t[i], feature_chunks_t[i].n_cols) / (2. * svr_parameters.get_svr_kernel_param() * svr_parameters.get_svr_kernel_param());
                    err2 = 2. - common::sum<double>(res_kernel_matrix % ref_kmatrices[i])/(arma::norm(res_kernel_matrix, "fro") * norm_ref_kmatrices[i]),
                    "Calc kernel matrix error " << err2 << ", chunk " << i << ", level " << svr_parameters.get_decon_level());
            //if (err1 != err2) LOG4_WARN("err1 " << err1 << " != err2 " << err2 << " diff " << err1 - err2);
            err += err2;
        }
        LOG4_DEBUG("Score " << err << ", gamma " << svr_parameters.get_svr_kernel_param() << ", lambda " << svr_parameters.get_svr_kernel_param2() << ", level " << svr_parameters.get_decon_level() << ", param values " << common::to_string(params));
        return err;
    };
    auto [best_err, opt_args] = (std::tuple<double, std::deque<double>>) svr::optimizer::firefly(
            2, 72, 45, FFA_ALPHA, FFA_BETAMIN, FFA_GAMMA, {TUNE_GAMMA_MIN, 0}, {p_model_parameters->get_decon_level() ? TUNE_GAMMA_MAX : TUNE_GAMMA_MAX, double(common::C_tune_lambdas_path.size())}, {8., 1.}, f_ideal);
    if (opt_args.size() >= 2) p_model_parameters->set_svr_kernel_param2(common::C_tune_lambdas_path[std::round(opt_args[1])]);
    p_model_parameters->set_svr_kernel_param(opt_args[0] * get_cached_gamma(*p_model_parameters, get_cached_Z(*p_model_parameters, last_features_chunk, train_len), train_len, meanabs_labels)); 

    LOG4_INFO("Final parameters " << p_model_parameters->to_string());
}

#endif

}
