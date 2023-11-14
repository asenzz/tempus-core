//
// Created by zarko on 9/29/22.
//


#include <algorithm>
#include <armadillo>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <map>
#include <math.h>
#include <limits>
#include <cmath>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <sys/types.h>
#include <utility>
#include <tuple>
#include <vector>
#include "common/defines.h"
#include "common/gpu_handler.hpp"
#include "common/gpu_handler.hpp"
#include "model/SVRParameters.hpp"
#include "onlinesvr.hpp"
#include "common/Logging.hpp"
#include "common/semaphore.hpp"
#include "common/constants.hpp"
#include "optimizer.hpp"
#include "firefly.hpp"
#include "calc_kernel_quantiles.hpp"
#include "calc_kernel_inversions.hpp"
#include "cuda_path.hpp"
#include "kernel_path.hpp"
#include "appcontext.hpp"
#include "cuqrsolve.hpp"
#include "util/math_utils.hpp"

#define TUNE_HYBRID_MAIN_THREADS 1 // Increase when code properly parallelized

namespace svr {

// Utility function used in tests, does predict, unscale and then validate
std::tuple<double, double, std::vector<double>, std::vector<double>, double, std::vector<double>>
OnlineMIMOSVR::future_validate(
        const size_t start_ix,
        svr::OnlineMIMOSVR &online_svr,
        const arma::mat &features,
        const arma::mat &labels,
        const arma::mat &last_knowns,
        const std::vector<bpt::ptime> &times,
        const bool single_pred,
        const double scale_label,
        const double dc_offset)
{
    if (labels.n_rows <= start_ix) {
        LOG4_WARN("Calling future validate at the end of labels array. MAE is 1000");
        return {1000, 1000, {}, {}, 0, {}};
    }

    const size_t fini_ix = std::min<size_t>(labels.n_rows - 1, single_pred ? start_ix : labels.n_rows - 1);
    LOG4_DEBUG("Validation start index " << start_ix << " and end index is " << fini_ix);
    const size_t num_preds = 1 + fini_ix - start_ix;
    arma::mat predicted_values;
#ifdef NO_ONLINE_LEARN
    PROFILE_EXEC_TIME(predicted_values = online_svr.chunk_predict(features.rows(start_ix, fini_ix), times[start_ix]),
                      "Chunk predict of " << num_preds << " rows, " << features.n_cols << " feature columns, " << PROPS.get_multistep_len() << " labels per row, level " << online_svr.get_svr_parameters().get_decon_level());
    if (predicted_values.n_rows != num_preds) LOG4_ERROR("predicted_values.n_rows " << predicted_values.n_rows << " != num_preds " << num_preds);
#endif
    std::vector<double> ret_predicted_values(num_preds), actual_values(num_preds), lin_pred_values(num_preds);
    double mae = 0, mae_lk = 0;
//#pragma omp parallel for reduction(+: mae, mae_lk) default(shared)
    for (size_t i_future = start_ix; i_future <= fini_ix; ++i_future) {
        const auto ix = i_future - start_ix;
#ifndef NO_ONLINE_LEARN
        PROFILE_EXEC_TIME(predicted_values = online_svr.chunk_predict(features.row(i_future), times[i_future]),
                          "Chunk predict " << ix << " of " << num_preds << " rows, " << features.n_cols << " feature columns, " << PROPS.get_multistep_len() << " labels per row, level " << online_svr.get_svr_parameters().get_decon_level());
        online_svr.learn(features.row(i_future), labels.row(i_future), false, true, std::numeric_limits<size_t>::max(), times[i_future]);
#endif
        lin_pred_values[ix] = last_knowns.at(i_future, 0) * scale_label;
#ifdef EMO_DIFF
        const auto predicted_val = predicted_values(ix, 0) + lin_pred_values[ix];
        const auto actual_val = labels.at(i_future, 0) + lin_pred_values[ix];
#else
        double predicted_val = predicted_values(ix, 0) * scale_label;
        double actual_val = labels.at(i_future, 0) * scale_label;
        if (online_svr.get_svr_parameters().get_decon_level() == 0) {
            predicted_val += dc_offset;
            actual_val += dc_offset;
            lin_pred_values[ix] += dc_offset;
        }
#endif
        ret_predicted_values[ix] = predicted_val;
        actual_values[ix] = actual_val;
        LOG4_DEBUG("Predicted " << predicted_val << ", actual " << actual_val << ", last known " << lin_pred_values[ix] << ", row " << ix << ", col " << 0 << ", level " << online_svr.get_svr_parameters().get_decon_level());
        mae += std::abs(actual_val - predicted_val);
#ifdef EMO_DIFF
        mae_lk += std::abs(actual_val - lin_pred_values[ix]);
#else
        mae_lk += std::abs(actual_val - lin_pred_values[ix]);
#endif
    }
    mae /= double(num_preds);
    mae_lk /= double(num_preds);
#ifdef EMO_DIFF
    const double avg_label = common::meanabs<double>(p_last_knowns + labels);
#else
    const double avg_label = common::meanabs(labels);
#endif
    const double mape = 100. * mae / avg_label;
    const double lin_mape = 100. * mae_lk / avg_label;
    const double alpha_pct = 100. * (lin_mape / mape - 1.);
    LOG4_DEBUG(
            "Future predict from row " << start_ix << " until " << fini_ix << ", MAE " << mae << ", MAPE " << mape << ", Lin MAE " << mae_lk << ", Lin MAPE " << lin_mape
            << ", alpha " << alpha_pct << " pct., parameters " << online_svr.get_svr_parameters().to_string());
#if 0
    LOG4_DEBUG("Actual values " << common::deep_to_string(actual_values) << ", level " << online_svr.get_svr_parameters().get_decon_level());
    LOG4_DEBUG("Predicted values " << common::deep_to_string(ret_predicted_values) << ", level " << online_svr.get_svr_parameters().get_decon_level());
    LOG4_DEBUG("Lin pred values " << common::deep_to_string(lin_pred_values) << ", level " << online_svr.get_svr_parameters().get_decon_level());
#endif
    return {mae, mape, ret_predicted_values, actual_values, lin_mape, lin_pred_values};
}


/* original features before matrix transposition
  = {   label 0 = { Level 0 lag 0, Level 1 lag 1, Level 1 lag 2, ... Level 1 lag 719, Level 2 lag 0, Level 2 lag 1, Level 2 lag 2, ... , Level 2 lag 719, ... Level 31 lag 0, Level 31 lag 1 ... Level 31 lag 31 } ,
        label 1 = { Level 0 lag 0, Level 1 lag 1, Level 1 lag 2, ... Level 1 lag 719, Level 2 lag 0, Level 2 lag 1, Level 2 lag 2, ... , Level 2 lag 719, ... Level 31 lag 0, Level 31 lag 1 ... Level 31 lag 31 } ..
        ...
        label 6000 = { Level 0 lag 0, Level 1 lag 1, Level 1 lag 2, ... Level 1 lag 719, Level 2 lag 0, Level 2 lag 1, Level 2 lag 2, ... , Level 2 lag 719, ... Level 31 lag 0, Level 31 lag 1 ... Level 31 lag 31 }
     }
*/
std::vector<arma::mat> OnlineMIMOSVR::prepare_cumulatives(const SVRParameters &params, const arma::mat &features /* transposed */)
{
    const auto lag = params.get_lag_count();
    const auto levels = features.n_rows / lag;
    LOG4_TRACE("Preparing " << levels << " cumulatives with " << lag << " lag, parameters " << params.to_string() << ", from features " << arma::size(features));
    std::vector<arma::mat> cums_X(levels);
#pragma omp parallel for num_threads(levels)
    for (size_t i = 0; i < levels; ++i)
        cums_X[i] = arma::cumsum(features.rows(i * lag, (i + 1) * lag - 1));
    return cums_X;
}

std::vector<arma::mat> OnlineMIMOSVR::get_cached_cumulatives(const SVRParameters &params, const arma::mat &features /* transposed */, const bpt::ptime &pred_time)
{
    typedef std::tuple<std::string /* queue name */, std::string /* column */, size_t /* lag */, size_t /* decrement */, bpt::ptime /* pred time */ > cml_cache_key_t;
    typedef std::map<cml_cache_key_t, std::vector<arma::mat>> cml_cache_t;
    static cml_cache_t cml_cache;
    static std::mutex cml_mx;
    std::vector<arma::mat> res;

    const cml_cache_key_t params_key {
            params.get_input_queue_table_name(),
            params.get_input_queue_column_name(),
            params.get_lag_count(),
            features.n_cols,
            pred_time};

    std::scoped_lock l(cml_mx);
    auto it_cml = cml_cache.find(params_key);
    if (it_cml == cml_cache.end()) {
        std::pair<cml_cache_t::iterator, bool> ins_res;
        PROFILE_EXEC_TIME(ins_res = cml_cache.emplace(params_key, prepare_cumulatives(params, features)), "Prepare cumulatives");
        if (ins_res.second) it_cml = ins_res.first;
        else LOG4_THROW("Error inserting Z matrix in cache");
    }
    return it_cml->second;
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
        cu_distances_xx(ctx.id(), features_len, levels, short_size_X, 0, 0, short_size_X, short_size_X, cum_XX.memptr(), params.get_svr_kernel_param2(), DEFAULT_TAU_TUNE, w_sum_sym,
                       this_first_Z.memptr(), d_Z), "CUDA kernel XX " << params.to_string());
*/


double calc_gamma(const arma::mat &Z, const size_t train_len, const double meanabs_labels)
{
    return 2. * meanabs_labels * (.5 * std::sqrt(2. * common::meanabs(Z)) - 1. / double(train_len));
    // return 2. * meanabs_labels * (.5 * std::sqrt(2. * arma::mean(arma::abs(arma::vectorise(Z.submat(Z.n_rows - CHUNK_DECREMENT, Z.n_cols - CHUNK_DECREMENT, Z.n_rows - 1, Z.n_cols - 1))))) - 1. / double(train_len));
}


double get_cached_gamma(const SVRParameters &params, const arma::mat &Z, const size_t train_len, const double meanabs_labels)
{
    typedef std::tuple<size_t /* train len */, size_t /* level */, size_t /* dataset id */, std::string /* queue name */, std::string /* column */, size_t /* lambda */, size_t /* lag */, size_t /* decrement */, size_t /* n samples */> gamma_cache_key_t;
    typedef std::map<gamma_cache_key_t, double> gamma_cache_t;
    static gamma_cache_t gamma_cache;
    static std::mutex gamma_mx;

    const gamma_cache_key_t params_key {
            train_len,
            params.get_decon_level(),
            params.get_dataset_id(),
            params.get_input_queue_table_name(),
            params.get_input_queue_column_name(),
            std::round(1e5 * params.get_svr_kernel_param2()),
            params.get_lag_count(),
            params.get_svr_decremental_distance(),
            Z.n_cols};

    const std::scoped_lock sl(gamma_mx);
    auto it_gamma = gamma_cache.find(params_key);
    if (it_gamma == gamma_cache.end()) {
        std::pair<gamma_cache_t::iterator, bool> ins_res;
        PROFILE_EXEC_TIME(ins_res = gamma_cache.emplace(params_key, calc_gamma(Z, train_len, meanabs_labels)), "Calc gamma min");
        if (ins_res.second) {
            it_gamma = ins_res.first;
            LOG4_DEBUG("Returning min gamma " << ins_res.first->second << " for " << params.to_string());
        } else
            LOG4_THROW("Error inserting gamma min in cache");
    }
    return it_gamma->second;
}


arma::mat *OnlineMIMOSVR::prepare_K(const SVRParameters &params, const arma::mat &features_t, const double meanabs_labels, const size_t train_len, const arma::mat &labels)
{
    if (train_len > features_t.n_cols) LOG4_THROW("Invalid train len " << train_len << " for features of size " << arma::size(features_t));
    const arma::mat *p_Z = prepare_Z(params, features_t, labels);
    const double opt_gamma = calc_gamma(*p_Z, train_len, meanabs_labels);
    LOG4_DEBUG("Optimal gamma for lambda " << params.get_svr_kernel_param2() << " is " << opt_gamma << ", level " << params.get_decon_level());

    arma::mat *p_K = new arma::mat(features_t.n_cols, features_t.n_cols, arma::fill::zeros);
    *p_K = 1. - *p_Z / (2. * opt_gamma * opt_gamma);
    delete p_Z;
    LOG4_DEBUG("Returning K of size " << arma::size(*p_K));
    return p_K;
}

arma::mat *OnlineMIMOSVR::prepare_Z(const SVRParameters &params, const arma::mat &features_t, const arma::mat &labels, size_t len)
{
    if (!len) len = features_t.n_cols;
    arma::mat *p_Z = nullptr;
    switch (params.get_kernel_type()) {
        case kernel_type_e::PATH: {
            const auto cums_X = get_cached_cumulatives(params, features_t);
            if (cums_X.empty()) LOG4_THROW("Failed preparing cumulatives matrices.");
            p_Z = new arma::mat(len, len, arma::fill::zeros);
#pragma omp parallel for num_threads(cums_X.size())
            for (size_t i = 0; i < cums_X.size(); ++i) {
                arma::mat z = arma::mat(arma::size(*p_Z));
                kernel::path::cu_distances_xx(params.get_lag_count(), 1, len, 0, 0, len, len, cums_X[i].memptr(), params.get_svr_kernel_param2(), DEFAULT_TAU_TUNE, 1, z.memptr());
#pragma omp critical(prepare_Z)
                *p_Z += z;
            }
            LOG4_DEBUG("Returning Z " << arma::size(*p_Z) << " for " << params.to_string() << ", cumulative matrices " << cums_X.size() << ", of dimensions " << arma::size(cums_X.front()) << ", features " << arma::size(features_t));
            break;
        }
        case kernel_type_e::DEEP_PATH:
        case kernel_type_e::DEEP_PATH2:
            p_Z = new arma::mat(get_reference_distance_matrix(labels));
            LOG4_DEBUG("Returning Z " << arma::size(*p_Z) << " for " << params.to_string() <<  ", labels " << arma::size(labels) << ", features " << arma::size(features_t));
            break;
        default:
            LOG4_THROW("Kernel type " << int(params.get_kernel_type()) << " not handled!");
    }
    return p_Z;
}

arma::mat &OnlineMIMOSVR::get_cached_K(const SVRParameters &params, const arma::mat &features_t, const arma::mat &labels, const double meanabs_labels, const size_t train_len)
{
    typedef std::tuple<size_t /* dataset id */, std::string /* queue name */, std::string /* column */, size_t /* lambda */, size_t /* lag */, size_t /* decrement */, size_t /* n samples */, size_t /* level */> K_cache_key_t;
    typedef std::map<K_cache_key_t, arma::mat *> K_cache_t;
    static K_cache_t K_cache;
    static std::mutex K_mx;

    const K_cache_key_t params_key {
            params.get_dataset_id(),
            params.get_input_queue_table_name(),
            params.get_input_queue_column_name(),
            std::round(1e5 * params.get_svr_kernel_param2()),
            params.get_lag_count(),
            params.get_svr_decremental_distance(),
            features_t.n_cols,
            params.get_decon_level()};

    std::scoped_lock sl(K_mx);
    auto it_K = K_cache.find(params_key);
    if (it_K == K_cache.end()) {
        std::pair<K_cache_t::iterator, bool> ins_res;
        PROFILE_EXEC_TIME(ins_res = K_cache.emplace(params_key, prepare_K(params, features_t, meanabs_labels, train_len, labels)), "Prepare K matrix");
        if (ins_res.second) it_K = ins_res.first;
        else
            LOG4_THROW("Error inserting K matrix in cache");
    }
    return *it_K->second;
}

arma::mat &OnlineMIMOSVR::get_cached_Z(const SVRParameters &params, const arma::mat &features_t, const arma::mat &labels, const size_t short_size_X)
{
    typedef std::tuple<size_t /* dataset id */, std::string /* queue name */, std::string /* column */, size_t /* lambda */, size_t /* lag */, size_t /* decrement */, size_t /* n samples */ > Z_cache_key_t;
    typedef std::map<Z_cache_key_t, arma::mat *> Z_cache_t;
    static Z_cache_t Z_cache;
    static std::mutex Z_mx;

    const Z_cache_key_t params_key {
            params.get_dataset_id(),
            params.get_input_queue_table_name(),
            params.get_input_queue_column_name(),
            std::round(1e5 * params.get_svr_kernel_param2()),
            params.get_lag_count(),
            params.get_svr_decremental_distance(),
            features_t.n_cols};

    const std::scoped_lock l(Z_mx);//, std::defer_lock);
    // if (Z_cache.size() < common::C_tune_lambdas_count) ul.lock();
    auto it_Z = Z_cache.find(params_key);
    if (it_Z == Z_cache.end()) {
        std::pair<Z_cache_t::iterator, bool> ins_res;
        PROFILE_EXEC_TIME(ins_res = Z_cache.emplace(params_key, prepare_Z(params, features_t, labels, short_size_X)), "Prepare Z matrix ");
        if (ins_res.second) it_Z = ins_res.first;
        else
            LOG4_THROW("Error inserting Z matrix in cache");
    }
    //if (ul.owns_lock()) ul.unlock();
    return *it_Z->second;
}


arma::mat *OnlineMIMOSVR::prepare_Zy(const SVRParameters &params, const arma::mat &features/* transposed*/, const arma::mat &predict_features /* transposed*/, const bpt::ptime &pred_time)
{
#ifdef NO_ONLINE_LEARN
    const auto cums_X = get_cached_cumulatives(params, features, pred_time);
#else
    const auto cums_X = get_cached_cumulatives(params, features, pred_time);
#endif
    if (cums_X.empty()) LOG4_THROW("Failed preparing feature cumulative matrices.");
    const auto cums_Y = prepare_cumulatives(params, predict_features);
    if (cums_Y.empty()) LOG4_THROW("Failed preparing predict cumulative matrices.");

    arma::mat *p_Zy = new arma::mat(predict_features.n_cols, features.n_cols, arma::fill::zeros);
#pragma omp parallel for num_threads(cums_X.size())
    for (size_t i = 0; i < cums_X.size(); ++i) {
        arma::mat z = arma::mat(arma::size(*p_Zy));
        kernel::path::cu_distances_xy(
                params.get_lag_count(), 1, features.n_cols, predict_features.n_cols, 0, 0, features.n_cols, predict_features.n_cols,
                cums_X[i].memptr(), cums_Y[i].memptr(), params.get_svr_kernel_param2(), DEFAULT_TAU_TUNE, 1, z.memptr());
#pragma omp critical(__omp_prepare_Zy)
        *p_Zy += z;
    }
    LOG4_DEBUG("Returning Zy " << arma::size(*p_Zy) << " for " << params.to_string() << ", cumulative matrices " << cums_X.size() << ", of dimensions " << arma::size(cums_X.front()) << " and " << arma::size(cums_Y.front()));
    return p_Zy;
}


// TODO Create an ensemble::cache class and move the above and below methods there
namespace {
    typedef std::tuple<std::string /* queue name */, std::string /* column */, size_t /* lambda */, size_t /* lag */, size_t /* decrement */, size_t /* p_predictions count */, bpt::ptime /* prediction time */, size_t /* level */> Zy_cache_key_t;
    typedef std::map<Zy_cache_key_t, arma::mat *> Zy_cache_t;
    Zy_cache_t Zy_cache;
}

void OnlineMIMOSVR::clear_Zy_cache()
{
    Zy_cache.clear();
}

arma::mat &OnlineMIMOSVR::get_cached_Zy(const SVRParameters &params, const arma::mat &features /* transposed */, const arma::mat &predict_features /* transposed */, const bpt::ptime &pred_time)
{
    static std::mutex Zy_mx;

    const Zy_cache_key_t params_key {
            params.get_input_queue_table_name(),
            params.get_input_queue_column_name(),
            std::round(1e5 * params.get_svr_kernel_param2()),
            params.get_lag_count(),
            features.n_cols,
            predict_features.n_cols,
            pred_time,
            params.get_decon_level()};

    arma::mat res;
    std::scoped_lock l(Zy_mx);
    auto it_Zy = Zy_cache.find(params_key);
    if (it_Zy == Zy_cache.end()) {
        std::pair<Zy_cache_t::iterator, bool> ins_res;
        PROFILE_EXEC_TIME(ins_res = Zy_cache.emplace(params_key, prepare_Zy(params, features, predict_features, pred_time)), "Prepare Zy matrix");
        if (ins_res.second)
            it_Zy = ins_res.first;
        else
            LOG4_THROW("Error inserting Zy matrix in cache");
    }
    return *it_Zy->second;
}




#if !defined(HYBRID_QUANTILES) && defined(TUNE_HYBRID)

double
OnlineMIMOSVR::produce_kernel_inverse_order(
        const SVRParameters &svr_parameters,
        const arma::mat &x_train,
        const arma::mat &y_train)
{
    const auto p_manifold = init_manifold(x_train, y_train, svr_parameters);
    const auto indexes = get_indexes(x_train.n_rows, svr_parameters);
    const auto num_chunks = std::min<size_t>(indexes.size(), svr::common::MAX_TUNE_CHUNKS);
    const auto overall_score = tbb::parallel_reduce(
        tbb::blocked_range<std::size_t>(0, num_chunks), double(0), [&] (const tbb::blocked_range<size_t> &r, double err) {
            for (std::size_t i = r.begin(); i < r.end(); ++i) {
                arma::mat res_kernel_matrix;
                PROFILE_EXEC_TIME(
                        init_kernel_matrix(svr_parameters, x_train.rows(indexes[i]), y_train.rows(indexes[i]), res_kernel_matrix, p_manifold),
                        "Init kernel matrix " << arma::size(res_kernel_matrix));
                double cur_err = 0;
                PROFILE_EXEC_TIME(
                        cur_err = calc_kernel_inversions(res_kernel_matrix.memptr(), y_train.rows(indexes[i])),
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
#pragma omp parallel for reduce(+:overall_score) default(shared)
    for (size_t i = 0; i < num_chunks; ++i) {
        arma::mat res_kernel_matrix;
        PROFILE_EXEC_TIME(
                init_kernel_matrix(svr_parameters, x_train.rows(indexes[i]), y_train.rows(indexes[i]), res_kernel_matrix, p_manifold),
                "init_kernel_matrix");
        overall_score += calc_kernel_quantiles(res_kernel_matrix, y_train.rows(indexes[i]));
    }

    LOG4_DEBUG("Quantiles error " << overall_score << ", gamma " << svr_parameters.get_svr_kernel_param() << ", lambda " << svr_parameters.get_svr_kernel_param2());
    return overall_score;
}


double
opt_loss_quantiles_lambda(
        std::vector<double> &params,
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

static std::tuple<std::vector<arma::mat> */* p_predictions */, std::vector<arma::mat> */* labels */, std::vector<arma::mat> */* lastknowns */, double>
calc_direction(const arma::mat &K, const double epsco, const arma::mat &labels, const arma::mat &last_knowns, const size_t train_len, const double meanabs_labels)
{
    const off_t start_point_K = K.n_rows - EMO_TUNE_TEST_SIZE - train_len;
    const off_t start_point_labels = labels.n_rows - EMO_TUNE_TEST_SIZE - train_len;
    if (start_point_K < 0 || start_point_labels < 0 || labels.n_rows != K.n_rows)
        LOG4_THROW("Shorter K " << start_point_K << " or labels " << start_point_labels << " for K " << arma::size(K) << ", labels " << arma::size(labels));
    auto p_out_preds = new std::vector<arma::mat>(EMO_MAX_J);
    auto p_out_labels = new std::vector<arma::mat>(EMO_MAX_J);
    auto p_out_last_knowns = new std::vector<arma::mat>(EMO_MAX_J);
    double score = 0;
#pragma omp parallel for reduction(+:score) num_threads(common::gpu_handler::get_instance().get_max_running_gpu_threads_number() / CTX_PER_GPU)
    for (size_t j = 0; j < EMO_MAX_J; ++j) {
        const size_t x_train_start = j * EMO_SLIDE_SKIP;
        const size_t x_train_final = x_train_start + train_len - 1;
        const size_t x_test_start = x_train_final + 1;
        const size_t x_test_final = x_test_start + EMO_TUNE_TEST_SIZE - j * EMO_SLIDE_SKIP - 1;
        const size_t x_test_len = x_train_final - x_train_start + 1;

        LOG4_TRACE("Try " << j << ", K " << arma::size(K) << ", train start " << x_train_start << ", train final " << x_train_final << ", test start " << x_test_start << ", test final " << x_test_final);
        try {
            double this_score = 0;
            const arma::mat eye_epsco = arma::eye(x_test_len, x_test_len) * epsco;
#ifdef TAIL_VALIDATION
            p_out_labels->at(j) = labels.rows(labels.n_rows - 1 - start_point_labels - x_test_final, labels.n_rows - 1 - start_point_labels - x_test_start);
            p_out_preds->at(j) = K.submat(K.n_rows - 1 - start_point_K - x_test_final, K.n_rows - 1 - start_point_K - x_train_final, K.n_rows - 1  - start_point_K - x_test_start, K.n_rows - 1 - start_point_K - x_train_start) *
                    OnlineMIMOSVR::call_gpu_dynsolve(eye_epsco + K.submat(K.n_rows - 1 - start_point_K - x_train_final, K.n_rows - 1 - start_point_K - x_train_final,
                                                                          K.n_rows - 1 - start_point_K - x_train_start, K.n_rows - 1 - start_point_K - x_train_start),
                    labels.rows(labels.n_rows - 1 - start_point_labels - x_train_final, labels.n_rows - 1 - start_point_labels - x_train_start));
            this_score += common::meanabs<double>(p_out_preds->at(j) - p_out_labels->at(j)) / meanabs_labels;
#endif
            p_out_labels->at(j) = labels.rows(start_point_labels + x_test_start, start_point_labels + x_test_final);
            p_out_last_knowns->at(j) = last_knowns.rows(start_point_labels + x_test_start, start_point_labels + x_test_final);
            p_out_preds->at(j) = K.submat(start_point_K + x_test_start, start_point_K + x_train_start, start_point_K + x_test_final, start_point_K + x_train_final) * OnlineMIMOSVR::call_gpu_dynsolve(
                    eye_epsco + K.submat(start_point_K + x_train_start, start_point_K + x_train_start, start_point_K + x_train_final, start_point_K + x_train_final),
                    labels.rows(start_point_labels + x_train_start, start_point_labels + x_train_final));
            this_score += common::meanabs<double>(p_out_preds->at(j) - p_out_labels->at(j)) / meanabs_labels;

            score += this_score;
        } catch (const std::exception &ex) {
            LOG4_ERROR("Error solving matrix, try "  << j << ", K " << arma::size(K) << ", " << x_train_start << ", " << x_train_final << ", " << x_test_start << ", " << x_test_final << ", error " << ex.what());
            p_out_preds->at(j) = arma::mat(x_test_len, labels.n_cols, arma::fill::value(BAD_VALIDATION));
            p_out_labels->at(j) = p_out_preds->at(j);
            p_out_last_knowns->at(j) = p_out_preds->at(j);
            score += BAD_VALIDATION;
        }
    }
    return {p_out_preds, p_out_labels, p_out_last_knowns, score};
}


void
OnlineMIMOSVR::tune_kernel_params(
        param_preds_set_t &predictions,
        SVRParameters_ptr &p_model_parameters,
        const arma::mat &features,
        const arma::mat &labels,
        const arma::mat &last_knowns,
        size_t chunk_ix)
{
    // TODO Fix tuning for multiple chunks, indexes are not properly validated in loss function
    LOG4_DEBUG("Tuning parameters " << *p_model_parameters << ", labels " << common::present(labels) << ", features " << common::present(features) << ", last-knowns " << common::present(last_knowns) << ", chunk index " << chunk_ix << ", EMO_SLIDE_SKIP " << EMO_SLIDE_SKIP << /* ", MULTIPLE_EPSCO " << MULTIPLE_EPSCO << */ ", EMO_MAX_J " << EMO_MAX_J << ", EMO_TUNE_VALIDATION_WINDOW " << EMO_TUNE_VALIDATION_WINDOW << ", TUNE_EPSCOST_MAX " << common::C_tune_crass_epscost.front() << ", TUNE_EPSCOST_MIN " << common::C_tune_crass_epscost.back());
    const std::string original_input_queue_column_name = p_model_parameters->get_input_queue_column_name();
    const std::string original_input_queue_table_name = p_model_parameters->get_input_queue_table_name();
    p_model_parameters->set_input_queue_column_name("TUNE_COLUMN_" + p_model_parameters->get_input_queue_column_name());
    p_model_parameters->set_input_queue_table_name("TUNE_TABLE_" + p_model_parameters->get_input_queue_table_name());
    const double meanabs_labels = common::meanabs(labels); // .rows(labels.n_rows - p_model_parameters->get_svr_decremental_distance(), labels.n_rows - 1));
    // const auto meanabs_labels = common::meanabs<double>(labels.rows(labels.n_rows - EMO_TUNE_TEST_SIZE, labels.n_rows - 1));
    // const double medianabs_labels = common::medianabs(labels); // .rows(labels.n_rows - EMO_TUNE_TEST_SIZE, labels.n_rows - 1)
    const auto indexes = get_indexes(labels.n_rows, *p_model_parameters);
    if (chunk_ix == std::numeric_limits<size_t>::max()) chunk_ix = indexes.size() - 1;
    const size_t train_len = indexes[chunk_ix].n_elem;
    std::vector<arma::mat> feature_chunks_t, label_chunks, lastknown_chunks;
    for (const auto &chunk_ixs: indexes) {
        feature_chunks_t.emplace_back(features.rows(chunk_ixs).t());
        label_chunks.emplace_back(labels.rows(chunk_ixs));
        lastknown_chunks.emplace_back(last_knowns.rows(chunk_ixs));
        feature_chunks_t.back() = arma::join_rows(feature_chunks_t.back(), features.rows(features.n_rows - EMO_TUNE_TEST_SIZE, features.n_rows - 1).t());
        label_chunks.back() = arma::join_cols(label_chunks.back(), labels.rows(labels.n_rows - EMO_TUNE_TEST_SIZE, labels.n_rows - 1));
        lastknown_chunks.back() = arma::join_cols(lastknown_chunks.back(), last_knowns.rows(last_knowns.n_rows - EMO_TUNE_TEST_SIZE, last_knowns.n_rows - 1));
    }

    double best_score = std::numeric_limits<double>::max();
    const arma::mat k_eye = arma::eye(train_len, train_len);
    SVRParameters best_params = *p_model_parameters;
    std::mutex add_predictions_mx;
    double gamma_mult_fine_start, gamma_mult_fine_end, epsco_fine_start, epsco_fine_end;
    const auto validate_gammas = [&](const double min_gamma, const arma::mat &Z, const std::deque<double> &gamma_multis, const SVRParameters &score_params, const std::deque<double> &epscos) -> double {
        LOG4_DEBUG("Validating " << gamma_multis.size() << " gamma multipliers, starting from " << gamma_multis.front() << " to " << gamma_multis.back() << ", min gamma " <<
                    min_gamma << ", Z " << arma::size(Z) << ", using " << epscos.size() << " epscos starting " << epscos.front() << ", last epsco " << epscos.back() << ", level " << p_model_parameters->get_decon_level());
        double lambda_score = 0;
#pragma omp parallel for num_threads(TUNE_HYBRID_MAIN_THREADS)
        for (const double gamma_mult: gamma_multis) {
            auto gamma_params = score_params;
            gamma_params.set_svr_kernel_param(gamma_mult * min_gamma);
            const arma::mat K = 1. - Z / (2. * std::pow(gamma_params.get_svr_kernel_param(), 2));
#pragma omp parallel for num_threads(TUNE_HYBRID_MAIN_THREADS)
            for (const auto epsco: epscos) {
                auto p_cost_params = std::make_shared<SVRParameters>(gamma_params);
                p_cost_params->set_svr_C(1. / (2. * epsco));
                std::vector<arma::mat> *p_out_preds, *p_out_labels, *p_out_last_knowns;
                double score;
                std::tie(p_out_preds, p_out_labels, p_out_last_knowns, score) = calc_direction(K, epsco, label_chunks[chunk_ix], lastknown_chunks[chunk_ix], train_len, meanabs_labels);
                const std::scoped_lock sl(add_predictions_mx);
                if (score < best_score) {
                    best_score = score;
                    gamma_mult_fine_start = gamma_mult / C_fine_gamma_div;
                    gamma_mult_fine_end = gamma_mult * C_fine_gamma_mult;
                    epsco_fine_start = epsco;
                    epsco_fine_end = epsco;
                    best_params = *p_cost_params;
                    LOG4_DEBUG("Lambda, gamma tune best score " << best_score << ", for " << best_params.to_string());
               }
               lambda_score += score;
               p_cost_params->set_input_queue_table_name(original_input_queue_table_name);
               p_cost_params->set_input_queue_column_name(original_input_queue_column_name);
               const uint64_t epsco_key = 1e6 * epsco;
               predictions[epsco_key].emplace(std::make_shared<t_param_preds>(score, p_cost_params, p_out_preds, p_out_labels, p_out_last_knowns));
               while (predictions[epsco_key].size() > size_t(common::C_tune_keep_preds))
                   predictions[epsco_key].erase(std::prev(predictions[epsco_key].end()));
            }
        }
        return lambda_score;
    };

    std::deque<double> tuned_lambdas;
    switch (p_model_parameters->get_kernel_type()) {
        case kernel_type_e::PATH:
            tuned_lambdas = common::C_tune_lambdas_path;
            break;
        default:
            tuned_lambdas.emplace_back(0);
    }

#pragma omp parallel for num_threads(TUNE_HYBRID_MAIN_THREADS)
    for (const double lambda: tuned_lambdas) {
        auto lambda_params = *p_model_parameters;
        lambda_params.set_svr_kernel_param2(lambda);
        const arma::mat &Z = get_cached_Z(lambda_params, feature_chunks_t[chunk_ix], label_chunks[chunk_ix]);
        const double min_gamma = get_cached_gamma(lambda_params, Z, train_len, meanabs_labels);
        PROFILE_EXEC_TIME(validate_gammas(min_gamma, Z, C_gamma_multis, lambda_params, common::C_tune_crass_epscost), "Validate gammas " << lambda_params.to_string());
    }
#ifdef FINER_GAMMA_TUNE
    LOG4_DEBUG("First pass score " << best_score << ", best parameters " << best_params.to_string());
    *p_model_parameters = best_params;
    const arma::mat &Z = get_cached_Z(best_params, feature_chunks_t[chunk_ix], label_chunks[chunk_ix]);

#if 0
    std::deque<double> epsco_fine_grid;
    for (double epsco = epsco_fine_start; epsco <= epsco_fine_end; epsco *= MULTIPLE_EPSCO_FINE)
        epsco_fine_grid.emplace_back(epsco);
#endif

    const double min_gamma = get_cached_gamma(best_params, Z, train_len, meanabs_labels);
    std::deque<double> gamma_fine_multis;
    for (double gamma_mult = gamma_mult_fine_start; gamma_mult < gamma_mult_fine_end; gamma_mult *= FINE_GAMMA_MULTIPLE)
        gamma_fine_multis.emplace_back(gamma_mult);
    validate_gammas(min_gamma, Z, gamma_fine_multis, *p_model_parameters, common::C_tune_crass_epscost);
    gamma_fine_multis.clear();
    *p_model_parameters = best_params;

#endif
#if 0
    // or remove every nth prediction
    size_t ctr = 0;
    const size_t keep_every = predictions.size() / C_tune_keep_preds;
    for (auto prediter = predictions.begin(); prediter != predictions.end(); ++prediter, ++ctr)
        if (ctr++ % keep_every) predictions.erase(prediter--);
#endif
    for (auto &cost_preds: predictions) {
        if (cost_preds.second.size() > common::C_tune_keep_preds)
            for_each(std::next(cost_preds.second.begin(), common::C_tune_keep_preds), cost_preds.second.end(), [&](auto &tune_prediction) { tune_prediction->clear(); });
        cost_preds.second.erase(std::next(cost_preds.second.begin(), common::C_tune_keep_preds), cost_preds.second.end());

        size_t ctr = 0;
        for (const auto &preds_cost: cost_preds.second) {
            LOG4_INFO("Best " << ctr << " score " << preds_cost->score << ", final parameters " << preds_cost->p_params->to_string() << ", returning best " << predictions.begin()->second.size() << " parameter sets.");
            ++ctr;
        }
    }
}

#elif defined(TUNE_IDEAL)

void
OnlineMIMOSVR::tune_kernel_params(SVRParameters_ptr &p_model_parameters, const arma::mat &features, const arma::mat &labels)
{
    std::vector<arma::mat> ref_kmatrices;
    std::vector<double> norm_ref_kmatrices;
    std::tie(ref_kmatrices, norm_ref_kmatrices) = get_reference_matrices(labels, *p_model_parameters);
    const auto indexes = get_indexes(labels.n_rows, *p_model_parameters);
    const auto num_chunks = indexes.size();
    std::vector<arma::mat> feature_chunks_t(num_chunks);
    __tbb_pfor_i(num_chunks - MAX_TUNE_CHUNKS, num_chunks, feature_chunks_t[i] = features.rows(indexes[i]).t())
    const size_t train_len = feature_chunks_t.back().n_cols;
    const double meanabs_labels = common::meanabs(labels);
    auto f_ideal = [&](std::vector<double> &params) -> double
    {
        SVRParameters svr_parameters = *p_model_parameters;
        svr_parameters.set_svr_kernel_param2(common::C_tune_lambdas_path[std::round(params[1])]);
        svr_parameters.set_svr_kernel_param(params[0] * get_cached_gamma(*p_model_parameters, get_cached_Z(svr_parameters, last_features_chunk, train_len), train_len, meanabs_labels));
        double err = 0;
        for (size_t i = num_chunks - MAX_TUNE_CHUNKS; i < num_chunks; ++i) {
            /* TODO Test CUDA port seems buggy !
             * double err1;
               PROFILE_EXEC_TIME(
                    err1 = svr::solvers::score_kernel(ref_kmatrices[i].memptr(), norm_ref_kmatrices[i], Z.memptr(), k_stride, svr_parameters.get_svr_kernel_param()),
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
        LOG4_DEBUG("Score " << err << ", gamma " << svr_parameters.get_svr_kernel_param() << ", lambda " << svr_parameters.get_svr_kernel_param2() << ", level " << svr_parameters.get_decon_level() << ", param values " << common::deep_to_string(params));
        return err;
    };
    std::vector<double> opt_args;
    double best_err;
    std::tie(best_err, opt_args) = (std::tuple<double, std::vector<double>>) svr::optimizer::firefly(
            2, 72, 45, FFA_ALPHA, FFA_BETAMIN, FFA_GAMMA, {TUNE_GAMMA_MIN, 0}, {p_model_parameters->get_decon_level() ? TUNE_GAMMA_MAX : TUNE_GAMMA_MAX, double(common::C_tune_lambdas_path.size())}, {8., 1.}, f_ideal);
    if (opt_args.size() >= 2) p_model_parameters->set_svr_kernel_param2(common::C_tune_lambdas_path[std::round(opt_args[1])]);
    p_model_parameters->set_svr_kernel_param(opt_args[0] * get_cached_gamma(*p_model_parameters, get_cached_Z(*p_model_parameters, last_features_chunk, train_len), train_len, meanabs_labels)); 

    LOG4_INFO("Final parameters " << p_model_parameters->to_string());
}

#endif

}
