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
#include <oneapi/tbb/parallel_reduce.h>

#include "common/defines.h"
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

#define TUNE_HYBRID_MAIN_THREADS 4

namespace svr {

// Utility function used in tests, does predict, unscale and then validate
std::tuple<double, double, std::deque<double>, std::deque<double>, double, std::deque<double>>
OnlineMIMOSVR::future_validate(
        const size_t start_ix,
        svr::OnlineMIMOSVR &online_svr,
        const arma::mat &features,
        const arma::mat &labels,
        const arma::mat &last_knowns,
        const std::deque<bpt::ptime> &times,
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
    PROFILE_EXEC_TIME(predicted_values = online_svr.predict(features.rows(start_ix, fini_ix), times[start_ix]),
                      "Chunk predict of " << num_preds << " rows, " << features.n_cols << " feature columns, " << PROPS.get_multistep_len() << " labels per row, level " << online_svr.get_params().get_decon_level());
    if (predicted_values.n_rows != num_preds) LOG4_ERROR("predicted_values.n_rows " << predicted_values.n_rows << " != num_preds " << num_preds);
#endif
    std::deque<double> ret_predicted_values(num_preds), actual_values(num_preds), lin_pred_values(num_preds);
    double mae = 0, mae_lk = 0;
//#pragma omp parallel for reduction(+: mae, mae_lk) default(shared)
    for (size_t i_future = start_ix; i_future <= fini_ix; ++i_future) {
        const auto ix = i_future - start_ix;
#ifndef NO_ONLINE_LEARN
        PROFILE_EXEC_TIME(predicted_values = online_svr.predict(features.row(i_future), times[i_future]),
                          "Chunk predict " << ix << " of " << num_preds << " rows, " << features.n_cols << " feature columns, " << PROPS.get_multistep_len() << " labels per row, level " << online_svr.get_param_set().get_decon_level());
        online_svr.learn(features.row(i_future), labels.row(i_future), false, true, std::numeric_limits<size_t>::max(), times[i_future]);
#endif
        lin_pred_values[ix] = arma::mean(last_knowns.row(i_future)) * scale_label;
#ifdef EMO_DIFF
        const auto predicted_val = predicted_values(ix, 0) + lin_pred_values[ix];
        const auto actual_val = labels.at(i_future, 0) + lin_pred_values[ix];
#else
        double predicted_val = arma::mean(predicted_values.row(ix)) * scale_label;
        double actual_val = arma::mean(labels.row(i_future)) * scale_label;
        if (online_svr.get_params().get_decon_level() == 0) {
            predicted_val += dc_offset;
            actual_val += dc_offset;
            lin_pred_values[ix] += dc_offset;
        }
#endif
        ret_predicted_values[ix] = predicted_val;
        actual_values[ix] = actual_val;
        LOG4_DEBUG("Predicted " << predicted_val << ", actual " << actual_val << ", last known " << lin_pred_values[ix] << ", row " << ix << ", cols " << labels.n_cols << ", level " << online_svr.get_params().get_decon_level());
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
            << ", alpha " << alpha_pct << " pct., parameters " << online_svr.get_params());
#if 0
    LOG4_DEBUG("Actual values " << common::deep_to_string(actual_values) << ", level " << online_svr.get_svr_parameters().get_decon_level());
    LOG4_DEBUG("Predicted values " << common::to_string(ret_predicted_values) << ", level " << online_svr.get_param_set().get_decon_level());
    LOG4_DEBUG("Lin pred values " << common::to_string(lin_pred_values) << ", level " << online_svr.get_param_set().get_decon_level());
#endif
    return {mae, mape, ret_predicted_values, actual_values, lin_mape, lin_pred_values};
}


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
    LOG4_TRACE("Preparing " << levels << " cumulatives with " << lag << " lag, parameters " << params << ", from features_t " << arma::size(features_t));
    const auto p_cums_X = new std::deque<arma::mat>(levels);
#pragma omp parallel for num_threads(levels)
    for (size_t i = 0; i < levels; ++i)
        p_cums_X->at(i) = arma::cumsum(features_t.rows(i * lag, (i + 1) * lag - 1));
    return p_cums_X;
}

std::deque<arma::mat> &OnlineMIMOSVR::get_cached_cumulatives(const SVRParameters &params, const arma::mat &features_t, const bpt::ptime &pred_time)
{
    typedef std::tuple<std::string /* queue name */, std::string /* column */, size_t /* lag */, size_t /* decrement */, bpt::ptime /* pred time */, size_t /* chunk */,
                        size_t /* gradient */ > cml_cache_key_t;
    typedef std::map<cml_cache_key_t, std::deque<arma::mat> *> cml_cache_t;
    static cml_cache_t cml_cache;
    static std::mutex cml_mx;

    const cml_cache_key_t params_key{
            params.get_input_queue_table_name(),
            params.get_input_queue_column_name(),
            params.get_lag_count(),
            features_t.n_cols,
            pred_time,
            params.get_chunk_ix(),
            params.get_grad_level()};

    const auto prepare_f = [&params, &features_t](){ return prepare_cumulatives(params, features_t); };
    return *cached(cml_cache, cml_mx, params_key, prepare_f);
}

template<typename rT, typename kT, typename fT>
rT OnlineMIMOSVR::cached(std::map<kT, rT> &cache_cont, std::mutex &mx, const kT &cache_key, const fT &f)
{
    auto it_cml = cache_cont.find(cache_key);
    if (it_cml != cache_cont.end()) goto __bail;
    {
        const std::scoped_lock l(mx);
        it_cml = cache_cont.find(cache_key);
        if (it_cml == cache_cont.end()) {
            typename std::map<kT, rT>::mapped_type r;
            PROFILE_EXEC_TIME(r = f(), "Prepare cumulatives");
            const auto [ins, rc] = cache_cont.emplace(cache_key, r);
            if (rc) it_cml = ins;
            else
                LOG4_THROW("Error inserting Z matrix in cache");
        }
    }
__bail:
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
        cu_distances_xx(ctx.id(), features_len, levels, short_size_X, 0, 0, short_size_X, short_size_X, cum_XX.mem, params.get_svr_kernel_param2(), DEFAULT_TAU_TUNE, w_sum_sym,
                       this_first_Z.memptr(), d_Z), "CUDA kernel XX " << params);
*/

double calc_gamma(const arma::mat &Z, const double train_len, const double mean_L)
{
    // return 2. * meanabs_labels * (.5 * std::sqrt(2. * common::meanabs<double>(arma::trimatl(Z.submat(Z.n_rows - CHUNK_DECREMENT, Z.n_cols - CHUNK_DECREMENT, Z.n_rows - 1, Z.n_cols - 1)))) - 1. / train_len);
    // return 2. * meanabs_labels * (.5 * std::sqrt(2. * common::medianabs(Z)) - 1. / train_len);
    const auto mean_Z = arma::mean(arma::vectorise(Z.submat(Z.n_rows - CHUNK_DECREMENT, Z.n_cols - CHUNK_DECREMENT, Z.n_rows - 1, Z.n_cols - 1)));
    const auto res = std::sqrt(train_len * mean_Z / (2. * (train_len - mean_L)));
    LOG4_DEBUG("Mean Z " << mean_Z << ", mean L " << mean_L << ", train len " << train_len << ", gamma " << res);
    return res;
}


double OnlineMIMOSVR::get_cached_gamma(const SVRParameters &params, const arma::mat &Z, const size_t train_len, const double meanabs_labels)
{
    typedef std::tuple<
            size_t /* train len */, size_t /* level */, size_t /* chunk */, size_t /* grad */, size_t /* dataset id */, std::string /* queue name */, std::string /* column */,
            size_t /* lambda */, size_t /* lag */, size_t /* decrement */, size_t /* n samples */> gamma_cache_key_t;
    typedef std::map<gamma_cache_key_t, double> gamma_cache_t;
    static gamma_cache_t gamma_cache;
    static std::mutex gamma_mx;

    const gamma_cache_key_t params_key {
            train_len,
            params.get_decon_level(),
            params.get_chunk_ix(),
            params.get_grad_level(),
            params.get_dataset_id(),
            params.get_input_queue_table_name(),
            params.get_input_queue_column_name(),
            std::round(1e5 * params.get_svr_kernel_param2()),
            params.get_lag_count(),
            params.get_svr_decremental_distance(),
            Z.n_cols};

    const auto prepare_f = [&Z, &train_len, &meanabs_labels](){ return calc_gamma(Z, train_len, meanabs_labels); };
    return cached(gamma_cache, gamma_mx, params_key, prepare_f);
}


arma::mat *OnlineMIMOSVR::prepare_Z(const SVRParameters &params, const arma::mat &features_t, const arma::mat &labels, size_t len)
{
    if (!len) len = features_t.n_cols;
    arma::mat *p_Z;
    switch (params.get_kernel_type()) {
        case kernel_type_e::PATH: {
            const auto cums_X = get_cached_cumulatives(params, features_t);
            if (cums_X.empty()) LOG4_THROW("Failed preparing cumulatives matrices.");
            p_Z = new arma::mat(len, len, arma::fill::zeros);
#pragma omp parallel for num_threads(cums_X.size())
            for (const auto &c_X: cums_X) {
                arma::mat z = arma::mat(arma::size(*p_Z));
                kernel::path::cu_distances_xx(c_X.n_rows, 1, len, 0, 0, len, len, c_X.mem, params.get_svr_kernel_param2(), DEFAULT_TAU_TUNE, 1, z.memptr());
#pragma omp critical(__prepare_Z)
                *p_Z += z;
            }
            *p_Z = p_Z->t();
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

arma::mat &OnlineMIMOSVR::get_cached_Z(const SVRParameters &params, const arma::mat &features_t, const arma::mat &labels, const size_t short_size_X)
{
    typedef std::tuple<
            size_t /* dataset id */, size_t /* chunk */, size_t /* grad */, std::string /* queue name */, std::string /* column */, size_t /* lambda */,
            size_t /* lag */, size_t /* decrement */, size_t /* n samples */ > Z_cache_key_t;
    typedef std::map<Z_cache_key_t, arma::mat *> Z_cache_t;
    static Z_cache_t Z_cache;
    static std::mutex Z_mx;

    const Z_cache_key_t params_key {
            params.get_dataset_id(),
            params.get_chunk_ix(),
            params.get_grad_level(),
            params.get_input_queue_table_name(),
            params.get_input_queue_column_name(),
            std::round(1e5 * params.get_svr_kernel_param2()),
            params.get_lag_count(),
            params.get_svr_decremental_distance(),
            features_t.n_cols};

    const auto prepare_f = [&params, &features_t, &labels, &short_size_X](){ return prepare_Z(params, features_t, labels, short_size_X); };
    return *cached(Z_cache, Z_mx, params_key, prepare_f);
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

            p_Zy = new arma::mat(features_t.n_cols, predict_features_t.n_cols, arma::fill::zeros);
#pragma omp parallel for num_threads(cums_X.size())
            for (size_t i = 0; i < cums_X.size(); ++i) {
                arma::mat z = arma::mat(arma::size(*p_Zy));
                if (cums_X[i].n_rows != features_t.n_cols || p_cums_Y_tmp->at(i).n_cols != predict_features_t.n_cols)
                    LOG4_WARN("Incompatible sizes cums X " << arma::size(cums_X[i]) << ", features transposed " << arma::size(features_t)
                        << ", cums Y " << arma::size(p_cums_Y_tmp->at(i)) << ", predict features transposed " << arma::size(predict_features_t));
                kernel::path::cu_distances_xy(
                        cums_X[i].n_rows, 1, cums_X[i].n_cols, p_cums_Y_tmp->at(i).n_cols, 0, 0, features_t.n_cols, predict_features_t.n_cols,
                        cums_X[i].mem, p_cums_Y_tmp->at(i).mem, params.get_svr_kernel_param2(), DEFAULT_TAU_TUNE, 1, z.memptr());
                p_cums_Y_tmp->at(i).clear();
#pragma omp critical
                *p_Zy += z;
            }
            *p_Zy = p_Zy->t();
            LOG4_DEBUG("Returning Zy " << arma::size(*p_Zy) << " for " << params << ", cumulative matrices " << cums_X.size() << ", of dimensions "
                                       << arma::size(cums_X.front()) << " and " << arma::size(p_cums_Y_tmp->front()));
            delete p_cums_Y_tmp;
            break;
        }
        case kernel_type_e::DEEP_PATH:
        case kernel_type_e::DEEP_PATH2:
#if 0
            p_Zy = new arma::mat(get_reference_distances(labels));
            LOG4_DEBUG("Returning Z " << arma::size(*p_Zy) << " for " << params <<  ", labels " << arma::size(labels) << ", features " << arma::size(features_t));
#endif
            LOG4_THROW("Manifold not handled!");
            break;
        default:
            LOG4_THROW("Kernel type " << int(params.get_kernel_type()) << " not handled!");
    }

    return p_Zy;
}


// TODO Create an ensemble::cache class and move the above and below methods there
namespace {
    typedef std::tuple<std::string /* queue name */, std::string /* column */, size_t /* lambda */, size_t /* lag */, size_t /* decrement */,
            size_t /* p_predictions count */, bpt::ptime /* prediction time */, size_t /* level */, size_t /* chunk */, size_t /* gradient */> Zy_cache_key_t;
    typedef std::map<Zy_cache_key_t, arma::mat *> Zy_cache_t;
    Zy_cache_t Zy_cache;
}

void OnlineMIMOSVR::clear_Zy_cache()
{
    Zy_cache.clear();
}

arma::mat &OnlineMIMOSVR::get_cached_Zy(const SVRParameters &params, const arma::mat &features_t /* transposed */, const arma::mat &predict_features_t /* transposed */, const bpt::ptime &pred_time)
{
    static std::mutex Zy_mx;

    const Zy_cache_key_t params_key {
            params.get_input_queue_table_name(),
            params.get_input_queue_column_name(),
            std::round(1e5 * params.get_svr_kernel_param2()),
            params.get_lag_count(),
            features_t.n_cols,
            predict_features_t.n_cols,
            pred_time,
            params.get_decon_level(),
            params.get_chunk_ix(),
            params.get_grad_level() };

    const auto prepare_f = [&params, &features_t, &predict_features_t, &pred_time](){ return prepare_Zy(params, features_t, predict_features_t, pred_time); };
    return *cached(Zy_cache, Zy_mx, params_key, prepare_f);
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
                        res_kernel_matrix = init_kernel_matrix(svr_parameters, x_train.rows(indexes[i]), y_train.rows(indexes[i])),
                        "Init kernel matrix " << arma::size(res_kernel_matrix));
                double cur_err = 0;
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
#pragma omp parallel for reduce(+:overall_score) default(shared)
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
std::tuple<
    std::shared_ptr<std::deque<arma::mat>> /* p_predictions */,
    std::shared_ptr<std::deque<arma::mat>> /* labels */,
    std::shared_ptr<std::deque<arma::mat>> /* lastknowns */,
    double> /* score */
calc_direction(const arma::mat &K, const double epsco, const arma::mat &labels, const arma::mat &last_knowns, const size_t train_len, const double meanabs_labels, const bool is_psd)
{
    const off_t start_point_K = K.n_rows - EMO_TUNE_TEST_SIZE - train_len;
    const off_t start_point_labels = labels.n_rows - EMO_TUNE_TEST_SIZE - train_len;
    if (start_point_K < 0 || start_point_labels < 0 || labels.n_rows != K.n_rows)
        LOG4_THROW("Shorter K " << start_point_K << " or labels " << start_point_labels << " for K " << arma::size(K) << ", labels " << arma::size(labels));
    auto p_out_preds = std::make_shared<std::deque<arma::mat>>(EMO_MAX_J);
    auto p_out_labels = std::make_shared<std::deque<arma::mat>>(EMO_MAX_J);
    auto p_out_last_knowns = std::make_shared<std::deque<arma::mat>>(EMO_MAX_J);
    double score = 0;
#pragma omp parallel for reduction(+:score) num_threads(TUNE_HYBRID_MAIN_THREADS)
    for (size_t j = 0; j < EMO_MAX_J; ++j) {
        const size_t x_train_start = j * EMO_SLIDE_SKIP;
        const size_t x_train_final = x_train_start + train_len - 1;
        const size_t x_test_start = x_train_final + 1;
        const size_t x_test_final = x_test_start + EMO_TUNE_TEST_SIZE - j * EMO_SLIDE_SKIP - 1;
        const size_t x_test_len = x_train_final - x_train_start + 1;

        LOG4_TRACE("Try " << j << ", K " << arma::size(K) << ", train start " << x_train_start << ", train final " << x_train_final << ", test start " << x_test_start << ", test final " << x_test_final);
        try {
            double this_score = 0;
#ifdef TAIL_VALIDATION
            p_out_labels->at(j) = labels.rows(labels.n_rows - 1 - start_point_labels - x_test_final, labels.n_rows - 1 - start_point_labels - x_test_start);
            p_out_preds->at(j).set_size(arma::size(p_out_labels->at(j)));
            p_out_preds->at(j) = K.submat(K.n_rows - 1 - start_point_K - x_test_final, K.n_rows - 1 - start_point_K - x_train_final, K.n_rows - 1  - start_point_K - x_test_start, K.n_rows - 1 - start_point_K - x_train_start) *
                    OnlineMIMOSVR::call_gpu_dynsolve(eye_epsco + K.submat(K.n_rows - 1 - start_point_K - x_train_final, K.n_rows - 1 - start_point_K - x_train_final,
                                                                          K.n_rows - 1 - start_point_K - x_train_start, K.n_rows - 1 - start_point_K - x_train_start),
                    labels.rows(labels.n_rows - 1 - start_point_labels - x_train_final, labels.n_rows - 1 - start_point_labels - x_train_start));
            this_score += common::meanabs<double>(p_out_preds->at(j) - p_out_labels->at(j)) / meanabs_labels;
#endif
            p_out_labels->at(j) = labels.rows(start_point_labels + x_test_start, start_point_labels + x_test_final);
            p_out_preds->at(j).set_size(arma::size(p_out_labels->at(j)));
            const arma::mat weights = OnlineMIMOSVR::solve_dispatch(
                    arma::eye(x_test_len, x_test_len) * epsco,
                    K.submat(start_point_K + x_train_start, start_point_K + x_train_start, start_point_K + x_train_final, start_point_K + x_train_final),
                    labels.rows(start_point_labels + x_train_start, start_point_labels + x_train_final),
                    IRWLS_ITER_TUNE,
                    is_psd);
            const arma::mat K_submat_1 = K.submat(start_point_K + x_test_start, start_point_K + x_train_start, start_point_K + x_test_final, start_point_K + x_train_final);
            p_out_preds->at(j) = K_submat_1 * weights;
            this_score += common::meanabs<double>(p_out_labels->at(j) - p_out_preds->at(j)) / meanabs_labels;

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
    return {p_out_preds, p_out_labels, p_out_last_knowns, score};
}
}

void OnlineMIMOSVR::tune_kernel_params(
        t_grad_preds &predictions,
        t_param_set &params, const arma::mat &features, const arma::mat &labels, const arma::mat &last_knowns,
        const size_t chunk_size)
{
    const auto p_head_parameters = *params.begin();
    if (p_head_parameters->is_manifold()) {
        LOG4_DEBUG("Skipping tuning of manifold kernel!");
        return;
    }
    // TODO Fix tuning for multiple chunks, indexes_tune are not properly validated in loss function
    LOG4_DEBUG("Tuning parameters " << *p_head_parameters << ", labels " << common::present(labels) << ", features " << common::present(features) << ", last-knowns " << common::present(last_knowns) << ", EMO_SLIDE_SKIP " << EMO_SLIDE_SKIP << /* ", MULTIPLE_EPSCO " << MULTIPLE_EPSCO << */ ", EMO_MAX_J " << EMO_MAX_J << ", EMO_TUNE_VALIDATION_WINDOW " << EMO_TUNE_VALIDATION_WINDOW << ", TUNE_EPSCOST_MAX " << common::C_tune_crass_epscost.front() << ", TUNE_EPSCOST_MIN " << common::C_tune_crass_epscost.back());
    const std::string original_input_queue_column_name = p_head_parameters->get_input_queue_column_name();
    const std::string original_input_queue_table_name = p_head_parameters->get_input_queue_table_name();
    p_head_parameters->set_input_queue_column_name("TUNE_COLUMN_" + p_head_parameters->get_input_queue_column_name());
    p_head_parameters->set_input_queue_table_name("TUNE_TABLE_" + p_head_parameters->get_input_queue_table_name());
    const double meanabs_all_labels = common::meanabs(labels);
    const auto indexes_tune = get_indexes(features.n_rows, *p_head_parameters, chunk_size);
    std::deque<arma::ucolvec> indexes_train;
    for (const auto &ixtu: indexes_tune) indexes_train.emplace_back(ixtu + EMO_TUNE_TEST_SIZE);
    const size_t train_len = indexes_tune.front().n_elem;
    std::deque<arma::mat> feature_chunks_t, label_chunks, lastknown_chunks;
    for (const auto &chunk_ixs: indexes_tune) {
        feature_chunks_t.emplace_back(features.rows(chunk_ixs).t());
        label_chunks.emplace_back(labels.rows(chunk_ixs));
        lastknown_chunks.emplace_back(last_knowns.rows(chunk_ixs));
        feature_chunks_t.back() = arma::join_rows(feature_chunks_t.back(), features.rows(features.n_rows - EMO_TUNE_TEST_SIZE, features.n_rows - 1).t());
        label_chunks.back() = arma::join_cols(label_chunks.back(), labels.rows(labels.n_rows - EMO_TUNE_TEST_SIZE, labels.n_rows - 1));
        lastknown_chunks.back() = arma::join_cols(lastknown_chunks.back(), last_knowns.rows(last_knowns.n_rows - EMO_TUNE_TEST_SIZE, last_knowns.n_rows - 1));
    }
    const arma::mat k_eye = arma::eye(train_len, train_len);
#ifdef FINER_GAMMA_TUNE
    double gamma_mult_fine_start, gamma_mult_fine_end, epsco_fine_start, epsco_fine_end;
#endif
    const auto validate_gammas = [&](const double mingamma, const arma::mat &Z, const arma::mat *p_Ztrain,
            const std::deque<double> &gamma_multis, const SVRParameters &score_params, const size_t chunk_ix)
    {
        LOG4_DEBUG("Validating " << gamma_multis.size() << " gamma multipliers, starting from " << gamma_multis.front() << " to " << gamma_multis.back() <<
                    ", min gamma " << mingamma << ", Z " << arma::size(Z) << ", params " << score_params);
#pragma omp parallel for num_threads(TUNE_HYBRID_MAIN_THREADS)
        for (const double gamma_mult: gamma_multis) {
            auto gamma_params = score_params;
            gamma_params.set_svr_kernel_param(gamma_mult * mingamma);

            arma::mat K(arma::size(Z));
            arma::mat *p_Ktrain = new arma::mat(arma::size(Z));
            // const arma::mat K = 1. - Z / (2. * std::pow<double>(gamma_params.get_svr_kernel_param(), 2.));
            PROFILE_EXEC_TIME(solvers::kernel_from_distances(p_Ktrain->memptr(), p_Ztrain->mem, p_Ztrain->n_rows, p_Ztrain->n_cols, gamma_params.get_svr_kernel_param()), "Kernel from train distances");
            PROFILE_EXEC_TIME(solvers::kernel_from_distances(K.memptr(), Z.mem, Z.n_rows, Z.n_cols, gamma_params.get_svr_kernel_param()), "Kernel from tune distances");
            constexpr bool is_psd = false; // K.is_symmetric() && K.min() >= 0;
            const auto epscos_gen = [&gamma_params, &p_Ktrain, &train_len]() -> std::deque<double> {
                double epsco;
                switch (gamma_params.get_kernel_type()) {
                    case kernel_type_e::PATH: // Path matrix is not PSD
                    case kernel_type_e::DEEP_PATH:
                    case kernel_type_e::DEEP_PATH2:
                        epsco = arma::mean(arma::vectorise(*p_Ktrain));
                        break;
                    default:
                        epsco = arma::mean(arma::vectorise(arma::trimatl(*p_Ktrain)));
                        break;
                }
                return {epsco, epsco / double(train_len)};
            } ();
            delete p_Ktrain;
            delete p_Ztrain;
#pragma omp parallel for num_threads(TUNE_HYBRID_MAIN_THREADS)
            for (const auto epsco: epscos_gen) {
                if (epsco <= 0) LOG4_WARN("Auto epsco is negative indefinite " << epsco << ",  K: " << common::present(K) << ", gamma params " << gamma_params);// << ", contents of K: " << arma::trimatl(K));
                auto p_cost_params = std::make_shared<SVRParameters>(gamma_params);
                const auto [p_out_preds, p_out_labels, p_out_last_knowns, score] = calc_direction(K, epsco, label_chunks[chunk_ix], lastknown_chunks[chunk_ix], train_len, meanabs_all_labels, is_psd);
#ifdef SEPARATE_PREDICTIONS_BY_COST
                const uint64_t epsco_key = 1e6 * epsco;
#else
                constexpr uint64_t epsco_key = 0;
#endif
#pragma omp critical
                {
                    if (!predictions.contains(chunk_ix)
                        || predictions[chunk_ix].empty()
                        || predictions[chunk_ix][epsco_key].size() < size_t(common::C_tune_keep_preds)
                        || score < predictions[chunk_ix][epsco_key].begin()->get()->score) {
                        p_cost_params->set_svr_C(1. / (2. * epsco));
                        p_cost_params->set_input_queue_table_name(original_input_queue_table_name);
                        p_cost_params->set_input_queue_column_name(original_input_queue_column_name);
                        predictions[chunk_ix][epsco_key].emplace(std::make_shared<t_param_preds>(score, p_cost_params, p_out_preds, p_out_labels, p_out_last_knowns));
                        LOG4_DEBUG("Lambda, gamma tune best score " << score << ", for " << *p_cost_params);
                        while (predictions[chunk_ix][epsco_key].size() > size_t(common::C_tune_keep_preds))
                            predictions[chunk_ix][epsco_key].unsafe_erase(std::prev(predictions[chunk_ix][epsco_key].end()));
#ifdef FINER_GAMMA_TUNE
                        gamma_mult_fine_start = gamma_mult / C_fine_gamma_div;
                        gamma_mult_fine_end = gamma_mult * C_fine_gamma_mult;
                        epsco_fine_start = epsco;
                        epsco_fine_end = epsco;
#endif
                    }
                }
            }
        }
    };

#pragma omp parallel for num_threads(TUNE_HYBRID_MAIN_THREADS)
    for (size_t chunk_ix = 0; chunk_ix < indexes_tune.size(); ++chunk_ix) {
        const auto chunk_params = business::SVRParametersService::find(params, chunk_ix);
        std::deque<double> tuned_lambdas;
        switch (chunk_params->get_kernel_type()) {
            case kernel_type_e::PATH:
                tuned_lambdas = common::C_tune_lambdas_path;
                break;
            default:
                tuned_lambdas.emplace_back(0);
        }

#pragma omp parallel for num_threads(TUNE_HYBRID_MAIN_THREADS)
        for (const double lambda: tuned_lambdas) {
            auto lambda_params = *chunk_params;
            lambda_params.set_chunk_ix(chunk_ix);
            lambda_params.set_svr_kernel_param2(lambda);
            // common::memory_manager::get().barrier();

            const arma::mat &Z = get_cached_Z(lambda_params, feature_chunks_t[chunk_ix], label_chunks[chunk_ix]);

            const double mean_train_labels = arma::mean(arma::vectorise(labels.rows(indexes_train[chunk_ix])));
            auto tmp_params = lambda_params;
            tmp_params.set_input_queue_table_name("TMP_" + tmp_params.get_input_queue_table_name());
            tmp_params.set_input_queue_column_name("TMP_" + tmp_params.get_input_queue_column_name());
            const arma::mat *p_Ztrain = prepare_Z(tmp_params, features.rows(indexes_train[chunk_ix]).t(), labels.rows(indexes_train[chunk_ix]), train_len);
            const auto mingamma = calc_gamma(*p_Ztrain, train_len, mean_train_labels);

            PROFILE_EXEC_TIME(validate_gammas(mingamma, Z, p_Ztrain, C_gamma_multis, lambda_params, chunk_ix), "Validate gammas " << lambda_params);
            for (const auto &p: predictions[chunk_ix])
                LOG4_DEBUG("Crass gamma pass score " << p.second.begin()->get()->score << ", mingamma " << mingamma << ", best parameters " << *p.second.begin()->get()->p_params);
#ifdef NEW_FINER_GAMMA_TUNE
            std::deque<double> gamma_fine_multis;
            for (double gamma_mult = gamma_mult_fine_start; gamma_mult < gamma_mult_fine_end; gamma_mult *= FINE_GAMMA_MULTIPLE)
                gamma_fine_multis.emplace_back(gamma_mult);
            validate_gammas(mingamma, Z, gamma_fine_multis, best_params, common::C_tune_crass_epscost);
            for (const auto &p: predictions)
                LOG4_DEBUG("Fine gamma pass score " << p.second.begin()->get()->score << ", mingamma " << mingamma << ", best parameters " << *p.second.begin()->get()->p_params);
#endif
        }
        for (const auto &p: predictions[chunk_ix])
            LOG4_DEBUG("First pass score " << p.second.begin()->get()->score << ", best parameters " << *p.second.begin()->get()->p_params);
#ifdef FINER_GAMMA_TUNE
        const arma::mat &Z = get_cached_Z(*p.second.begin()->get()->p_params, feature_chunks_t[chunk_ix], label_chunks[chunk_ix]);

#if 0 // Fine epsco tuning
        std::deque<double> epsco_fine_grid;
        for (double epsco = epsco_fine_start; epsco <= epsco_fine_end; epsco *= MULTIPLE_EPSCO_FINE)
            epsco_fine_grid.emplace_back(epsco);
#endif

    const double min_gamma = get_cached_gamma(*p.second.begin()->get()->p_params, Z, train_len, mean_train_labels);
    std::deque<double> gamma_fine_multis;
    for (double gamma_mult = gamma_mult_fine_start; gamma_mult < gamma_mult_fine_end; gamma_mult *= FINE_GAMMA_MULTIPLE)
        gamma_fine_multis.emplace_back(gamma_mult);
    validate_gammas(min_gamma, Z, gamma_fine_multis, *p.second.begin()->get()->p_params, common::C_tune_crass_epscost);
    gamma_fine_multis.clear();
    *p_head_parameters = best_params;
#endif

        for (auto &p: predictions[chunk_ix]) {
#pragma omp critical
            p.second.unsafe_erase(std::next(p.second.begin(), common::C_tune_keep_preds), p.second.end());

            size_t ctr = 0;
            for (const auto &pr: p.second)
                LOG4_INFO("Final best " << ctr++ << " score " << pr->score << ", final parameters " << *pr->p_params);
        }
    }
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
