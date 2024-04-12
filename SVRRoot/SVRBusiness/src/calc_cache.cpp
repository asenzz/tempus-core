//
// Created by zarko on 2/21/24.
//
#include "calc_cache.tpp"
#include "calc_cache.hpp"
#include "common/types.hpp"
#include "model/Dataset.hpp"
#include "model/SVRParameters.hpp"
#include "common/compatibility.hpp"
#include "onlinesvr.hpp"
#include "appcontext.hpp"

namespace svr {
namespace business {


calc_cache::calc_cache()
{}


double calc_cache::get_cached_gamma(const datamodel::SVRParameters &params, const arma::mat &Z, const double meanabs_labels)
{
    const gamma_cache_key_t params_key{
            params.get_input_queue_column_name(),
            params.get_decon_level(),
            params.get_grad_level(),
            params.get_chunk_ix(),
            common::hash_lambda(params.get_svr_kernel_param2()),
            params.get_lag_count(),
            params.get_svr_decremental_distance(),
            params.get_adjacent_levels(),
            arma::size(Z)};

    auto prepare_f = [&Z, &meanabs_labels]() { return datamodel::OnlineMIMOSVR::calc_gamma(Z, meanabs_labels); };
    return cached(gamma_cache, gamma_mx, params_key, prepare_f);
}


std::deque<arma::mat> &calc_cache::get_cached_cumulatives(const datamodel::SVRParameters &params, const arma::mat &features_t, const bpt::ptime &time)
{
    const cml_cache_key_t params_key{
            params.get_input_queue_column_name(),
            params.get_grad_level(),
            params.get_chunk_ix(),
            params.get_lag_count(),
            params.get_svr_decremental_distance(),
            params.get_adjacent_levels(),
            arma::size(features_t),
            time};

    auto prepare_f = [&params, &features_t]() { return datamodel::OnlineMIMOSVR::prepare_cumulatives(params, features_t); };
    return *cached(cml_cache, cml_mx, params_key, prepare_f);
}


arma::mat &calc_cache::get_cached_Z(const datamodel::SVRParameters &params, const arma::mat &features_t, const bpt::ptime &time)
{
    const Z_cache_key_t params_key{
            params.get_input_queue_column_name(),
            params.get_grad_level(),
            params.get_chunk_ix(),
            common::hash_lambda(params.get_svr_kernel_param2()),
            params.get_lag_count(),
            params.get_svr_decremental_distance(),
            params.get_adjacent_levels(),
            arma::size(features_t),
            time};

    auto prepare_f = [&params, &features_t, &time, this](){ return datamodel::OnlineMIMOSVR::prepare_Z(*this, params, features_t, time); };
    return *cached(Z_cache, Z_mx, params_key, prepare_f);
}


arma::mat &calc_cache::get_cached_Zy(
        const datamodel::SVRParameters &params, const arma::mat &features_t /* transposed */, const arma::mat &predict_features_t /* transposed */, const bpt::ptime &time)
{
    const Zy_cache_key_t params_key{
            params.get_input_queue_column_name(),
            params.get_decon_level(),
            params.get_grad_level(),
            params.get_chunk_ix(),
            common::hash_lambda(params.get_svr_kernel_param2()),
            params.get_lag_count(),
            params.get_svr_decremental_distance(),
            params.get_adjacent_levels(),
            arma::size(features_t),
            arma::size(predict_features_t),
            time};

    auto prepare_f = [&params, &features_t, &predict_features_t, &time, this]() {
        return datamodel::OnlineMIMOSVR::prepare_Zy(*this, params, features_t, predict_features_t, time);
    };
    return *cached(Zy_cache, Zy_mx, params_key, prepare_f);
}

mat_ptr calc_cache::get_cached_features(
    const std::deque<bpt::ptime> &label_times, const std::deque<datamodel::DeconQueue_ptr> &features_aux, const size_t lag, const std::set<size_t> &adjacent_levels,
    const bpt::time_duration &max_gap, const bpt::time_duration &aux_queue_res, const bpt::time_duration &main_queue_resolution)
{
    const features_cache_key_t features_key {lag, adjacent_levels, aux_queue_res, label_times.front(), main_queue_resolution, label_times.back(), label_times.size()};

    auto prepare_f = [&label_times, &features_aux, &lag, &adjacent_levels, &max_gap, &aux_queue_res, &main_queue_resolution] () {
        auto p_features = ptr<arma::mat>();
        PROFILE_EXEC_TIME(
                business::ModelService::prepare_features(*p_features, label_times, features_aux, lag, adjacent_levels, max_gap, aux_queue_res, main_queue_resolution)
        , "Get " << label_times.size() << " features, on levels " << adjacent_levels << ", starting " << label_times.front() << ", until " << label_times.back());
        return p_features;
    };
    return cached(features_cache, features_mx, features_key, prepare_f);
}


datamodel::t_tuned_parameters_ptr calc_cache::checkin_tuner(const datamodel::OnlineMIMOSVR &svr)
{
    const auto &column_name = svr.get_params_ptr()->get_input_queue_column_name();
    const std::scoped_lock l(tuners_mx);
    auto tune_iter = tune_predictions.find(column_name);
    if (tune_iter == tune_predictions.end()) {
        const auto rc = tune_predictions.emplace(
                column_name, tune_data{0, 0, otr<datamodel::t_tuned_parameters>(), otr<std::condition_variable>(), otr<std::mutex>(), false});
        if (!rc.second) LOG4_THROW("Failed creating tune data for column " << column_name);
        tune_iter = rc.first;
    };
    ++tune_iter->second.started_tuners;
    return tune_iter->second.p_tune_predictions;
}


#define TUNERS_DONE(iter_state, svr) \
    iter_state->second.started_tuners == iter_state->second.completed_tuners && iter_state->second.completed_tuners == ssize_t(svr.get_dataset()->get_model_count())


void calc_cache::checkout_tuner(const datamodel::OnlineMIMOSVR &svr)
{
    const auto p_params = svr.get_params_ptr();
    const std::scoped_lock l(tuners_mx);
    auto tune_iter = tune_predictions.find(p_params->get_input_queue_column_name());
    if (tune_iter == tune_predictions.end()) return;
    ++tune_iter->second.completed_tuners;
    LOG4_DEBUG("Checking out tuner " << *p_params << ", started tuners " << tune_iter->second.started_tuners << ", completed tuners " << tune_iter->second.completed_tuners);
    tune_iter->second.p_tuners_done->notify_all();
}


bool calc_cache::recombine_go(const datamodel::OnlineMIMOSVR &svr)
{
    const auto p_params = svr.get_params_ptr();
    const auto tune_iter = tune_predictions.find(p_params->get_input_queue_column_name());
    if (tune_iter == tune_predictions.end()) return false;
    std::unique_lock ul(*tune_iter->second.p_mx);
    tune_iter->second.p_tuners_done->wait(ul, [&tune_iter, &svr] { return TUNERS_DONE(tune_iter, svr); });
    bool res = false;
    if (!tune_iter->second.recombining) {
        res = tune_iter->second.recombining = true;
        clear_tune_cache(p_params->get_input_queue_column_name());
    }
    ul.unlock();
    return res;
}


datamodel::t_tuned_parameters &calc_cache::get_tuner_state(const std::string &column_name)
{
    const auto tune_iter = tune_predictions.find(column_name);
    if (tune_iter == tune_predictions.end()) LOG4_THROW("Could not find tuner data for ensemble " << column_name);
    return *tune_iter->second.p_tune_predictions;
}


datamodel::t_param_set calc_cache::get_best_parameters(const std::string &column_name, const size_t decon_level, const size_t gradient_level, const size_t num_chunks)
{
    datamodel::t_param_set res;
    for (size_t chunk_ix = 0; chunk_ix < num_chunks; ++chunk_ix) {
        const auto p_best_params = (**tune_predictions[column_name].p_tune_predictions->at({decon_level, gradient_level, chunk_ix})->cbegin()).p_params;
        res.emplace(p_best_params);
        LOG4_DEBUG("Best params " << *p_best_params);
    }
    return res;
}


void calc_cache::clear_tune_cache(const std::string &column_name)
{
    const auto tune_column_name = "TUNE_" + column_name;
    auto find_tune = [&tune_column_name](auto &e) { return std::get<0>(e.first).find(tune_column_name) != std::string::npos; };
#pragma omp parallel num_threads(adj_threads(4))
    {
#pragma omp task
        remove_if(gamma_cache, find_tune);
#pragma omp task
        remove_if(cml_cache, find_tune);
#pragma omp task
        remove_if(Zy_cache, find_tune);
#pragma omp task
        remove_if(Z_cache, find_tune);
    }
}

void calc_cache::clear()
{
#pragma omp parallel num_threads(adj_threads(4))
    {
#pragma omp task
        gamma_cache.clear();
#pragma omp task
        cml_cache.clear();
#pragma omp task
        Z_cache.clear();
#pragma omp task
        Zy_cache.clear();
#pragma omp task
        features_cache.clear();
    }
}

}
}
