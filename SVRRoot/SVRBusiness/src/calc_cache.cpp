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

tbb::concurrent_unordered_set<cached_iface *> cached_register::cr;
std::mutex cached_register::erase_mx;

calc_cache::calc_cache()
{}


double calc_cache::get_cached_gamma(const datamodel::SVRParameters &params, const arma::mat &Z, const double meanabs_labels)
{
    const gamma_cache_key_t params_key{
            params.get_input_queue_column_name(),
            params.get_decon_level(),
            params.get_grad_level(),
            params.get_chunk_index(),
            common::hash_lambda(params.get_svr_kernel_param2()),
            params.get_adjacent_levels(),
            arma::size(Z)};

    const auto prepare_f = [&Z, &meanabs_labels]() { return datamodel::OnlineMIMOSVR::calc_gamma(Z, meanabs_labels); };
    return cached<double, dtype(params_key), dtype(prepare_f)>::get()(params_key, prepare_f);
}


arma::mat &calc_cache::get_cached_cumulatives(const datamodel::SVRParameters &params, const arma::mat &features_t, const bpt::ptime &time)
{
    const cml_cache_key_t params_key{
            params.get_grad_level(),
            params.get_chunk_index(),
            params.get_adjacent_levels(),
            arma::size(features_t),
            time};

    const auto prepare_f = [&params, &features_t]() { return datamodel::OnlineMIMOSVR::all_cumulatives(params, features_t); };
    return *cached<mat_ptr, dtype(params_key), dtype(prepare_f)>::get()(params_key, prepare_f);
}


arma::mat &calc_cache::get_cached_Z(const datamodel::SVRParameters &params, const arma::mat &features_t, const bpt::ptime &time)
{
    const Z_cache_key_t params_key{
            params.get_grad_level(),
            params.get_chunk_index(),
            common::hash_lambda(params.get_svr_kernel_param2()),
            params.get_adjacent_levels(),
            arma::size(features_t),
            time};

    const auto prepare_f = [&params, &features_t, &time, this](){ return datamodel::OnlineMIMOSVR::prepare_Z(*this, params, features_t, time); };
    return *cached<mat_ptr, dtype(params_key), dtype(prepare_f)>::get()(params_key, prepare_f);
}


arma::mat &calc_cache::get_cached_Zy(
        const datamodel::SVRParameters &params, const arma::mat &features_t /* transposed */, const arma::mat &predict_features_t /* transposed */, const bpt::ptime &time)
{
    const Zy_cache_key_t params_key{
            params.get_grad_level(),
            params.get_chunk_index(),
            common::hash_lambda(params.get_svr_kernel_param2()),
            params.get_adjacent_levels(),
            arma::size(features_t),
            arma::size(predict_features_t),
            time};

    const auto prepare_f = [&params, &features_t, &predict_features_t, &time, this]() {
        return datamodel::OnlineMIMOSVR::prepare_Zy(*this, params, features_t, predict_features_t, time);
    };
    return *cached<mat_ptr, dtype(params_key), dtype(prepare_f)>::get()(params_key, prepare_f);
}

arma::mat &calc_cache::get_cached_K(const datamodel::SVRParameters &params, const arma::mat &features_t, const bpt::ptime &time)
{
    const K_cache_key_t params_key{
            params.get_grad_level(),
            params.get_chunk_index(),
            common::hash_lambda(params.get_svr_kernel_param2()),
            common::hash_lambda(params.get_svr_kernel_param()),
            params.get_adjacent_levels(),
            arma::size(features_t),
            time};

    const auto prepare_f = [&params, &features_t, &time, this](){ return datamodel::OnlineMIMOSVR::prepare_K(*this, params, features_t, time); };
    return *cached<mat_ptr, dtype(params_key), dtype(prepare_f)>::get()(params_key, prepare_f);
}

mat_ptr calc_cache::get_cached_features(
    const std::deque<bpt::ptime> &label_times, const std::deque<datamodel::DeconQueue_ptr> &features_aux, const size_t lag, const double quantize, const std::set<size_t> &adjacent_levels,
    const bpt::time_duration &max_gap, const bpt::time_duration &aux_queue_res, const bpt::time_duration &main_queue_resolution)
{
    const features_cache_key_t features_key {lag, common::hash_lambda(quantize), adjacent_levels, aux_queue_res, label_times.front(), main_queue_resolution,
                                             label_times.back(), label_times.size()};

    const auto prepare_f = [&label_times, &features_aux, &lag, &quantize, &adjacent_levels, &max_gap, &aux_queue_res, &main_queue_resolution] () {
        auto p_features = ptr<arma::mat>();
        PROFILE_EXEC_TIME(
                business::ModelService::prepare_features(*p_features, label_times, features_aux, lag, quantize, adjacent_levels, max_gap, aux_queue_res, main_queue_resolution)
        , "Get " << label_times.size() << " features, on levels " << adjacent_levels << ", starting " << label_times.front() << ", until " << label_times.back());
        return p_features;
    };
    return cached<mat_ptr, dtype(features_key), dtype(prepare_f)>::get()(features_key, prepare_f);
}


datamodel::t_parameter_predictions_set &calc_cache::checkin_tuner(const datamodel::OnlineMIMOSVR &svr, const size_t chunk_ix)
{
    const auto tune_predictions_key = std::make_tuple(svr.get_params_ptr()->get_input_queue_column_name(), svr.get_gradient_level(), chunk_ix);
    const std::scoped_lock l(tuners_mx);
    auto tune_iter = tune_results.find(tune_predictions_key);
    if (tune_iter == tune_results.end()) {
        bool rc;
        std::tie(tune_iter, rc) = tune_results.emplace(tune_predictions_key, levels_tune_data{0, 0, {}, ptr<std::condition_variable>(), ptr<std::mutex>(), false});
        if (!rc) LOG4_THROW("Failed creating tune data for " << tune_predictions_key);
    };
    ++tune_iter->second.started_tuners;
    auto level_iter = tune_iter->second.level_predictions.find(svr.get_decon_level());
    if (level_iter == tune_iter->second.level_predictions.cend()) {
        bool rc;
        std::tie(level_iter, rc) = tune_iter->second.level_predictions.emplace(svr.get_decon_level(), datamodel::t_parameter_predictions_set{});
        if (!rc) LOG4_THROW("Failed creating level predictions for " << svr.get_decon_level());
    }
    return level_iter->second;
}


void calc_cache::checkout_tuner(const datamodel::OnlineMIMOSVR &svr, const size_t chunk_ix)
{
    const auto tune_predictions_key = std::make_tuple(svr.get_params_ptr()->get_input_queue_column_name(), svr.get_gradient_level(), chunk_ix);
    const auto tune_iter = tune_results.find(tune_predictions_key);
    if (tune_iter == tune_results.end()) return;
    const std::scoped_lock l(tuners_mx);
    ++tune_iter->second.completed_tuners;
    LOG4_DEBUG("Checking out tuner " << tune_predictions_key << ", started tuners " << tune_iter->second.started_tuners << ", completed tuners " << tune_iter->second.completed_tuners);
    tune_iter->second.p_tuners_done->notify_one();
}


const datamodel::t_level_tuned_parameters *calc_cache::recombine_go(const datamodel::OnlineMIMOSVR &svr, const size_t chunk_ix)
{
    const auto tune_predictions_key = std::make_tuple(svr.get_params_ptr()->get_input_queue_column_name(), svr.get_gradient_level(), chunk_ix);
    const auto tune_iter = tune_results.find(tune_predictions_key);
    if (tune_iter == tune_results.end()) {
        LOG4_WARN("Tuner results for " << tune_predictions_key << " not found.");
        return nullptr;
    }
    std::unique_lock ul(*tune_iter->second.p_mx);
    tune_iter->second.p_tuners_done->wait(ul, [&tune_iter, &svr] {
        return tune_iter->second.started_tuners == tune_iter->second.completed_tuners &&
            tune_iter->second.completed_tuners == ssize_t(svr.get_dataset()->get_model_count());
    });
    if (tune_iter->second.recombining) {
        LOG4_DEBUG("Already recombining " << tune_predictions_key);
        ul.unlock();
        return nullptr;
    }
    tune_iter->second.recombining = true;
//        clear_tune_cache(svr.get_params_ptr()->get_input_queue_column_name()); // Cache is not used while tuning at present
    ul.unlock();
    return &tune_iter->second.level_predictions;
}

void calc_cache::clear_tune_cache(const std::string &column_name)
{
    // Not needed for the time being
}

void calc_cache::clear()
{
    std::for_each(std::execution::par_unseq, cached_register::cr.begin(), cached_register::cr.end(), [](auto &i) { i->clear(); });
}

}
}
