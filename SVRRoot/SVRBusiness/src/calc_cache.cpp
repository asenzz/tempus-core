//
// Created by zarko on 2/21/24.
//
#include "calc_cache.tpp"
#include "common/types.hpp"
#include "model/Dataset.hpp"
#include "model/SVRParameters.hpp"
#include "common/compatibility.hpp"
#include "onlinesvr.hpp"
#include "appcontext.hpp"
#include "WScalingFactorService.hpp"

namespace svr {
namespace business {

tbb::concurrent_unordered_set<cached_iface *> cached_register::cr;
tbb::mutex cached_register::erase_mx;

calc_cache::calc_cache()
{}


double calc_cache::get_gamma(const datamodel::SVRParameters &params, const arma::mat &Z, const arma::mat &L)
{
    const auto k = std::tuple{
            params.get_input_queue_column_name(),
            params.get_decon_level(),
            params.get_step(),
            params.get_grad_level(),
            params.get_chunk_index(),
            common::hash_lambda(params.get_svr_kernel_param2()),
            common::hash_lambda(params.get_kernel_param3()),
            params.get_adjacent_levels(),
            arma::size(Z)};
    const auto prepare_f = [&Z, &L] { return datamodel::OnlineMIMOSVR::calc_gamma(Z, L); };
    return cached<DTYPE(k), DTYPE(prepare_f)>::get()(k, prepare_f);
}


arma::mat &calc_cache::get_cumulatives(const datamodel::SVRParameters &params, const arma::mat &features_t, const bpt::ptime &time)
{
    const auto k = std::tuple{
            params.get_input_queue_column_name(),
            params.get_grad_level(),
            params.get_chunk_index(),
            params.get_adjacent_levels(),
            arma::size(features_t),
            time};
    const auto prepare_f = [&params, &features_t] { return datamodel::OnlineMIMOSVR::all_cumulatives(params, features_t); };
    return *cached<DTYPE(k), DTYPE(prepare_f)>::get()(k, prepare_f);
}


arma::mat &calc_cache::get_Z(datamodel::SVRParameters &params, const arma::mat &features_t, const bpt::ptime &time)
{
    const auto k = std::tuple{
            params.get_input_queue_column_name(),
            params.get_grad_level(),
            params.get_chunk_index(),
            common::hash_lambda(params.get_svr_kernel_param2()),
            common::hash_lambda(params.get_kernel_param3()),
            params.get_adjacent_levels(),
            arma::size(features_t),
            time};
    const auto prepare_f = [&params, &features_t, &time, this]{
        const auto p_Z = datamodel::OnlineMIMOSVR::prepare_Z(*this, params, features_t, time);
        return std::tuple{params.get_min_Z(), params.get_max_Z(), p_Z};
    };
    const auto [min_Z, max_Z, p_Z] = cached<DTYPE(k), DTYPE(prepare_f)>::get()(k, prepare_f);
    params.set_min_Z(min_Z);
    params.set_max_Z(max_Z);
    return *p_Z;
}


arma::mat &calc_cache::get_Zy(const datamodel::SVRParameters &params, const arma::mat &features_t, const arma::mat &predict_features_t, const bpt::ptime &predict_time,
                                     const bpt::ptime &trained_time)
{
    const auto k = std::tuple{
            params.get_input_queue_column_name(),
            params.get_grad_level(),
            params.get_chunk_index(),
            common::hash_lambda(params.get_svr_kernel_param2()),
            common::hash_lambda(params.get_kernel_param3()),
            params.get_adjacent_levels(),
            arma::size(features_t),
            arma::size(predict_features_t),
            predict_time,
            trained_time};

    const auto prepare_f = [&params, &features_t, &predict_features_t, &predict_time, &trained_time, this] {
        return datamodel::OnlineMIMOSVR::prepare_Zy(*this, params, features_t, predict_features_t, predict_time, trained_time);
    };
    return *cached<DTYPE(k), DTYPE(prepare_f)>::get()(k, prepare_f);
}

arma::mat &calc_cache::get_K(datamodel::SVRParameters &params, const arma::mat &features_t, const bpt::ptime &time)
{
    const auto k = std::tuple{
            params.get_input_queue_column_name(),
            params.get_grad_level(),
            params.get_chunk_index(),
            common::hash_lambda(params.get_kernel_param3()),
            common::hash_lambda(params.get_svr_kernel_param2()),
            common::hash_lambda(params.get_svr_kernel_param()),
            params.get_adjacent_levels(),
            arma::size(features_t),
            time};

    const auto prepare_f = [&params, &features_t, &time, this](){ return datamodel::OnlineMIMOSVR::prepare_K(*this, params, features_t, time); };
    return *cached<DTYPE(k), DTYPE(prepare_f)>::get()(k, prepare_f);
}


std::tuple<mat_ptr, vec_ptr, data_row_container_ptr>
calc_cache::get_labels(const std::string &column_name, const uint16_t step, const datamodel::datarow_crange &main_data, const datamodel::datarow_crange &labels_aux,
                       const bpt::time_duration &max_gap, const uint16_t level, const uint16_t multistep, const bpt::time_duration &aux_queue_res, const bpt::ptime &last_modeled_value_time,
                       const bpt::time_duration &main_resolution, const uint16_t lag)
{
    const auto k = std::tuple{column_name, (**main_data.begin()).get_value_time(), main_data.distance(), level, main_resolution, aux_queue_res};
    LOG4_TRACE("Getting labels for " << column_name << " at " << last_modeled_value_time << " with " << main_data.distance() << " rows, level " << level << ", step " << step <<
        ", aux last values " << labels_aux.back()->to_string());
    const auto prepare_f = [&] {
        auto p_labels = ptr<arma::mat>();
        auto p_last_knowns = ptr<arma::vec>();
        auto p_label_times = ptr<data_row_container>();
        ModelService::prepare_labels(*p_labels, *p_last_knowns, *p_label_times, main_data, labels_aux, max_gap, level, aux_queue_res, last_modeled_value_time,
                                     main_resolution, multistep, lag, column_name.find("_ask") != std::string::npos);
        return std::tuple{p_labels, p_last_knowns, p_label_times};
    };
    const auto [p_labels, p_last_knowns, p_label_times] = cached<DTYPE(k), DTYPE(prepare_f)>::get()(k, prepare_f);
    LOG4_TRACE("For step " << step << ", column " << column_name << ", level returning labels " << common::present(p_labels->col(step)));
    return {ptr<arma::mat>(p_labels->col(step)), p_last_knowns, p_label_times};
}


mat_ptr
calc_cache::get_features(
        const arma::mat &labels, const std::deque<datamodel::DeconQueue_ptr> &aux_decon_queues, datamodel::SVRParameters &params, const bpt::time_duration &aux_resolution,
        const bpt::time_duration &main_resolution, const bpt::time_duration &max_lookback_time_gap, const data_row_container &label_times)
{
    const auto k = std::tuple{
        params.get_step(), params.get_input_queue_column_name(), label_times.front()->get_value_time(), label_times.back()->get_value_time(), label_times.size(),
        params.get_adjacent_levels(), params.get_lag_count(), aux_decon_queues.size(), main_resolution, aux_resolution};
    LOG4_TRACE("Getting features for with " << label_times.size() << " rows, parameters " << params << ", queues " << aux_decon_queues.size());
    const auto prepare_f = [&] {
        auto p_features = ptr<arma::mat>();
        if (params.get_feature_mechanics().needs_tuning()) {
            PROFILE_EXEC_TIME(
                    ModelService::tune_features(*p_features, labels, params, label_times, aux_decon_queues, max_lookback_time_gap, aux_resolution, main_resolution),
                    "Tune features for " << params);
        } else PROFILE_EXEC_TIME(
                ModelService::prepare_features(*p_features, label_times, aux_decon_queues, params, max_lookback_time_gap, aux_resolution, main_resolution),
                "Prepare features " << params);
        const auto p_feature_mechanics = ptr<datamodel::t_feature_mechanics>(params.get_feature_mechanics());
        return std::pair{p_features, p_feature_mechanics};
    };
    const auto [p_features, p_feature_mechanics] = cached<DTYPE(k), DTYPE(prepare_f)>::get()(k, prepare_f);
    if (params.get_feature_mechanics().needs_tuning()) params.set_feature_mechanics(*p_feature_mechanics);
    return p_features;
}

// Input weights
mat_ptr calc_cache::get_weights(
        const bigint dataset_id, const data_row_container &times, const std::deque<datamodel::InputQueue_ptr> &aux_inputs, const uint16_t step, const uint16_t steps,
        const bpt::time_duration &resolution_main)
{
    const auto k = std::tuple{aux_inputs.front()->get_table_name(), (**times.cbegin()).get_value_time(), times.size(), steps, resolution_main};
    const auto prepare_f = [&] {
        auto p_weights = ptr<arma::mat>();
        ModelService::prepare_weights(*p_weights, times, aux_inputs, steps, resolution_main);
        APP.w_scaling_factor_service.scale(dataset_id, *p_weights);
        return p_weights;
    };
    const auto p_weights = cached<DTYPE(k), DTYPE(prepare_f)>::get()(k, prepare_f);
    return ptr<arma::mat>(p_weights->col(step));
}

datamodel::t_parameter_predictions_set &calc_cache::checkin_tuner(const datamodel::OnlineMIMOSVR &svr, const uint16_t chunk_ix)
{
    bool rc;
    const auto tune_predictions_key = std::tuple{svr.get_params_ptr()->get_input_queue_column_name(), svr.get_step(), svr.get_gradient_level(), chunk_ix};
    const tbb::mutex::scoped_lock l(tuners_mx);
    auto tune_iter = tune_results.find(tune_predictions_key);
    if (tune_iter == tune_results.end()) {
        std::tie(tune_iter, rc) = tune_results.emplace(tune_predictions_key, levels_tune_data{{}, {}, {}, ptr<std::condition_variable>(), ptr<std::mutex>(), false});
        if (!rc) LOG4_THROW("Failed creating tune data for " << tune_predictions_key);
    }
    auto iter_counter = tune_iter->second.started_tuners.find(chunk_ix);
    if (iter_counter == tune_iter->second.started_tuners.cend()) {
        std::tie(iter_counter, rc) = tune_iter->second.started_tuners.emplace(chunk_ix, 0);
        if (!rc) LOG4_THROW("Failed creating counter for chunk " << chunk_ix);
    }
    ++iter_counter->second;
    const auto levix = svr.get_decon_level();
    auto level_iter = tune_iter->second.level_predictions.find(levix);
    if (level_iter == tune_iter->second.level_predictions.cend()) {
        std::tie(level_iter, rc) = tune_iter->second.level_predictions.emplace(levix, datamodel::t_parameter_predictions_set{});
        if (!rc) LOG4_THROW("Failed creating level predictions for " << levix);
    }
    return level_iter->second;
}


void calc_cache::checkout_tuner(const datamodel::OnlineMIMOSVR &svr, const uint16_t chunk_ix)
{
    bool rc;
    const auto tune_predictions_key = std::make_tuple(svr.get_params_ptr()->get_input_queue_column_name(), svr.get_step(), svr.get_gradient_level(), chunk_ix);
    const auto tune_iter = tune_results.find(tune_predictions_key);
    if (tune_iter == tune_results.end()) return;
    const tbb::mutex::scoped_lock l(tuners_mx);
    auto iter_counter = tune_iter->second.completed_tuners.find(chunk_ix);
    if (iter_counter == tune_iter->second.completed_tuners.cend()) {
        std::tie(iter_counter, rc) = tune_iter->second.completed_tuners.emplace(chunk_ix, 0);
        if (!rc) LOG4_THROW("Failed creating counter for chunk " << chunk_ix);
    }
    ++iter_counter->second;
    LOG4_DEBUG("Checking out tuner " << tune_predictions_key << ", started tuners " << tune_iter->second.started_tuners[chunk_ix] <<
        ", completed tuners " << tune_iter->second.completed_tuners[chunk_ix]);
    tune_iter->second.p_tuners_done->notify_all();
}

// Decrement distance and max chunk size of all levels need to be equal
datamodel::t_level_tuned_parameters *calc_cache::recombine_go(const datamodel::OnlineMIMOSVR &svr, const uint16_t chunk_ix)
{
    const auto tune_predictions_key = std::make_tuple(svr.get_params_ptr()->get_input_queue_column_name(), svr.get_step(), svr.get_gradient_level(), chunk_ix);
    const auto tune_iter = tune_results.find(tune_predictions_key);
    if (tune_iter == tune_results.end()) {
        LOG4_WARN("Tuner results for " << tune_predictions_key << " not found.");
        return nullptr;
    }
    std::unique_lock ul(*tune_iter->second.p_mx);
    tune_iter->second.p_tuners_done->wait(ul, [&chunk_ix, &tune_iter, &svr] {
        const bool res = tune_iter->second.started_tuners[chunk_ix] == tune_iter->second.completed_tuners[chunk_ix] &&
                         tune_iter->second.completed_tuners[chunk_ix] == ssize_t(svr.get_dataset()->get_model_count());
        LOG4_TRACE("Result " << res << ", started tuners " << tune_iter->second.started_tuners[chunk_ix] << ", completed tuners " <<
            tune_iter->second.completed_tuners[chunk_ix] << ", model count " << ssize_t(svr.get_dataset()->get_model_count()));
        return res;
    });
    if (tune_iter->second.recombining) {
        LOG4_DEBUG("Already recombining " << tune_predictions_key);
        ul.unlock();
        return nullptr;
    }
    tune_iter->second.recombining = true;
//        clear_tune_cache(svr.get_params_ptr()->get_input_queue_column_name()); // Cache is not used while tuning at present
    ul.unlock();
    LOG4_DEBUG("Recombine go " << tune_iter->second.level_predictions.size() << " models.");
    return &tune_iter->second.level_predictions;
}

void calc_cache::clear_tune_cache(const std::string &column_name)
{
    // Not needed for the time being
}

void calc_cache::clear()
{
    std::for_each(C_default_exec_policy, cached_register::cr.begin(), cached_register::cr.end(), [](auto &i) { i->clear(); });
}

}
}
