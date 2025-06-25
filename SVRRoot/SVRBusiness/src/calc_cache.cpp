//
// Created by zarko on 2/21/24.
//
#include <boost/unordered/unordered_flat_map.hpp>
#include <highwayhash/highwayhash.h>
#include "common/types.hpp"
#include "model/Dataset.hpp"
#include "model/SVRParameters.hpp"
#include "onlinesvr.hpp"
#include "appcontext.hpp"
#include "WScalingFactorService.hpp"
#include "kernel_path.hpp"
#include "kernel_factory.hpp"
#include "common/compatibility.hpp"

const uint16_t C_cached_tuple_len_limit = 0; // 1 or less disables caching // TODO Fix buggy, cached calculation are giving different outcomes

namespace svr {
namespace business {
struct cached_iface
{
    virtual void clear() = 0;
};

struct cached_register
{
    static tbb::mutex erase_mx;
    static tbb::concurrent_unordered_set<cached_iface *> cr;
};

tbb::concurrent_unordered_set<cached_iface *> cached_register::cr;
tbb::mutex cached_register::erase_mx;

template<typename kT, typename fT> struct cached : cached_iface
{
    using rT = typename return_type<fT>::type;

    static constexpr uint32_t reserved_cache_size = 1000;

    static cached<kT, fT> &get();

    static boost::unordered_flat_map<kT, rT, common::hash_tuple> cache_cont;

    cached();

    ~cached();

    void clear() override;

    rT operator()(const kT &cache_key, const fT &f);
};


// cached functor implementation
#define __rT typename cached<kT, fT>::rT

template<typename kT, typename fT> boost::unordered_flat_map<kT, __rT, common::hash_tuple> cached<kT, fT>::cache_cont;

template<typename kT, typename fT> __rT cached<kT, fT>::operator()(const kT &cache_key, const fT &f)
{
    constexpr auto key_len = std::tuple_size_v<kT>;
    if constexpr(key_len >= C_cached_tuple_len_limit) {
        LOG4_TRACE("Key complexity " << key_len << " is higher or equal to " << C_cached_tuple_len_limit << ", not caching");
        return f();
    }
    LOG4_TRACE("Looking for " << cache_key << " in cache");

    auto iter = cache_cont.find(cache_key); // TODO Risky remove this it may cause issues with concurrent access
    if (iter != cache_cont.cend()) goto __bail;
    {
        static tbb::mutex mx;
        const tbb::mutex::scoped_lock lk(mx);
        LOG4_TRACE("Looking for " << cache_key << " in cache");
        iter = cache_cont.find(cache_key);
        if (iter != cache_cont.cend()) goto __bail;
        {
            PROFILE_BEGIN;
            const auto r = f();
            PROFILE_END("Preparing cached key " << cache_key);

            if (cache_cont.empty()) cache_cont.reserve(reserved_cache_size);
            if (cache_cont.size() + 1 >= common::capacity(cache_cont))
                LOG4_THROW("Cache container for key " << cache_key << ", size " << cache_cont.size() << " exceeds capacity " << common::capacity(cache_cont));

            bool rc;
            std::tie(iter, rc) = cache_cont.emplace(cache_key, r);
            if (!rc)
                LOG4_THROW("Error inserting entry in cache");
            LOG4_TRACE("Inserted cached key " << cache_key);
        }
    }
__bail:
    LOG4_TRACE("Returning element for " << cache_key);
    return iter->second;
}

template<typename kT, typename fT> void cached<kT, fT>::clear()
{
    RELEASE_CONT(cache_cont);
}

template<typename kT, typename fT> cached<kT, fT>::cached()
{
    cached_register::cr.emplace(this);
}

template<typename kT, typename fT> cached<kT, fT> &cached<kT, fT>::get()
{
    static cached<kT, fT> o;
    return o;
}

template<typename kT, typename fT> cached<kT, fT>::~cached()
{
    const tbb::mutex::scoped_lock l(cached_register::erase_mx);
    cached_register::cr.unsafe_erase(this);
}

// calc_cache
std::tuple<mat_ptr, data_row_container_ptr, data_row_container_ptr> calc_cache::get_manifold_labels(
    datamodel::Model &model, const std::string &column_name, const uint16_t step, const datamodel::datarow_crange &main_data, const datamodel::datarow_crange &labels_aux,
    const bpt::time_duration &max_gap, const uint16_t level, const uint16_t multistep, const bpt::time_duration &aux_queue_res, const bpt::ptime &last_modeled_value_time,
    const bpt::time_duration &main_resolution, const uint16_t lag)
{
    LOG4_TRACE("Getting labels for " << column_name << " at " << last_modeled_value_time << " with " << main_data.distance() << " rows, level " << level << ", step " << step <<
        ", aux last values " << labels_aux.back()->to_string());
    const auto prepare_f = [&] {
        auto p_labels = ptr<arma::mat>();
        auto p_label_times_left = ptr<data_row_container>();
        auto p_label_times_right = ptr<data_row_container>();
        ModelService::prepare_manifold_labels(model, *p_labels, *p_label_times_left, *p_label_times_right, main_data, labels_aux, max_gap, level, aux_queue_res, last_modeled_value_time,
            main_resolution, multistep, lag);
        return std::make_tuple(p_labels, p_label_times_left, p_label_times_right);
    };
    const auto k = std::make_tuple(column_name, (*main_data.cbegin())->get_value_time(), main_data.distance(), level, main_resolution, aux_queue_res);
    const auto [p_labels, p_label_times_left, p_label_times_right] = cached<DTYPE(k), DTYPE(prepare_f) >::get()(k, prepare_f);
    return {ptr<arma::mat>(p_labels->col(step)), p_label_times_left, p_label_times_right};
}

std::tuple<mat_ptr, vec_ptr, data_row_container_ptr> calc_cache::get_labels(
    const std::string &column_name, const uint16_t step, const datamodel::datarow_crange &main_data, const datamodel::datarow_crange &labels_aux,
    const bpt::time_duration &max_gap, const uint16_t level, const uint16_t multistep, const bpt::time_duration &aux_queue_res,
    const bpt::ptime &last_modeled_value_time, const bpt::time_duration &main_resolution, const uint16_t lag)
{
    LOG4_TRACE("Getting labels for " << column_name << " at " << last_modeled_value_time << " with " << main_data.distance() << " rows, level " << level << ", step " << step <<
        ", aux last values " << labels_aux.back()->to_string());
    const auto prepare_f = [&] {
        auto p_labels = ptr<arma::mat>();
        auto p_last_knowns = ptr<arma::vec>();
        auto p_label_times = ptr<data_row_container>();
        ModelService::prepare_labels(*p_labels, *p_last_knowns, *p_label_times, main_data, labels_aux, max_gap, level, aux_queue_res, last_modeled_value_time,
                                     main_resolution, multistep, lag);
        return std::make_tuple(p_labels, p_last_knowns, p_label_times);
    };
    const auto k = std::make_tuple(column_name, (*main_data.cbegin())->get_value_time(), main_data.distance(), level, main_resolution, aux_queue_res);
    const auto [p_labels, p_last_knowns, p_label_times] = cached<DTYPE(k), DTYPE(prepare_f) >::get()(k, prepare_f);
    return {ptr<arma::mat>(p_labels->col(step)), p_last_knowns, p_label_times};
}


mat_ptr calc_cache::get_features(
    const arma::mat &labels, const std::deque<datamodel::DeconQueue_ptr> &aux_decon_queues, datamodel::SVRParameters &params, const bpt::time_duration &aux_resolution,
    const bpt::time_duration &main_resolution, const bpt::time_duration &max_lookback_time_gap, const data_row_container &label_times)
{
    LOG4_TRACE("Getting features for with " << label_times.size() << " rows, parameters " << params << ", queues " << aux_decon_queues.size());
    const auto needs_tuning = params.get_feature_mechanics().needs_tuning();
    const auto prepare_f = [&, needs_tuning] {
        auto p_features = ptr<arma::mat>();
        if (needs_tuning) {
            PROFILE_MSG(
                ModelService::tune_features(*p_features, labels, params, label_times, aux_decon_queues, max_lookback_time_gap, aux_resolution, main_resolution),
                "Tune features for " << params);
        } else PROFILE_MSG(
            ModelService::prepare_features(*p_features, label_times, aux_decon_queues, params, max_lookback_time_gap, aux_resolution, main_resolution),
            "Prepare features " << params);
        const auto p_feature_mechanics = ptr<datamodel::t_feature_mechanics>(params.get_feature_mechanics());
        return std::make_pair(p_features, p_feature_mechanics);
    };
    const auto k = std::make_tuple(
        params.get_step(), params.get_input_queue_column_name(), label_times.front()->get_value_time(), label_times.back()->get_value_time(), label_times.size(),
        params.get_adjacent_levels(), params.get_lag_count(), aux_decon_queues.size(), main_resolution, aux_resolution);

    const auto [p_features, p_feature_mechanics] = cached<DTYPE(k), DTYPE(prepare_f) >::get()(k, prepare_f);
    if (needs_tuning) params.set_feature_mechanics(*p_feature_mechanics);
    return p_features;
}

// Input weights
mat_ptr calc_cache::get_weights(
    const bigint dataset_id, const data_row_container &times, const std::deque<datamodel::InputQueue_ptr> &aux_inputs, const uint16_t step, const uint16_t steps,
    const bpt::time_duration &resolution_main)
{
    LOG4_BEGIN();

    const auto prepare_f = [&] {
        auto p_weights = ptr<arma::mat>();
        ModelService::prepare_weights(*p_weights, times, aux_inputs, steps, resolution_main);
        APP.w_scaling_factor_service.scale(dataset_id, *p_weights);
        return p_weights;
    };
    const auto k = std::make_tuple(aux_inputs.front()->get_table_name(), (**times.cbegin()).get_value_time(), times.size(), steps, resolution_main);
    const auto p_weights = cached<DTYPE(k), DTYPE(prepare_f) >::get()(k, prepare_f);

    LOG4_END();

    return ptr<arma::mat>(p_weights->col(step));
}

[[maybe_unused]] void calc_cache::clear_tune_cache(const std::string &column_name)
{
    // Not needed for the time being
}

void calc_cache::clear()
{
    std::for_each(C_default_exec_policy, cached_register::cr.begin(), cached_register::cr.end(), [](auto &i) { i->clear(); });
}

#define T double

template<> arma::Mat<T> calc_cache::get_Zy(const kernel::kernel_base<T> &kernel_ftor, const arma::Mat<T> &X, const arma::Mat<T> &Xy, const bpt::ptime &X_time, const bpt::ptime &Xy_time)
{
    const auto &params = kernel_ftor.get_parameters();
    LOG4_TRACE("Cache look up " << params);
    const auto k = std::make_tuple(
        params.get_kernel_type(),
        params.get_lag_count(),
        params.get_input_queue_column_name(),
        params.get_grad_level(),
        params.get_chunk_index(),
        common::hash_lambda(params.get_svr_kernel_param2()),
        common::hash_lambda(params.get_kernel_param3()),
        params.get_adjacent_levels(),
        arma::size(X),
        arma::size(Xy),
        X_time,
        Xy_time);

    const auto prepare_f = [&params, &X, &Xy, &kernel_ftor] {
        switch (params.get_kernel_type()) {
            case datamodel::e_kernel_type::DEEP_PATH:
                LOG4_THROW("Unhandled kernel type " << params.get_kernel_type());

            default:
                const auto Zy = kernel_ftor.distances(X, Xy);
                return ptr<arma::Mat<T> >(Zy);
        }
    };

    LOG4_END();

    return *cached<DTYPE(k), DTYPE(prepare_f) >::get()(k, prepare_f);
}


template<> arma::Mat<T> calc_cache::get_Ky(const kernel::kernel_base<T> &kernel_ftor, const arma::Mat<T> &X, const arma::Mat<T> &Xy, const bpt::ptime &time_X, const bpt::ptime &time_Xy)
{
    const auto &params = kernel_ftor.get_parameters();
    LOG4_TRACE("Cache look up " << params);
    const auto k = std::make_tuple(
        params.get_kernel_type(),
        params.get_lag_count(),
        params.get_input_queue_column_name(),
        params.get_grad_level(),
        params.get_chunk_index(),
        common::hash_lambda(params.get_kernel_param3()),
        common::hash_lambda(params.get_svr_kernel_param2()),
        common::hash_lambda(params.get_svr_kernel_param()),
        params.get_adjacent_levels(),
        arma::size(X),
        arma::size(Xy),
        time_X,
        time_Xy);

    const auto prepare_f = [&params, &X, &Xy, &time_X, &time_Xy, &kernel_ftor, this] {
        switch (params.get_kernel_type()) {
            case datamodel::e_kernel_type::TFT:
            case datamodel::e_kernel_type::GBM:
            case datamodel::e_kernel_type::DEEP_PATH:
                return ptr(kernel_ftor.kernel(X, Xy));
            default:
                const auto Zy = get_Zy(kernel_ftor, X, Xy, time_X, time_Xy);
                const auto K = kernel_ftor.kernel_from_distances(Zy);
                return ptr(K);
        }
    };

    LOG4_END();
    return *cached<DTYPE(k), DTYPE(prepare_f) >::get()(k, prepare_f);
}

#undef T
}
}
