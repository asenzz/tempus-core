//
// Created by zarko on 2/21/24.
//

#ifndef SVR_CALC_CACHE_HPP
#define SVR_CALC_CACHE_HPP

#include <condition_variable>
#include <oneapi/tbb/concurrent_unordered_map.h>
#include <oneapi/tbb/mutex.h>
#include <functional>
#include <memory>
#include <set>
#include <deque>
#include <armadillo>
#include <string>
#include <tuple>
#include <mutex>
#include <unordered_map>
#include "model/SVRParameters.hpp"
#include "onlinesvr.hpp"
#include "model/InputQueue.hpp"

namespace svr {
namespace datamodel {

class Dataset;

using Dataset_ptr = std::shared_ptr<Dataset>;

class SVRParameters;

using SVRParameters_ptr = std::shared_ptr<SVRParameters>;

class DeconQueue;

using DeconQueue_ptr = std::shared_ptr<DeconQueue>;

}

namespace business {

struct levels_tune_data
{
    tbb::concurrent_unordered_map<size_t, ssize_t> started_tuners, completed_tuners;
    datamodel::t_level_tuned_parameters level_predictions;
    std::shared_ptr<std::condition_variable> p_tuners_done;
    std::shared_ptr<std::mutex> p_mx;
    bool recombining = false;
};

struct cached_iface
{
    virtual void clear() = 0;
};

struct cached_register
{
    static tbb::mutex erase_mx;
    static tbb::concurrent_unordered_set<cached_iface *> cr;
};

template<typename kT, typename fT>
struct cached : cached_iface
{
    using rT = typename return_type<fT>::type;

    static cached<kT, fT> &get();

    static std::unordered_map<kT, rT> cache_cont;
    static tbb::concurrent_unordered_map<kT, tbb::mutex> mx_map;

    cached();

    ~cached();

    void clear() override;

    rT &operator()(const kT &cache_key, const fT &f);
};

class calc_cache
{
    tbb::mutex tuners_mx;
    std::unordered_map<std::tuple<std::string /* ensemble */, uint16_t /* step */, uint16_t /* gradient */, uint16_t /* chunk */>, levels_tune_data> tune_results;

public:
    explicit calc_cache();

    double get_gamma(const datamodel::SVRParameters &params, const arma::mat &Z, const arma::mat &L);

    arma::mat &get_cumulatives(const datamodel::SVRParameters &params, const arma::mat &features_t, const bpt::ptime &time);

    arma::mat &get_Z(datamodel::SVRParameters &params, const arma::mat &features_t, const bpt::ptime &time);

    arma::mat &get_Zy(const datamodel::SVRParameters &params, const arma::mat &features_t, const arma::mat &predict_features_t, const bpt::ptime &predict_time,
                             const bpt::ptime &trained_time);

    arma::mat &get_K(datamodel::SVRParameters &params, const arma::mat &features_t, const bpt::ptime &time);

    std::tuple<mat_ptr, vec_ptr, data_row_container_ptr>
    get_labels(
            const std::string &column_name, const uint16_t step, const datamodel::datarow_crange &main_data, const datamodel::datarow_crange &labels_aux,
            const bpt::time_duration &max_gap, const uint16_t level, const uint16_t multistep, const bpt::time_duration &aux_queue_res, const bpt::ptime &last_modeled_value_time,
            const bpt::time_duration &main_queue_resolution, const uint16_t lag);

    mat_ptr
    get_features(
            const arma::mat &labels, const std::deque<datamodel::DeconQueue_ptr> &aux_decon_queues, datamodel::SVRParameters &params, const bpt::time_duration &aux_resolution,
            const bpt::time_duration &main_resolution, const bpt::time_duration &max_lookback_time_gap, const data_row_container &label_times);

    mat_ptr get_weights(
            const bigint dataset_id, const data_row_container &times, const std::deque<datamodel::InputQueue_ptr> &aux_inputs, const uint16_t step, const uint16_t steps,
            const bpt::time_duration &resolution_main);

    void clear_tune_cache(const std::string &column_name);

    void clear();

    datamodel::t_parameter_predictions_set &checkin_tuner(const datamodel::OnlineMIMOSVR &svr, const uint16_t chunk_ix);

    void checkout_tuner(const datamodel::OnlineMIMOSVR &svr, const uint16_t chunk_ix);

    datamodel::t_level_tuned_parameters *recombine_go(const datamodel::OnlineMIMOSVR &svr, const uint16_t chunk_ix);
};

typedef std::shared_ptr<calc_cache> t_calc_cache_ptr;

}
}

#endif //SVR_CALC_CACHE_HPP
