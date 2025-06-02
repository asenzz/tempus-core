//
// Created by zarko on 2/21/24.
//

#ifndef SVR_CALC_CACHE_HPP
#define SVR_CALC_CACHE_HPP

#include <condition_variable>
#include <oneapi/tbb/concurrent_unordered_map.h>
#include <oneapi/tbb/mutex.h>
#include <memory>
#include <deque>
#include <armadillo>
#include <string>
#include <tuple>
#include <mutex>
#include <boost/unordered/unordered_flat_map.hpp>
#include "model/SVRParameters.hpp"
#include "onlinesvr.hpp"
#include "model/InputQueue.hpp"

#define BYPASS_CALC_CACHE

namespace svr {

namespace kernel {

template<typename T> class kernel_base;

template<typename T> class kernel_path;

}

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
    std::shared_ptr<std::condition_variable> p_tuners_done;
    std::shared_ptr<std::mutex> p_mx;
    bool recombining = false;
};

class calc_cache
{
    tbb::mutex tuners_mx;
    boost::unordered_flat_map<std::tuple<std::string /* ensemble */, uint16_t /* step */, uint16_t /* gradient */, uint16_t /* chunk */>, levels_tune_data> tune_results;

public:
    explicit calc_cache() = default;

    template<typename T> arma::Mat<T> get_Zy(
            const kernel::kernel_base<T> &kernel_ftor, const arma::Mat<T> &X, const arma::Mat<T> &Xy, const bpt::ptime &X_time, const bpt::ptime &Xy_time);

    template<typename T> arma::Mat<T> get_Ky(
            const kernel::kernel_base<T> &kernel_ftor, const arma::Mat<T> &X, const arma::Mat<T> &Xy, const bpt::ptime &time_X, const bpt::ptime &time_Xy);

    std::tuple<mat_ptr, vec_ptr, data_row_container_ptr> get_labels(
            const std::string &column_name, const uint16_t step, const datamodel::datarow_crange &main_data, const datamodel::datarow_crange &labels_aux,
            const bpt::time_duration &max_gap, const uint16_t level, const uint16_t multistep, const bpt::time_duration &aux_queue_res, const bpt::ptime &last_modeled_value_time,
            const bpt::time_duration &main_queue_resolution, const uint16_t lag);

    mat_ptr get_features(
            const arma::mat &labels, const std::deque<datamodel::DeconQueue_ptr> &aux_decon_queues, datamodel::SVRParameters &params, const bpt::time_duration &aux_resolution,
            const bpt::time_duration &main_resolution, const bpt::time_duration &max_lookback_time_gap, const data_row_container &label_times);

    mat_ptr get_weights(
            const bigint dataset_id, const data_row_container &times, const std::deque<datamodel::InputQueue_ptr> &aux_inputs, const uint16_t step, const uint16_t steps,
            const bpt::time_duration &resolution_main);

    void clear_tune_cache(const std::string &column_name);

    void clear();
};

typedef std::shared_ptr<calc_cache> t_calc_cache_ptr;

}
}

#endif //SVR_CALC_CACHE_HPP
