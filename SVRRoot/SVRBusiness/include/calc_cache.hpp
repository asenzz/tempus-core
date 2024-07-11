//
// Created by zarko on 2/21/24.
//

#ifndef SVR_CALC_CACHE_HPP
#define SVR_CALC_CACHE_HPP

#include <condition_variable>
#include </opt/intel/oneapi/tbb/latest/include/tbb/concurrent_unordered_map.h>
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
#include "model/DeconQueue.hpp"

namespace svr {
namespace datamodel {

class Dataset;

using Dataset_ptr = std::shared_ptr<Dataset>;

class SVRParameters;

using SVRParameters_ptr = std::shared_ptr<SVRParameters>;
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
    std::unordered_map<std::tuple<std::string /* ensemble */, size_t /* step */, size_t /* gradient */, size_t /* chunk */>, levels_tune_data> tune_results;

public:
    explicit calc_cache();

    double get_cached_gamma(const datamodel::SVRParameters &params, const arma::mat &Z, const double meanabs_labels);

    arma::mat &get_cached_cumulatives(const datamodel::SVRParameters &params, const arma::mat &features_t, const bpt::ptime &time);

    arma::mat &get_cached_Z(const datamodel::SVRParameters &params, const arma::mat &features_t, const bpt::ptime &time);

    arma::mat &get_cached_Zy(const datamodel::SVRParameters &params, const arma::mat &features_t, const arma::mat &predict_features_t, const bpt::ptime &time,
                             const bpt::ptime &trained_time);

    arma::mat &get_cached_K(const datamodel::SVRParameters &params, const arma::mat &features_t, const bpt::ptime &time);

    std::tuple<mat_ptr, vec_ptr, times_ptr> get_cached_labels(
            const size_t step, const datamodel::datarow_crange &main_data, const datamodel::datarow_crange &labels_aux, const bpt::time_duration &max_gap, const size_t level,
            const bpt::time_duration &aux_queue_res, const bpt::ptime &last_modeled_value_time, const bpt::time_duration &main_queue_resolution, const size_t multistep);

    void clear_tune_cache(const std::string &column_name);

    void clear();

    datamodel::t_parameter_predictions_set &checkin_tuner(const datamodel::OnlineMIMOSVR &svr, const size_t chunk_ix);

    void checkout_tuner(const datamodel::OnlineMIMOSVR &svr, const size_t chunk_ix);

    datamodel::t_level_tuned_parameters *recombine_go(const datamodel::OnlineMIMOSVR &svr, const size_t chunk_ix);
};

typedef std::shared_ptr<calc_cache> t_calc_cache_ptr;

}
}

#endif //SVR_CALC_CACHE_HPP
