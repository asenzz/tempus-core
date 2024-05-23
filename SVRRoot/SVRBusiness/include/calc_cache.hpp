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

typedef std::tuple<
        std::string /* column */, size_t /* level */, size_t /* gradient */, size_t /* chunk */, double /* lambda */,
        std::set<size_t> /* feature levels */, arma::SizeMat /* feature matrix size */>
        gamma_cache_key_t;
typedef std::unordered_map<gamma_cache_key_t, double> gamma_cache_t;


typedef std::tuple<
        size_t /* gradient */, size_t /* chunk */,
        std::set<size_t> /* feature levels */, arma::SizeMat /* feature matrix size */, bpt::ptime>
        cml_cache_key_t;
typedef std::unordered_map<cml_cache_key_t, std::shared_ptr<arma::mat>> cml_cache_t;


typedef std::tuple<
        size_t /* gradient */, size_t /* chunk */, size_t /* lambda */,
        std::set<size_t> /* feature levels */, arma::SizeMat /* feature matrix size */, bpt::ptime>
        Z_cache_key_t;
typedef std::unordered_map<Z_cache_key_t, mat_ptr> Z_cache_t;


typedef std::tuple<
        size_t /* gradient */, size_t /* chunk */, size_t /* lambda */,
        std::set<size_t> /* feature levels */, arma::SizeMat /* feature matrix size */, arma::SizeMat /* predict matrix size */, bpt::ptime>
        Zy_cache_key_t;
typedef std::unordered_map<Zy_cache_key_t, mat_ptr> Zy_cache_t;


typedef std::tuple<
        size_t /* gradient */, size_t /* chunk */, size_t /* lambda */, size_t /* gamma */,
        std::set<size_t> /* feature levels */, arma::SizeMat /* feature matrix size */, bpt::ptime>
        K_cache_key_t;
typedef std::unordered_map<K_cache_key_t, mat_ptr> K_cache_t;


typedef std::tuple<size_t, size_t, std::set<size_t>, bpt::time_duration, bpt::ptime, bpt::time_duration, bpt::ptime, size_t> features_cache_key_t;
typedef std::unordered_map<features_cache_key_t, mat_ptr> features_cache_t;


struct levels_tune_data
{
    ssize_t started_tuners = 0, completed_tuners = 0;
    datamodel::t_level_tuned_parameters level_predictions;
    std::shared_ptr<std::condition_variable> p_tuners_done;
    std::shared_ptr<std::mutex> p_mx;
    bool recombining = false;
};

struct cached_iface {
    virtual void clear() = 0;
};

struct cached_register
{
    static std::mutex erase_mx;
    static tbb::concurrent_unordered_set<cached_iface *> cr;
};

template<typename rT, typename kT, typename fT> struct cached : cached_iface
{
    static cached<rT, kT, fT> &get();

    static std::unordered_map<kT, rT> cache_cont;
    static tbb::concurrent_unordered_map<kT, std::mutex> mx_map;

    cached();
    ~cached();
    void clear() override;
    rT &operator() (const kT &cache_key, const fT &f);
};

class calc_cache
{
    std::mutex tuners_mx;
    std::unordered_map<std::tuple<std::string /* ensemble */, size_t /* gradient */, size_t /* chunk */>, levels_tune_data> tune_results;

public:
    explicit calc_cache();

    double get_cached_gamma(const datamodel::SVRParameters &params, const arma::mat &Z, const double meanabs_labels);

    arma::mat &get_cached_cumulatives(const datamodel::SVRParameters &params, const arma::mat &features_t, const bpt::ptime &time);

    arma::mat &get_cached_Z(const datamodel::SVRParameters &params, const arma::mat &features_t, const bpt::ptime &time);

    arma::mat &get_cached_Zy(const datamodel::SVRParameters &params, const arma::mat &features_t, const arma::mat &predict_features_t, const bpt::ptime &time);

    arma::mat &get_cached_K(const datamodel::SVRParameters &params, const arma::mat &features_t, const bpt::ptime &time);

    mat_ptr get_cached_features(
        const std::deque<bpt::ptime> &label_times, const std::deque<datamodel::DeconQueue_ptr> &features_aux, const size_t lag, const double quantize,
        const std::set<size_t> &adjacent_levels, const bpt::time_duration &max_gap, const bpt::time_duration &aux_queue_res, const bpt::time_duration &main_queue_resolution);

    void clear_tune_cache(const std::string &column_name);

    void clear();

    datamodel::t_parameter_predictions_set &checkin_tuner(const datamodel::OnlineMIMOSVR &svr, const size_t chunk_ix);

    void checkout_tuner(const datamodel::OnlineMIMOSVR &svr, const size_t chunk_ix);

    const datamodel::t_level_tuned_parameters *recombine_go(const datamodel::OnlineMIMOSVR &svr, const size_t chunk_ix);
};

typedef std::shared_ptr<calc_cache> t_calc_cache_ptr;

}
}

#endif //SVR_CALC_CACHE_HPP
