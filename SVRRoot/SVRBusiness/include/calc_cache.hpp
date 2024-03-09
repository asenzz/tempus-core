//
// Created by zarko on 2/21/24.
//

#ifndef SVR_CALC_CACHE_HPP
#define SVR_CALC_CACHE_HPP

#include <condition_variable>
#include </opt/intel/oneapi/tbb/latest/include/tbb/concurrent_unordered_map.h>
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
        size_t /* lag */, size_t /* decrement */, std::set<size_t> /* feature levels */, arma::SizeMat /* feature matrix size */>
        gamma_cache_key_t;
typedef std::unordered_map<gamma_cache_key_t, double> gamma_cache_t;


typedef std::tuple<
        std::string /* column */, size_t /* gradient */, size_t /* chunk */,
        size_t /* lag */, size_t /* decrement */, std::set<size_t> /* feature levels */, arma::SizeMat /* feature matrix size */, bpt::ptime>
        cml_cache_key_t;
typedef std::unordered_map<cml_cache_key_t, std::shared_ptr<std::deque<arma::mat>>> cml_cache_t;


typedef std::tuple<
        std::string /* column */, size_t /* gradient */, size_t /* chunk */, size_t /* lambda */,
        size_t /* lag */, size_t /* decrement */, std::set<size_t> /* feature levels */, arma::SizeMat /* feature matrix size */, bpt::ptime>
        Z_cache_key_t;
typedef std::unordered_map<Z_cache_key_t, matrix_ptr> Z_cache_t;


typedef std::tuple<
        std::string /* column */, size_t /* level */, size_t /* gradient */, size_t /* chunk */, size_t /* lambda */,
        size_t /* lag */, size_t /* decrement */, std::set<size_t> /* feature levels */, arma::SizeMat /* feature matrix size */, arma::SizeMat /* predict matrix size */, bpt::ptime>
        Zy_cache_key_t;
typedef std::unordered_map<Zy_cache_key_t, matrix_ptr> Zy_cache_t;


typedef std::tuple<size_t, std::set<size_t>, bpt::time_duration, bpt::ptime, bpt::time_duration, bpt::ptime, size_t> features_cache_key_t;
typedef std::unordered_map<features_cache_key_t, matrix_ptr> features_cache_t;


struct tune_data
{
    ssize_t started_tuners = 0, completed_tuners = 0;
    datamodel::t_tuned_parameters_ptr p_tune_predictions;
    std::shared_ptr<std::condition_variable> p_tuners_done;
    std::shared_ptr<std::mutex> p_mx;
    bool recombining = false;
};


class calc_cache
{
    std::mutex tuners_mx;

    gamma_cache_t gamma_cache;
    tbb::concurrent_unordered_map<gamma_cache_key_t, std::mutex> gamma_mx;

    cml_cache_t cml_cache;
    tbb::concurrent_unordered_map<cml_cache_key_t, std::mutex> cml_mx;

    Z_cache_t Z_cache;
    tbb::concurrent_unordered_map<Z_cache_key_t, std::mutex> Z_mx;

    Zy_cache_t Zy_cache;
    tbb::concurrent_unordered_map<Zy_cache_key_t, std::mutex> Zy_mx;

    features_cache_t features_cache;
    tbb::concurrent_unordered_map<features_cache_key_t, std::mutex> features_mx;

    std::unordered_map<std::string /* ensemble */, tune_data, std::hash<std::string>> tune_predictions;
    const datamodel::Dataset &dataset; // owner

public:
    explicit calc_cache(const datamodel::Dataset &dataset);

    double get_cached_gamma(const datamodel::SVRParameters &params, const arma::mat &Z, const double meanabs_labels);

    std::deque<arma::mat> &get_cached_cumulatives(const datamodel::SVRParameters &params, const arma::mat &features_t, const bpt::ptime &time);

    arma::mat &get_cached_Z(const datamodel::SVRParameters &params, const arma::mat &features_t, const bpt::ptime &time);

    arma::mat &get_cached_Zy(const datamodel::SVRParameters &params, const arma::mat &features_t, const arma::mat &predict_features_t, const bpt::ptime &time);

    matrix_ptr get_cached_features(
        const std::deque<bpt::ptime> &label_times, const std::deque<datamodel::DeconQueue_ptr> &features_aux, const size_t lag, const std::set<size_t> &adjacent_levels,
        const bpt::time_duration &max_gap, const bpt::time_duration &aux_queue_res, const bpt::time_duration &main_queue_resolution);

    datamodel::t_tuned_parameters &get_tuner_state(const std::string &column_name);

    void clear_tune_cache(const std::string &column_name);

    void clear();

    datamodel::t_tuned_parameters_ptr checkin_tuner(const datamodel::OnlineMIMOSVR &svr);

    void checkout_tuner(const datamodel::OnlineMIMOSVR &svr);

    bool recombine_go(const datamodel::OnlineMIMOSVR &svr);

    datamodel::t_param_set get_best_parameters(const std::string &column_name, const size_t decon_level, const size_t gradient_level, const size_t num_chunks);

    template<typename rT, typename kT, typename fT> rT cached(std::unordered_map<kT, rT> &cache_cont, tbb::concurrent_unordered_map<kT, std::mutex> &mx, const kT &cache_key, const fT &f);
};

typedef std::shared_ptr<calc_cache> t_calc_cache_ptr;

}
}

#endif //SVR_CALC_CACHE_HPP
