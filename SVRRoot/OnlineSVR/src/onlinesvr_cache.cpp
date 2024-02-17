//
// Created by zarko on 2/6/24.
//

#ifndef SVR_ONLINESVR_CACHE_CPP
#define SVR_ONLINESVR_CACHE_CPP


#include "onlinesvr.hpp"
#include "common/compatibility.hpp"

namespace svr {
// TODO Create an ensemble::cache class and move the above and below methods there
namespace {

typedef std::tuple<
        size_t /* dataset id */, size_t /* chunk */, size_t /* grad */, std::string /* queue name */, std::string /* column */, size_t /* lambda */,
        size_t /* lag */, size_t /* decrement */, size_t /* n samples */ > Z_cache_key_t;
typedef std::map<Z_cache_key_t, arma::mat *> Z_cache_t;
Z_cache_t Z_cache;
std::mutex Z_mx;


typedef std::tuple<std::string /* queue name */, std::string /* column */, size_t /* lambda */, size_t /* lag */, size_t /* decrement */,
        size_t /* p_predictions count */, bpt::ptime /* prediction time */, size_t /* level */, size_t /* chunk */, size_t /* gradient */> Zy_cache_key_t;
typedef std::map<Zy_cache_key_t, arma::mat *> Zy_cache_t;
Zy_cache_t Zy_cache;
std::mutex Zy_mx;


typedef std::tuple<std::string /* queue name */, std::string /* column */, size_t /* lag */, size_t /* decrement */, bpt::ptime /* pred time */, size_t /* chunk */,
        size_t /* gradient */ > cml_cache_key_t;
typedef std::map<cml_cache_key_t, std::deque<arma::mat> *> cml_cache_t;
cml_cache_t cml_cache;
std::mutex cml_mx;


typedef std::tuple<
        size_t /* train len */, size_t /* level */, size_t /* chunk */, size_t /* grad */, size_t /* dataset id */, std::string /* queue name */, std::string /* column */,
        size_t /* lambda */, size_t /* lag */, size_t /* decrement */, size_t /* n samples */> gamma_cache_key_t;
typedef std::map<gamma_cache_key_t, double> gamma_cache_t;
gamma_cache_t gamma_cache;
std::mutex gamma_mx;


template<typename rT, typename kT, typename fT>
rT cached(std::map<kT, rT> &cache_cont, std::mutex &mx, const kT &cache_key, const fT &f)
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

}

std::deque<arma::mat> &OnlineMIMOSVR::get_cached_cumulatives(const datamodel::SVRParameters &params, const arma::mat &features_t, const bpt::ptime &pred_time)
{
    const cml_cache_key_t params_key{
            params.get_input_queue_table_name(),
            params.get_input_queue_column_name(),
            params.get_lag_count(),
            features_t.n_cols,
            pred_time,
            params.get_chunk_ix(),
            params.get_grad_level()};

    const auto prepare_f = [&params, &features_t]() { return prepare_cumulatives(params, features_t); };
    return *cached(cml_cache, cml_mx, params_key, prepare_f);
}

double OnlineMIMOSVR::get_cached_gamma(const datamodel::SVRParameters &params, const arma::mat &Z, const size_t train_len, const double meanabs_labels)
{
    const gamma_cache_key_t params_key{
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

    const auto prepare_f = [&Z, &train_len, &meanabs_labels]() { return calc_gamma(Z, train_len, meanabs_labels); };
    return cached(gamma_cache, gamma_mx, params_key, prepare_f);
}


arma::mat &OnlineMIMOSVR::get_cached_Z(const datamodel::SVRParameters &params, const arma::mat &features_t, const size_t short_size_X)
{
    const Z_cache_key_t params_key{
            params.get_dataset_id(),
            params.get_chunk_ix(),
            params.get_grad_level(),
            params.get_input_queue_table_name(),
            params.get_input_queue_column_name(),
            std::round(1e5 * params.get_svr_kernel_param2()),
            params.get_lag_count(),
            params.get_svr_decremental_distance(),
            features_t.n_cols};

    const auto prepare_f = [&params, &features_t, &short_size_X]() { return prepare_Z(params, features_t, short_size_X); };
    return *cached(Z_cache, Z_mx, params_key, prepare_f);
}


arma::mat &OnlineMIMOSVR::get_cached_Zy(const datamodel::SVRParameters &params, const arma::mat &features_t /* transposed */, const arma::mat &predict_features_t /* transposed */,
                                        const bpt::ptime &pred_time)
{
    const Zy_cache_key_t params_key{
            params.get_input_queue_table_name(),
            params.get_input_queue_column_name(),
            std::round(1e5 * params.get_svr_kernel_param2()),
            params.get_lag_count(),
            features_t.n_cols,
            predict_features_t.n_cols,
            pred_time,
            params.get_decon_level(),
            params.get_chunk_ix(),
            params.get_grad_level()};

    const auto prepare_f = [&params, &features_t, &predict_features_t, &pred_time]() { return prepare_Zy(params, features_t, predict_features_t, pred_time); };
    return *cached(Zy_cache, Zy_mx, params_key, prepare_f);
}


void OnlineMIMOSVR::clear_gradient_caches(const datamodel::SVRParameters &p)
{
#pragma omp parallel num_threads(adj_threads(4))
    {
#pragma omp task
        { // Cumulatives cache
            const std::scoped_lock sl(cml_mx);
            remove_if(cml_cache, [&p](auto &e) {
                return std::get<0>(e.first) == p.get_input_queue_table_name()
                       && std::get<1>(e.first) == p.get_input_queue_column_name()
                       && std::get<6>(e.first) == p.get_grad_level();
            });
        }
#pragma omp task
        { // Distance kernel matrix
            const std::scoped_lock sl(Z_mx);
            remove_if(Z_cache, [&p](auto &e) {
                return (p.get_dataset_id() && std::get<0>(e.first) && std::get<0>(e.first) == p.get_dataset_id())
                       && std::get<2>(e.first) == p.get_grad_level()
                       && std::get<3>(e.first) == p.get_input_queue_table_name()
                       && std::get<4>(e.first) == p.get_input_queue_column_name();
            });
        }
#pragma omp task
        { // Distance predict matrix
            const std::scoped_lock sl(Zy_mx);
            remove_if(Zy_cache, [&p](auto &e) {
                return std::get<0>(e.first) == p.get_input_queue_table_name()
                       && std::get<1>(e.first) == p.get_input_queue_column_name()
                       && std::get<7>(e.first) == p.get_decon_level()
                       && std::get<9>(e.first) == p.get_grad_level();
            });
        }
#pragma omp task
        {
            const std::scoped_lock sl(gamma_mx);
            remove_if(gamma_cache, [&p](auto &e) {
                return std::get<1>(e.first) == p.get_decon_level()
                       && std::get<3>(e.first) == p.get_grad_level()
                       && (p.get_dataset_id() && std::get<4>(e.first) && std::get<4>(e.first) == p.get_dataset_id())
                       && std::get<5>(e.first) == p.get_input_queue_table_name()
                       && std::get<6>(e.first) == p.get_input_queue_column_name();
            });
        }
    }
}

}

#endif //SVR_ONLINESVR_CACHE_CPP
