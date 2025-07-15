//
// Created by zarko on 2/22/24.
//

#ifndef SVR_CALC_CACHE_TPP
#define SVR_CALC_CACHE_TPP

#include <unordered_map>
#include "calc_cache.hpp"
#include "common/compatibility.hpp"
#include "common/logging.hpp"
#include "kernel_factory.hpp"

namespace svr {
namespace business {

#define __rT typename cached<kT, fT>::rT

template<typename kT, typename fT> std::unordered_map<kT, __rT> cached<kT, fT>::cache_cont;
template<typename kT, typename fT> tbb::concurrent_unordered_map<kT, tbb::mutex> cached<kT, fT>::mx_map;


template<typename kT, typename fT> __rT &cached<kT, fT>::operator()(const kT &cache_key, const fT &f)
{
    auto iter = cache_cont.find(cache_key);
    if (iter != cache_cont.cend()) goto __bail;
    {
        const tbb::mutex::scoped_lock l(mx_map[cache_key]);
        iter = cache_cont.find(cache_key);
        if (iter == cache_cont.cend()) {
            typename std::unordered_map<kT, __rT>::mapped_type r;
            PROFILE_MSG(r = f(), "Prepare");
            bool rc;
            std::tie(iter, rc) = cache_cont.emplace(cache_key, r); // If rehashing occurs here, we are doomed, in that case replace mx_map with plain mx
            if (!rc) LOG4_THROW("Error inserting entry in cache");
        }
    }
    __bail:
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


template<typename T> // X samples are columns, features are rows
arma::Mat<T> &calc_cache::get_cumulatives(const kernel::kernel_path<T> &kernel_ftor, const arma::Mat<T> &X, const bpt::ptime &time)
{
    const auto &params = kernel_ftor.get_parameters();
    const auto k = std::tuple{
            params.get_input_queue_column_name(),
            params.get_grad_level(),
            params.get_chunk_index(),
            params.get_adjacent_levels(),
            arma::size(X),
            time};
    const auto prepare_f = [&X, &kernel_ftor] { return kernel_ftor.all_cumulatives(X); };
    return cached<DTYPE(k), DTYPE(prepare_f)>::get()(k, X);
}

template<typename T>
arma::Mat<T> &calc_cache::get_cumulatives(const datamodel::SVRParameters &parameters, const arma::Mat<T> &X, const bpt::ptime &time)
{
    get_cumulatives(*kernel::IKernel<double>::get(parameters), X, time);
}


template<typename T>
arma::Mat<T> &calc_cache::get_Z(const kernel::kernel_base<T> &kernel_ftor, const arma::Mat<T> &X, const bpt::ptime &time)
{
    const auto &params = kernel_ftor.get_parameters();
    const auto k = std::tuple{
            params.get_input_queue_column_name(),
            params.get_grad_level(),
            params.get_chunk_index(),
            common::hash_lambda(params.get_svr_kernel_param2()),
            common::hash_lambda(params.get_kernel_param3()),
            params.get_adjacent_levels(),
            arma::size(X),
            time};
    const auto prepare_f = [&params, &X, &time, &kernel_ftor, this]{
        switch (params.get_kernel_type()) {
            case datamodel::e_kernel_type::DEEP_PATH:
                LOG4_THROW("Unhandled kernel type " << params.get_kernel_type());

            case datamodel::e_kernel_type::PATH:
                return kernel_ftor.distances(get_cumulatives(kernel_ftor, params, X, time));

            default:
                return kernel_ftor.distances(X);
        }
    };
    return cached<DTYPE(k), DTYPE(prepare_f)>::get()(k, prepare_f);
}


template<typename T>
arma::Mat<T> &calc_cache::get_Zy(const kernel::kernel_base<T> &kernel_ftor, const arma::Mat<T> &X, const arma::Mat<T> &Xy,
                                 const bpt::ptime &X_time, const bpt::ptime &Xy_time)
{
    const auto &params = kernel_ftor.get_parameters();
    const auto k = std::tuple{
            params.get_input_queue_column_name(),
            params.get_grad_level(),
            params.get_chunk_index(),
            common::hash_lambda(params.get_svr_kernel_param2()),
            common::hash_lambda(params.get_kernel_param3()),
            params.get_adjacent_levels(),
            arma::size(X),
            arma::size(Xy),
            X_time,
            Xy_time};

    const auto prepare_f = [&params, &X, &Xy, &X_time, &Xy_time, &kernel_ftor, this] {
        switch (params.get_kernel_type()) {
            case datamodel::e_kernel_type::DEEP_PATH:
                LOG4_THROW("Unhandled kernel type " << params.get_kernel_type());

            case datamodel::e_kernel_type::PATH:
                return kernel_ftor.distances(get_cumulatives(kernel_ftor, params, X, X_time), get_cumulatives(kernel_ftor, params, Xy, Xy_time));

            default:
                return kernel_ftor.distances(X, Xy);
        }
    };
    return *cached<DTYPE(k), DTYPE(prepare_f)>::get()(k, prepare_f);
}


template<typename T>
arma::Mat<T> &calc_cache::get_K(const kernel::kernel_base<T> &kernel_ftor, const arma::Mat<T> &X, const bpt::ptime &time)
{
    const auto &params = kernel_ftor.get_parameters();
    const auto k = std::tuple{
            params.get_input_queue_column_name(),
            params.get_grad_level(),
            params.get_chunk_index(),
            common::hash_lambda(params.get_kernel_param3()),
            common::hash_lambda(params.get_svr_kernel_param2()),
            common::hash_lambda(params.get_svr_kernel_param()),
            params.get_adjacent_levels(),
            arma::size(X),
            time};

    const auto prepare_f = [&params, &X, &time, &kernel_ftor, this](){
        switch(params.get_kernel_type()) {
            case datamodel::e_kernel_type::DEEP_PATH:
                LOG4_THROW("Unhandled kernel type " << params.get_kernel_type());
            default:
                return kernel_ftor.kernel_from_distances(kernel_ftor.distances(*this, X, time));
        }
    };
    return cached<DTYPE(k), DTYPE(prepare_f)>::get()(k, prepare_f);
}

template<typename T>
arma::Mat<T> &calc_cache::get_Ky(const kernel::kernel_base<T> &kernel_ftor, const arma::Mat<T> &X, const arma::Mat<T> &Xy, const bpt::ptime &time_X, const bpt::ptime &time_Xy)
{
    const auto &params = kernel_ftor.get_parameters();
    const auto k = std::tuple{
            params.get_kernel_type(),
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
            time_Xy};

    const auto prepare_f = [&params, &X, &Xy, &time_X, &time_Xy, &kernel_ftor, this]{
        switch(params.get_kernel_type()) {
            case datamodel::e_kernel_type::DEEP_PATH:
                LOG4_THROW("Unhandled kernel type " << params.get_kernel_type());
            default:
                return kernel_ftor.kernel_from_distances(kernel_ftor.distances(*this, X, Xy, time_X, time_Xy));
        }
    };

    return *cached<DTYPE(k), DTYPE(prepare_f)>::get()(k, prepare_f);
}

}
}

#endif //SVR_CALC_CACHE_TPP
