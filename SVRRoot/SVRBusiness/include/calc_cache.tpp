//
// Created by zarko on 2/22/24.
//

#ifndef SVR_CALC_CACHE_TPP
#define SVR_CALC_CACHE_TPP

#include "calc_cache.hpp"
#include "common/compatibility.hpp"
#include "common/logging.hpp"
#include <unordered_map>

namespace svr {
namespace business {

#define __rT typename cached<kT, fT>::rT

template<typename kT, typename fT> std::unordered_map<kT, __rT> cached<kT, fT>::cache_cont;
template<typename kT, typename fT> tbb::concurrent_unordered_map<kT, tbb::mutex> cached<kT, fT>::mx_map;


template<typename kT, typename fT> __rT &cached<kT, fT>::operator()(const kT &cache_key, const fT &f)
{
    auto iter = cache_cont.find(cache_key);
    if (iter != cache_cont.end()) goto __bail;
    {
        const tbb::mutex::scoped_lock l(mx_map[cache_key]);
        iter = cache_cont.find(cache_key);
        if (iter == cache_cont.end()) {
            typename std::unordered_map<kT, __rT>::mapped_type r;
            PROFILE_(r = f());
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
    release_cont(cache_cont);
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

}
}

#endif //SVR_CALC_CACHE_TPP
