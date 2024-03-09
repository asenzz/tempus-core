//
// Created by zarko on 2/22/24.
//

#ifndef SVR_CALC_CACHE_TPP
#define SVR_CALC_CACHE_TPP

#include "calc_cache.hpp"
#include "common/Logging.hpp"

namespace svr {
namespace business {

template<typename rT, typename kT, typename fT>
rT calc_cache::cached(std::unordered_map<kT, rT> &cache_cont, tbb::concurrent_unordered_map<kT, std::mutex> &mx_map, const kT &cache_key, const fT &f)
{
    auto it_cml = cache_cont.find(cache_key);
    if (it_cml != cache_cont.end()) goto __bail;
    {
        const std::scoped_lock l(mx_map[cache_key]);
        it_cml = cache_cont.find(cache_key);
        if (it_cml == cache_cont.end()) {
            typename std::unordered_map<kT, rT>::mapped_type r;
            PROFILE_EXEC_TIME(r = f(), "Prepare");
            const auto [ins, rc] = cache_cont.emplace(cache_key, r); // If rehashing occurs here, we are doomed, in that case replace mx_map with plain mx
            if (rc) it_cml = ins;
            else
                LOG4_THROW("Error inserting entry in cache");
        }
    }
__bail:
    return it_cml->second;
}

}
}

#endif //SVR_CALC_CACHE_TPP