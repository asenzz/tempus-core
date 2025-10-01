//
// Created by zarko on 27/07/2025.
//

#ifndef DATAROWSERVICE_TPP
#define DATAROWSERVICE_TPP

#include "util/time_utils.hpp"
#include "common/parallelism.hpp"

namespace svr {
namespace business {

inline double get_value(const datamodel::DataRow::container::const_iterator &it, const uint16_t level)
{
    return (**it).get_value(level);
}

inline const bpt::ptime &get_time(const datamodel::DataRow::container::const_iterator &it)
{
    return (**it).get_value_time();
}

inline const bpt::ptime &get_time(const std::deque<bpt::ptime>::const_iterator &it)
{
    return *it;
}

inline bool is_valid(const datamodel::DataRow::container::const_iterator &it)
{
    return it->operator bool();
}

inline bool is_valid(const std::deque<bpt::ptime>::const_iterator &it)
{
    return !it->is_special();
}

template<typename I> inline void generate_twap_indexes(
        const I &cbegin, // Begin of container
        const I &start_it, // At start time or before
        const I &it_end, // At end time or after
        const bpt::ptime &start_time, // Exact start time
        const bpt::time_duration &duration, // Exact end time
        const uint32_t n_out, // Count of positions to output
        RPTR(uint32_t) out)
{
    assert(it_end >= start_it);
    assert(end_time >= start_time);
    auto it = start_it;
    for (DTYPE(n_out) outctr = 0; outctr < n_out; ++outctr) {
        const auto time_iter = start_time + duration * outctr / n_out;
        while (it < it_end && is_valid(it) && get_time(it) < time_iter) ++it;
        out[outctr] = it - cbegin - (it > cbegin && (it == it_end || !is_valid(it) || get_time(it) > time_iter));
    }
#ifndef NDEBUG
    if (const auto dist_it = it - start_it; dist_it < 1) LOG4_THROW("Could not calculate TWAP indexes for " << start_time << ", distance " << dist_it);
#endif
}

template<typename I> inline uint32_t /* index of extrema */ generate_twap_bias(
        uint32_t *const out, // Output array
        const bool maxmin, // Min or max
        const I &cbegin, // Begin of container
        const I &start_it, // At start time or before
        const I &it_end, // At end time or after
        const bpt::ptime &start_time, // Exact start time
        const bpt::time_duration &duration, // Exact end time
        const uint32_t n_out, // Count of positions to output
        const uint16_t level // Level
        )
{
    assert(it_end >= start_it);
    auto it = start_it;
    auto maxmin_v = maxmin ? std::numeric_limits<double>::max() : std::numeric_limits<double>::min();
    uint32_t maxmin_i = start_it - cbegin;
    for (DTYPE(n_out) outctr = 0; outctr < n_out; ++outctr) {
        const auto time_iter = start_time + duration * outctr / n_out;
        while (it < it_end && is_valid(it) && get_time(it) < time_iter) ++it;
        out[outctr] = it - cbegin - (it > cbegin && (it == it_end || !is_valid(it) || get_time(it) > time_iter));
        const auto v = get_value(cbegin + out[outctr], level);
        if ((maxmin && v > maxmin_v) || (!maxmin && v < maxmin_v)) {
            maxmin_v = v;
            maxmin_i = out[outctr];
        }
    }
#ifndef NDEBUG
    if (const auto dist_it = it - start_it; dist_it < 1) LOG4_THROW("Could not calculate TWAP indexes for " << start_time << ", distance " << dist_it);
#endif
    return maxmin_i;
}


}
}

#endif //DATAROWSERVICE_TPP
