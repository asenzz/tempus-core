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
        const bpt::ptime &end_time, // Exact end time
        const bpt::time_duration &resolution, // Aux input queue resolution
        const uint32_t n_out, // Count of positions to output
        uint32_t *const out)
{
    assert(it_end >= start_it);
    assert(end_time >= start_time);
    auto it = start_it;
    const uint32_t inlen = (end_time - start_time) / resolution;
    uint32_t inctr = 0;
    const auto inout_ratio = double(n_out) / double(inlen);
    UNROLL()
    for (auto time_iter = start_time; time_iter < end_time; time_iter += resolution, ++inctr) {
        while (it != it_end && is_valid(it) && get_time(it) < time_iter) ++it;
        out[uint32_t(inctr * inout_ratio)] = it - cbegin - (it >= it_end || !is_valid(it) || (get_time(it) != time_iter && it != start_it && it != cbegin));
    }
#ifndef NDEBUG
    const auto dist_it = it - start_it;
    if (dist_it < 1) LOG4_THROW("Could not calculate TWAP indexes for " << start_time << ", resolution " << resolution << ", distance " << dist_it);
#endif
}

template<typename I> inline uint32_t /* index of extrema */ generate_twap_bias(
        uint32_t *const out, // Output array
        const bool maxmin, // Min or max
        const I &cbegin, // Begin of container
        const I &start_it, // At start time or before
        const I &it_end, // At end time or after
        const bpt::ptime &start_time, // Exact start time
        const bpt::ptime &end_time, // Exact end time
        const bpt::time_duration &resolution, // Aux input queue resolution
        const uint32_t n_out, // Count of positions to output
        const uint16_t level // Level
        )
{
    assert(it_end >= start_it);
    assert(end_time >= start_time);
    auto it = start_it;
    const uint32_t inlen = (end_time - start_time) / resolution;
    uint32_t inctr = 0;
    const auto inout_ratio = double(n_out) / double(inlen);
    auto maxmin_v = maxmin ? std::numeric_limits<double>::max() : std::numeric_limits<double>::min();
    uint32_t maxmin_i = start_it - cbegin;
    UNROLL()
    for (auto time_iter = start_time; time_iter < end_time; time_iter += resolution, ++inctr) {
        while (it != it_end && is_valid(it) && get_time(it) < time_iter) ++it;
        const auto out_i = uint32_t(inctr * inout_ratio);
        out[out_i] = it - cbegin - (it >= it_end || !is_valid(it) || (get_time(it) != time_iter && it != start_it && it != cbegin));
        const auto v = get_value(cbegin + out[out_i], level);
        if ((maxmin && v > maxmin_v) || (!maxmin && v < maxmin_v)) {
            maxmin_v = v;
            maxmin_i = out[out_i];
        }
    }
#ifndef NDEBUG
    const auto dist_it = it - start_it;
    if (dist_it < 1) LOG4_THROW("Could not calculate TWAP indexes for " << start_time << ", resolution " << resolution << ", distance " << dist_it);
#endif
    return maxmin_i;
}

inline std::vector<uint32_t> generate_twap_indexes(
        const datamodel::DataRow::container::const_iterator &cbegin, // Begin of container
        const datamodel::DataRow::container::const_iterator &start_it, // At start time or before
        const datamodel::DataRow::container::const_iterator &it_end, // At end time or after
        const bpt::ptime &start_time, // Exact start time
        const bpt::ptime &end_time, // Exact end time
        const bpt::time_duration &resolution, // Aux input queue resolution
        const uint32_t n_out)
{
    std::vector<uint32_t> out(n_out);
    generate_twap_indexes(cbegin, start_it, it_end, start_time, end_time, resolution, n_out, out.data());
    return out;
}

}
}

#endif //DATAROWSERVICE_TPP
