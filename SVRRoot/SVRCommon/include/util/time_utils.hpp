#pragma once

#include "common/types.hpp"

namespace bpt = boost::posix_time;

namespace svr {
const bpt::seconds onesec(1);
const bpt::hours onehour(1);

double operator /(const boost::posix_time::time_duration &lhs, const boost::posix_time::time_duration &rhs);

// boost::posix_time::time_duration operator /(const boost::posix_time::time_duration &lhs, const boost::posix_time::time_duration &rhs);

template<typename T, std::enable_if_t<std::is_same_v<T, double>, bool>  = false>
boost::posix_time::time_duration operator /(const boost::posix_time::time_duration &lhs, const T rhs);

template<typename T, std::enable_if_t<std::is_same_v<T, double>, bool>  = false>
boost::posix_time::time_duration operator *(const boost::posix_time::time_duration &lhs, const T rhs);

template<typename T, std::enable_if_t<std::is_same_v<T, double>, bool>  = false>
boost::posix_time::time_duration operator *(const T lhs, const boost::posix_time::time_duration &rhs);

template<typename T, std::enable_if_t<std::is_same_v<T, unsigned>, bool>  = false>
boost::posix_time::time_duration operator *(const T lhs, const boost::posix_time::time_duration &rhs);

template<typename T, std::enable_if_t<std::is_same_v<T, uint16_t>, bool>  = false>
boost::posix_time::time_duration operator *(const T lhs, const boost::posix_time::time_duration &rhs);

namespace common {
boost::posix_time::seconds date_time_string_to_seconds(const std::string &date_time);

bpt::time_period adjust_time_period_to_frame_size(const bpt::time_period &time_range, const bpt::time_duration &resolution, const size_t frame_size);

boost::posix_time::ptime round_second(const boost::posix_time::ptime &t);

boost::posix_time::ptime round_millisecond(const boost::posix_time::ptime &t);

boost::posix_time::ptime round_hour(const boost::posix_time::ptime &t);

boost::posix_time::ptime round_minute(const boost::posix_time::ptime &t);
} // namespace common
} // namespace svr

#include "time_utils.tpp"