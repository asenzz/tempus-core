#pragma once

#include <chrono>
#include "common/types.hpp"

namespace bpt = boost::posix_time;

namespace svr {

const bpt::seconds onesec(1);
const bpt::hours onehour(1);

double operator /(const boost::posix_time::time_duration &lhs, const boost::posix_time::time_duration &rhs);
//boost::posix_time::time_duration operator /(const boost::posix_time::time_duration &lhs, const boost::posix_time::time_duration &rhs);

template<typename T, std::enable_if_t<std::is_same<T, double>::value, bool> = false>
boost::posix_time::time_duration operator /(const boost::posix_time::time_duration &lhs, const T rhs)
{
    return boost::posix_time::microseconds(size_t(lhs.total_microseconds() / rhs));
}

template<typename T, std::enable_if_t<std::is_same<T, double>::value, bool> = false>
boost::posix_time::time_duration operator *(const boost::posix_time::time_duration &lhs, const T rhs)
{
    return boost::posix_time::microseconds(size_t(lhs.total_microseconds() * rhs));
}

template<typename T, std::enable_if_t<std::is_same<T, double>::value, bool> = false>
boost::posix_time::time_duration operator *(const T lhs, const boost::posix_time::time_duration &rhs)
{
    return boost::posix_time::microseconds(size_t(rhs.total_microseconds() * lhs));
}

template<typename T, std::enable_if_t<std::is_same<T, unsigned>::value, bool> = false>
boost::posix_time::time_duration operator *(const T lhs, const boost::posix_time::time_duration &rhs)
{
    return boost::posix_time::microseconds(size_t(rhs.total_microseconds() * lhs));
}

template<typename T, std::enable_if_t<std::is_same<T, uint16_t>::value, bool> = false>
boost::posix_time::time_duration operator *(const T lhs, const boost::posix_time::time_duration &rhs)
{
    return boost::posix_time::microseconds(size_t(rhs.total_microseconds() * lhs));
}

namespace common {

boost::posix_time::seconds date_time_string_to_seconds(const std::string & date_time);

bpt::time_period adjust_time_period_to_frame_size(const bpt::time_period & time_range,
                                                  const bpt::time_duration & resolution,
                                                  const size_t frame_size);


boost::posix_time::ptime round_second(const boost::posix_time::ptime &t);

boost::posix_time::ptime round_millisecond(const boost::posix_time::ptime &t);

boost::posix_time::ptime round_hour(const boost::posix_time::ptime &t);

boost::posix_time::ptime round_minute(const boost::posix_time::ptime &t);

} // namespace common
} // namespace svr
