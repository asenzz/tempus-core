//
// Created by zarko on 27/07/2025.
//

#ifndef TIME_UTILS_TPP
#define TIME_UTILS_TPP

#include "common/types.hpp"

namespace svr {

template<typename T, std::enable_if_t<std::is_same_v<T, double>, bool>>
boost::posix_time::time_duration operator /(const boost::posix_time::time_duration &lhs, const T rhs)
{
    return boost::posix_time::microseconds(size_t(lhs.total_microseconds() / rhs));
}

template<typename T, std::enable_if_t<std::is_same_v<T, double>, bool>>
boost::posix_time::time_duration operator *(const boost::posix_time::time_duration &lhs, const T rhs)
{
    return boost::posix_time::microseconds(size_t(lhs.total_microseconds() * rhs));
}

template<typename T, std::enable_if_t<std::is_same_v<T, double>, bool>>
boost::posix_time::time_duration operator *(const T lhs, const boost::posix_time::time_duration &rhs)
{
    return boost::posix_time::microseconds(size_t(rhs.total_microseconds() * lhs));
}

template<typename T, std::enable_if_t<std::is_same_v<T, unsigned>, bool>>
boost::posix_time::time_duration operator *(const T lhs, const boost::posix_time::time_duration &rhs)
{
    return boost::posix_time::microseconds(size_t(rhs.total_microseconds() * lhs));
}

template<typename T, std::enable_if_t<std::is_same_v<T, uint16_t>, bool>>
boost::posix_time::time_duration operator *(const T lhs, const boost::posix_time::time_duration &rhs)
{
    return boost::posix_time::microseconds(size_t(rhs.total_microseconds() * lhs));
}

}

#endif //TIME_UTILS_TPP
