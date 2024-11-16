#pragma once

#include <stdexcept>
#include <string>
#include <memory>
#include <boost/date_time.hpp>
#include "common/compatibility.hpp"


namespace svr {
namespace common {

// SFINAE test
template<typename T>
class has_size
{
    typedef char one;
    typedef long two;

    template<typename C>
    static one test(DTYPE(&C::size));

    template<typename C>
    static two test(...);

public:
    enum
    {
        value = sizeof(test<T>(0)) == sizeof(char)
    };
};

template<typename T>
inline void reject_nullptr(const std::shared_ptr<T> &ptr, const std::string &msg = "Nullptr passed as an argument!")
{
    if (ptr.get() == nullptr) {
        throw std::invalid_argument(msg);
    }
}

inline void
reject_not_a_date_time(const bpt::ptime &time, const std::string &msg = "Invalid Time passed as an argument!")
{
    if (time.is_not_a_date_time()) {
        throw std::invalid_argument(msg);
    }
}

inline void
reject_special_date_time(const bpt::ptime &time, const std::string &msg = "Invalid Time passed as an argument!")
{
    if (time.is_special()) {
        throw std::invalid_argument(msg);
    }
}

template<typename T>
inline void reject_empty
        (const T &obj, const std::string &msg = "The argument cannot be empty!")
{
    if (!has_size<T>::value || obj.size() == 0) {
        throw std::invalid_argument(msg);
    }
}

template<typename T>
inline void
reject_not_positive(T const &value, const std::string &msg = "The argument should be positive.")
{
    if (value <= 0)
        throw std::invalid_argument(msg);
}

template<typename X>
inline void
throwx(const std::string &msg = "Unspecified error occured.")
{
    throw X(msg);
}

}
}