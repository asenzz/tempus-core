#pragma once

#include <stdexcept>
#include <string>
#include <memory>
#include <boost/date_time.hpp>
#include "../common/compatibility.hpp"
#include "../common/logging.hpp"

// TODO Wrap reject functions below with macros that call the corresponding function only if NDEBUG is not defined.
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


template<typename T> inline constexpr void reject_nullptr(const std::shared_ptr<T> &ptr, const std::string &msg = "Nullptr passed as an argument!")
{
    if (!ptr) throw std::invalid_argument(msg);
}

#ifdef NDEBUG
#define REJECT_NULLPTR_1(ptr)
#define REJECT_NULLPTR_2(ptr, msg)
#else
#define REJECT_NULLPTR_2(ptr, msg) reject_nullptr(ptr, msg)
#define REJECT_NULLPTR_1(ptr) reject_nullptr(ptr)
#endif
#define REJECT_NULLPTR(...) PP_MACRO_OVERLOAD(REJECT_NULLPTR, __VA_ARGS__)


inline constexpr void reject_not_a_date_time(const bpt::ptime &time, const std::string &msg = "Invalid Time passed as an argument!")
{
    if (time.is_not_a_date_time()) throw std::invalid_argument(msg);
}

#ifdef NDEBUG
#define REJECT_NOT_A_DATE_TIME(time, msg)
#else
#define REJECT_NOT_A_DATE_TIME(time, msg) reject_not_a_date_time(time, msg)
#endif


inline constexpr void reject_special_date_time(const bpt::ptime &time, const std::string &msg = "Invalid Time passed as an argument!")
{
    if (time.is_special()) throw std::invalid_argument(msg);
}

#ifdef NDEBUG
#define REJECT_SPECIAL_DATE_TIME(time, msg)
#else
#define REJECT_SPECIAL_DATE_TIME(time, msg) reject_special_date_time(time, msg)
#endif


template<typename T> inline constexpr std::enable_if_t<has_size<T>::value> reject_empty(const T &obj, const std::string &msg = "The argument cannot be empty!")
{
    if (!has_size<T>::value || obj.size() == 0) throw std::invalid_argument(msg);
}

#ifdef NDEBUG
#define REJECT_EMPTY_1(obj)
#define REJECT_EMPTY_2(obj, msg)
#else
#define REJECT_EMPTY_1(obj) common::reject_empty(obj)
#define REJECT_EMPTY_2(obj, msg) common::reject_empty(obj, msg)
#endif
#define REJECT_EMPTY(...) PP_MACRO_OVERLOAD(REJECT_EMPTY, __VA_ARGS__)


template<typename T> inline constexpr void reject_not_positive(T const &value, const std::string &msg = "The argument should be positive.")
{
    if (value <= 0) throw std::invalid_argument(msg);
}

#ifdef NDEBUG
#define REJECT_NOT_POSITIVE(value, msg)
#else
#define REJECT_NOT_POSITIVE(value, msg) reject_not_positive(value, msg)
#endif


template<typename X> inline void throwx(const std::string &msg = "Unspecified error occured.")
{
    throw X(msg);
}

}
}