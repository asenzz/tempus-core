#ifndef PERFORMANCEUTILS_HPP
#define PERFORMANCEUTILS_HPP

#include <memory>
#include <vector>
#include <deque>

#define ATOMIC_FLAG_UNLOCK(atomic_flag) (atomic_flag).clear(std::memory_order_release)
#define ATOMIC_FLAG_LOCK(atomic_flag) while ((atomic_flag).test_and_set(std::memory_order_acquire))

template <typename T> std::vector<T>
get_all_except(const std::vector<T> &all_elements, const T &excepted_element)
{
    std::vector<T> result;
    for (const auto &el: all_elements)
        if (el != excepted_element)
            result.push_back(el);
    return result;
}

template <typename T> std::deque<T>
get_all_except(const std::deque<T> &all_elements, const T &excepted_element)
{
    std::deque<T> result;
    for (const auto &el: all_elements)
        if (el != excepted_element)
            result.push_back(el);
    return result;
}

template <typename T> void
do_nothing_deleter(T *)
{
    return;
}

template <typename T> std::shared_ptr<T>
shared_observer_ptr(T &v)
{
    return std::shared_ptr<T>(&v, do_nothing_deleter<T>);
}


//#define PERFORMANCE_TESTING
#ifdef PERFORMANCE_TESTING

#include <chrono>
#include <string>
#include <sstream>

#include "StringUtils.hpp"

namespace svr {
namespace common {

template <class metric_holder>
class stream_provider
{
public:
    stream_provider();
    std::basic_ostream<char> & get_stream();
private:
    std::ofstream stream;
};

class performance_timer
{
public:
    performance_timer(std::string name, std::basic_ostream<char> & out);
    template<class metric_holder> performance_timer(std::string name, stream_provider<metric_holder> *);
    ~performance_timer();

    std::basic_ostream<char> & get_comment_stream();
private:
    std::string name;
    std::basic_ostream<char> & out;
    std::chrono::time_point<std::chrono::steady_clock> start;
    std::ostringstream comment;
};

template<class metric_holder>
performance_timer::performance_timer(std::string name, stream_provider<metric_holder> * sp)
: performance_timer(std::move(name), sp->get_stream())
{
}


template<class T>
inline performance_timer & operator<<(performance_timer & pt, T const & t)
{
    pt.get_comment_stream() << t;
    return pt;
}


template <class metric_holder>
stream_provider<metric_holder>::stream_provider()
{
    std::ostringstream fname_str;
    fname_str << svr::common::demangle(typeid(metric_holder).name()) << "_" << std::hex << this << ".perflog";

    stream.open(fname_str.str().c_str());

    if(stream.bad())
        throw std::runtime_error("Cannot open the output file.");
}

template <class metric_holder>
std::basic_ostream<char> & stream_provider<metric_holder>::get_stream()
{
    return stream;
}

}
}

#endif

#endif /* PERFORMANCEUTILS_HPP */

