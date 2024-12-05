//
// Created by zarko on 2/3/24.
//

#ifndef SVR_DATAROW_TPP
#define SVR_DATAROW_TPP

#include "util/time_utils.hpp"
#include "common/parallelism.hpp"

namespace svr {
namespace datamodel {

template<typename C, typename C_range_iter, typename T>
container_range<C, C_range_iter, T>::container_range(C &container) :
        begin_(container.begin()), end_(container.end()), container_(container)
{
    reinit();
}

template<typename C, typename C_range_iter, typename T>
container_range<C, C_range_iter, T>::container_range(
        const C_range_iter &start, const C_range_iter &end, C &container) :
        begin_(start), end_(end), container_(container)
{
    reinit();
}

template<typename C, typename C_range_iter, typename T>
container_range<C, C_range_iter, T>::container_range(C_range_iter start, C &container) :
        begin_(start), end_(container.end()), container_(container)
{
    reinit();
}

template<typename C, typename C_range_iter, typename T>
container_range<C, C_range_iter, T>::container_range(C &container, C_range_iter end) :
        begin_(container.start()), end_(end), container_(container)
{
    reinit();
}

template<typename C, typename C_range_iter, typename T>
container_range<C, C_range_iter, T>::container_range(const container_range &rhs) :
        begin_(rhs.begin_), end_(rhs.end_), container_(rhs.container_)
{
    reinit();
}

template<typename C, typename C_range_iter, typename T>
container_range<C, C_range_iter, T>::container_range(container_range &rhs) :
        begin_(rhs.begin_), end_(rhs.end_), container_(rhs.container_)
{
    reinit();
}

template<typename C, typename C_range_iter, typename T>
container_range<C, C_range_iter, T> &container_range<C, C_range_iter, T>::operator=(const container_range &rhs)
{
    if (this == &rhs) return *this;
    begin_ = rhs.begin_;
    end_ = rhs.end_;
    container_ = rhs.container_;
    reinit();
    return *this;
}

template<typename C, typename C_range_iter, typename T> T &
container_range<C, C_range_iter, T>::operator[](const size_t index)
{
    return *(begin_ + index);
}


template<typename C, typename C_range_iter, typename T> T
container_range<C, C_range_iter, T>::operator[](const size_t index) const
{
    return *(begin_ + index);
}


template<typename C, typename C_range_iter, typename T> C_range_iter
container_range<C, C_range_iter, T>::operator()(const ssize_t index) const
{
    return it(index);
}

template<typename C, typename C_range_iter, typename T> C_range_iter
container_range<C, C_range_iter, T>::it(const ssize_t index) const
{
    if (index > 0)
        return index < distance_ ? begin_ + index : container_.end();
    else if (index < 0)
        return index > std::distance(begin_, container_.begin()) ? begin_ + index : container_.begin();
    else
        return begin_;
}

template<typename C, typename C_range_iter, typename T> C_range_iter
container_range<C, C_range_iter, T>::contbegin() const
{
    return container_.begin();
}

template<typename C, typename C_range_iter, typename T> container_range<C, C_range_iter, T>::C_citer
container_range<C, C_range_iter, T>::contcbegin() const
{
    return container_.cbegin();
}

template<typename C, typename C_range_iter, typename T> C_range_iter
container_range<C, C_range_iter, T>::contend() const
{
    return container_.end();
}

template<typename C, typename C_range_iter, typename T> container_range<C, C_range_iter, T>::C_citer
container_range<C, C_range_iter, T>::contcend() const
{
    return container_.cend();
}

template<typename C, typename C_range_iter, typename T> unsigned
container_range<C, C_range_iter, T>::contsize() const
{
    return container_.size();
}

template<typename C, typename C_range_iter, typename T> ssize_t
container_range<C, C_range_iter, T>::distance() const
{
    return distance_;
}

template<typename C, typename C_range_iter, typename T>
C_range_iter container_range<C, C_range_iter, T>::begin() const
{ return begin_; }

template<typename C, typename C_range_iter, typename T>
C_range_iter container_range<C, C_range_iter, T>::end() const
{
    return end_;
}

template<typename C, typename C_range_iter, typename T>
typename container_range<C, C_range_iter, T>::C_citer container_range<C, C_range_iter, T>::cbegin() const
{ return begin_; }

template<typename C, typename C_range_iter, typename T>
typename container_range<C, C_range_iter, T>::C_citer container_range<C, C_range_iter, T>::cend() const
{
    return end_;
}

template<typename C, typename C_range_iter, typename T>
typename container_range<C, C_range_iter, T>::C_riter container_range<C, C_range_iter, T>::rbegin() const
{
    return end_ == container_.end() ? container_.rbegin() : std::make_reverse_iterator(end_);
}

template<typename C, typename C_range_iter, typename T>
typename container_range<C, C_range_iter, T>::C_criter container_range<C, C_range_iter, T>::crbegin() const
{
    return end_ == container_.end() ? container_.rbegin() : std::make_reverse_iterator(end_);
}

template<typename C, typename C_range_iter, typename T>
typename container_range<C, C_range_iter, T>::C_riter container_range<C, C_range_iter, T>::rend() const
{
    return begin_ == container_.end() ? container_.rend() : C_riter(begin_);
}

template<typename C, typename C_range_iter, typename T>
typename container_range<C, C_range_iter, T>::C_criter container_range<C, C_range_iter, T>::crend() const
{
    return begin_ == container_.end() ? container_.rend() : C_criter(begin_);
}


template<typename C, typename C_range_iter, typename T>
T &container_range<C, C_range_iter, T>::front()
{
    return *container_.begin();
}


template<typename C, typename C_range_iter, typename T>
T container_range<C, C_range_iter, T>::front() const
{
    return *container_.begin();
}

template<typename C, typename C_range_iter, typename T>
T &container_range<C, C_range_iter, T>::back()
{
    return *container_.end();
}

template<typename C, typename C_range_iter, typename T>
T container_range<C, C_range_iter, T>::back() const
{
    return *container_.end();
}


template<typename C, typename C_range_iter, typename T>
C &container_range<C, C_range_iter, T>::get_container() const
{
    return container_;
}

template<typename C, typename C_range_iter, typename T>
void container_range<C, C_range_iter, T>::set_range(const C_range_iter &start, const C_range_iter &end)
{
    begin_ = start;
    end_ = end;
    reinit();
}

template<typename C, typename C_range_iter, typename T>
void container_range<C, C_range_iter, T>::set_begin(C_range_iter &begin)
{
    begin_ = begin;
    reinit();
}

template<typename C, typename C_range_iter, typename T>
void container_range<C, C_range_iter, T>::set_end(C_range_iter &end)
{
    end_ = end;
    reinit();
}

template<typename C, typename C_range_iter, typename T>
void container_range<C, C_range_iter, T>::reinit() // Call whenever changes to the container are made
{
    distance_ = std::distance(begin_, end_);
}

template<typename C, typename C_range_iter, typename T>
size_t container_range<C, C_range_iter, T>::levels() const
{
    return container_.size() ? container_.front()->size() : 0;
}

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
        out[uint32_t(inctr * inout_ratio)] = it - cbegin - (!is_valid(it) || it == it_end || (get_time(it) != time_iter && it != start_it && it != cbegin));
    }
#ifndef NDEBUG
    const auto dist_it = it - start_it;
    if (inctr != inlen || dist_it < 1) LOG4_THROW("Could not calculate TWAP indexes for " << start_time << ", resolution " << resolution << ", distance " << dist_it);
#endif
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

#endif // SVR_DATAROW_TPP
