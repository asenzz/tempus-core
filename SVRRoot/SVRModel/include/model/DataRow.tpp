//
// Created by zarko on 2/3/24.
//

#ifndef SVR_DATAROW_TPP
#define SVR_DATAROW_TPP

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

template<typename C, typename C_range_iter, typename T> C_range_iter
container_range<C, C_range_iter, T>::contend() const
{
    return container_.end();
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
    return container_.size() ? container_.front()->get_values().size() : 0;
}

}
}

#endif //SVR_DATAROW_TPP
