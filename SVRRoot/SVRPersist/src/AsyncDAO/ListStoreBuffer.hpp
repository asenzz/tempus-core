#ifndef LFQUEUEStoreBuffer_H
#define LFQUEUEStoreBuffer_H

#include <list>
#include <algorithm>
#include <mutex>

#include <model/StoreBufferPushMerge.hpp>

class ListStoreBufferTests_BasicTests_Test;

namespace svr { namespace dao {

template <class T, class ShallowPredicate>
class ListStoreBuffer
{
    friend class ::ListStoreBufferTests_BasicTests_Test;
public:
    using my_cont_t=std::list<T>;

    ListStoreBuffer(size_t size, ShallowPredicate shallow);

    /*
     * Tries pushing the elem into the queue. If there is a congestion
     * i.e. no enough space in the queue, returns false. DOES NOT push
     * value in this case.
     * Returns true if there is no congestion.
     */
    bool push(T const & elem);

    /*
     * Returns true if where was a value at the queue back. False - otherwise.
     */
    bool pop (T & elem);

    /*
     * Searches for the elements to be removed using the ShallowPredicate.
     */
    void remove(T const & elem);

    /*
     * Searches for the elements to be removed using the Pred.
     */
    template<class Pred>
    void remove_if(Pred);

    size_t size() const;
private:
    my_cont_t cont;
    mutable std::mutex mutex;
    typename my_cont_t::size_type const sz;
    ShallowPredicate shallow;
};

template <class T, class ShallowPredicate>
ListStoreBuffer<T, ShallowPredicate>::ListStoreBuffer(size_t size, ShallowPredicate shallow)
: cont(), sz(size), shallow(std::move(shallow))
{}

template <class T, class ShallowPredicate>
bool ListStoreBuffer<T, ShallowPredicate>::push(T const & elem)
{
    std::scoped_lock lg(mutex);
    if(!cont.empty() && shallow(cont.front(), elem))
    {
        store_buffer_push_merge(cont.front(), elem);
        return true;
    }

    if(cont.size() >= sz)
        return false;

    cont.push_front(elem);
    return true;
}

template <class T, class ShallowPredicate>
bool ListStoreBuffer<T, ShallowPredicate>::pop (T & elem)
{
    std::scoped_lock lg(mutex);
    if(cont.empty())
        return false;

    elem = cont.back();
    cont.pop_back();
    return true;
}

template <class T, class ShallowPredicate>
void ListStoreBuffer<T, ShallowPredicate>::remove(T const & elem)
{
    std::scoped_lock lg(mutex);
    auto iter = std::find_if(cont.begin(), cont.end(), [this, &elem](T const & what){return shallow(elem, what); });
    while( iter != cont.end() )
    {
        cont.erase(iter++);
        iter = std::find_if(iter, cont.end(), [this, &elem](T const & what){return shallow(elem, what); });
    }
}

template <class T, class ShallowPredicate>
template<class Pred>
void ListStoreBuffer<T, ShallowPredicate>::remove_if(Pred pred)
{
    auto lpred = [&pred](T const & what){return pred(what); };
    std::scoped_lock lg(mutex);
    auto iter = std::find_if(cont.begin(), cont.end(), lpred);
    while( iter != cont.end() )
    {
        cont.erase(iter++);
        iter = std::find_if(iter, cont.end(), lpred);
    }
}

template <class T, class ShallowPredicate>
size_t ListStoreBuffer<T, ShallowPredicate>::size() const
{
    std::scoped_lock lg(mutex);
    return cont.size();
}


} }

#endif /* LFQUEUEStoreBuffer_H */
