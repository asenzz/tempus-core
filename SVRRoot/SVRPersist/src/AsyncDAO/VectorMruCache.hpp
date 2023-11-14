#ifndef VECTOR_MRU_CACHE_H
#define VECTOR_MRU_CACHE_H

#include <vector>
#include <algorithm>

class VectorCacheTests_BasicTests_Test;
class VectorCacheTests_RandomTests_Test;

namespace svr { namespace dao {

template<class T, class ShallowPredicate, class DeepPredicate>
class VectorMruCache
{
    friend class ::VectorCacheTests_BasicTests_Test;
    friend class ::VectorCacheTests_RandomTests_Test;
public:
    using my_cont_t = std::vector<T>;
    using iterator = typename my_cont_t::iterator;
    using const_iterator = typename my_cont_t::const_iterator;

private:
    my_cont_t cont;
    ShallowPredicate const shallow;
    DeepPredicate const deep;

    void recache(T const & what, iterator where);
    void emplace(T const & what);
    void emplace(T const & what, iterator where);

public:
    VectorMruCache(size_t size, ShallowPredicate shallow, DeepPredicate deep);

    /*
     * Caches the elem.
     * Applies ShallowPredicate to search for a cached element. If found, applies
     * DeepPredicate to check if it requires updating.
     * Returns true if both predicates returned true, false otherwise.
     */
    bool cache(T const & elem);

    /*
     * Checks if the elem was cached previously.
     * Applies ShallowPredicate to search for a cached element.
     * Returns true if found. Assigns elem to the found element
     */
    bool cached(T & elem) const ;

    /*
     * Replaces the elem with T() if it was cached previously.
     * Applies ShallowPredicate to search for a cached element.
     * Returns true if found.
     */
    bool remove(T const & elem);

    /*
     * Replaces the elem with T() if it was cached previously.
     * Applies Pred to search for a cached element.
     * Returns true if found.
     */
    template<class Pred>
    bool remove_if(Pred);

};

template<class T, class ShallowPredicate, class DeepPredicate>
VectorMruCache<T, ShallowPredicate, DeepPredicate>::VectorMruCache(size_t size, ShallowPredicate shallow, DeepPredicate deep)
: cont(size, T()), shallow(shallow), deep(deep)
{}

template<class T, class ShallowPredicate, class DeepPredicate>
bool VectorMruCache<T, ShallowPredicate, DeepPredicate>::cache(T const & elem)
{
    iterator iter = std::find_if(cont.begin(), cont.end(), [this, &elem](T const & other){ return shallow(elem, other); });
    if(iter == cont.end())
    {
        emplace(elem);
        return false;
    }
    if(deep(elem, *iter))
    {
        recache(elem, iter);
        return true;
    }

    emplace(elem, iter);
    return false;
}

template<class T, class ShallowPredicate, class DeepPredicate>
bool VectorMruCache<T, ShallowPredicate, DeepPredicate>::cached(T & elem) const
{
    const_iterator iter = std::find_if(cont.cbegin(), cont.cend(), [this, &elem](T const & other){ return shallow(elem, other); });

    if(iter == cont.cend())
        return false;

    elem = *iter;
    return true;
}

template<class T, class ShallowPredicate, class DeepPredicate>
bool VectorMruCache<T, ShallowPredicate, DeepPredicate>::remove(T const & elem)
{
    iterator iter = std::find_if(cont.begin(), cont.end(), [this, &elem](T const & other){ return shallow(elem, other); });

    if(iter == cont.cend())
        return false;

    *iter = T();
    return true;
}

template<class T, class ShallowPredicate, class DeepPredicate>
template<class Pred>
bool VectorMruCache<T, ShallowPredicate, DeepPredicate>::remove_if(Pred pred)
{
    iterator iter = std::find_if(cont.begin(), cont.end(), [&pred](T const & other){ return pred(other); });

    if(iter == cont.cend())
        return false;

    *iter = T();
    return true;
}

template<class T, class ShallowPredicate, class DeepPredicate>
void VectorMruCache<T, ShallowPredicate, DeepPredicate>::recache(T const & what, iterator where)
{
    if(where - cont.begin() > long(cont.size() / 2))
        emplace(what, where);
}

template<class T, class ShallowPredicate, class DeepPredicate>
void VectorMruCache<T, ShallowPredicate, DeepPredicate>::emplace(T const & what)
{
    std::move(cont.rbegin()+1, cont.rend(), cont.rbegin());
    cont.front() = what;
}

template<class T, class ShallowPredicate, class DeepPredicate>
void VectorMruCache<T, ShallowPredicate, DeepPredicate>::emplace(T const & what, iterator where)
{
    std::move(typename my_cont_t::reverse_iterator(where), cont.rend(), typename my_cont_t::reverse_iterator(where+1));
    cont.front() = what;
}

} }

#endif /* VECTOR_MRU_CACHE_H */
