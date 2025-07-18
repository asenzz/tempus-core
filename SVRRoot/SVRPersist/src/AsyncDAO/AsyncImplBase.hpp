#ifndef ASYNCIMPL_HPP
#define ASYNCIMPL_HPP

#include "StoreBufferInterface.hpp"
#include "StoreBufferController.hpp"
#include "ListStoreBuffer.hpp"
#include "VectorMruCache.hpp"

namespace svr {
namespace dao {

template<class Elem, class ShallowPredicate, class DeepPredicate, class SynchronousDao>
class AsyncImplBase : public StoreBufferInterface {
public:
    using my_value_t = Elem;
    using my_mru_cache_t = VectorMruCache<Elem, ShallowPredicate, DeepPredicate>;
    using my_store_buffer_t = ListStoreBuffer<Elem, ShallowPredicate>;

    SynchronousDao pgDao;
    std::mutex pgMutex;

private:
    my_mru_cache_t mru_cache;
    my_store_buffer_t store_buffer;

public:
    AsyncImplBase(common::PropertiesReader &tempus_config, dao::DataSource &data_source, ShallowPredicate shallow, DeepPredicate deep,
                  const size_t mruCacheSize, const size_t storeBufferSize)
            : pgDao(tempus_config, data_source), mru_cache(mruCacheSize, shallow, deep), store_buffer(storeBufferSize, shallow)
    {
    }

    ~AsyncImplBase()
    {
    }

    uint64_t storeOne()
    {
        my_value_t value;
        if (store_buffer.pop(value)) {
            std::scoped_lock lg(pgMutex);
            pgDao.save(value);
        }
        return 0;
    }

    template<class SearchMethod>
    void seekAndCache(my_value_t &what, SearchMethod search, std::string const &name)
    {
        if (mru_cache.cached(what)) return;

        const std::scoped_lock lg(pgMutex);
        what = (pgDao.*search)(name);
        mru_cache.cache(what);
    }

    void cache(my_value_t const &what)
    {
        const my_value_t value = what;
        mru_cache.cache(value);
        while (!store_buffer.push(value)) flush();
        StoreBufferController::get_instance().addStoreBuffer(*this);
    }

    void cache_no_store(my_value_t const &what)
    {
        mru_cache.cache(what);
    }

    int remove(my_value_t const &what)
    {
        int result = 0;
        my_value_t value = what;
        if (mru_cache.cached(value)) {
            flush();
            cache_remove(value);
            result = 1;
        }

        const std::scoped_lock lg(pgMutex);
        if (pgDao.remove(value)) result = 1;
        return result;
    }

    void cache_remove(my_value_t const &what)
    {
        mru_cache.remove(what);
        store_buffer.remove(what);
    }

    template<class Pred> void cache_remove_if(const Pred p)
    {
        mru_cache.remove_if(p);
        store_buffer.remove_if(p);
    }


    bool cached(my_value_t &what)
    {
        return mru_cache.cached(what);
    }

    void flush()
    {
        StoreBufferController::get_instance().flush();
    }
};

}
}

#endif /* ASYNCIMPL_HPP */

