//
// Created by zarko on 2/25/24.
//

#ifndef SVR_DBCACHE_TPP
#define SVR_DBCACHE_TPP

#include <boost/unordered/unordered_flat_map.hpp>
#include <boost/functional/hash.hpp>
#include "common/compatibility.hpp"

namespace svr {

template<typename T> using registered_ptr = std::shared_ptr<T>;

namespace datamodel {

template<typename T> class dbcache
{
#define actual_T std::decay_t<T>
#define actual_T_id typeid(actual_T).hash_code()
#define __MAKE_SHARED_OBJ(__ID) std::make_shared<actual_T>(__ID, args...)

    static boost::unordered_flat_map<bigint, registered_ptr<actual_T>> cached_db_entities;
    static std::mutex mx_obj;

public:

    template<typename... Ts> static registered_ptr<actual_T>
    __ptr(const bool overwrite, const bigint id, Ts &&... args)
    {
        if (!id) return std::make_shared<T>(id, args...);

        typename DTYPE(cached_db_entities)::iterator search_obj;
        std::unique_lock ul(mx_obj);
        if ((search_obj = cached_db_entities.find(id)) != cached_db_entities.end()) {
            ul.unlock();
            if (overwrite) search_obj->second = __MAKE_SHARED_OBJ(id);
        } else {
            const auto rc = cached_db_entities.emplace(id, __MAKE_SHARED_OBJ(id));
            ul.unlock();
            search_obj = rc.first;
        }
        return search_obj->second;
    }


    template<typename... Ts> static registered_ptr<actual_T>
    __ptr(const bool overwrite, Ts &&... args)
    {
        const auto new_obj_ptr = std::make_shared<actual_T>(args...);
        const auto new_obj_id = new_obj_ptr->get_id();
        if (!new_obj_id) return new_obj_ptr;

        typename DTYPE(cached_db_entities)::iterator search_obj;
        std::unique_lock ul(mx_obj);
        if ((search_obj = cached_db_entities.find(new_obj_id)) != cached_db_entities.end()) {
            ul.unlock();
            if (overwrite) search_obj->second = new_obj_ptr;
        } else {
            const auto rc = cached_db_entities.emplace(new_obj_id, new_obj_ptr);
            ul.unlock();
            search_obj = rc.first;
        }
        return search_obj->second;
    }


    static registered_ptr<actual_T>
    __ptr(const bool overwrite, T &&arg)
    {
        const auto new_obj_id = arg.get_id();
        if (!new_obj_id) return std::make_shared<actual_T>(arg);

        typename DTYPE(cached_db_entities)::iterator search_obj;
        std::unique_lock ul(mx_obj);
        if ((search_obj = cached_db_entities.find(new_obj_id)) != cached_db_entities.end()) {
            ul.unlock();
            if (overwrite) search_obj->second = std::make_shared<actual_T>(arg);
        } else {
            const auto rc = cached_db_entities.emplace(new_obj_id, std::make_shared<actual_T>(arg));
            ul.unlock();
            search_obj = rc.first;
        }
        return search_obj->second;
    }


    static void print_cache()
    {
        for (const auto &el: cached_db_entities)
            LOG4_DEBUG("DB cache contents: " << el.first << ", " << el.second);
    }
};

template<typename T> boost::unordered_flat_map<bigint, registered_ptr<actual_T>> dbcache<T>::cached_db_entities;
template<typename T> std::mutex dbcache<T>::mx_obj;

}

template<typename T, typename... Ts> registered_ptr<std::enable_if_t<std::is_base_of_v<datamodel::Entity, actual_T>, actual_T>> inline
ptr(Ts &&... args) // Overwrites any existing database objects in cache with newly created instance and returns a pointer to the new instance
{
    return datamodel::dbcache<actual_T>::__ptr(true, args...);
}

template<typename T> registered_ptr<std::enable_if_t<std::is_base_of_v<datamodel::Entity, actual_T>, actual_T>> inline
ptr(T &&arg) // Overwrites any existing database objects in cache with newly created instance and returns a pointer to the new instance
{
    return datamodel::dbcache<actual_T>::__ptr(true, arg);
}

template<typename T, typename... Ts> registered_ptr<typename std::enable_if_t<std::is_base_of_v<datamodel::Entity, actual_T>, actual_T>> inline
dtr(Ts &&... args) // Deter creating a new object if an object with the same ID is already present in cache
{
    return datamodel::dbcache<actual_T>::__ptr(false, args...);
}

template<typename T> registered_ptr<typename std::enable_if_t<std::is_base_of_v<datamodel::Entity, actual_T>, actual_T>> inline
dtr(T &&arg) // Deter creating a new object if an object with the same ID is already present in cache
{
    return datamodel::dbcache<actual_T>::__ptr(false, arg);
}

// Compatibility
template<typename T, typename... Ts> std::shared_ptr<typename std::enable_if_t<std::is_base_of<datamodel::Entity, actual_T>::value == false, actual_T>> inline
ptr(Ts &&... args)
{
    return std::make_shared<actual_T>(args...);
}

template<typename T> std::shared_ptr<typename std::enable_if_t<std::is_base_of<datamodel::Entity, actual_T>::value == false, actual_T>> inline
ptr(T &&arg)
{
    return std::make_shared<actual_T>(arg);
}

template<typename T, typename... Ts> std::shared_ptr<actual_T> inline
otr(Ts &&... args)
{
    return std::make_shared<actual_T>(args...);
}

template<typename T> std::shared_ptr<actual_T> inline
otr(T &&arg)
{
    return std::make_shared<actual_T>(arg);
}

}

#endif //SVR_DBCACHE_TPP