#include "AsyncDecrementTaskDAO.hpp"
#include "AsyncImplBase.hpp"
#include <model/DecrementTask.hpp>
#include <DAO/DataSource.hpp>
#include <DAO/DecrementTaskRowMapper.hpp>
#include "../PgDAO/PgDecrementTaskDAO.hpp"

namespace svr {
namespace dao {

namespace
{
    static const auto cmp_primary_key = [] (DecrementTask_ptr const & lhs, DecrementTask_ptr const & rhs)
    {
        return reinterpret_cast<unsigned long>(lhs.get()) && reinterpret_cast<unsigned long>(rhs.get())
                && lhs->get_id() == rhs->get_id();
    };
    static const auto cmp_whole_value = [] (DecrementTask_ptr const & lhs, DecrementTask_ptr const & rhs)
    {
        return reinterpret_cast<unsigned long>(lhs.get()) && reinterpret_cast<unsigned long>(rhs.get())
                && *lhs == *rhs;
    };
}

struct AsyncDecrementTaskDAO::AsyncImpl
    : AsyncImplBase<DecrementTask_ptr, dtype(cmp_primary_key), dtype(cmp_whole_value), PgDecrementTaskDAO>
{
    AsyncImpl(svr::common::PropertiesFileReader& tempus_config, svr::dao::DataSource& data_source)
    :AsyncImplBase(tempus_config, data_source, cmp_primary_key, cmp_whole_value, 10, 10)
    {}
};

AsyncDecrementTaskDAO::AsyncDecrementTaskDAO(svr::common::PropertiesFileReader& tempus_config, svr::dao::DataSource& data_source)
: DecrementTaskDAO (tempus_config, data_source), pImpl(* new AsyncImpl(tempus_config, data_source))
{}

AsyncDecrementTaskDAO::~AsyncDecrementTaskDAO()
{
    delete & pImpl;
}

bigint AsyncDecrementTaskDAO::get_next_id()
{
    std::scoped_lock scope_guard(pImpl.pgMutex);
    return pImpl.pgDao.get_next_id();
}

bool AsyncDecrementTaskDAO::exists(const bigint id)
{
    std::scoped_lock scope_guard(pImpl.pgMutex);
    return pImpl.pgDao.exists(id);
}

int AsyncDecrementTaskDAO::save(const DecrementTask_ptr& decrementTask)
{
    std::scoped_lock scope_guard(pImpl.pgMutex);
    return pImpl.pgDao.save(decrementTask);
}

int AsyncDecrementTaskDAO::remove(const DecrementTask_ptr& decrementTask)
{
    std::scoped_lock scope_guard(pImpl.pgMutex);
    return pImpl.pgDao.remove(decrementTask);
}

DecrementTask_ptr AsyncDecrementTaskDAO::get_by_id(const bigint id)
{
    std::scoped_lock scope_guard(pImpl.pgMutex);
    return pImpl.pgDao.get_by_id(id);
}

}
}

