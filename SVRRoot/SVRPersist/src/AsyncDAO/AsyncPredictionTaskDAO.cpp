#include "AsyncPredictionTaskDAO.hpp"
#include "AsyncImplBase.hpp"
#include "../PgDAO/PgPredictionTaskDAO.hpp"
#include <model/PredictionTask.hpp>

namespace svr {
namespace dao {

namespace
{
    static bool cmp_primary_key(PredictionTask_ptr const & lhs, PredictionTask_ptr const & rhs)
    {
        return reinterpret_cast<unsigned long>(lhs.get()) && reinterpret_cast<unsigned long>(rhs.get())
                && lhs->get_id() == rhs->get_id();
    }
    static bool cmp_whole_value(PredictionTask_ptr const & lhs, PredictionTask_ptr const & rhs)
    {
        return reinterpret_cast<unsigned long>(lhs.get()) && reinterpret_cast<unsigned long>(rhs.get())
                && *lhs == *rhs;
    }
}

struct AsyncPredictionTaskDAO::AsyncImpl
    : AsyncImplBase<PredictionTask_ptr, decltype(std::ptr_fun(cmp_primary_key)), decltype(std::ptr_fun(cmp_whole_value)), PgPredictionTaskDAO>
{
    AsyncImpl(svr::common::PropertiesFileReader& sqlProperties, svr::dao::DataSource& dataSource)
    :AsyncImplBase(sqlProperties, dataSource, std::ptr_fun(cmp_primary_key), std::ptr_fun(cmp_whole_value), 10, 10)
    {}
};

AsyncPredictionTaskDAO::AsyncPredictionTaskDAO(svr::common::PropertiesFileReader& sqlProperties, svr::dao::DataSource& dataSource)
: PredictionTaskDAO(sqlProperties, dataSource), pImpl(* new AsyncImpl(sqlProperties, dataSource))
{}

AsyncPredictionTaskDAO::~AsyncPredictionTaskDAO()
{
    delete & pImpl;
}

bigint AsyncPredictionTaskDAO::get_next_id()
{
    std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.get_next_id();
}

bool AsyncPredictionTaskDAO::exists(const bigint id)
{
   AsyncImpl::my_value_t value {new typename AsyncImpl::my_value_t::element_type() };
    value->set_id(id);

    if(pImpl.cached(value))
        return true;

    std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.exists(id);
}

int AsyncPredictionTaskDAO::save(const PredictionTask_ptr &predictionTask)
{
    pImpl.cache(predictionTask);
    return 1;
}

int AsyncPredictionTaskDAO::remove(const PredictionTask_ptr &predictionTask)
{
    return pImpl.remove(predictionTask);
}

PredictionTask_ptr AsyncPredictionTaskDAO::get_by_id(const bigint id)
{
    AsyncImpl::my_value_t value {new typename AsyncImpl::my_value_t::element_type() };
    value->set_id(id);

    if(pImpl.cached(value))
        return value;

    std::scoped_lock lg(pImpl.pgMutex);
    value = pImpl.pgDao.get_by_id(id);

    pImpl.cache_no_store(value);
    return value;
}

} /* namespace dao */
} /* namespace svr */

