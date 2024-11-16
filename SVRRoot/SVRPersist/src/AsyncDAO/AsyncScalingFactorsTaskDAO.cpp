#include "AsyncScalingFactorsTaskDAO.hpp"
#include "AsyncImplBase.hpp"
#include "../PgDAO/PgScalingFactorsTaskDAO.hpp"
#include <model/ScalingFactorsTask.hpp>

namespace svr {
namespace dao {

namespace
{
    static const auto cmp_primary_key = [] (ScalingFactorsTask_ptr const & lhs, ScalingFactorsTask_ptr const & rhs)
    {
        return reinterpret_cast<uint64_t>(lhs.get()) && reinterpret_cast<uint64_t>(rhs.get())
                && lhs->get_id() == rhs->get_id();
    };
    static const auto cmp_whole_value = [] (ScalingFactorsTask_ptr const & lhs, ScalingFactorsTask_ptr const & rhs)
    {
        return reinterpret_cast<uint64_t>(lhs.get()) && reinterpret_cast<uint64_t>(rhs.get())
                && *lhs == *rhs;
    };
}

struct AsyncScalingFactorsTaskDAO::AsyncImpl
    : AsyncImplBase<ScalingFactorsTask_ptr, DTYPE(cmp_primary_key), DTYPE(cmp_whole_value), PgScalingFactorsTaskDAO>
{
    AsyncImpl(svr::common::PropertiesFileReader& tempus_config, svr::dao::DataSource& data_source)
    :AsyncImplBase(tempus_config, data_source, cmp_primary_key, cmp_whole_value, 10, 10)
    {}
};

AsyncScalingFactorsTaskDAO::AsyncScalingFactorsTaskDAO(svr::common::PropertiesFileReader& tempus_config, svr::dao::DataSource& data_source)
: ScalingFactorsTaskDAO(tempus_config, data_source), pImpl(* new AsyncImpl(tempus_config, data_source))
{}

AsyncScalingFactorsTaskDAO::~AsyncScalingFactorsTaskDAO()
{
    delete & pImpl;
}

bigint AsyncScalingFactorsTaskDAO::get_next_id()
{
    std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.get_next_id();
}

bool AsyncScalingFactorsTaskDAO::exists(const bigint id)
{
   AsyncImpl::my_value_t value {new typename AsyncImpl::my_value_t::element_type() };
    value->set_id(id);

    if(pImpl.cached(value))
        return true;

    std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.exists(id);
}

int AsyncScalingFactorsTaskDAO::save(const ScalingFactorsTask_ptr &scalingFactorsTask)
{
    pImpl.cache(scalingFactorsTask);
    return 1;
}

int AsyncScalingFactorsTaskDAO::remove(const ScalingFactorsTask_ptr &scalingFactorsTask)
{
    return pImpl.remove(scalingFactorsTask);
}

    ScalingFactorsTask_ptr AsyncScalingFactorsTaskDAO::get_by_id(const bigint id)
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

