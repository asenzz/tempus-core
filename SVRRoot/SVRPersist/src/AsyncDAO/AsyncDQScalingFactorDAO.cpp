#include "AsyncDQScalingFactorDAO.hpp"
#include "AsyncImplBase.hpp"
#include <model/DQScalingFactor.hpp>
#include <DAO/DataSource.hpp>
#include <DAO/DQScalingFactorRowMapper.hpp>
#include "../PgDAO/PgDQScalingFactorDAO.hpp"

namespace svr {
namespace dao {

namespace
{
    static bool cmp_primary_key(datamodel::DQScalingFactor_ptr const & lhs, datamodel::DQScalingFactor_ptr const & rhs)
    {
        return reinterpret_cast<unsigned long>(lhs.get()) && reinterpret_cast<unsigned long>(rhs.get())
                && lhs->get_id() == rhs->get_id();
    }
    static bool cmp_whole_value(datamodel::DQScalingFactor_ptr const & lhs, datamodel::DQScalingFactor_ptr const & rhs)
    {
        return reinterpret_cast<unsigned long>(lhs.get()) && reinterpret_cast<unsigned long>(rhs.get())
                && *lhs == *rhs;
    }
}

struct AsyncDQScalingFactorDAO::AsyncImpl
    : AsyncImplBase<datamodel::DQScalingFactor_ptr, decltype(std::ptr_fun(cmp_primary_key)), decltype(std::ptr_fun(cmp_whole_value)), PgDQScalingFactorDAO>
{
    AsyncImpl(svr::common::PropertiesFileReader& sqlProperties, svr::dao::DataSource& dataSource)
    :AsyncImplBase(sqlProperties, dataSource, std::ptr_fun(cmp_primary_key), std::ptr_fun(cmp_whole_value), 10, 10)
    {}
};

AsyncDQScalingFactorDAO::AsyncDQScalingFactorDAO(svr::common::PropertiesFileReader& sqlProperties, svr::dao::DataSource& dataSource)
: DQScalingFactorDAO (sqlProperties, dataSource), pImpl(* new AsyncImpl(sqlProperties, dataSource))
{}

AsyncDQScalingFactorDAO::~AsyncDQScalingFactorDAO()
{
    delete & pImpl;
}

bigint AsyncDQScalingFactorDAO::get_next_id()
{
    std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.get_next_id();
}

bool AsyncDQScalingFactorDAO::exists(const datamodel::DQScalingFactor_ptr& dq_scaling_factor)
{
    std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.exists(dq_scaling_factor);
}

int AsyncDQScalingFactorDAO::save(const datamodel::DQScalingFactor_ptr& dq_scaling_factor)
{
    std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.save(dq_scaling_factor);
}

int AsyncDQScalingFactorDAO::remove(const datamodel::DQScalingFactor_ptr& dq_scaling_factor)
{
    std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.remove(dq_scaling_factor);
}

svr::datamodel::dq_scaling_factor_container_t AsyncDQScalingFactorDAO::find_all_by_dataset_id(const bigint dataset_id)
{
    std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.find_all_by_dataset_id(dataset_id);
}


}
}

