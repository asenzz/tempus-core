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
        return reinterpret_cast<uint64_t>(lhs.get()) && reinterpret_cast<uint64_t>(rhs.get())
                && lhs->get_id() == rhs->get_id();
    }
    static bool cmp_whole_value(datamodel::DQScalingFactor_ptr const & lhs, datamodel::DQScalingFactor_ptr const & rhs)
    {
        return reinterpret_cast<uint64_t>(lhs.get()) && reinterpret_cast<uint64_t>(rhs.get())
                && *lhs == *rhs;
    }
}

struct AsyncDQScalingFactorDAO::AsyncImpl
    : AsyncImplBase<datamodel::DQScalingFactor_ptr, DTYPE(cmp_primary_key), DTYPE(cmp_whole_value), PgDQScalingFactorDAO>
{
    AsyncImpl(svr::common::PropertiesFileReader& tempus_config, svr::dao::DataSource& data_source)
    :AsyncImplBase(tempus_config, data_source, cmp_primary_key, cmp_whole_value, 10, 10)
    {}
};

AsyncDQScalingFactorDAO::AsyncDQScalingFactorDAO(svr::common::PropertiesFileReader& tempus_config, svr::dao::DataSource& data_source)
: DQScalingFactorDAO (tempus_config, data_source), pImpl(* new AsyncImpl(tempus_config, data_source))
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

svr::datamodel::dq_scaling_factor_container_t AsyncDQScalingFactorDAO::find_all_by_model_id(const bigint model_id)
{
    std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.find_all_by_model_id(model_id);
}


}
}

