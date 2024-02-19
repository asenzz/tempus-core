#include "AsyncIQScalingFactorDAO.hpp"
#include "AsyncImplBase.hpp"
#include <model/IQScalingFactor.hpp>
#include <DAO/DataSource.hpp>
#include <DAO/IQScalingFactorRowMapper.hpp>
#include "../PgDAO/PgIQScalingFactorDAO.hpp"

namespace svr {
namespace dao {

namespace {
static bool cmp_primary_key(IQScalingFactor_ptr const &lhs, IQScalingFactor_ptr const &rhs)
{
    return reinterpret_cast<unsigned long>(lhs.get()) && reinterpret_cast<unsigned long>(rhs.get())
           && lhs->get_id() == rhs->get_id();
}

static bool cmp_whole_value(IQScalingFactor_ptr const &lhs, IQScalingFactor_ptr const &rhs)
{
    return reinterpret_cast<unsigned long>(lhs.get()) && reinterpret_cast<unsigned long>(rhs.get())
           && *lhs == *rhs;
}
}

struct AsyncIQScalingFactorDAO::AsyncImpl
        : AsyncImplBase<IQScalingFactor_ptr, decltype(std::ptr_fun(cmp_primary_key)), decltype(std::ptr_fun(cmp_whole_value)), PgIQScalingFactorDAO>
{
    AsyncImpl(svr::common::PropertiesFileReader &tempus_config, svr::dao::DataSource &data_source)
            : AsyncImplBase(tempus_config, data_source, std::ptr_fun(cmp_primary_key), std::ptr_fun(cmp_whole_value), 10, 10)
    {}
};

AsyncIQScalingFactorDAO::AsyncIQScalingFactorDAO(svr::common::PropertiesFileReader &tempus_config, svr::dao::DataSource &data_source)
        : IQScalingFactorDAO(tempus_config, data_source), pImpl(*new AsyncImpl(tempus_config, data_source))
{}

AsyncIQScalingFactorDAO::~AsyncIQScalingFactorDAO()
{
    delete &pImpl;
}

bigint AsyncIQScalingFactorDAO::get_next_id()
{
    const std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.get_next_id();
}

bool AsyncIQScalingFactorDAO::exists(const bigint id)
{
    const std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.exists(id);
}

int AsyncIQScalingFactorDAO::save(const IQScalingFactor_ptr &p_iq_scaling_factor)
{
    const std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.save(p_iq_scaling_factor);
}

int AsyncIQScalingFactorDAO::remove(const IQScalingFactor_ptr &p_iq_scaling_factor)
{
    const std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.remove(p_iq_scaling_factor);
}

std::deque<IQScalingFactor_ptr> AsyncIQScalingFactorDAO::find_all_by_dataset_id(const bigint dataset_id)
{
    const std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.find_all_by_dataset_id(dataset_id);
}


}
}

