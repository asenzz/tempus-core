#include "AsyncIQScalingFactorDAO.hpp"
#include "AsyncImplBase.hpp"
#include "model/IQScalingFactor.hpp"
#include "DAO/DataSource.hpp"
#include "DAO/IQScalingFactorRowMapper.hpp"
#include "../PgDAO/PgIQScalingFactorDAO.hpp"

namespace svr {
namespace dao {

namespace {
static const auto cmp_primary_key = [](datamodel::IQScalingFactor_ptr const &lhs, datamodel::IQScalingFactor_ptr const &rhs) {
    return reinterpret_cast<uint64_t>(lhs.get()) && reinterpret_cast<uint64_t>(rhs.get())
           && lhs->get_id() == rhs->get_id();
};

static const auto cmp_whole_value = [](datamodel::IQScalingFactor_ptr const &lhs, datamodel::IQScalingFactor_ptr const &rhs) {
    return reinterpret_cast<uint64_t>(lhs.get()) && reinterpret_cast<uint64_t>(rhs.get())
           && *lhs == *rhs;
};
}

struct AsyncIQScalingFactorDAO::AsyncImpl
        : AsyncImplBase<datamodel::IQScalingFactor_ptr, DTYPE(cmp_primary_key), DTYPE(cmp_whole_value), PgIQScalingFactorDAO>
{
    AsyncImpl(common::PropertiesReader &tempus_config, dao::DataSource &data_source)
            : AsyncImplBase(tempus_config, data_source, cmp_primary_key, cmp_whole_value, 10, 10)
    {}
};

AsyncIQScalingFactorDAO::AsyncIQScalingFactorDAO(common::PropertiesReader &tempus_config, dao::DataSource &data_source)
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

int AsyncIQScalingFactorDAO::save(const datamodel::IQScalingFactor_ptr &p_iq_scaling_factor)
{
    const std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.save(p_iq_scaling_factor);
}

int AsyncIQScalingFactorDAO::remove(const datamodel::IQScalingFactor_ptr &p_iq_scaling_factor)
{
    const std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.remove(p_iq_scaling_factor);
}

std::deque<datamodel::IQScalingFactor_ptr> AsyncIQScalingFactorDAO::find_all_by_dataset_id(const bigint dataset_id)
{
    const std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.find_all_by_dataset_id(dataset_id);
}


}
}

