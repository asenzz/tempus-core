#include "AsyncWScalingFactorDAO.hpp"
#include "AsyncImplBase.hpp"
#include <model/WScalingFactor.hpp>
#include <DAO/DataSource.hpp>
#include <DAO/WScalingFactorRowMapper.hpp>
#include "../PgDAO/PgWScalingFactorDAO.hpp"

namespace svr {
namespace dao {

namespace {
static const auto cmp_primary_key = [](datamodel::WScalingFactor_ptr const &lhs, datamodel::WScalingFactor_ptr const &rhs) {
    return reinterpret_cast<uint64_t>(lhs.get()) && reinterpret_cast<uint64_t>(rhs.get()) && lhs->get_id() == rhs->get_id();
};

static const auto cmp_whole_value = [](datamodel::WScalingFactor_ptr const &lhs, datamodel::WScalingFactor_ptr const &rhs) {
    return reinterpret_cast<uint64_t>(lhs.get()) && reinterpret_cast<uint64_t>(rhs.get()) && *lhs == *rhs;
};
}

struct AsyncWScalingFactorDAO::AsyncImpl
        : AsyncImplBase<datamodel::WScalingFactor_ptr, DTYPE(cmp_primary_key), DTYPE(cmp_whole_value), PgWScalingFactorDAO>
{
    AsyncImpl(common::PropertiesFileReader &tempus_config, dao::DataSource &data_source)
            : AsyncImplBase(tempus_config, data_source, cmp_primary_key, cmp_whole_value, 10, 10)
    {}
};

AsyncWScalingFactorDAO::AsyncWScalingFactorDAO(common::PropertiesFileReader &tempus_config, dao::DataSource &data_source)
        : WScalingFactorDAO(tempus_config, data_source), pImpl(*new AsyncImpl(tempus_config, data_source))
{}

AsyncWScalingFactorDAO::~AsyncWScalingFactorDAO()
{
    delete &pImpl;
}

bigint AsyncWScalingFactorDAO::get_next_id()
{
    const std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.get_next_id();
}

bool AsyncWScalingFactorDAO::exists(const bigint id)
{
    const std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.exists(id);
}

int AsyncWScalingFactorDAO::save(const datamodel::WScalingFactor_ptr &p_w_scaling_factor)
{
    const std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.save(p_w_scaling_factor);
}

int AsyncWScalingFactorDAO::remove(const datamodel::WScalingFactor_ptr &p_w_scaling_factor)
{
    const std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.remove(p_w_scaling_factor);
}

std::deque<datamodel::WScalingFactor_ptr> AsyncWScalingFactorDAO::find_all_by_dataset_id(const bigint dataset_id)
{
    const std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.find_all_by_dataset_id(dataset_id);
}


}
}

