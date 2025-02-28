#include "AsyncModelDAO.hpp"
#include "AsyncImplBase.hpp"
#include "../PgDAO/PgModelDAO.hpp"
#include <model/Model.hpp>

namespace svr {
namespace dao {

namespace {
static const auto cmp_primary_key = [](datamodel::Model_ptr const &lhs, datamodel::Model_ptr const &rhs) {
    return reinterpret_cast<uint64_t>(lhs.get()) && reinterpret_cast<uint64_t>(rhs.get())
           && lhs->get_id() == rhs->get_id();
};
static const auto cmp_whole_value = [](datamodel::Model_ptr const &lhs, datamodel::Model_ptr const &rhs) {
    return reinterpret_cast<uint64_t>(lhs.get()) && reinterpret_cast<uint64_t>(rhs.get())
           && *lhs == *rhs;
};
}

struct AsyncModelDAO::AsyncImpl
        : AsyncImplBase<datamodel::Model_ptr, DTYPE(cmp_primary_key), DTYPE(cmp_whole_value), PgModelDAO> {
    AsyncImpl(common::PropertiesReader &tempus_config, dao::DataSource &data_source)
            : AsyncImplBase(tempus_config, data_source, cmp_primary_key, cmp_whole_value, 10, 10)
    {}
};

AsyncModelDAO::AsyncModelDAO(common::PropertiesReader &tempus_config, dao::DataSource &data_source)
        : ModelDAO(tempus_config, data_source), pImpl(*new AsyncImpl(tempus_config, data_source))
{}

AsyncModelDAO::~AsyncModelDAO()
{
    delete &pImpl;
}

bigint AsyncModelDAO::get_next_id()
{
    const std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.get_next_id();
}

bool AsyncModelDAO::exists(const bigint model_id)
{
    AsyncImpl::my_value_t value{new typename AsyncImpl::my_value_t::element_type()};
    value->set_id(model_id);

    if (pImpl.cached(value))
        return true;

    const std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.exists(model_id);
}

int AsyncModelDAO::save(const datamodel::Model_ptr &model)
{
    pImpl.cache(model);
    return 1;
}

int AsyncModelDAO::remove(const datamodel::Model_ptr &model)
{
    return pImpl.remove(model);
}

int AsyncModelDAO::remove_by_ensemble_id(const bigint ensemble_id)
{
    pImpl.cache_remove_if([&ensemble_id](datamodel::Model_ptr const &model) { return model && model->get_ensemble_id() == ensemble_id; });
    pImpl.flush();
    const std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.remove_by_ensemble_id(ensemble_id);
}

datamodel::Model_ptr AsyncModelDAO::get_by_id(const bigint id)
{
    AsyncImpl::my_value_t value{new typename AsyncImpl::my_value_t::element_type()};
    value->set_id(id);

    if (pImpl.cached(value))
        return value;

    const std::scoped_lock lg(pImpl.pgMutex);
    value = pImpl.pgDao.get_by_id(id);

    pImpl.cache_no_store(value);
    return value;
}

datamodel::Model_ptr AsyncModelDAO::get_by_ensemble_id_and_decon_level(const bigint ensemble_id, size_t decon_level)
{
    pImpl.flush();
    const std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.get_by_ensemble_id_and_decon_level(ensemble_id, decon_level);
}

std::deque<datamodel::Model_ptr> AsyncModelDAO::get_all_ensemble_models(const bigint ensemble_id)
{
    pImpl.flush();
    const std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.get_all_ensemble_models(ensemble_id);
}

std::deque<datamodel::OnlineMIMOSVR_ptr> AsyncModelDAO::get_svr_by_model_id(const bigint model_id)
{
    pImpl.flush();
    const std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.get_svr_by_model_id(model_id);
}

}
}
