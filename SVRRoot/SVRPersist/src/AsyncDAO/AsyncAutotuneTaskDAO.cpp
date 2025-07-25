#include "AsyncAutotuneTaskDAO.hpp"
#include "AsyncImplBase.hpp"
#include "../PgDAO/PgAutotuneTaskDAO.hpp"
#include <model/AutotuneTask.hpp>
#include <DAO/AutotuneTaskRowMapper.hpp>

namespace svr {
namespace dao {

namespace {
static const auto cmp_primary_key = [](AutotuneTask_ptr const &lhs, AutotuneTask_ptr const &rhs) {
    return reinterpret_cast<uint64_t>(lhs.get()) && reinterpret_cast<uint64_t>(rhs.get())
           && lhs->get_id() == rhs->get_id();
};
static const auto cmp_whole_value = [](AutotuneTask_ptr const &lhs, AutotuneTask_ptr const &rhs) {
    return reinterpret_cast<uint64_t>(lhs.get()) && reinterpret_cast<uint64_t>(rhs.get())
           && *lhs == *rhs;
};
}

struct AsyncAutotuneTaskDAO::AsyncImpl
        : AsyncImplBase<AutotuneTask_ptr, DTYPE(cmp_primary_key), DTYPE(cmp_whole_value), PgAutotuneTaskDAO> {
    AsyncImpl(common::PropertiesReader &tempus_config, dao::DataSource &data_source)
            : AsyncImplBase(tempus_config, data_source, cmp_primary_key, cmp_whole_value, 10, 10)
    {}
};

AsyncAutotuneTaskDAO::AsyncAutotuneTaskDAO(common::PropertiesReader &tempus_config, dao::DataSource &data_source)
        : AutotuneTaskDAO(tempus_config, data_source), pImpl(*new AsyncImpl(tempus_config, data_source))
{}

AsyncAutotuneTaskDAO::~AsyncAutotuneTaskDAO()
{
    delete &pImpl;
}

bigint AsyncAutotuneTaskDAO::get_next_id()
{
    const std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.get_next_id();
}

bool AsyncAutotuneTaskDAO::exists(const bigint id)
{
    AsyncImpl::my_value_t value{new typename AsyncImpl::my_value_t::element_type()};
    value->set_id(id);

    if (pImpl.cached(value)) return true;

    const std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.exists(id);
}

int AsyncAutotuneTaskDAO::save(const AutotuneTask_ptr &autotuneTask)
{
    pImpl.cache(autotuneTask);
    return 1;
}

int AsyncAutotuneTaskDAO::remove(const AutotuneTask_ptr &autotuneTask)
{
    return pImpl.remove(autotuneTask);
}

AutotuneTask_ptr AsyncAutotuneTaskDAO::get_by_id(const bigint id)
{
    AsyncImpl::my_value_t value{new typename AsyncImpl::my_value_t::element_type()};
    value->set_id(id);

    if (pImpl.cached(value)) return value;

    const std::scoped_lock lg(pImpl.pgMutex);
    value = pImpl.pgDao.get_by_id(id);

    pImpl.cache_no_store(value);
    return value;
}

std::vector<AutotuneTask_ptr> AsyncAutotuneTaskDAO::find_all_by_dataset_id(const bigint dataset_id)
{
    pImpl.flush();
    const std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.find_all_by_dataset_id(dataset_id);
}

}
}

