#include "AsyncEnsembleDAO.hpp"
#include "AsyncImplBase.hpp"
#include "../PgDAO/PgEnsembleDAO.hpp"
#include "model/Ensemble.hpp"

namespace svr {
namespace dao {

namespace {
static const auto cmp_primary_key = [](datamodel::Ensemble_ptr const &lhs, datamodel::Ensemble_ptr const &rhs) {
    return reinterpret_cast<uint64_t>(lhs.get()) && reinterpret_cast<uint64_t>(rhs.get())
           && lhs->get_id() == rhs->get_id();
};
static const auto cmp_whole_value = [](datamodel::Ensemble_ptr const &lhs, datamodel::Ensemble_ptr const &rhs) {
    return reinterpret_cast<uint64_t>(lhs.get()) && reinterpret_cast<uint64_t>(rhs.get())
           && *lhs == *rhs;
};
}

struct AsyncEnsembleDAO::AsyncImpl
        : AsyncImplBase<datamodel::Ensemble_ptr, DTYPE(cmp_primary_key), DTYPE(cmp_whole_value), PgEnsembleDAO> {
    AsyncImpl(common::PropertiesReader &tempus_config, dao::DataSource &data_source)
            : AsyncImplBase(tempus_config, data_source, cmp_primary_key, cmp_whole_value, 10, 10)
    {}
};

AsyncEnsembleDAO::AsyncEnsembleDAO(common::PropertiesReader &tempus_config, dao::DataSource &data_source)
        : EnsembleDAO(tempus_config, data_source), pImpl(*new AsyncImpl(tempus_config, data_source))
{}

AsyncEnsembleDAO::~AsyncEnsembleDAO()
{
    delete &pImpl;
}


bigint AsyncEnsembleDAO::get_next_id()
{
    std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.get_next_id();
}


datamodel::Ensemble_ptr AsyncEnsembleDAO::get_by_id(const bigint id)
{
    AsyncImpl::my_value_t value{new typename AsyncImpl::my_value_t::element_type()};
    value->set_id(id);

    if (pImpl.cached(value)) return value;

    const std::scoped_lock lg(pImpl.pgMutex);
    value = pImpl.pgDao.get_by_id(id);

    pImpl.cache_no_store(value);
    return value;
}

bool AsyncEnsembleDAO::exists(const bigint ensemble_id)
{
    AsyncImpl::my_value_t value{new typename AsyncImpl::my_value_t::element_type()};
    value->set_id(ensemble_id);
    return exists(value);
}

bool AsyncEnsembleDAO::exists(const datamodel::Ensemble_ptr &ensemble)
{
    AsyncImpl::my_value_t value = ensemble;
    if (pImpl.cached(value)) return true;
    const std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.exists(ensemble->get_id());
}

datamodel::Ensemble_ptr AsyncEnsembleDAO::get_by_dataset_and_decon_queue(const datamodel::Dataset_ptr &dataset, const datamodel::DeconQueue_ptr &decon_queue)
{
    pImpl.flush();
    const std::scoped_lock lg(pImpl.pgMutex);
    datamodel::Ensemble_ptr value = pImpl.pgDao.get_by_dataset_and_decon_queue(dataset, decon_queue);
    pImpl.cache_no_store(value);
    return value;
}

std::deque<datamodel::Ensemble_ptr> AsyncEnsembleDAO::find_all_ensembles_by_dataset_id(const bigint dataset_id)
{
    pImpl.flush();
    const std::scoped_lock lg(pImpl.pgMutex);
    std::deque<datamodel::Ensemble_ptr> values = pImpl.pgDao.find_all_ensembles_by_dataset_id(dataset_id);
    for (auto e: values) pImpl.cache_no_store(e);
    return values;
}

int AsyncEnsembleDAO::save(const datamodel::Ensemble_ptr &ensemble)
{
    pImpl.cache(ensemble);
    return 1;
}

int AsyncEnsembleDAO::remove(const datamodel::Ensemble_ptr &ensemble)
{
    return pImpl.remove(ensemble);
}

}
}
