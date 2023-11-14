#include "AsyncEnsembleDAO.hpp"
#include "AsyncImplBase.hpp"
#include "../PgDAO/PgEnsembleDAO.hpp"
#include <model/Ensemble.hpp>

namespace svr {
namespace dao {

namespace
{
    static bool cmp_primary_key(Ensemble_ptr const & lhs, Ensemble_ptr const & rhs)
    {
        return reinterpret_cast<unsigned long>(lhs.get()) && reinterpret_cast<unsigned long>(rhs.get())
                && lhs->get_id() == rhs->get_id();
    }
    static bool cmp_whole_value(Ensemble_ptr const & lhs, Ensemble_ptr const & rhs)
    {
        return reinterpret_cast<unsigned long>(lhs.get()) && reinterpret_cast<unsigned long>(rhs.get())
                && *lhs == *rhs;
    }
}

struct AsyncEnsembleDAO::AsyncImpl
    : AsyncImplBase<Ensemble_ptr, decltype(std::ptr_fun(cmp_primary_key)), decltype(std::ptr_fun(cmp_whole_value)), PgEnsembleDAO>
{
    AsyncImpl(svr::common::PropertiesFileReader& sqlProperties, svr::dao::DataSource& dataSource)
    :AsyncImplBase(sqlProperties, dataSource, std::ptr_fun(cmp_primary_key), std::ptr_fun(cmp_whole_value), 10, 10)
    {}
};

AsyncEnsembleDAO::AsyncEnsembleDAO(svr::common::PropertiesFileReader& sqlProperties, svr::dao::DataSource& dataSource)
:EnsembleDAO(sqlProperties, dataSource), pImpl(*new AsyncImpl(sqlProperties, dataSource))
{}

AsyncEnsembleDAO::~AsyncEnsembleDAO()
{
    delete & pImpl;
}


bigint AsyncEnsembleDAO::get_next_id()
{
    std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.get_next_id();
}


Ensemble_ptr AsyncEnsembleDAO::get_by_id(bigint id)
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

bool AsyncEnsembleDAO::exists(bigint ensemble_id)
{
    AsyncImpl::my_value_t value {new typename AsyncImpl::my_value_t::element_type() };
    value->set_id(ensemble_id);

    return exists(value);
}

bool AsyncEnsembleDAO::exists(const Ensemble_ptr &ensemble)
{
    AsyncImpl::my_value_t value = ensemble;

    if(pImpl.cached(value))
        return true;

    std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.exists(ensemble->get_id());
}

Ensemble_ptr AsyncEnsembleDAO::get_by_dataset_and_decon_queue(const Dataset_ptr &dataset, const DeconQueue_ptr& decon_queue)
{
    pImpl.flush();
    std::scoped_lock lg(pImpl.pgMutex);
    Ensemble_ptr value = pImpl.pgDao.get_by_dataset_and_decon_queue(dataset, decon_queue);
    pImpl.cache_no_store(value);
    return value;
}

std::vector<Ensemble_ptr> AsyncEnsembleDAO::find_all_ensembles_by_dataset_id(bigint dataset_id)
{
    pImpl.flush();
    std::scoped_lock lg(pImpl.pgMutex);
    std::vector<Ensemble_ptr> values = pImpl.pgDao.find_all_ensembles_by_dataset_id(dataset_id);
    for (auto e: values) pImpl.cache_no_store(e);
    return values;
}

int AsyncEnsembleDAO::save(const Ensemble_ptr &ensemble)
{
    pImpl.cache(ensemble);
    return 1;
}

int AsyncEnsembleDAO::remove(const Ensemble_ptr &ensemble)
{
    return pImpl.remove(ensemble);
}

}
}
