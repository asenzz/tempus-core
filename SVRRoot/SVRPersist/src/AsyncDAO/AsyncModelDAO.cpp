#include "AsyncModelDAO.hpp"
#include "AsyncImplBase.hpp"
#include "../PgDAO/PgModelDAO.hpp"
#include <model/Model.hpp>

namespace svr {
namespace dao {

namespace
{
    static bool cmp_primary_key(Model_ptr const & lhs, Model_ptr const & rhs)
    {
        return reinterpret_cast<unsigned long>(lhs.get()) && reinterpret_cast<unsigned long>(rhs.get())
                && lhs->get_id() == rhs->get_id();
    }
    static bool cmp_whole_value(Model_ptr const & lhs, Model_ptr const & rhs)
    {
        return reinterpret_cast<unsigned long>(lhs.get()) && reinterpret_cast<unsigned long>(rhs.get())
                && *lhs == *rhs;
    }
}

struct AsyncModelDAO::AsyncImpl
    : AsyncImplBase<Model_ptr, decltype(std::ptr_fun(cmp_primary_key)), decltype(std::ptr_fun(cmp_whole_value)), PgModelDAO>
{
    AsyncImpl(svr::common::PropertiesFileReader& sqlProperties, svr::dao::DataSource& dataSource)
    :AsyncImplBase(sqlProperties, dataSource, std::ptr_fun(cmp_primary_key), std::ptr_fun(cmp_whole_value), 10, 10)
    {}
};

AsyncModelDAO::AsyncModelDAO(svr::common::PropertiesFileReader& sqlProperties, svr::dao::DataSource& dataSource)
: ModelDAO(sqlProperties, dataSource), pImpl (* new  AsyncImpl(sqlProperties, dataSource))
{}

AsyncModelDAO::~AsyncModelDAO()
{
    delete & pImpl;
}

bigint AsyncModelDAO::get_next_id()
{
    std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.get_next_id();
}

bool AsyncModelDAO::exists(bigint model_id)
{
    AsyncImpl::my_value_t value {new typename AsyncImpl::my_value_t::element_type() };
    value->set_id(model_id);

    if(pImpl.cached(value))
        return true;

    std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.exists(model_id);
}

int AsyncModelDAO::save(const Model_ptr &model)
{
    pImpl.cache(model);
    return 1;
}

int AsyncModelDAO::remove(const Model_ptr &model)
{
    return pImpl.remove(model);
}

int AsyncModelDAO::remove_by_ensemble_id(bigint ensemble_id)
{
    pImpl.cache_remove_if([&ensemble_id](Model_ptr const & model){ return model && model->get_ensemble_id() == ensemble_id; });
    pImpl.flush();
    std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.remove_by_ensemble_id(ensemble_id);
}

Model_ptr AsyncModelDAO::get_by_id(bigint id)
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

Model_ptr AsyncModelDAO::get_by_ensemble_id_and_decon_level(bigint ensemble_id, size_t decon_level)
{
    pImpl.flush();
    std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.get_by_ensemble_id_and_decon_level(ensemble_id, decon_level);
}

std::vector<Model_ptr> AsyncModelDAO::get_all_ensemble_models(bigint ensemble_id)
{
    pImpl.flush();
    std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.get_all_ensemble_models(ensemble_id);
}

} }
