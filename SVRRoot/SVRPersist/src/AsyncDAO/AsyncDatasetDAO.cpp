#include "AsyncDatasetDAO.hpp"
#include "AsyncImplBase.hpp"
#include "../PgDAO/PgDatasetDAO.hpp"
#include <model/Dataset.hpp>

namespace svr { namespace dao {

namespace
{
    static bool cmp_primary_key(Dataset_ptr const & lhs, Dataset_ptr const & rhs)
    {
        return reinterpret_cast<unsigned long>(lhs.get()) && reinterpret_cast<unsigned long>(rhs.get())
                && lhs->get_id() == rhs->get_id();
    }
    static bool cmp_whole_value(Dataset_ptr const & lhs, Dataset_ptr const & rhs)
    {
        return reinterpret_cast<unsigned long>(lhs.get()) && reinterpret_cast<unsigned long>(rhs.get())
                && *lhs == *rhs;
    }
}

struct AsyncDatasetDAO::AsyncImpl 
    : AsyncImplBase<Dataset_ptr, decltype(std::ptr_fun(cmp_primary_key)), decltype(std::ptr_fun(cmp_whole_value)), PgDatasetDAO>
{
    AsyncImpl(svr::common::PropertiesFileReader& sqlProperties, svr::dao::DataSource& dataSource)
    :AsyncImplBase(sqlProperties, dataSource, std::ptr_fun(cmp_primary_key), std::ptr_fun(cmp_whole_value), 10, 10)
    {}
};

AsyncDatasetDAO::AsyncDatasetDAO(svr::common::PropertiesFileReader& sqlProperties, svr::dao::DataSource& dataSource)
:DatasetDAO(sqlProperties, dataSource), pImpl(*new AsyncDatasetDAO::AsyncImpl(sqlProperties, dataSource))
{}

AsyncDatasetDAO::~AsyncDatasetDAO()
{
    delete & pImpl;
}

bigint AsyncDatasetDAO::get_next_id()
{
    std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.get_next_id();
}

bool AsyncDatasetDAO::exists(bigint id)
{
    AsyncImpl::my_value_t value {new typename AsyncImpl::my_value_t::element_type() };
    value->set_id(id);

    if(pImpl.cached(value))
        return true;

    std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.exists(id);
}

bool AsyncDatasetDAO::exists(std::string user_name, std::string dataset_name)
{
    pImpl.flush();
    std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.exists(user_name, dataset_name);
}

int AsyncDatasetDAO::save(const Dataset_ptr &dataset)
{
    pImpl.cache(dataset);
    return 1;
}

int AsyncDatasetDAO::remove(const Dataset_ptr &dataset)
{
    return pImpl.remove(dataset);
}

Dataset_ptr AsyncDatasetDAO::get_by_id(bigint dataset_id)
{
    AsyncImpl::my_value_t value {new typename AsyncImpl::my_value_t::element_type() };
    value->set_id(dataset_id);

    if(pImpl.cached(value))
        return value;

    std::scoped_lock lg(pImpl.pgMutex);
    value = pImpl.pgDao.get_by_id(dataset_id);

    pImpl.cache_no_store(value);
    return value;
}

Dataset_ptr AsyncDatasetDAO::get_by_name(std::string user_name, std::string dataset_name)
{
    pImpl.flush();
    std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.get_by_name(user_name, dataset_name);
}

std::vector<Dataset_ptr> AsyncDatasetDAO::find_all_user_datasets(const std::string &user_name)
{
    pImpl.flush();
    std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.find_all_user_datasets(user_name);
}

bool AsyncDatasetDAO::link_user_to_dataset(const std::string& user_name, const Dataset_ptr & dataset)
{
    pImpl.flush();
    std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.link_user_to_dataset(user_name, dataset);
}

bool AsyncDatasetDAO::unlink_user_from_dataset(const std::string& user_name, const Dataset_ptr & dataset)
{
    pImpl.flush();
    std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.unlink_user_from_dataset(user_name, dataset);
}

PgDatasetDAO::UserDatasetPairs AsyncDatasetDAO::get_active_datasets()
{
    pImpl.flush();
    std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.get_active_datasets();
}

}
}
