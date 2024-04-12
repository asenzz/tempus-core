#include "AsyncDatasetDAO.hpp"
#include "AsyncImplBase.hpp"
#include "../PgDAO/PgDatasetDAO.hpp"
#include <model/Dataset.hpp>

namespace svr { namespace dao {

namespace
{
    static const auto cmp_primary_key = [] (datamodel::Dataset_ptr const & lhs, datamodel::Dataset_ptr const & rhs)
    {
        return reinterpret_cast<unsigned long>(lhs.get()) && reinterpret_cast<unsigned long>(rhs.get())
                && lhs->get_id() == rhs->get_id();
    };
    static const auto cmp_whole_value = [] (datamodel::Dataset_ptr const & lhs, datamodel::Dataset_ptr const & rhs)
    {
        return reinterpret_cast<unsigned long>(lhs.get()) && reinterpret_cast<unsigned long>(rhs.get())
                && *lhs == *rhs;
    };
}

struct AsyncDatasetDAO::AsyncImpl 
    : AsyncImplBase<datamodel::Dataset_ptr, dtype(cmp_primary_key), dtype(cmp_whole_value), PgDatasetDAO>
{
    AsyncImpl(svr::common::PropertiesFileReader& tempus_config, svr::dao::DataSource& data_source)
    :AsyncImplBase(tempus_config, data_source, cmp_primary_key, cmp_whole_value, 10, 10)
    {}
};

AsyncDatasetDAO::AsyncDatasetDAO(svr::common::PropertiesFileReader& tempus_config, svr::dao::DataSource& data_source)
:DatasetDAO(tempus_config, data_source), pImpl(*new AsyncDatasetDAO::AsyncImpl(tempus_config, data_source))
{}

AsyncDatasetDAO::~AsyncDatasetDAO()
{
    delete & pImpl;
}

bigint AsyncDatasetDAO::get_next_id()
{
    const std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.get_next_id();
}

size_t AsyncDatasetDAO::get_level_count(const bigint dataset_id)
{
    const std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.get_level_count(dataset_id);
}

bool AsyncDatasetDAO::exists(const bigint id)
{
    AsyncImpl::my_value_t value {new typename AsyncImpl::my_value_t::element_type() };
    value->set_id(id);

    if (pImpl.cached(value)) return true;

    const std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.exists(id);
}

bool AsyncDatasetDAO::exists(const std::string &user_name, const std::string &dataset_name)
{
    pImpl.flush();
    const std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.exists(user_name, dataset_name);
}

int AsyncDatasetDAO::save(const datamodel::Dataset_ptr &dataset)
{
    pImpl.cache(dataset);
    return 1;
}

int AsyncDatasetDAO::remove(const datamodel::Dataset_ptr &dataset)
{
    return pImpl.remove(dataset);
}

datamodel::Dataset_ptr AsyncDatasetDAO::get_by_id(const bigint dataset_id)
{
    AsyncImpl::my_value_t value {new typename AsyncImpl::my_value_t::element_type() };
    value->set_id(dataset_id);

    if(pImpl.cached(value)) return value;

    const std::scoped_lock lg(pImpl.pgMutex);
    value = pImpl.pgDao.get_by_id(dataset_id);

    pImpl.cache_no_store(value);
    return value;
}

datamodel::Dataset_ptr AsyncDatasetDAO::get_by_name(const std::string &user_name, const std::string &dataset_name)
{
    pImpl.flush();
    const std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.get_by_name(user_name, dataset_name);
}

std::deque<datamodel::Dataset_ptr> AsyncDatasetDAO::find_all_user_datasets(const std::string &user_name)
{
    pImpl.flush();
    const std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.find_all_user_datasets(user_name);
}

bool AsyncDatasetDAO::link_user_to_dataset(const std::string& user_name, const datamodel::Dataset_ptr & dataset)
{
    pImpl.flush();
    const std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.link_user_to_dataset(user_name, dataset);
}

bool AsyncDatasetDAO::unlink_user_from_dataset(const std::string& user_name, const datamodel::Dataset_ptr & dataset)
{
    pImpl.flush();
    const std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.unlink_user_from_dataset(user_name, dataset);
}

PgDatasetDAO::UserDatasetPairs AsyncDatasetDAO::get_active_datasets()
{
    pImpl.flush();
    const std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.get_active_datasets();
}

}
}
