#include "TsDatasetDAO.hpp"


namespace svr{
namespace dao{

DEFINE_THREADSAFE_DAO_CONSTRUCTOR (TsDatasetDAO, DatasetDAO)
{}

bigint TsDatasetDAO::get_next_id()
{
    return ts_call<bigint>(&DatasetDAO::get_next_id);
}


bool TsDatasetDAO::exists(const bigint dataset_id)
{
    std::scoped_lock<std::recursive_mutex> scoped_lock(mutex);
    return dao->exists(dataset_id);
}


bool TsDatasetDAO::exists(const std::string &user_name, const std::string &dataset_name)
{
    std::scoped_lock<std::recursive_mutex> scoped_lock(mutex);
    return dao->exists(user_name, dataset_name);
}


int TsDatasetDAO::save(const datamodel::Dataset_ptr& dataset)
{
    return ts_call<int>(&DatasetDAO::save, dataset);
}


int TsDatasetDAO::remove(const datamodel::Dataset_ptr& dataset)
{
    return ts_call<int>(&DatasetDAO::remove, dataset);
}


datamodel::Dataset_ptr TsDatasetDAO::get_by_id(const bigint dataset_id)
{
    return ts_call<datamodel::Dataset_ptr>(&DatasetDAO::get_by_id, dataset_id);
}


datamodel::Dataset_ptr TsDatasetDAO::get_by_name(const std::string &user_name, const std::string &dataset_name)
{
    return ts_call<datamodel::Dataset_ptr>(&DatasetDAO::get_by_name, user_name, dataset_name);
}


std::deque<datamodel::Dataset_ptr> TsDatasetDAO::find_all_user_datasets(const std::string& user_name)
{
    return ts_call<std::deque<datamodel::Dataset_ptr>>(&DatasetDAO::find_all_user_datasets, user_name);
}


bool TsDatasetDAO::link_user_to_dataset(const std::string& user_name, const datamodel::Dataset_ptr & dataset)
{
    return ts_call<bool>(&DatasetDAO::link_user_to_dataset, user_name, dataset);
}


bool TsDatasetDAO::unlink_user_from_dataset(const std::string& user_name, const datamodel::Dataset_ptr & dataset)
{
    return ts_call<bool>(&DatasetDAO::unlink_user_from_dataset, user_name, dataset);
}


TsDatasetDAO::UserDatasetPairs TsDatasetDAO::get_active_datasets()
{
    return ts_call<TsDatasetDAO::UserDatasetPairs>(&DatasetDAO::get_active_datasets);
}


}
}
