#include "TsDatasetDAO.hpp"


namespace svr{
namespace dao{

DEFINE_THREADSAFE_DAO_CONSTRUCTOR (TsDatasetDAO, DatasetDAO)
{}

bigint TsDatasetDAO::get_next_id()
{
    return ts_call<bigint>(&DatasetDAO::get_next_id);
}


bool TsDatasetDAO::exists(bigint dataset_id)
{
    std::scoped_lock<std::recursive_mutex> scoped_lock(mutex);
    return dao->exists(dataset_id);
}


bool TsDatasetDAO::exists(std::string user_name, std::string dataset_name)
{
    std::scoped_lock<std::recursive_mutex> scoped_lock(mutex);
    return dao->exists(user_name, dataset_name);
}


int TsDatasetDAO::save(const Dataset_ptr& dataset)
{
    return ts_call<int>(&DatasetDAO::save, dataset);
}


int TsDatasetDAO::remove(const Dataset_ptr& dataset)
{
    return ts_call<int>(&DatasetDAO::remove, dataset);
}


Dataset_ptr TsDatasetDAO::get_by_id(bigint dataset_id)
{
    return ts_call<Dataset_ptr>(&DatasetDAO::get_by_id, dataset_id);
}


Dataset_ptr TsDatasetDAO::get_by_name(std::string user_name, std::string dataset_name)
{
    return ts_call<Dataset_ptr>(&DatasetDAO::get_by_name, user_name, dataset_name);
}


std::vector<Dataset_ptr> TsDatasetDAO::find_all_user_datasets(const std::string& user_name)
{
    return ts_call<std::vector<Dataset_ptr>>(&DatasetDAO::find_all_user_datasets, user_name);
}


bool TsDatasetDAO::link_user_to_dataset(const std::string& user_name, const Dataset_ptr & dataset)
{
    return ts_call<bool>(&DatasetDAO::link_user_to_dataset, user_name, dataset);
}


bool TsDatasetDAO::unlink_user_from_dataset(const std::string& user_name, const Dataset_ptr & dataset)
{
    return ts_call<bool>(&DatasetDAO::unlink_user_from_dataset, user_name, dataset);
}


TsDatasetDAO::UserDatasetPairs TsDatasetDAO::get_active_datasets()
{
    return ts_call<TsDatasetDAO::UserDatasetPairs>(&DatasetDAO::get_active_datasets);
}


}
}
