#pragma once

#include "TsDaoBase.hpp"
#include <DAO/DatasetDAO.hpp>

namespace svr{
namespace dao{

THREADSAFE_DAO_CLASS_DECLARATION_HEADER (TsDatasetDAO, DatasetDAO)

    virtual bigint get_next_id();

    virtual bool exists(bigint dataset_id);
    virtual bool exists(std::string user_name, std::string dataset_name);

    virtual int save(const Dataset_ptr& dataset);
    virtual int remove(const Dataset_ptr& dataset);

    virtual Dataset_ptr get_by_id(bigint dataset_id);
    virtual Dataset_ptr get_by_name(std::string user_name, std::string dataset_name);

    virtual std::vector<Dataset_ptr> find_all_user_datasets(const std::string& user_name);

    virtual bool link_user_to_dataset(const std::string& user_name, const Dataset_ptr & dataset);
    virtual bool unlink_user_from_dataset(const std::string& user_name, const Dataset_ptr & dataset);

    virtual DatasetDAO::UserDatasetPairs get_active_datasets();
};

}
}

