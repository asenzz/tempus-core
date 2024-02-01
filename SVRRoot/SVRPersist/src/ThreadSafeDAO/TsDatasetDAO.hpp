#pragma once

#include "TsDaoBase.hpp"
#include <DAO/DatasetDAO.hpp>

namespace svr{
namespace dao{

THREADSAFE_DAO_CLASS_DECLARATION_HEADER (TsDatasetDAO, DatasetDAO)

    virtual bigint get_next_id();

    virtual bool exists(const bigint dataset_id);
    virtual bool exists(const std::string &user_name, const std::string &dataset_name);

    virtual int save(const datamodel::Dataset_ptr& dataset);
    virtual int remove(const datamodel::Dataset_ptr& dataset);

    virtual datamodel::Dataset_ptr get_by_id(const bigint dataset_id);
    virtual datamodel::Dataset_ptr get_by_name(const std::string &user_name, const std::string &dataset_name);

    virtual std::deque<datamodel::Dataset_ptr> find_all_user_datasets(const std::string& user_name);

    virtual bool link_user_to_dataset(const std::string& user_name, const datamodel::Dataset_ptr & dataset);
    virtual bool unlink_user_from_dataset(const std::string& user_name, const datamodel::Dataset_ptr & dataset);

    virtual DatasetDAO::UserDatasetPairs get_active_datasets();
};

}
}

