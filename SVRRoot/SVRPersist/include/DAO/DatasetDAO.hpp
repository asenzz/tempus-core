#pragma once

#include "DAO/AbstractDAO.hpp"
#include "model/Dataset.hpp"


namespace svr {
namespace dao {


class DatasetDAO : public AbstractDAO{
public:
    static DatasetDAO * build(svr::common::PropertiesFileReader& sql_properties, svr::dao::DataSource& data_source, svr::common::ConcreteDaoType daoType, bool use_threadsafe_dao);

    explicit DatasetDAO(svr::common::PropertiesFileReader& sql_properties, svr::dao::DataSource& data_source);

    virtual bigint get_next_id() = 0;

    virtual bool exists(const bigint dataset_id) = 0;
    virtual bool exists(const std::string &user_name, const std::string &dataset_name) = 0;

    virtual int save(const datamodel::Dataset_ptr& dataset) = 0;
    virtual int remove(const datamodel::Dataset_ptr& dataset) = 0;

    virtual datamodel::Dataset_ptr get_by_id(const bigint dataset_id) = 0;
    virtual datamodel::Dataset_ptr get_by_name(const std::string &user_name, const std::string &dataset_name) = 0;

    virtual std::deque<datamodel::Dataset_ptr> find_all_user_datasets(const std::string& user_name) = 0;

    virtual bool link_user_to_dataset(const std::string& user_name, const datamodel::Dataset_ptr & dataset) = 0;
    virtual bool unlink_user_from_dataset(const std::string& user_name, const datamodel::Dataset_ptr & dataset) = 0;

    typedef std::vector<std::pair<std::string, datamodel::Dataset_ptr>> UserDatasetPairs;

    /** @return Sorted by (user.priority, dataset.priority) vector of pairs(user_name, datamodel::Dataset_ptr)
     */
    virtual UserDatasetPairs get_active_datasets() = 0;
};

}
}

using DatasetDAO_ptr = std::shared_ptr<svr::dao::DatasetDAO>;
