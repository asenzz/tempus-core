#pragma once

#include <DAO/AbstractDAO.hpp>

namespace svr { namespace datamodel { class Dataset; } }
using Dataset_ptr = std::shared_ptr<svr::datamodel::Dataset>;

namespace svr{
namespace dao{

class DatasetDAO : public AbstractDAO{
public:
    static DatasetDAO * build(svr::common::PropertiesFileReader& sql_properties, svr::dao::DataSource& data_source, svr::common::ConcreteDaoType daoType, bool use_threadsafe_dao);

    explicit DatasetDAO(svr::common::PropertiesFileReader& sql_properties, svr::dao::DataSource& data_source);

    virtual bigint get_next_id() = 0;

    virtual bool exists(bigint dataset_id) = 0;
    virtual bool exists(std::string user_name, std::string dataset_name) = 0;

    virtual int save(const Dataset_ptr& dataset) = 0;
    virtual int remove(const Dataset_ptr& dataset) = 0;

    virtual Dataset_ptr get_by_id(bigint dataset_id) = 0;
    virtual Dataset_ptr get_by_name(std::string user_name, std::string dataset_name) = 0;

    virtual std::vector<Dataset_ptr> find_all_user_datasets(const std::string& user_name) = 0;

    virtual bool link_user_to_dataset(const std::string& user_name, const Dataset_ptr & dataset) = 0;
    virtual bool unlink_user_from_dataset(const std::string& user_name, const Dataset_ptr & dataset) = 0;

    typedef std::vector<std::pair<std::string, Dataset_ptr>> UserDatasetPairs;

    /** @return Sorted by (user.priority, dataset.priority) vector of pairs(user_name, Dataset_ptr)
     */
    virtual UserDatasetPairs get_active_datasets() = 0;
};

}
}

using DatasetDAO_ptr = std::shared_ptr<svr::dao::DatasetDAO>;
