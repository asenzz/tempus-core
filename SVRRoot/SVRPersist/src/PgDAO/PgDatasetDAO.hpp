#ifndef PGDATASETDAO_HPP
#define PGDATASETDAO_HPP

#include <DAO/DatasetDAO.hpp>

namespace svr{
namespace dao{

class PgDatasetDAO : public DatasetDAO{
public:
    explicit PgDatasetDAO(svr::common::PropertiesFileReader& sql_properties, svr::dao::DataSource& data_source);

    bigint get_next_id();

    bool exists(const bigint dataset_id);
    bool exists(const std::string &user_name, const std::string &dataset_name);

    int save(const datamodel::Dataset_ptr& dataset);
    int remove(const datamodel::Dataset_ptr& dataset);

    datamodel::Dataset_ptr get_by_id(const bigint dataset_id);
    datamodel::Dataset_ptr get_by_name(const std::string &user_name, const std::string &dataset_name);

    std::deque<datamodel::Dataset_ptr> find_all_user_datasets(const std::string& user_name);

    bool link_user_to_dataset(const std::string& user_name, const datamodel::Dataset_ptr & dataset);
    bool unlink_user_from_dataset(const std::string& user_name, const datamodel::Dataset_ptr & dataset);
    UserDatasetPairs get_active_datasets();
};

} }

#endif /* PGDATASETDAO_HPP */

