#ifndef ASYNCDATASETDAO_HPP
#define ASYNCDATASETDAO_HPP

#include <DAO/DatasetDAO.hpp>

namespace svr{
namespace dao{ 

class AsyncDatasetDAO : public DatasetDAO
{ 
public:
    explicit AsyncDatasetDAO(svr::common::PropertiesFileReader& sql_properties, svr::dao::DataSource& data_source);
    ~AsyncDatasetDAO();

    bigint get_next_id();

    bool exists(bigint dataset_id);
    bool exists(std::string user_name, std::string dataset_name);

    int save(const Dataset_ptr& dataset);
    int remove(const Dataset_ptr& dataset);

    Dataset_ptr get_by_id(bigint dataset_id);
    Dataset_ptr get_by_name(std::string user_name, std::string dataset_name);

    std::vector<Dataset_ptr> find_all_user_datasets(const std::string& user_name);

    bool link_user_to_dataset(const std::string& user_name, const Dataset_ptr & dataset);
    bool unlink_user_from_dataset(const std::string& user_name, const Dataset_ptr & dataset);
    UserDatasetPairs get_active_datasets();
private:
    struct AsyncImpl;
    AsyncImpl & pImpl;
};

} }

#endif /* ASYNCDATASETDAO_HPP */

