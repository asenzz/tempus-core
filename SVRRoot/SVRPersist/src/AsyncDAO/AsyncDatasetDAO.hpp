#ifndef ASYNCDATASETDAO_HPP
#define ASYNCDATASETDAO_HPP

#include <DAO/DatasetDAO.hpp>

namespace svr {
namespace dao {

class AsyncDatasetDAO : public DatasetDAO {
public:
    explicit AsyncDatasetDAO(common::PropertiesReader &sql_properties, dao::DataSource &data_source);

    ~AsyncDatasetDAO();

    bigint get_next_id();

    bool exists(const bigint dataset_id);

    bool exists(const std::string &user_name, const std::string &dataset_name);

    int save(const datamodel::Dataset_ptr &dataset);

    int remove(const datamodel::Dataset_ptr &dataset);

    datamodel::Dataset_ptr get_by_id(const bigint dataset_id);

    datamodel::Dataset_ptr get_by_name(const std::string &user_name, const std::string &dataset_name);

    std::deque<datamodel::Dataset_ptr> find_all_user_datasets(const std::string &user_name);

    bool link_user_to_dataset(const std::string &user_name, const datamodel::Dataset_ptr &dataset);

    bool unlink_user_from_dataset(const std::string &user_name, const datamodel::Dataset_ptr &dataset);

    UserDatasetPairs get_active_datasets();

    size_t get_level_count(const bigint dataset_id);

private:
    struct AsyncImpl;
    AsyncImpl &pImpl;
};

}
}

#endif /* ASYNCDATASETDAO_HPP */

