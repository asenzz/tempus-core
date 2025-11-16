#pragma once

#include "model/Dataset.hpp"
#include "common/types.hpp"

// #define TRIM_DATA

namespace svr {
namespace datamodel {
class SVRParameters;
using SVRParameters_ptr = std::shared_ptr<SVRParameters>;
class User;
using User_ptr = std::shared_ptr<User>;
struct MultivalRequest;
using MultivalRequest_ptr = std::shared_ptr<MultivalRequest>;
class DataRow;
using DataRow_ptr = std::shared_ptr<DataRow>;
}

namespace dao {
class DatasetDAO;
}

namespace business {
class EnsembleService;

class SVRParametersService;
}
}

namespace svr {
namespace business {
typedef boost::unordered_flat_map<std::string /* column */, std::deque<datamodel::DataRow_ptr> > t_stream_results, *t_stream_results_ptr;

class DatasetService
{
    dao::DatasetDAO &dataset_dao;
    EnsembleService &ensemble_service;
    SVRParametersService &svr_parameters_service;

public:
    struct DatasetUsers
    {
        datamodel::Dataset_ptr p_dataset;
        std::deque<datamodel::User_ptr> users;

        DatasetUsers(datamodel::Dataset_ptr const &dataset, std::deque<datamodel::User_ptr> &&users);
    };

    typedef std::deque<DatasetUsers> UserDatasetPairs;

    DatasetService(dao::DatasetDAO &datasetDao, EnsembleService &ensemble_service, SVRParametersService &svr_parameters_service);

    datamodel::Dataset_ptr get_user_dataset(const std::string &user_name, const std::string &dataset_name);

    std::deque<datamodel::Dataset_ptr> find_all_user_datasets(const std::string &username);

    datamodel::Dataset_ptr load(const bigint dataset_id, const bool load_dependencies = false);

    void load(datamodel::Dataset_ptr &p_dataset);

    bool save(const datamodel::Dataset_ptr &p_dataset);

    bool exists(const datamodel::Dataset_ptr &);

    bool exists(int dataset_id);

    bool exists(const std::string &user_name, const std::string &dataset_name);

    int remove(const datamodel::Dataset_ptr &);

    int remove(const datamodel::SVRParameters_ptr &);

    size_t get_level_count(const bigint dataset_id);

    bool link_user_to_dataset(datamodel::User_ptr const &user, const datamodel::Dataset_ptr &dataset);

    bool unlink_user_from_dataset(datamodel::User_ptr const &user, const datamodel::Dataset_ptr &dataset);

    void update_active_datasets(UserDatasetPairs &processed_user_dataset_pairs);

    static datamodel::t_dataset_train_data process(datamodel::Dataset &dataset);

    static void process_requests(const datamodel::User &user, datamodel::Dataset &dataset, const std::deque<datamodel::MultivalRequest_ptr> &requests, t_stream_results_ptr p_stream_results);
};
} /* namespace business */
} /* namespace svr */
