#pragma once

#include "common/types.hpp"
#include "model/Dataset.hpp"
#include "model/SVRParameters.hpp"
#include "model/Ensemble.hpp"
#include "model/Model.hpp"
#include "model/User.hpp"
#include </opt/intel/oneapi/tbb/latest/include/oneapi/tbb/concurrent_unordered_map.h>
#include <boost/date_time/posix_time/time_period.hpp>

// #define TRIM_DATA

namespace svr {
namespace datamodel {
class Dataset;

using Dataset_ptr = std::shared_ptr<Dataset>;
}
namespace dao { class DatasetDAO; }
namespace business {
class EnsembleService;

class SVRParametersService;
}
}

namespace svr {
namespace business {

struct t_training_data
{
    std::unordered_map<size_t /* level */, matrix_ptr> features, labels;
    std::unordered_map<size_t /* level */, vec_ptr> last_knowns;
    bpt::ptime last_row_time;
};


class DatasetService
{
    dao::DatasetDAO &dataset_dao;
    EnsembleService &ensemble_service;
    SVRParametersService &svr_parameters_service;

public:
    struct DatasetUsers
    {
        datamodel::Dataset_ptr dataset;
        std::deque<User_ptr> users;

        DatasetUsers(datamodel::Dataset_ptr const &dataset, std::deque<User_ptr> &&users);
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

    bool link_user_to_dataset(User_ptr const &user, const datamodel::Dataset_ptr &dataset);

    bool unlink_user_from_dataset(User_ptr const &user, const datamodel::Dataset_ptr &dataset);

    void update_active_datasets(UserDatasetPairs &processed_user_dataset_pairs);

    static std::unordered_map<size_t, matrix_ptr> join_features(
            std::unordered_map<std::string, t_training_data> &train_data, const size_t levct, const std::deque<datamodel::Ensemble_ptr> &ensembles);

    static bool prepare_training_data(datamodel::Dataset &dataset, datamodel::Ensemble &ensemble, t_training_data &train_data);

    static auto prepare_request_features(const datamodel::Dataset_ptr &p_dataset, const std::set<bpt::ptime> &predict_times);

    static void process(datamodel::Dataset_ptr &p_dataset);

    static void process_dataset_test_tune(datamodel::Dataset_ptr &p_dataset, datamodel::Ensemble_ptr &p_ensemble);

    static boost::posix_time::time_period get_training_range(const datamodel::Dataset_ptr &p_dataset);

    static void process_requests(const User_ptr &p_user, datamodel::Dataset_ptr &p_dataset);
};

} /* namespace business */
} /* namespace svr */