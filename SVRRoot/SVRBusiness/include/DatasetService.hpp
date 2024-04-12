#pragma once

#include "common/types.hpp"
#include "model/Dataset.hpp"
#include "model/SVRParameters.hpp"
#include "model/Ensemble.hpp"
#include "model/Model.hpp"
#include "model/User.hpp"
#include "EnsembleService.hpp"
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

class DatasetService
{
    dao::DatasetDAO &dataset_dao;
    EnsembleService &ensemble_service;
    SVRParametersService &svr_parameters_service;

public:
    struct DatasetUsers
    {
        datamodel::Dataset_ptr p_dataset;
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

    static datamodel::t_training_data prepare_training_data(datamodel::Dataset &dataset, datamodel::Ensemble &ensemble);

    static datamodel::t_predict_features prepare_prediction_data(datamodel::Dataset &dataset, const datamodel::Ensemble &ensemble, const std::deque<bpt::ptime> &predict_times);

    static void process(datamodel::Dataset &dataset);

    static void process_requests(const datamodel::User &user, datamodel::Dataset &dataset);
};

} /* namespace business */
} /* namespace svr */