#pragma once

#include "online_emd.hpp"
#include "common/types.hpp"
#include "model/Dataset.hpp"
#include "model/SVRParameters.hpp"
#include "model/Ensemble.hpp"
#include "model/DataRow.hpp"
#include "model/Model.hpp"
#include "model/User.hpp"
#include "onlinesvr.hpp"

// #define TRIM_DATA

namespace svr {
class online_emd;
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

template<typename T> using t_enscon = tbb::concurrent_map<std::string /* column */, tbb::concurrent_map<size_t /* level */, T>>;

class DatasetService
{
    svr::dao::DatasetDAO &dataset_dao;
    svr::business::EnsembleService &ensemble_service;
    svr::business::SVRParametersService &svr_parameters_service;

public:
    struct DatasetUsers
    {
        datamodel::Dataset_ptr dataset;
        std::deque<User_ptr> users;

        DatasetUsers(datamodel::Dataset_ptr const &dataset, std::deque<User_ptr> &&users);
    };

    typedef std::deque<DatasetUsers> UserDatasetPairs;

    DatasetService(
            svr::dao::DatasetDAO &datasetDao,
            svr::business::EnsembleService &ensemble_service,
            svr::business::SVRParametersService &svr_parameters_service) :
            dataset_dao(datasetDao),
            ensemble_service(ensemble_service),
            svr_parameters_service(svr_parameters_service)
    {}

    datamodel::Dataset_ptr get(const bigint dataset_id, const bool load = false);
    datamodel::Dataset_ptr get_user_dataset(const std::string& user_name, const std::string& dataset_name);
    std::deque<datamodel::Dataset_ptr> find_all_user_datasets(std::string username);

    void load(datamodel::Dataset_ptr &p_dataset);
    bool save(datamodel::Dataset_ptr &p_dataset);

    bool exists(const datamodel::Dataset_ptr &);
    bool exists(int dataset_id);
    bool exists(const std::string &user_name, const std::string &dataset_name);

    int remove(const datamodel::Dataset_ptr &);
    int remove(const datamodel::SVRParameters_ptr &);

    bool link_user_to_dataset(User_ptr const &user, const datamodel::Dataset_ptr &dataset);
    bool unlink_user_from_dataset(User_ptr const &user, const datamodel::Dataset_ptr &dataset);
    void update_active_datasets(UserDatasetPairs &processed_user_dataset_pairs);

    static tbb::concurrent_map<size_t, matrix_ptr> join_features(
            t_enscon<matrix_ptr> &features,
            const size_t levct,
            const std::deque<datamodel::Ensemble_ptr> &ensembles);

    static bool prepare_training_data(
            datamodel::Dataset_ptr &p_dataset,
            datamodel::Ensemble_ptr &p_ensemble,
            tbb::concurrent_map<size_t, matrix_ptr> &features,
            tbb::concurrent_map<size_t, matrix_ptr> &labels,
            tbb::concurrent_map<size_t, matrix_ptr> &last_knowns,
            tbb::concurrent_map<size_t, bpt::ptime> &last_row_time);

    static auto prepare_request_features(const datamodel::Dataset_ptr &p_dataset, const std::set<bpt::ptime> &predict_times);

    static void process(datamodel::Dataset_ptr &p_dataset);

    static void process_dataset_test_tune(datamodel::Dataset_ptr &p_dataset, datamodel::Ensemble_ptr &p_ensemble);

    static void recombine_params(
            const datamodel::Dataset_ptr &p_dataset, datamodel::Ensemble_ptr &p_ensemble, t_tuned_parameters &tune_predictions,
            const size_t chunk_ix, const size_t grad_level);
    static boost::posix_time::time_period get_training_range(const datamodel::Dataset_ptr &p_dataset);

    static void process_requests(const User_ptr &p_user, datamodel::Dataset_ptr &p_dataset);
};

} /* namespace business */
} /* namespace svr */

using DatasetService_ptr = std::shared_ptr<svr::business::DatasetService>;
