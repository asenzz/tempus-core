#pragma once

#include <common/types.hpp>
#include <unordered_map>
#include <model/SVRParameters.hpp>
#include <model/Model.hpp>
#include <model/Ensemble.hpp>
#include "model/DataRow.hpp"
#include "onlinesvr.hpp"

// #define TRIM_DATA

namespace svr { namespace dao { class DatasetDAO; }}
namespace svr {
namespace business {
class EnsembleService;

class SVRParametersService;
}
}

namespace svr {
namespace datamodel {
class DeconQueue;

class Dataset;

class Model;

class Ensemble;
}
}

#define TUNE_SVR_PARAM_FILE_NAME(svr_parameters) "/var/tmp/svr_parameters_" << (svr_parameters).get_input_queue_table_name() << "_" << (svr_parameters).get_input_queue_column_name() << "_" << (svr_parameters).get_decon_level() << ".tsv"

using DeconQueue_ptr = std::shared_ptr<svr::datamodel::DeconQueue>;
using Dataset_ptr = std::shared_ptr<svr::datamodel::Dataset>;
using Model_ptr = std::shared_ptr<svr::datamodel::Model>;
using Ensemble_ptr = std::shared_ptr<svr::datamodel::Ensemble>;

namespace svr {
namespace business {

typedef std::unordered_map<std::string, data_row_container_ptr> t_predict_result;
typedef t_predict_result *t_predict_result_ptr;

class DatasetService
{
public:
    DatasetService(
            svr::dao::DatasetDAO &datasetDao,
            svr::business::EnsembleService &ensemble_service,
            svr::business::SVRParametersService &svr_parameters_service) :
            dataset_dao(datasetDao),
            ensemble_service(ensemble_service),
            svr_parameters_service(svr_parameters_service)
    {}

    Dataset_ptr get(const bigint dataset_id, const bool load = false);
    Dataset_ptr get_user_dataset(const std::string& user_name, const std::string& dataset_name);
    std::vector<Dataset_ptr> find_all_user_datasets(std::string username);

    bool check_ensembles_svr_parameters(const Dataset_ptr &p_dataset);

    bool save(Dataset_ptr &);

    bool exists(const Dataset_ptr &);

    bool exists(int dataset_id);

    bool exists(const std::string &user_name, const std::string &dataset_name);

    int remove(const Dataset_ptr &);
    int remove(const SVRParameters_ptr &);

    bool link_user_to_dataset(User_ptr const &user, Dataset_ptr const &dataset);

    bool unlink_user_from_dataset(User_ptr const &user, Dataset_ptr const &dataset);

    struct DatasetUsers
    {
        Dataset_ptr dataset;
        std::vector<User_ptr> users;

        DatasetUsers(Dataset_ptr const &dataset, std::vector<User_ptr> &&users);
    };

    typedef std::vector<DatasetUsers> UserDatasetPairs;

    void update_active_datasets(UserDatasetPairs &processed_user_dataset_pairs);

    static void join_features(std::vector<std::vector<matrix_ptr>> &features, const size_t mod_ct, const size_t ens_ct);
    static bool prepare_models(Ensemble_ptr &p_ensemble, std::vector<SVRParameters_ptr> &ensemble_params, const size_t lev_ct);
    static bool prepare_data(
            Dataset_ptr &p_dataset,
            Ensemble_ptr &p_ensemble,
            std::vector<SVRParameters_ptr> &ensemble_params,
            std::vector<std::vector<matrix_ptr>> &features,
            std::vector<std::vector<matrix_ptr>> &labels,
            std::vector<std::vector<bpt::ptime>> &last_row_time,
            const size_t lev_ct,
            const size_t ens_ix);
    static bool prepare_dataset(Dataset_ptr &p_dataset);

    static void process_dataset(Dataset_ptr &dataset);

    static void process_dataset_test_tune(
            Dataset_ptr &p_dataset,
            Ensemble_ptr &p_ensemble,
            std::vector<SVRParameters_ptr> &ensemble_params,
            std::vector<std::vector<matrix_ptr>> &features,
            std::vector<std::vector<matrix_ptr>> &labels,
            std::vector<std::vector<bpt::ptime>> &last_row_time,
            const size_t lev_ct,
            const size_t ens_ix);

    static double
    recombine_params(
            predictions_t &tune_predictions,
            std::vector<SVRParameters_ptr> &ensemble_params,
            const size_t half_levct,
            const std::vector<double> &scale_label,
            const std::vector<double> &dc_offset,
            const uint64_t epsco_key);

    static void recombine_params(
            predictions_t &tune_predictions,
            std::vector<SVRParameters_ptr> &ensemble_params,
            const size_t half_levct,
            const std::vector<double> &scale_label,
            const std::vector<double> &dc_offset);
    static boost::posix_time::time_period get_training_range(const Dataset_ptr &p_dataset);

private:
    svr::dao::DatasetDAO &dataset_dao;
    svr::business::EnsembleService &ensemble_service;
    svr::business::SVRParametersService &svr_parameters_service;
};

} /* namespace business */
} /* namespace svr */

using DatasetService_ptr = std::shared_ptr<svr::business::DatasetService>;
