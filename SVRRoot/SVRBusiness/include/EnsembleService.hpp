#pragma once

#include <memory>
#include <common/types.hpp>
#include <oneapi/tbb/concurrent_unordered_map.h>
#include <boost/date_time/posix_time/time_period.hpp>
#include "model/InputQueue.hpp"
#include "DatasetService.hpp"
#include "model/Dataset.hpp"
#include "model/SVRParameters.hpp"
#include "model/Ensemble.hpp"
#include "model/Model.hpp"
#include "model/User.hpp"

namespace svr {
namespace dao { class EnsembleDAO; }

namespace datamodel {
class Model;

class Ensemble;

class Dataset;

class DeconQueue;

using Ensemble_ptr = std::shared_ptr<Ensemble>;
using Dataset_ptr = std::shared_ptr<Dataset>;
using DeconQueue_ptr = std::shared_ptr<DeconQueue>;
}

namespace business {

class ModelService;

class DeconQueueService;

class EnsembleService {
    svr::dao::EnsembleDAO &ensemble_dao_;
    svr::business::ModelService &model_service_;
    svr::business::DeconQueueService &decon_queue_service_;

public:
    EnsembleService(svr::dao::EnsembleDAO &ensemble_dao, svr::business::ModelService &model_service, svr::business::DeconQueueService &decon_queue_service);

    bigint get_next_id();

    datamodel::Ensemble_ptr get(const bigint ensemble_id);

    datamodel::Ensemble_ptr get(const datamodel::Dataset_ptr &p_dataset, const datamodel::DeconQueue_ptr &p_decon_queue);

    std::deque<datamodel::Ensemble_ptr> get_all_by_dataset_id(const bigint dataset_id);

    static void load(const datamodel::Dataset_ptr &p_dataset, datamodel::Ensemble &ensemble, const bool load_decon_data = false);

    static void load_decon(const datamodel::Ensemble &ensemble);

    static void init_ensembles(datamodel::Dataset_ptr &p_dataset, const bool load_data = false);

    int save(const datamodel::Ensemble_ptr &p_ensemble);

    bool save_ensembles(const std::deque<datamodel::Ensemble_ptr> &ensembles, bool save_decon_queues = false);

    bool exists(const datamodel::Ensemble &ensemble);

    bool exists(const bigint ensemble_id);

    size_t remove_by_dataset_id(const bigint dataset_id);

    int remove(const datamodel::Ensemble_ptr &p_ensemble);

    static bool check(const std::deque<datamodel::Ensemble_ptr> &ensembles, const std::deque<std::string> &value_columns);

    static void get_decon_queues_from_input_queue(const datamodel::Dataset &dataset, const datamodel::InputQueue &input_queue, std::deque<datamodel::DeconQueue_ptr> &decon_queues);

    static void train(datamodel::Dataset &dataset, datamodel::Ensemble &ensemble);

    static datamodel::DeconQueue_ptr predict(const datamodel::Dataset &dataset, const datamodel::Ensemble &ensemble, const datamodel::t_predict_features &features);

    static datamodel::DeconQueue_ptr predict_noexcept(datamodel::Dataset &dataset, const datamodel::Ensemble &ensemble, const data_row_container &times) noexcept;

    static bool is_ensemble_input_queue(const datamodel::Ensemble &ensemble, const datamodel::InputQueue &input_queue);

    static void update_ensemble_decon_queues(const std::deque<datamodel::Ensemble_ptr> &ensembles, const std::deque<datamodel::DeconQueue_ptr> &new_decon_queues);

    static datamodel::t_predict_features prepare_prediction_data(datamodel::Dataset &dataset, const datamodel::Ensemble &ensemble, const data_row_container &times);
};
} /* namespace business */
} /* namespace svr */
