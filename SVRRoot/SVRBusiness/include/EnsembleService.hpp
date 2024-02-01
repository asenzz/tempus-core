#pragma once

#include <memory>
#include <common/types.hpp>
#include "model/DataRow.hpp"
#include "model/InputQueue.hpp"


namespace svr { namespace dao { class EnsembleDAO; } }
namespace svr { namespace business {
    class ModelService;
    class SVRParametersService;
    class DeconQueueService;

} }
namespace svr { namespace datamodel {
    class Model;
    class Ensemble;
    class Dataset;
    class DeconQueue;
    class SVRParameters;
    using Ensemble_ptr = std::shared_ptr<Ensemble>;
    using Dataset_ptr = std::shared_ptr<Dataset>;
    using DeconQueue_ptr = std::shared_ptr<DeconQueue>;
} }

#define DEFAULT_MAX_GAP (24) /* hours */

#define ALL_MODELS (std::numeric_limits<size_t>::max()) /* train or predict all models, instead of a specific level one. */

namespace svr {
namespace business {

class EnsembleService {
    friend struct pdtm_task;

    svr::dao::EnsembleDAO &ensemble_dao_;
    svr::business::ModelService &model_service_;
    svr::business::DeconQueueService &decon_queue_service_;

public:
    EnsembleService(svr::dao::EnsembleDAO & ensemble_dao,
            svr::business::ModelService & model_service,
            svr::business::DeconQueueService & decon_queue_service)
            :
            ensemble_dao_(ensemble_dao),
            model_service_(model_service),
            decon_queue_service_(decon_queue_service)
    {   }

    bigint get_next_id();

    datamodel::Ensemble_ptr get(const bigint ensemble_id);
    datamodel::Ensemble_ptr get(const datamodel::Dataset_ptr &p_dataset, const datamodel::DeconQueue_ptr &p_decon_queue);

    std::deque<datamodel::Ensemble_ptr> get_all_by_dataset_id(const bigint dataset_id);

    void load(const datamodel::Dataset_ptr &p_dataset, const datamodel::Ensemble_ptr& p_ensemble, const bool load_decon_data = false);
    int save(const datamodel::Ensemble_ptr &p_ensemble);
    bool save_ensembles(const std::deque<datamodel::Ensemble_ptr> &ensembles, bool save_decon_queues = false);
    bool exists(const datamodel::Ensemble_ptr &ensemble);
    bool exists(const bigint ensemble_id);
    size_t remove_by_dataset_id(const bigint dataset_id);
    int remove(const datamodel::Ensemble_ptr & p_ensemble);

    static data_row_container::iterator
    get_start(
            data_row_container &cont,
            const size_t decremental_offset,
            const size_t lag_count,
            const boost::posix_time::ptime &model_last_time,
            const boost::posix_time::time_duration &resolution);

    void init_ensembles(datamodel::Dataset_ptr &p_dataset);

 	//create ensembles with decons and svrParameters from dataset
    std::deque<datamodel::Ensemble_ptr> init_ensembles_from_dataset(const datamodel::Dataset_ptr &p_dataset);

    std::deque<datamodel::Ensemble_ptr> init_ensembles_from_dataset(const datamodel::Dataset_ptr &p_dataset,
                                                              const std::deque<datamodel::DeconQueue_ptr> &decon_queues);

    void train(datamodel::Dataset_ptr &p_dataset, datamodel::Ensemble_ptr &p_ensemble);

    static void predict(
            const datamodel::Ensemble_ptr &p_ensemble,
            const boost::posix_time::time_period &range,
            const boost::posix_time::time_duration &resolution,
            const bpt::time_duration &max_gap,
            svr::datamodel::DataRow::container &decon_data,
            std::deque<data_row_container_ptr> &aux_decon_data);

    static bool
    is_ensemble_input_queue(const datamodel::Ensemble_ptr &p_ensemble, const datamodel::InputQueue_ptr &p_input_queue);

    static void
    update_ensemble_decon_queues(const std::deque<datamodel::Ensemble_ptr> &ensembles, const std::deque<datamodel::DeconQueue_ptr> &new_decon_queues);
};
} /* namespace business */
} /* namespace svr */

using EnsembleService_ptr = std::shared_ptr<svr::business::EnsembleService>;
