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
} }

using Model_ptr = std::shared_ptr<svr::datamodel::Model>;
using Ensemble_ptr = std::shared_ptr<svr::datamodel::Ensemble>;
using Dataset_ptr = std::shared_ptr<svr::datamodel::Dataset>;
using DeconQueue_ptr = std::shared_ptr<svr::datamodel::DeconQueue>;
using SVRParameters_ptr = std::shared_ptr<svr::datamodel::SVRParameters>;


#define DEFAULT_MAX_GAP (24) /* hours */

#define ALL_MODELS (std::numeric_limits<size_t>::max()) /* train or predict all models, instead of a specific level one. */

namespace svr {
namespace business {

class EnsembleService {
    friend struct pdtm_task;

    svr::dao::EnsembleDAO &ensemble_dao_;
    svr::business::ModelService &model_service_;
    svr::business::DeconQueueService &decon_queue_service_;

    /**
     * @brief load decon_queue and aux_decon_queues by table names
     * @param ensemble shared pointer
     */
    void load_svr_params_and_models(Ensemble_ptr p_ensemble);
    void prepare_data_and_train_model(
            Model_ptr &p_model,
            const DeconQueue_ptr &p_main_delta_decon_queue,
            const std::vector<DeconQueue_ptr> &p_aux_delta_decon_queues,
            const SVRParameters_ptr &p_svr_parameters,
            const bpt::time_duration &max_gap);

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

    Ensemble_ptr get(bigint ensemble_id);
    Ensemble_ptr get(const Dataset_ptr &p_dataset, const DeconQueue_ptr &p_decon_queue);

    std::vector<Ensemble_ptr> get_all_by_dataset_id(const bigint dataset_id);

    int save(const Ensemble_ptr &p_ensemble);
    bool save_ensembles(const std::vector<Ensemble_ptr> &ensembles, bool save_decon_queues = false);
    bool exists(const Ensemble_ptr &ensemble);
    bool exists(bigint ensemble_id);
    int remove_by_dataset_id(bigint dataset_id);
    int remove(const Ensemble_ptr & p_ensemble);

    static data_row_container::iterator
    get_start(
            data_row_container &cont,
            const size_t decremental_offset,
            const size_t lag_count,
            const boost::posix_time::ptime &model_last_time,
            const boost::posix_time::time_duration &resolution);

    void init_ensembles(Dataset_ptr &p_dataset);

 	//create ensembles with decons and svrParameters from dataset
    std::vector<Ensemble_ptr> init_ensembles_from_dataset(const Dataset_ptr &p_dataset);

    std::vector<Ensemble_ptr> init_ensembles_from_dataset(const Dataset_ptr &p_dataset,
                                                              const std::vector<DeconQueue_ptr> &decon_queues);

    void train(
            Dataset_ptr &p_dataset,
            Ensemble_ptr &p_ensemble,
            const std::vector<SVRParameters_ptr> &model_parameters,
            const bpt::time_duration &max_gap,
            const size_t model_numb = std::numeric_limits<size_t>::max());

    static void
    predict(Ensemble_ptr &p_ensemble,
            const boost::posix_time::time_period &range,
            const boost::posix_time::time_duration &resolution,
            const bpt::time_duration &max_gap = bpt::hours(DEFAULT_MAX_GAP));

    static void predict(
            const Ensemble_ptr &p_ensemble,
            const boost::posix_time::time_period &range,
            const boost::posix_time::time_duration &resolution,
            const bpt::time_duration &max_gap,
            svr::datamodel::DataRow::container &decon_data,
            std::vector<data_row_container_ptr> &aux_decon_data);

    static void
    init_decon_data(std::vector<Ensemble_ptr> &ensembles, const std::vector<DeconQueue_ptr> &decon_queues);

    static bool
    is_ensemble_input_queue(const Ensemble_ptr &p_ensemble, const InputQueue_ptr &p_input_queue);

    static void
    update_ensemble_decon_queues(const std::vector<Ensemble_ptr> &ensembles, const std::vector<DeconQueue_ptr> &new_decon_queues);
};
} /* namespace business */
} /* namespace svr */

using EnsembleService_ptr = std::shared_ptr<svr::business::EnsembleService>;
