#include "EnsembleService.hpp"
#include <util/TimeUtils.hpp>
#include <util/PerformanceUtils.hpp>

#include "appcontext.hpp"
#include <model/Ensemble.hpp>
#include <model/Dataset.hpp>
#include <DAO/EnsembleDAO.hpp>
#include "DeconQueueService.hpp"
#include "ModelService.hpp"
#include "InputQueueService.hpp"
#include "SVRParametersService.hpp"
#include <common/thread_pool.hpp>
#include "onlinesvr.hpp"


using namespace svr::common;
using namespace svr::datamodel;


namespace svr {
namespace business {


void EnsembleService::load(const datamodel::Dataset_ptr &p_dataset, const datamodel::Ensemble_ptr &p_ensemble, const bool load_decon_data)
{
    reject_nullptr(p_ensemble);

    if (!p_ensemble->get_id()) {
        LOG4_ERROR("Ensemble is not configured in database ensembles table, aborting.");
        return;
    }

    p_ensemble->set_models(model_service_.get_all_models_by_ensemble_id(p_ensemble->get_id()));
    if (p_ensemble->get_models().size() != p_dataset->get_model_count()) {
        std::deque<datamodel::Model_ptr> e_models(p_dataset->get_model_count());
#pragma omp parallel for
        for (size_t modix = 0; modix < e_models.size(); ++modix)
            e_models[modix] = std::make_shared<Model>(
                    0, p_ensemble->get_id(), DatasetService::to_levix(modix), p_dataset->get_multiout(), p_dataset->get_gradients(), p_dataset->get_chunk_size());
        p_ensemble->set_models(e_models);
    }
#pragma omp parallel for
    for (auto &m: p_ensemble->get_models())
        APP.model_service.load(p_dataset, p_ensemble, m);

    if (load_decon_data && p_ensemble->get_decon_queue())
        APP.decon_queue_service.load_latest(p_ensemble->get_decon_queue());
#pragma omp parallel for
    for (auto &p_aux_decon_queue: p_ensemble->get_aux_decon_queues())
        if (load_decon_data && p_aux_decon_queue)
            APP.decon_queue_service.load_latest(p_aux_decon_queue);
}


datamodel::Ensemble_ptr EnsembleService::get(const bigint ensemble_id)
{
    datamodel::Ensemble_ptr p_ensemble = ensemble_dao_.get_by_id(ensemble_id);

    return p_ensemble;
}


datamodel::Ensemble_ptr
EnsembleService::get(
        const datamodel::Dataset_ptr &p_dataset,
        const datamodel::DeconQueue_ptr &p_decon_queue)
{
    reject_nullptr(p_dataset);
    reject_nullptr(p_decon_queue);
    reject_empty(p_decon_queue->get_table_name());

    return ensemble_dao_.get_by_dataset_and_decon_queue(p_dataset, p_decon_queue);
}


bool
EnsembleService::is_ensemble_input_queue(
        const datamodel::Ensemble_ptr &p_ensemble,
        const datamodel::InputQueue_ptr &p_input_queue)
{

    bool res = std::any_of(
            p_input_queue->get_value_columns().begin(),
            p_input_queue->get_value_columns().end(),
            [&p_ensemble](const std::string &column_name) -> bool { return column_name == p_ensemble->get_decon_queue()->get_input_queue_column_name(); }
    );

    if (!res)
        LOG4_DEBUG(
                "Skipping ensemble for column " << p_ensemble->get_decon_queue()->get_input_queue_column_name() <<
                                                " of input queue "
                                                << p_ensemble->get_decon_queue()->get_input_queue_table_name()
                                                << ". Not a value column.");

    return res;
}


std::deque<datamodel::Ensemble_ptr> EnsembleService::get_all_by_dataset_id(const bigint dataset_id)
{
    auto ensembles = ensemble_dao_.find_all_ensembles_by_dataset_id(dataset_id);
    return ensembles;
}


int EnsembleService::save(const datamodel::Ensemble_ptr &p_ensemble)
{
    LOG4_BEGIN();

    if (!p_ensemble) {
        LOG4_ERROR("Ensemble is null! Aborting!");
        return 0;
    }

    if (p_ensemble->get_id() <= 0) {
        p_ensemble->set_id(ensemble_dao_.get_next_id());
        for (datamodel::Model_ptr &p_model: p_ensemble->get_models())
            p_model->set_ensemble_id(p_ensemble->get_id());
    }
    int ret_value = ensemble_dao_.save(p_ensemble);
    model_service_.remove_by_ensemble_id(p_ensemble->get_id());
    for (datamodel::Model_ptr &p_model: p_ensemble->get_models())
        model_service_.save(p_model);

    LOG4_END();

    return ret_value;
}


bool EnsembleService::save_ensembles(const std::deque<datamodel::Ensemble_ptr> &ensembles, bool save_decon_queues)
{
    LOG4_BEGIN();

    if (save_decon_queues)
        for (const datamodel::Ensemble_ptr &e: ensembles)
            if (e->get_decon_queue())
                decon_queue_service_.save(e->get_decon_queue());

    for (const datamodel::Ensemble_ptr &e: ensembles)
        if (save(e) == 0)
            return false;

    LOG4_END();

    return true;
}


bool EnsembleService::exists(const datamodel::Ensemble_ptr &ensemble)
{
    reject_nullptr(ensemble);
    if (ensemble->get_id() <= 0) return false;
    return ensemble_dao_.exists(ensemble->get_id());
}


size_t EnsembleService::remove_by_dataset_id(const bigint dataset_id)
{
    auto ensembles = get_all_by_dataset_id(dataset_id);
    //iterate only by decon_queue (not aux_decon) because each aux_decons must be in another ensemble as decon_queue
    size_t ret_val = 0;
    for (const auto &e: ensembles) ret_val += remove(e);
    return ret_val;
}


int EnsembleService::remove(const datamodel::Ensemble_ptr &p_ensemble)
{
    reject_nullptr(p_ensemble);
    model_service_.remove_by_ensemble_id(p_ensemble->get_id());
    int ret_val = ensemble_dao_.remove(p_ensemble); // First we need to remove ensembles due "REFERENCE" in db
    if (p_ensemble->get_decon_queue())
        decon_queue_service_.remove(p_ensemble->get_decon_queue());
    return ret_val;
}

void EnsembleService::init_ensembles(datamodel::Dataset_ptr &p_dataset)
{
    p_dataset->set_ensembles(init_ensembles_from_dataset(p_dataset));
}

void get_decon_queues_from_input_queue(const datamodel::Dataset_ptr &p_dataset, const datamodel::InputQueue_ptr &p_input_queue, std::deque<datamodel::DeconQueue_ptr> &decon_queues)
{
    for (size_t i = 0; i < p_input_queue->get_value_columns().size(); i++) {
        const std::string column_name = p_input_queue->get_value_columns()[i];
        svr::datamodel::DeconQueue_ptr p_decon_queue = std::make_shared<DeconQueue>(
                "", p_input_queue->get_table_name(), column_name, p_dataset->get_id(),
                p_dataset->get_transformation_levels());
        decon_queues.push_back(p_decon_queue);
    }
}

std::deque<datamodel::Ensemble_ptr> EnsembleService::init_ensembles_from_dataset(const datamodel::Dataset_ptr &p_dataset)
{
    datamodel::InputQueue_ptr p_input_queue = p_dataset->get_input_queue();
    std::deque<datamodel::DeconQueue_ptr> decon_queues;
    get_decon_queues_from_input_queue(p_dataset, p_input_queue, decon_queues);
    for (const auto &p_aux_input_queue: p_dataset->get_aux_input_queues())
        get_decon_queues_from_input_queue(p_dataset, p_aux_input_queue, decon_queues);

    return init_ensembles_from_dataset(p_dataset, decon_queues);
}


std::deque<datamodel::Ensemble_ptr> EnsembleService::init_ensembles_from_dataset(
        const datamodel::Dataset_ptr &p_dataset,
        const std::deque<svr::datamodel::DeconQueue_ptr> &decon_queues)
{
    LOG4_WARN("Wrong decon queues are guessed per ensemble!");
    // init ensembles
    std::deque<datamodel::Ensemble_ptr> ensembles;
    for (const auto &main_value_column: p_dataset->get_input_queue()->get_value_columns()) {
        std::deque<datamodel::DeconQueue_ptr> decon_queues_aux;
        for (const auto &p_aux_input_queue: p_dataset->get_aux_input_queues()) {
            for (const auto &value_column_aux: p_aux_input_queue->get_value_columns())
                if (const auto p_aux_decon_queue = APP.decon_queue_service.find_decon_queue(decon_queues, p_aux_input_queue->get_table_name(), value_column_aux))
                    decon_queues_aux.push_back(p_aux_decon_queue);
        }
        if (const auto &p_main_decon_queue = APP.decon_queue_service.find_decon_queue(decon_queues, p_dataset->get_input_queue()->get_table_name(), main_value_column))
            ensembles.push_back(std::make_shared<Ensemble>(
                    0, p_dataset->get_id(), std::deque<datamodel::Model_ptr>{}, p_main_decon_queue, decon_queues_aux));
        else
            LOG4_ERROR("Could not find decon queue for " << p_dataset->get_input_queue()->get_table_name() << " " << main_value_column);
    }

    LOG4_END();

    return ensembles;
}

#include "DaemonFacade.hpp"

void
EnsembleService::update_ensemble_decon_queues(
        const std::deque<datamodel::Ensemble_ptr> &ensembles,
        const std::deque<datamodel::DeconQueue_ptr> &new_decon_queues)
{
    LOG4_BEGIN();

    if (ensembles.size() != new_decon_queues.size())
        LOG4_WARN(
                "Number of ensembles " << ensembles.size() << " and new decon queues " << new_decon_queues.size() << " differ.");

    for (auto p_ensemble: ensembles) {
        {
            const auto p_decon_queue = DeconQueueService::find_decon_queue(
                    new_decon_queues,
                    p_ensemble->get_decon_queue()->get_input_queue_table_name(),
                    p_ensemble->get_decon_queue()->get_input_queue_column_name());
            p_ensemble->get_decon_queue()->update_data(p_decon_queue->get_data());
        }
        for (auto p_ensemble_aux_decon_queue: p_ensemble->get_aux_decon_queues()) {
            const auto p_decon_queue = DeconQueueService::find_decon_queue(
                    new_decon_queues,
                    p_ensemble_aux_decon_queue->get_input_queue_table_name(),
                    p_ensemble_aux_decon_queue->get_input_queue_column_name());
            p_ensemble_aux_decon_queue->update_data(p_decon_queue->get_data());
        }
    }

    LOG4_END();
}


bool EnsembleService::exists(const bigint ensemble_id)
{
    return ensemble_dao_.exists(ensemble_id);
}


DataRow::container::iterator
EnsembleService::get_start(
        DataRow::container &cont,
        const size_t decremental_offset,
        const size_t lag_count,
        const boost::posix_time::ptime &model_last_time,
        const boost::posix_time::time_duration &resolution)
{
    if (decremental_offset < 1) {
        LOG4_ERROR("Decremental offset " << decremental_offset << " lag " << lag_count << ", returning end.");
        return cont.end();
    }
    // Returns an iterator to a DeconQueue with the earliest value_time needed for training a model.
    // An additional lag_count number of values are needed to produce autoregressive features.
    const auto needed_size = decremental_offset + lag_count;
    LOG4_DEBUG("Size is " << cont.size() << " decrement " << decremental_offset << " lag " << lag_count);
    if (cont.size() <= needed_size) {
        LOG4_WARN("cont.size() <= needed_size " << cont.size() << " <= " << needed_size);
        return cont.begin();
    } else if (model_last_time == boost::posix_time::min_date_time)
        return std::next(cont.begin(), cont.size() - needed_size);
    else
        return std::next(find_nearest(cont, model_last_time + resolution), -lag_count);
}


void EnsembleService::train(datamodel::Dataset_ptr &p_dataset, datamodel::Ensemble_ptr &p_ensemble)
{
    LOG4_BEGIN();
}


void
EnsembleService::predict(
        const datamodel::Ensemble_ptr &p_ensemble,
        const boost::posix_time::time_period &range,
        const boost::posix_time::time_duration &resolution,
        const bpt::time_duration &max_gap,
        datamodel::DataRow::container &decon_data,
        std::deque<data_row_container_ptr> &aux_decon_data)
{
    LOG4_DEBUG("Request to predict in place time period new function " << range);
    std::deque<data_row_container_ptr> all_decon_data;
    all_decon_data.push_back(shared_observer_ptr<data_row_container>(decon_data));
    for (const auto &p_aux_data: aux_decon_data) all_decon_data.push_back(p_aux_data);
    for (auto current_time = range.begin(); current_time < range.end();) { // Predict every time
        bpt::ptime last_predicted_time;
        for (auto p_decon_data: all_decon_data) { // Predict every column
            /*
            auto priv_aux_decon_data = get_all_except(all_decon_data, p_decon_data);
            const auto prediction_results = APP.model_service.predict(
                    p_ensemble->get_models(),
                    current_time,
                    resolution,
                    max_gap,
                    *p_decon_data,
                    priv_aux_decon_data);
            last_predicted_time = datamodel::DataRow::insert_rows(*p_decon_data, prediction_results, current_time, resolution);
             */
        }
        current_time = last_predicted_time;
    }
    LOG4_END();
}


bigint EnsembleService::get_next_id()
{
    return ensemble_dao_.get_next_id();
}

}
}
