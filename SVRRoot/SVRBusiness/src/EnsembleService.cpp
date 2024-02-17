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
#include "DaemonFacade.hpp"


using namespace svr::common;
using namespace svr::datamodel;


namespace svr {
namespace business {


void EnsembleService::load(const datamodel::Dataset_ptr &p_dataset, datamodel::Ensemble_ptr &p_ensemble, const bool load_decon_data)
{
    reject_nullptr(p_ensemble);

    if (!p_ensemble->get_id()) {
        LOG4_ERROR("Ensemble is not configured in database ensembles table.");
        return;
    }

    p_ensemble->set_models(model_service_.get_all_models_by_ensemble_id(p_ensemble->get_id()));
    if (p_ensemble->get_models().size() != p_dataset->get_model_count())
        ModelService::init_default_models(p_dataset, p_ensemble);

#pragma omp parallel for num_threads(adj_threads(p_ensemble->get_models().size()))
    for (auto &m: p_ensemble->get_models())
        APP.model_service.configure(p_dataset, p_ensemble, m);

    if (load_decon_data && p_ensemble->get_decon_queue())
        APP.decon_queue_service.load_latest(p_ensemble->get_decon_queue());
#pragma omp parallel for num_threads(adj_threads(p_ensemble->get_aux_decon_queues().size()))
    for (auto &p_aux_decon_queue: p_ensemble->get_aux_decon_queues())
        if (load_decon_data && p_aux_decon_queue)
            APP.decon_queue_service.load_latest(p_aux_decon_queue);
}


void EnsembleService::train(
        const datamodel::Dataset_ptr &p_dataset,
        datamodel::Ensemble_ptr &p_ensemble,
        const tbb::concurrent_map<size_t, matrix_ptr> &dataset_features,
        const tbb::concurrent_map<size_t, matrix_ptr> &labels,
        const tbb::concurrent_map<size_t, matrix_ptr> &last_knowns,
        const tbb::concurrent_map<size_t, bpt::ptime> &last_row_time)
{
    t_tuned_parameters tuned_parameters;
#pragma omp parallel for num_threads(adj_threads(p_ensemble->get_models().size()))
    for (auto p_model: p_ensemble->get_models()) {
        tuned_parameters[p_model->get_decon_level()] = std::deque<t_gradient_tuned_parameters>(p_model->get_gradient_count());
        ModelService::tune(
                p_model,
                tuned_parameters[p_model->get_decon_level()],
                dataset_features.at(p_model->get_decon_level()),
                labels.at(p_model->get_decon_level()),
                last_knowns.at(p_model->get_decon_level()));
    }

// Recombine parameters works only on the first gradient and with same number of chunks (decrement distance) across all models
#pragma omp parallel for schedule(static, 1) num_threads(adj_threads(tuned_parameters.begin()->second.size()))
    for (size_t chunk_ix = 0; chunk_ix < tuned_parameters.begin()->second.size(); ++chunk_ix)
        DatasetService::recombine_params(p_dataset, p_ensemble, tuned_parameters, chunk_ix, 0);

#pragma omp parallel for num_threads(adj_threads(p_ensemble->get_models().size()))
    for (auto p_model: p_ensemble->get_models())
        ModelService::train(
                p_model,
                dataset_features.at(p_model->get_decon_level()),
                labels.at(p_model->get_decon_level()),
                last_row_time.at(p_model->get_decon_level()));
}

datamodel::DeconQueue_ptr EnsembleService::predict_noexcept(
        const datamodel::Dataset_ptr &p_dataset,
        const datamodel::Ensemble_ptr &p_ensemble,
        const t_predict_features &dataset_features) noexcept
{
    try {
        return predict(p_dataset, p_ensemble, dataset_features);
    } catch (const std::exception &ex) {
        LOG4_ERROR("Failed predicting " << p_ensemble->get_column_name() << ", " << ex.what());
        return nullptr;
    }
}

datamodel::DeconQueue_ptr EnsembleService::predict(
        const datamodel::Dataset_ptr &p_dataset,
        const datamodel::Ensemble_ptr &p_ensemble,
        const t_predict_features &dataset_features)
{
    LOG4_BEGIN();

    auto p_aux_decon = p_ensemble->get_aux_decon_queue(p_ensemble->get_column_name())->clone_empty();
    if (!p_aux_decon) LOG4_THROW("Auxiliary decon queue for column " << p_ensemble->get_column_name() << " not found.");

    const auto aux_dq_scaling_factors = APP.dq_scaling_factor_service.slice(
            p_dataset->get_dq_scaling_factors(), p_dataset->get_id(), p_ensemble->get_column_name(), {});

#pragma omp parallel for num_threads(adj_threads(p_ensemble->get_models().size()))
    for (const auto &p_model: p_ensemble->get_models()) {
        const auto it_level_feats = dataset_features.find(p_model->get_decon_level());
        if (it_level_feats == dataset_features.end())
            LOG4_THROW("Features for level " << p_model->get_decon_level() << " are missing.");

        const auto level_preds = ModelService::predict(p_ensemble, p_model, aux_dq_scaling_factors, it_level_feats->second,
                                                       p_dataset->get_input_queue()->get_resolution());
        for (const auto &pred_row: level_preds) {
            data_row_container::iterator it_row;
#pragma omp critical
            {
                if ((it_row = lower_bound_back(p_aux_decon->get_data(), pred_row->get_value_time())) == p_aux_decon->end()) {
                    auto p_new_row = std::make_shared<DataRow>(pred_row->get_value_time(), bpt::second_clock::local_time(), 1,
                                                               std::vector<double>(p_dataset->get_transformation_levels(), 0.));
                    it_row = p_aux_decon->get_data().insert(it_row, p_new_row);
                }
            }
            it_row->get()->set_value(p_model->get_decon_level(), pred_row->get_value(0));
        }
    }

    LOG4_END();

    return p_aux_decon;
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

void EnsembleService::init_default_ensembles(datamodel::Dataset_ptr &p_dataset)
{
    datamodel::InputQueue_ptr p_input_queue = p_dataset->get_input_queue();
    std::deque<datamodel::DeconQueue_ptr> main_decon_queues, aux_decon_queues;
    get_decon_queues_from_input_queue(p_dataset, p_input_queue, main_decon_queues);
    for (const auto &p_aux_input_queue: p_dataset->get_aux_input_queues())
        get_decon_queues_from_input_queue(p_dataset, p_aux_input_queue, aux_decon_queues);
    std::deque<datamodel::Ensemble_ptr> ensembles;
#pragma omp parallel for num_threads(adj_threads(main_decon_queues.size()))
    for (const auto &p_main_decon: main_decon_queues) {
        auto p_ensemble = std::make_shared<Ensemble>(
                0, p_dataset->get_id(), std::deque<datamodel::Model_ptr>{}, p_main_decon, aux_decon_queues);
        ModelService::init_default_models(p_dataset, p_ensemble);
        ensembles.emplace_back(p_ensemble);
    }

    p_dataset->set_ensembles(ensembles);

    LOG4_END();
}

void EnsembleService::get_decon_queues_from_input_queue(
        const datamodel::Dataset_ptr &p_dataset, const datamodel::InputQueue_ptr &p_input_queue, std::deque<datamodel::DeconQueue_ptr> &decon_queues)
{
    for (size_t i = 0; i < p_input_queue->get_value_columns().size(); i++) {
        const std::string column_name = p_input_queue->get_value_column(i);
        svr::datamodel::DeconQueue_ptr p_decon_queue = std::make_shared<DeconQueue>(
                DeconQueueService::make_queue_table_name(p_input_queue->get_table_name(), p_dataset->get_id(), column_name),
                p_input_queue->get_table_name(), column_name, p_dataset->get_id(),
                p_dataset->get_transformation_levels());
        decon_queues.emplace_back(p_decon_queue);
    }
}


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
        const boost::posix_time::ptime &model_last_time,
        const boost::posix_time::time_duration &resolution)
{
    if (decremental_offset < 1) {
        LOG4_ERROR("Decremental offset " << decremental_offset << " returning end.");
        return cont.end();
    }
    // Returns an iterator with the earliest value time needed to train a model with the most current data.
    LOG4_DEBUG("Size is " << cont.size() << " decrement " << decremental_offset);
    if (cont.size() <= decremental_offset) {
        LOG4_WARN("Container size " << cont.size() << " is less or equal to needed size " << decremental_offset);
        return cont.begin();
    } else if (model_last_time == boost::posix_time::min_date_time)
        return std::next(cont.begin(), cont.size() - decremental_offset);
    else
        return find_nearest(cont, model_last_time + resolution);
}


bigint EnsembleService::get_next_id()
{
    return ensemble_dao_.get_next_id();
}

}
}
