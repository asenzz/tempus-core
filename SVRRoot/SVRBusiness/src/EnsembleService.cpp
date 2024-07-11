#include "EnsembleService.hpp"
#include <util/time_utils.hpp>
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

namespace svr {
namespace business {

EnsembleService::EnsembleService(
        svr::dao::EnsembleDAO &ensemble_dao,
        svr::business::ModelService &model_service,
        svr::business::DeconQueueService &decon_queue_service) :
        ensemble_dao_(ensemble_dao),
        model_service_(model_service),
        decon_queue_service_(decon_queue_service)
{}


void EnsembleService::load_decon(const datamodel::Ensemble_ptr &p_ensemble)
{
#pragma omp parallel num_threads(adj_threads(1 + p_ensemble->get_aux_decon_queues().size()))
    {
#pragma omp task
        if (p_ensemble->get_decon_queue())
            APP.decon_queue_service.load_latest(p_ensemble->get_decon_queue());
#pragma omp parallel for num_threads(adj_threads(p_ensemble->get_aux_decon_queues().size()))
        for (auto &p_aux_decon_queue: p_ensemble->get_aux_decon_queues())
            if (p_aux_decon_queue)
                APP.decon_queue_service.load_latest(p_aux_decon_queue);
    }
}

void EnsembleService::load(const datamodel::Dataset_ptr &p_dataset, datamodel::Ensemble_ptr &p_ensemble, const bool load_decon_data)
{
    common::reject_nullptr(p_ensemble);

    if (!p_ensemble->get_id()) {
        LOG4_ERROR("Ensemble is not configured in database ensembles table.");
        return;
    }

    APP.model_service.init_models(p_dataset, p_ensemble);
    if (load_decon_data) load_decon(p_ensemble);
}


void EnsembleService::train(datamodel::Dataset &dataset, datamodel::Ensemble &ensemble)
{
#pragma omp parallel for num_threads(adj_threads(std::min<size_t>(C_parallel_train_models, ensemble.get_model_ct()))) schedule(static, 1)
    for (auto p_model: ensemble.get_models()) ModelService::train(dataset, ensemble, *p_model);
}

datamodel::DeconQueue_ptr EnsembleService::predict_noexcept(
        const datamodel::Dataset &dataset, const datamodel::Ensemble &ensemble, const datamodel::t_predict_features &features) noexcept
{
    try {
        return predict(dataset, ensemble, features);
    } catch (const std::exception &ex) {
        LOG4_ERROR("Failed predicting " << ensemble.get_column_name() << ", " << ex.what());
        return nullptr;
    }
}

datamodel::DeconQueue_ptr EnsembleService::predict(const datamodel::Dataset &dataset, const datamodel::Ensemble &ensemble, const datamodel::t_predict_features &features)
{
    LOG4_BEGIN();

    auto p_aux_decon = ensemble.get_label_aux_decon()->clone_empty();
    tbb::mutex insert_mx;
#pragma omp parallel for num_threads(adj_threads(ensemble.get_model_ct())) schedule(static, 1)
    for (auto &p_model: ensemble.get_models())
        ModelService::predict(ensemble, *p_model, features.at(std::tuple{p_model->get_decon_level(), p_model->get_step()}), dataset.get_input_queue()->get_resolution(), insert_mx, *p_aux_decon);

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
    common::reject_nullptr(p_dataset);
    common::reject_nullptr(p_decon_queue);
    common::reject_empty(p_decon_queue->get_table_name());

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
        LOG4_DEBUG("Skipping ensemble for column " << p_ensemble->get_decon_queue()->get_input_queue_column_name() <<
                                                   " of input queue " << p_ensemble->get_decon_queue()->get_input_queue_table_name() << ". Not a value column.");

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
    common::reject_nullptr(ensemble);
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
    common::reject_nullptr(p_ensemble);
    model_service_.remove_by_ensemble_id(p_ensemble->get_id());
    int ret_val = ensemble_dao_.remove(p_ensemble); // First we need to remove ensembles due "REFERENCE" in db
    if (p_ensemble->get_decon_queue())
        decon_queue_service_.remove(p_ensemble->get_decon_queue());
    return ret_val;
}


bool EnsembleService::check(const std::deque<datamodel::Ensemble_ptr> &ensembles, const std::deque<std::string> &value_columns)
{
    tbb::concurrent_vector<bool> present(value_columns.size(), false);
#pragma omp parallel for collapse(2) num_threads(adj_threads(ensembles.size() * value_columns.size())) schedule(static, 1)
    for (const auto &e: ensembles)
        for (size_t i = 0; i < value_columns.size(); ++i)
            if (e->get_column_name() == value_columns[i])
                present[i] = true;
    return std::all_of(std::execution::par_unseq, present.begin(), present.end(), [](const auto &p) { return p == true; });
}


void EnsembleService::init_ensembles(datamodel::Dataset_ptr &p_dataset, const bool load_data)
{
    auto p_input_queue = p_dataset->get_input_queue();
    std::deque<datamodel::DeconQueue_ptr> main_decon_queues, aux_decon_queues;
    if (load_data) PROFILE_EXEC_TIME(APP.input_queue_service.load(*p_input_queue), "Loading " << p_input_queue->get_table_name());
    get_decon_queues_from_input_queue(*p_dataset, *p_input_queue, main_decon_queues);
    for (const auto &p_aux_input_queue: p_dataset->get_aux_input_queues()) {
        if (load_data) PROFILE_EXEC_TIME(APP.input_queue_service.load(*p_aux_input_queue), "Loading " << p_aux_input_queue->get_table_name());
        get_decon_queues_from_input_queue(*p_dataset, *p_aux_input_queue, aux_decon_queues);
    }
#pragma omp parallel num_threads(adj_threads(main_decon_queues.size() + aux_decon_queues.size()))
#pragma omp single
    if (load_data) {
        for (auto &dq: main_decon_queues)
#pragma omp task
            APP.decon_queue_service.load(dq);
        for (auto &dq: aux_decon_queues)
#pragma omp task
            APP.decon_queue_service.load(dq);
    }

    if (!check(p_dataset->get_ensembles(), p_dataset->get_input_queue()->get_value_columns()) && p_dataset->get_id())
        p_dataset->set_ensembles(APP.ensemble_service.get_all_by_dataset_id(p_dataset->get_id()), true);

    t_omp_lock ens_emplace_l;
    auto &ensembles = p_dataset->get_ensembles();
#pragma omp parallel for num_threads(adj_threads(main_decon_queues.size())) schedule(static, 1)
    for (size_t i = 0; i < main_decon_queues.size(); ++i) {
        const auto &p_main_decon = main_decon_queues[i];
        std::decay_t<dtype(ensembles)>::iterator ens_iter;
        datamodel::Ensemble_ptr p_ensemble;
        ens_emplace_l.set();
        if ((ens_iter = std::find_if(std::execution::par_unseq, ensembles.begin(), ensembles.end(),
                                     [&p_main_decon](const auto &p_ensemble) {
                                         return p_ensemble->get_decon_queue()->get_input_queue_table_name() == p_main_decon->get_input_queue_table_name() &&
                                                p_ensemble->get_decon_queue()->get_input_queue_column_name() == p_main_decon->get_input_queue_column_name();
                                     })) == ensembles.end())
            p_ensemble = ensembles.emplace_back(ptr<datamodel::Ensemble>(0, p_dataset->get_id(), std::deque<datamodel::Model_ptr>{}, p_main_decon, aux_decon_queues));
        else
            p_ensemble = *ens_iter;
        ens_emplace_l.unset();
        APP.model_service.init_models(p_dataset, p_ensemble);
    }

    LOG4_END();
}

void EnsembleService::get_decon_queues_from_input_queue(
        const datamodel::Dataset &dataset, const datamodel::InputQueue &input_queue, std::deque<datamodel::DeconQueue_ptr> &decon_queues)
{
    const auto prev_size = decon_queues.size();
#pragma omp parallel for num_threads(adj_threads(input_queue.get_value_columns().size())) schedule(static, 1)
    for (const auto &column_name: input_queue.get_value_columns()) {
        std::atomic<bool> skip = false;
#pragma omp parallel for num_threads(adj_threads(decon_queues.size())) schedule(static, 1)
        for (size_t j = 0; j < prev_size; ++j) {
            const auto &dq = decon_queues[j];
            if (dq->get_input_queue_table_name() == input_queue.get_table_name() && dq->get_input_queue_column_name() == column_name)
                skip.store(true, std::memory_order_relaxed);
        }
        if (skip) continue;
        datamodel::DeconQueue_ptr p_decon_queue;
        if (dataset.get_id())
            p_decon_queue = APP.decon_queue_service.get_by_table_name(input_queue.get_table_name(), dataset.get_id(), column_name);
        if (!p_decon_queue)
            p_decon_queue = dtr<datamodel::DeconQueue>(
                    DeconQueueService::make_queue_table_name(input_queue.get_table_name(), dataset.get_id(), column_name),
                    input_queue.get_table_name(), column_name, dataset.get_id(), dataset.get_transformation_levels());
#pragma omp critical
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


bigint EnsembleService::get_next_id()
{
    return ensemble_dao_.get_next_id();
}

}
}
