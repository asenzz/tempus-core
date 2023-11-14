#include "EnsembleService.hpp"
#include <util/ValidationUtils.hpp>
#include <util/TimeUtils.hpp>
#include <util/MemoryManager.hpp>
#include <util/PerformanceUtils.hpp>
#include <util/string_utils.hpp>

#include <model/Ensemble.hpp>
#include <model/Dataset.hpp>
#include <DAO/EnsembleDAO.hpp>
#include "DeconQueueService.hpp"
#include "ModelService.hpp"
#include "InputQueueService.hpp"
#include "SVRParametersService.hpp"
#include <common/exceptions.hpp>
#include <common/thread_pool.hpp>
#include "appcontext.hpp"

#include <common/rtp_thread_pool.hpp>


using namespace svr::common;
using namespace svr::datamodel;

using namespace std;

namespace svr {
namespace business {


void EnsembleService::load_svr_params_and_models(Ensemble_ptr p_ensemble)
{
    reject_nullptr(p_ensemble);
    p_ensemble->set_models(model_service_.get_all_models_by_ensemble_id(p_ensemble->get_id()));
}


Ensemble_ptr EnsembleService::get(bigint ensemble_id)
{
    Ensemble_ptr p_ensemble = ensemble_dao_.get_by_id(ensemble_id);
    load_svr_params_and_models(p_ensemble);

    return p_ensemble;
}


Ensemble_ptr
EnsembleService::get(
        const Dataset_ptr &p_dataset,
        const DeconQueue_ptr &p_decon_queue)
{
    reject_nullptr(p_dataset);
    reject_nullptr(p_decon_queue);
    reject_empty(p_decon_queue->get_table_name());

    return ensemble_dao_.get_by_dataset_and_decon_queue(p_dataset, p_decon_queue);
}


bool
EnsembleService::is_ensemble_input_queue(
        const Ensemble_ptr &p_ensemble,
        const InputQueue_ptr &p_input_queue)
{

    bool res = std::any_of(
            p_input_queue->get_value_columns().begin(),
            p_input_queue->get_value_columns().end(),
            [&p_ensemble](const std::string &column_name) -> bool
            { return column_name == p_ensemble->get_decon_queue()->get_input_queue_column_name(); }
    );

    if (!res)
        LOG4_DEBUG(
                "Skipping ensemble for column " << p_ensemble->get_decon_queue()->get_input_queue_column_name() <<
                                                " of input queue "
                                                << p_ensemble->get_decon_queue()->get_input_queue_table_name()
                                                << ". Not a value column.");

    return res;
}


std::vector<Ensemble_ptr> EnsembleService::get_all_by_dataset_id(const bigint dataset_id)
{
    LOG4_BEGIN();

    std::vector<Ensemble_ptr> ensembles = ensemble_dao_.find_all_ensembles_by_dataset_id(dataset_id);
    for (Ensemble_ptr &p_ensemble: ensembles) load_svr_params_and_models(p_ensemble);

    // Get all decon table names from ensembles
    std::map<std::string, DeconQueue_ptr> decon_queues;
    for (Ensemble_ptr &p_ensemble: ensembles) {
        const DeconQueue_ptr &p_decon_queue = p_ensemble->get_decon_queue();
        if (p_decon_queue) decon_queues[p_decon_queue->get_table_name()] = nullptr;
        for (const auto &p_aux_decon_queue: p_ensemble->get_aux_decon_queues())
            if (p_aux_decon_queue) decon_queues[p_aux_decon_queue->get_table_name()] = nullptr;
    }

    // Load decon data
    __pxt_pfor_i(0, decon_queues.size(),
        const auto it_decon_queue = std::next(decon_queues.begin(), i);
        it_decon_queue->second = decon_queue_service_.get_by_table_name(it_decon_queue->first);
        if (not it_decon_queue->second) THROW_EX_FS(std::runtime_error, "Failed loading decon queue " << it_decon_queue->first);
        try {
            APP.decon_queue_service.load_decon_data(it_decon_queue->second);
        } catch (const std::exception &ex) {
            LOG4_WARN("Failed loading data for decon queue " << it_decon_queue->first << ". " << ex.what());
        }
    )

    // Set decon queues to ensembles
    for (Ensemble_ptr p_ensemble : ensembles) {
        if (p_ensemble->get_decon_queue())
            p_ensemble->set_decon_queue(decon_queues[p_ensemble->get_decon_queue()->get_table_name()]);

        for (DeconQueue_ptr &p_aux_decon_queue: p_ensemble->get_aux_decon_queues())
            p_aux_decon_queue = decon_queues[p_aux_decon_queue->get_table_name()];
    }

    LOG4_END();

    return ensembles;
}


int EnsembleService::save(const Ensemble_ptr &p_ensemble)
{
    LOG4_BEGIN();

    if (!p_ensemble) {
        LOG4_ERROR("Ensemble is null! Aborting!");
        return 0;
    }

    if (p_ensemble->get_id() <= 0) {
        p_ensemble->set_id(ensemble_dao_.get_next_id());
        for (Model_ptr &p_model: p_ensemble->get_models())
            p_model->set_ensemble_id(p_ensemble->get_id());
    }
    int ret_value = ensemble_dao_.save(p_ensemble);
    model_service_.remove_by_ensemble_id(p_ensemble->get_id());
    for (Model_ptr &p_model: p_ensemble->get_models())
        model_service_.save(p_model);

    LOG4_END();

    return ret_value;
}


bool EnsembleService::save_ensembles(const std::vector<Ensemble_ptr> &ensembles, bool save_decon_queues)
{
    LOG4_BEGIN();

    if (save_decon_queues)
        for (const Ensemble_ptr &p_ensemble: ensembles)
            if (p_ensemble->get_decon_queue())
                decon_queue_service_.save(p_ensemble->get_decon_queue());

    for (const Ensemble_ptr &p_ensemble: ensembles)
        if (save(p_ensemble) == 0)
            return false;

    LOG4_END();

    return true;
}


bool EnsembleService::exists(const Ensemble_ptr &ensemble)
{
    reject_nullptr(ensemble);
    if (ensemble->get_id() <= 0) return false;
    return ensemble_dao_.exists(ensemble->get_id());
}


int EnsembleService::remove_by_dataset_id(bigint dataset_id)
{
    std::vector<Ensemble_ptr> ensembles = this->get_all_by_dataset_id(dataset_id);
    //iterate only by decon_queue (not aux_decon) because each aux_decons must be in another ensemble as decon_queue
    int ret_val = 0;
    for (Ensemble_ptr p_ensemble : ensembles) ret_val += this->remove(p_ensemble);
    return ret_val;
}


int EnsembleService::remove(const Ensemble_ptr &p_ensemble)
{
    reject_nullptr(p_ensemble);
    model_service_.remove_by_ensemble_id(p_ensemble->get_id());
    int ret_val = ensemble_dao_.remove(p_ensemble); // First we need to remove ensembles due "REFERENCE" in db
    if (p_ensemble->get_decon_queue() != nullptr)
        decon_queue_service_.remove(p_ensemble->get_decon_queue());
    return ret_val;
}

void EnsembleService::init_ensembles(Dataset_ptr &p_dataset)
{
    p_dataset->set_ensembles(init_ensembles_from_dataset(p_dataset));
}

void get_decon_queues_from_input_queue(const Dataset_ptr &p_dataset, const InputQueue_ptr &p_input_queue, std::vector<DeconQueue_ptr> &decon_queues)
{
    for (size_t i = 0; i < p_input_queue->get_value_columns().size(); i++) {
        const std::string column_name = p_input_queue->get_value_columns()[i];
        svr::datamodel::DeconQueue_ptr p_decon_queue = std::make_shared<DeconQueue>(
                "", p_input_queue->get_table_name(), column_name, p_dataset->get_id(),
                p_dataset->get_transformation_levels());
        decon_queues.push_back(p_decon_queue);
    }
}

std::vector<Ensemble_ptr> EnsembleService::init_ensembles_from_dataset(const Dataset_ptr &p_dataset)
{
    InputQueue_ptr p_input_queue = p_dataset->get_input_queue();
    std::vector<DeconQueue_ptr> decon_queues;
    get_decon_queues_from_input_queue(p_dataset, p_input_queue, decon_queues);
    for (const auto &p_aux_input_queue: p_dataset->get_aux_input_queues())
        get_decon_queues_from_input_queue(p_dataset, p_aux_input_queue, decon_queues);

    return init_ensembles_from_dataset(p_dataset, decon_queues);
}


std::vector<Ensemble_ptr> EnsembleService::init_ensembles_from_dataset(
        const Dataset_ptr &p_dataset,
        const std::vector<svr::datamodel::DeconQueue_ptr> &decon_queues)
{
    LOG4_WARN("Wrong decon queues are guessed per ensemble!");
    // init ensembles
    std::vector<Ensemble_ptr> ensembles;
    for (const auto &main_value_column: p_dataset->get_input_queue()->get_value_columns()) {
        std::vector<DeconQueue_ptr> decon_queues_aux;
        for (const auto &p_aux_input_queue: p_dataset->get_aux_input_queues()) {
            for (const auto &value_column_aux: p_aux_input_queue->get_value_columns())
                if (const auto p_aux_decon_queue = APP.decon_queue_service.find_decon_queue(decon_queues, p_aux_input_queue->get_table_name(), value_column_aux))
                    decon_queues_aux.push_back(p_aux_decon_queue);
        }
        if (const auto &p_main_decon_queue = APP.decon_queue_service.find_decon_queue(decon_queues, p_dataset->get_input_queue()->get_table_name(), main_value_column))
            ensembles.push_back(std::make_shared<Ensemble>(
                    0, p_dataset->get_id(), std::vector<Model_ptr>{}, p_main_decon_queue, decon_queues_aux));
        else
            LOG4_ERROR("Could not find decon queue for " << p_dataset->get_input_queue()->get_table_name() << " " << main_value_column);
    }

    LOG4_END();

    return ensembles;
}

#include "DaemonFacade.hpp"

void
EnsembleService::update_ensemble_decon_queues(
        const vector<Ensemble_ptr> &ensembles,
        const std::vector<DeconQueue_ptr> &new_decon_queues)
{
    LOG4_BEGIN();

    if (ensembles.size() != new_decon_queues.size()) LOG4_WARN(
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


void
EnsembleService::init_decon_data(
        std::vector<Ensemble_ptr> &ensembles,
        const std::vector<DeconQueue_ptr> &decon_queues)
{
    LOG4_BEGIN();

    for (auto &p_ensemble: ensembles) {
        // Init main decon queue
        for (const auto &p_decon_queue: decon_queues)
            if (
                    p_ensemble->get_decon_queue()->get_input_queue_column_name() ==
                    p_decon_queue->get_input_queue_column_name() &&
                    p_ensemble->get_decon_queue()->get_input_queue_table_name() ==
                    p_decon_queue->get_input_queue_table_name()) {
                p_ensemble->get_decon_queue()->set_data(p_decon_queue->get_data());
                break;
            }

        // Aux columns data from main input queue. These are added when first creating the ensemble using paramtune and aux columns features option on.
        for (auto &p_ensemble_aux_decon_queue: p_ensemble->get_aux_decon_queues())
            for (const auto &p_decon_queue: decon_queues)
                if (
                        p_ensemble_aux_decon_queue->get_input_queue_table_name() ==
                        p_decon_queue->get_input_queue_table_name() &&
                        p_ensemble_aux_decon_queue->get_input_queue_column_name() ==
                        p_decon_queue->get_input_queue_column_name()) {
                    p_ensemble_aux_decon_queue->set_data(p_decon_queue->get_data());
                    break;
                }
    }

    LOG4_END();
}


bool EnsembleService::exists(bigint ensemble_id)
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
        return std::next(cont.begin(), cont.size() - MODEL_TRAIN_OFFSET - needed_size);
    else
        return std::next(find_nearest(cont, model_last_time + resolution), - lag_count - MODEL_TRAIN_OFFSET);
}


void EnsembleService::prepare_data_and_train_model(
        Model_ptr &p_model,
        const DeconQueue_ptr &p_main_delta_decon_queue,
        const std::vector<DeconQueue_ptr> &p_aux_delta_decon_queues,
        const SVRParameters_ptr &p_svr_parameters,
        const bpt::time_duration &max_gap)
{
    /*
    LOG4_DEBUG("Training model " << p_model->to_string() << " with parameters " << p_svr_parameters->to_string());
    datarow_range main_data_range(
            get_start(
                    p_main_delta_decon_queue->get_data(),
                    p_svr_parameters->get_svr_decremental_distance(),
                    p_svr_parameters->get_lag_count(),
                    p_model->get_last_modeled_value_time()),
            p_main_delta_decon_queue->get_data().end(),
            p_main_delta_decon_queue->get_data());

    // get aux_data_row vector
    std::vector<datarow_range> aux_data_ranges;
    for (const auto &p_aux_decon_queue: p_aux_delta_decon_queues)
        aux_data_ranges.emplace_back(
                datarow_range(
                        get_start(
                                p_aux_decon_queue->get_data(),
                                p_svr_parameters->get_svr_decremental_distance(),
                                p_svr_parameters->get_lag_count(),
                                p_model->get_last_modeled_value_time()),
                        p_aux_decon_queue->get_data().end(),
                        p_aux_decon_queue->get_data()));

    arma::mat features, labels;
    try {
        // Create training data
        model_service_.get_training_data(
                features, labels,
                main_data_range,
                p_svr_parameters->get_lag_count(),
                p_model->get_learning_levels(),
                max_gap,
                p_model->get_decon_level(),
                p_model->get_last_modeled_value_time());

        LOG4_DEBUG(
                "Decon queue from " << p_main_delta_decon_queue->get_data().begin()->first << " to "
                                    << p_main_delta_decon_queue->get_data().rbegin()->first
                                    << " model " << p_model->to_string()
//                                    << " aux decon queues "
//                                    << ::svr::common::deep_to_string(p_ensemble->get_aux_decon_table_names())
                                    << " parameters " << p_svr_parameters->to_string()
                                    << " ensemble " << p_model->get_ensemble_id()
                                    << " features number of columns " << features.n_cols
                                    << " rows " << features.n_rows
                                    << " training labels cols " << labels.n_cols << " rows " << labels.n_rows);
        model_service_.train(
                *p_svr_parameters, p_model, features, labels, main_data_range.get_container().rbegin()->first);
    } catch (const std::runtime_error &ex) {
        LOG4_ERROR(ex.what());
    }
*/
    LOG4_END();
}

// TODO Train with a time range!
// TODO Split in two, one for per model train one for all models in ensemble
void EnsembleService::train(
        Dataset_ptr &p_dataset,
        Ensemble_ptr &p_ensemble,
        const std::vector<SVRParameters_ptr> &model_parameters,
        const bpt::time_duration &max_gap,
        const size_t model_numb)
{
    LOG4_BEGIN();

    if (model_numb != std::numeric_limits<size_t>::max()) LOG4_DEBUG("Training only model " << model_numb);
    const auto models_count = model_parameters.size();
    if (models_count != p_ensemble->get_decon_queue()->get_column_count())
        LOG4_THROW("Number of SVR parameters " << models_count
                                             << ", does not equal decon queue columns count "
                                             << p_ensemble->get_decon_queue()->get_column_count() << ".");
    if (p_ensemble->get_decon_queue()->get_data().empty())
        LOG4_THROW("Decon queue for ensemble " <<
                                             p_ensemble->get_decon_queue()->get_input_queue_table_name() << " " <<
                                             p_ensemble->get_decon_queue()->get_input_queue_column_name() << " is empty!");

    std::vector<::svr::future<void>> pdtm_tasks;
    std::vector<Model_ptr> models;
#ifdef OUTPUT_TRAINING_DATA
    {
        static size_t dq_ct = 0;
        static std::mutex mx_out;
        {
            std::scoped_lock lg(mx_out);
            if (dq_ct++ == 0) {
                LOG4_FILE("/mnt/faststore/main_decon_queue_before_scaling.txt",
                          p_ensemble->get_decon_queue()->data_to_string(p_ensemble->get_decon_queue()->get_data().size()));
            }
        }
    }
#endif
    // Initialize if models are non existent
    if (not p_ensemble->get_models().empty() && p_ensemble->get_models().size() == models_count) goto __train;

    LOG4_DEBUG("Ensemble models are empty. Initializing " << models_count << " models with default values.");
    for (size_t model_counter = 0; model_counter < models_count; ++model_counter)
        models.push_back(std::make_shared<Model>(
                0,
                p_ensemble->get_id(),
                model_counter,
                get_adjacent_indexes(model_counter, model_parameters[model_counter]->get_svr_adjacent_levels_ratio(), models_count),
                nullptr,
                bpt::min_date_time,
                bpt::min_date_time));
    p_ensemble->set_models(models);

    __train:

    for (size_t model_counter = 0; model_counter < models_count; ++model_counter) {
        // continue if we need train only one model and current current model_counter != model_numb which need to train
        if (model_numb != ALL_MODELS && model_counter != model_numb) {
            LOG4_DEBUG("Skipping training on model " << model_counter << " because we want model " <<
                                 model_numb << " out of " << models_count << " models trained.");
            continue;
        } else
            LOG4_DEBUG("Training model " << model_counter << " ensemble " << p_ensemble->get_id());

        pdtm_tasks.push_back(
                ::svr::async(
                        &EnsembleService::prepare_data_and_train_model,
                        this,
                        std::ref(p_ensemble->get_models()[model_counter]),
                        std::ref(p_ensemble->get_decon_queue()),
                        std::ref(p_ensemble->get_aux_decon_queues()),
                        std::ref(model_parameters[model_counter]),
                        std::ref(max_gap))
        );
    }
    for (auto &tsk: pdtm_tasks) tsk.get();
    LOG4_DEBUG("Finished training ensemble for " << p_ensemble->get_decon_queue()->get_input_queue_table_name() <<
                                                 " " << p_ensemble->get_decon_queue()->get_input_queue_column_name());
}

// Warning predict in ensemble decon queues
void EnsembleService::predict(
        Ensemble_ptr &p_ensemble,
        const boost::posix_time::time_period &range,
        const boost::posix_time::time_duration &resolution,
        const bpt::time_duration &max_gap)
{
    auto aux_datas = p_ensemble->get_aux_decon_datas();
    predict(p_ensemble, range, resolution, max_gap, p_ensemble->get_decon_queue()->get_data(), aux_datas);
}


void
EnsembleService::predict(
        const Ensemble_ptr &p_ensemble,
        const boost::posix_time::time_period &range,
        const boost::posix_time::time_duration &resolution,
        const bpt::time_duration &max_gap,
        ::svr::datamodel::DataRow::container &decon_data,
        std::vector<data_row_container_ptr> &aux_decon_data)
{
    LOG4_DEBUG("Request to predict in place time period new function " << range);
    std::vector<data_row_container_ptr> all_decon_data;
    all_decon_data.push_back(shared_observer_ptr<data_row_container>(decon_data));
    for (const auto &p_aux_data: aux_decon_data) all_decon_data.push_back(p_aux_data);
    for (auto current_time = range.begin(); current_time < range.end();) { // Predict every time
        bpt::ptime last_predicted_time;
        for (auto p_decon_data: all_decon_data) { // Predict every column
            auto priv_aux_decon_data = get_all_except(all_decon_data, p_decon_data);
            const auto prediction_results = ::APP.model_service.predict(
                    p_ensemble->get_models(),
                    current_time,
                    resolution,
                    max_gap,
                    *p_decon_data,
                    priv_aux_decon_data);
            last_predicted_time = datamodel::DataRow::insert_rows(*p_decon_data, prediction_results, current_time, resolution);
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
