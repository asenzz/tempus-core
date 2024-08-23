//
// Created by zarko on 2/14/23.
//

#include "appcontext.hpp"
#include "model/Ensemble.hpp"
#include "common/logging.hpp"
#include "model/Model.hpp"
#include "onlinesvr.hpp"
#include "ModelService.hpp"
#include <iterator>


namespace svr {
namespace datamodel {


bool Ensemble::operator==(const Ensemble &o) const
{
    return dataset_id == o.dataset_id && models == o.models && p_decon_queue == o.p_decon_queue && aux_decon_queues == o.aux_decon_queues;
}

Ensemble::Ensemble(const bigint id, const bigint dataset_id, const std::deque<datamodel::Model_ptr> &models,
         const datamodel::DeconQueue_ptr &p_decon_queue, const std::deque<datamodel::DeconQueue_ptr> &aux_decon_queues)
        : Entity(id), dataset_id(dataset_id), models(models),
          p_decon_queue(p_decon_queue), aux_decon_queues(aux_decon_queues)
{
#ifdef ENTITY_INIT_ID
    init_id();
#endif
    if (dataset_id) set_dataset_id(dataset_id);
}

Ensemble::Ensemble(const bigint id, const bigint dataset_id, const std::string &decon_queue_table_name,
                   const std::deque<std::string> &aux_decon_queue_table_names, const bool load_decon_data) : Entity(id), dataset_id(dataset_id)
{
    if (!decon_queue_table_name.empty()) {
        p_decon_queue = APP.decon_queue_service.get_by_table_name(decon_queue_table_name);
        if (!p_decon_queue) {
            p_decon_queue = std::make_shared<DeconQueue>(decon_queue_table_name, "", "", dataset_id);
            LOG4_DEBUG("Decon queue " << decon_queue_table_name << " does not exist in database, creating with default configuration.");
        } else if (load_decon_data)
            try {
                APP.decon_queue_service.load(*p_decon_queue);
            } catch (const std::exception &ex) {
                LOG4_ERROR("Loading decon data for " << *p_decon_queue << " failed , error " << ex.what());
            }
    }

    for (const std::string &aux_decon_queue_table_name: aux_decon_queue_table_names) {
        auto p_aux_decon_queue = APP.decon_queue_service.get_by_table_name(aux_decon_queue_table_name);
        if (!p_aux_decon_queue) {
            p_aux_decon_queue = ptr<DeconQueue>(aux_decon_queue_table_name, "", "", dataset_id);
            LOG4_DEBUG("Aux decon queue " << decon_queue_table_name << " does not exist in database, creating with default configuration.");
        } else if (load_decon_data)
            try {
                APP.decon_queue_service.load(*p_aux_decon_queue);
            } catch (const std::exception &ex) {
                LOG4_ERROR("Loading decon data for " << *p_aux_decon_queue << "failed , error " << ex.what());
            }
        aux_decon_queues.emplace_back(p_aux_decon_queue);
    }
#ifdef ENTITY_INIT_ID
    init_id();
#endif
    if (dataset_id) set_dataset_id(dataset_id);
}

void Ensemble::init_id()
{
    if (!id) {
        boost::hash_combine(id, dataset_id);
        if (p_decon_queue) boost::hash_combine(id, p_decon_queue->get_id());
    }
}

void Ensemble::set_dataset_id(const bigint dataset_id_)
{
    dataset_id = dataset_id_;
    if (p_decon_queue)
        p_decon_queue->set_dataset_id(dataset_id_);
#pragma omp parallel for num_threads(adj_threads(aux_decon_queues.size()))
    for (const auto &p_aux_decon: aux_decon_queues)
        if (p_aux_decon)
            p_aux_decon->set_dataset_id(dataset_id_);
#pragma omp parallel for num_threads(adj_threads(models.size()))
    for (auto &p_model: models) {
        if (!p_model) continue;
        p_model->set_ensemble_id(id);
        for (auto &p_svr_model: p_model->get_gradients()) {
            if (!p_svr_model) continue;
            p_svr_model->set_model_id(p_model->get_id());
            for (auto &p_param: p_svr_model->get_param_set()) {
                if (!p_param) continue;
                p_param->set_dataset_id(dataset_id_);
                p_param->set_input_queue_table_name(p_decon_queue->get_input_queue_table_name());
            }
        }
    }
}

datamodel::Model_ptr Ensemble::get_model(const size_t levix, const size_t stepix) const
{
    return business::ModelService::find(models, levix, stepix);
}

bigint Ensemble::get_dataset_id() const
{
    return dataset_id;
}

std::string Ensemble::get_column_name() const
{
    return p_decon_queue ? p_decon_queue->get_input_queue_column_name() : "";
}

std::deque<datamodel::Model_ptr> &Ensemble::get_models()
{
    return models;
}

const std::deque<datamodel::Model_ptr> &Ensemble::get_models() const
{
    return models;
}

std::deque<bigint> Ensemble::get_models_ids()
{
    std::deque<bigint> res;
    for (const auto &m: models) res.emplace_back(m->get_id());
    return res;
}

void Ensemble::set_models(const std::deque<datamodel::Model_ptr> &new_models, const bool overwrite)
{
    const auto prev_size = models.size();
    for (const auto &p_new_model: new_models) {
        std::atomic<bool> found = false;
#pragma omp parallel for num_threads(adj_threads(prev_size))
        for (size_t i = 0; i < prev_size; ++i)
            if (models[i]->get_decon_level() == p_new_model->get_decon_level() &&
                (!models[i]->get_ensemble_id() || !p_new_model->get_ensemble_id() || models[i]->get_ensemble_id() == p_new_model->get_ensemble_id())) {
                if (overwrite || models[i]->get_gradients().empty()) {
                    models[i] = p_new_model;
                    models[i]->set_ensemble_id(id);
                }
                found.store(true, std::memory_order_relaxed);
            }
        if (found) continue;
        models.emplace_back(p_new_model);
        models.back()->set_ensemble_id(id);
    }
}

datamodel::DeconQueue_ptr Ensemble::get_decon_queue() const
{
    return p_decon_queue;
}

datamodel::DeconQueue_ptr &Ensemble::get_decon_queue()
{
    return p_decon_queue;
}

datamodel::DeconQueue_ptr Ensemble::get_label_aux_decon() const
{
    return business::DeconQueueService::find_decon_queue(
            aux_decon_queues, aux_decon_queues.front()->get_input_queue_table_name(), p_decon_queue->get_input_queue_column_name());
}

void Ensemble::set_decon_queue(const datamodel::DeconQueue_ptr &p_decon_queue_)
{
    p_decon_queue = p_decon_queue_;
}

std::deque<datamodel::DeconQueue_ptr> &Ensemble::get_aux_decon_queues()
{
    return aux_decon_queues;
}

datamodel::DeconQueue_ptr Ensemble::get_aux_decon_queue(const size_t i) const
{
    if (i >= aux_decon_queues.size()) LOG4_THROW("Illegal index " << i);
    return aux_decon_queues[i];
}

datamodel::DeconQueue_ptr Ensemble::get_aux_decon_queue(const std::string &column_name) const
{
    const auto res = std::find_if(C_default_exec_policy, aux_decon_queues.begin(), aux_decon_queues.end(),
                                  [&column_name](const datamodel::DeconQueue_ptr &d) { return d->get_input_queue_column_name() == column_name; });
    if (res == aux_decon_queues.end()) {
        LOG4_ERROR("Invalid column name " << column_name);
        return nullptr;
    }
    return *res;
}

std::deque<data_row_container_ptr> Ensemble::get_aux_decon_datas()
{
    std::deque<data_row_container_ptr> result;
    for (const auto &aux_decon_queue: aux_decon_queues)
        if (aux_decon_queue)
            result.emplace_back(shared_observer_ptr<data_row_container>(aux_decon_queue->get_data()));
    return result;
}

const std::deque<datamodel::DeconQueue_ptr> &Ensemble::get_aux_decon_queues() const
{
    return aux_decon_queues;
}

std::deque<std::string> Ensemble::get_aux_decon_table_names() const
{
    std::deque<std::string> res;
    std::transform(aux_decon_queues.begin(), aux_decon_queues.end(), std::back_inserter(res),
                   [](const datamodel::DeconQueue_ptr &decon){ return decon->get_table_name(); });
    return res;
}

void Ensemble::set_aux_decon_queues(const std::deque<datamodel::DeconQueue_ptr> &aux_decon_queues_)
{
    aux_decon_queues = aux_decon_queues_;
}

std::string Ensemble::to_string() const
{
    std::stringstream s;
    s << "Ensemble ID " << id << ", dataset ID " << dataset_id << ", models:\n";
    for (const auto &p_model: models)
        if (p_model) s << "\t" << *p_model << "\n";
    if (p_decon_queue)
        s << "Main decon queue:\n" << *p_decon_queue << "\n, aux decon queues:\n";
    for (const auto &p_aux_decon_queue: aux_decon_queues)
        if (p_aux_decon_queue) s << '\t' << *p_aux_decon_queue << '\n';
    return s.str();
}

size_t Ensemble::get_level_ct() const
{
    return business::ModelService::to_level_ct(models.size());
}

size_t Ensemble::get_model_ct() const
{
    return models.size();
}

}
}