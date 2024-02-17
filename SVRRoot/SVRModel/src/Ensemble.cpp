//
// Created by zarko on 2/14/23.
//

#include "appcontext.hpp"
#include "model/Ensemble.hpp"
#include "model/Model.hpp"
#include "onlinesvr.hpp"
#include "ModelService.hpp"
#include <iterator>


namespace svr {
namespace datamodel {


Ensemble::Ensemble(const bigint id, const bigint dataset_id, const std::string &decon_queue_table_name,
                   const std::deque<std::string> &aux_decon_queue_table_names, const bool load_decon_data) : Entity(id), dataset_id(dataset_id)
{
    if (!decon_queue_table_name.empty()) {
        p_decon_queue = APP.decon_queue_service.get_by_table_name(decon_queue_table_name);
        if (!p_decon_queue) {
            p_decon_queue = std::make_shared<DeconQueue>();
            p_decon_queue->set_dataset_id(dataset_id);
            p_decon_queue->set_table_name(decon_queue_table_name);
            LOG4_DEBUG("Decon queue " << decon_queue_table_name << " does not exist in database, creating with default configuration.");
        } else if (load_decon_data)
            try {
                APP.decon_queue_service.load(p_decon_queue);
            } catch (const std::exception &ex) {
                LOG4_ERROR("Loading decon data for " << *p_decon_queue << " failed , error " << ex.what());
            }
    }

    for (const std::string &aux_decon_queue_table_name: aux_decon_queue_table_names) {
        auto p_aux_decon_queue = APP.decon_queue_service.get_by_table_name(aux_decon_queue_table_name);
        if (!p_aux_decon_queue) {
            p_aux_decon_queue = std::make_shared<DeconQueue>();
            p_aux_decon_queue->set_dataset_id(dataset_id);
            p_decon_queue->set_table_name(aux_decon_queue_table_name);
            LOG4_DEBUG("Aux decon queue " << decon_queue_table_name << " does not exist in database, creating with default configuration.");
        } else if (load_decon_data)
            try {
                APP.decon_queue_service.load(p_aux_decon_queue);
            } catch (const std::exception &ex) {
                LOG4_ERROR("Loading decon data for " << *p_aux_decon_queue << "failed , error " << ex.what());
            }
        aux_decon_queues.emplace_back(p_aux_decon_queue);
    }

    if (dataset_id) set_dataset_id(dataset_id);
}


void Ensemble::set_dataset_id(const bigint dataset_id_)
{
    dataset_id = dataset_id_;
    if (p_decon_queue) p_decon_queue->set_dataset_id(dataset_id_);
    for (const auto &p_aux_decon: aux_decon_queues)
        if (p_aux_decon)
            p_aux_decon->set_dataset_id(dataset_id_);
    for (auto &p_model: models)
        if (p_model && p_model->get_gradients().size())
            for (auto &p_param: p_model->get_param_set())
                p_param->set_dataset_id(dataset_id_);
}

datamodel::Model_ptr Ensemble::get_model(const size_t levix)
{
    return business::ModelService::find(models, levix);
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

void Ensemble::set_models(const std::deque<datamodel::Model_ptr> &m)
{
    models = m;
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

datamodel::DeconQueue_ptr &Ensemble::get_aux_decon_queue(const size_t i)
{
    if (i >= aux_decon_queues.size()) LOG4_THROW("Illegal i " << i);
    return aux_decon_queues[i];
}

datamodel::DeconQueue_ptr Ensemble::get_aux_decon_queue(const std::string &column_name)
{
    const auto res = std::find_if(std::execution::par_unseq,
                                  aux_decon_queues.begin(), aux_decon_queues.end(),
                                  [&column_name](const datamodel::DeconQueue_ptr &d) { return d->get_input_queue_column_name() == column_name; });
    if (res == aux_decon_queues.end())
        return nullptr;
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
        if (p_model) s << "\t" << p_model->to_string() << "\n";
    if (p_decon_queue)
        s << "Main decon queue:\n" << p_decon_queue->to_string() << "\n, aux decon queues:\n";
    for (const auto &p_aux_decon_queue: aux_decon_queues)
        if (p_aux_decon_queue)
            s << "\t" << p_aux_decon_queue->to_string() << "\n";
    return s.str();
}

}
}