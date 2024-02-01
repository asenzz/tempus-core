//
// Created by zarko on 2/14/23.
//

#include "appcontext.hpp"
#include "model/Ensemble.hpp"
#include "model/Model.hpp"
#include "onlinesvr.hpp"
#include "ModelService.hpp"


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

}
}