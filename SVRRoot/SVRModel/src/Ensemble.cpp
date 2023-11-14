//
// Created by zarko on 2/14/23.
//

#include "appcontext.hpp"
#include "model/Ensemble.hpp"
#include "model/Model.hpp"
#include "onlinesvr.hpp"


namespace svr::datamodel {


Ensemble::Ensemble(const bigint id, const bigint dataset_id, const std::string &decon_queue_table_name,
         const std::vector<std::string> &aux_decon_queue_table_names) : Entity(id), dataset_id(dataset_id)
{
    if (!decon_queue_table_name.empty()) {
        p_decon_queue = APP.decon_queue_service.get_by_table_name(decon_queue_table_name);
        if (!p_decon_queue) {
            p_decon_queue = std::make_shared<DeconQueue>();
            p_decon_queue->set_dataset_id(dataset_id);
            p_decon_queue->set_table_name(decon_queue_table_name);
        }
    }

    for (const std::string &aux_decon_queue_table_name: aux_decon_queue_table_names) {
        auto p_aux_decon_queue = APP.decon_queue_service.get_by_table_name(aux_decon_queue_table_name);
        if (!p_aux_decon_queue) {
            p_aux_decon_queue = std::make_shared<DeconQueue>();
            p_aux_decon_queue->set_dataset_id(dataset_id);
            p_decon_queue->set_table_name(aux_decon_queue_table_name);
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
    for (const auto &p_model: models)
        if (p_model && p_model->get_svr_model())
            p_model->get_svr_model()->get_svr_parameters().set_dataset_id(dataset_id_);
}

}