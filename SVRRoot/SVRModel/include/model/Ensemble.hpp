#pragma once

#include <util/PerformanceUtils.hpp>
#include "model/Entity.hpp"
#include "model/Model.hpp"
#include "model/DeconQueue.hpp"
#include "model/SVRParameters.hpp"
#include "util/string_utils.hpp"


namespace svr {
namespace datamodel {

class Ensemble : public Entity
{

private:
    bigint dataset_id; /* TODO Replace with pointer to a dataset */
//    std::vector<SVRParameters_ptr> vec_svr_parameters;
    std::vector<Model_ptr> models;
    DeconQueue_ptr p_decon_queue;
    std::vector<DeconQueue_ptr> aux_decon_queues;

public:
    bool operator==(Ensemble const &o)
    {
        return dataset_id == o.dataset_id && models == o.models && p_decon_queue == o.p_decon_queue && aux_decon_queues == o.aux_decon_queues;
    }

    Ensemble() : Entity() {}

    Ensemble(const bigint id, const bigint dataset_id, const std::string &decon_queue_table_name, const std::vector<std::string> &aux_decon_queue_table_names);

    Ensemble(const bigint id, const bigint dataset_id, const std::vector<Model_ptr> &models,
             const DeconQueue_ptr &p_decon_queue, const std::vector<DeconQueue_ptr> &aux_decon_queues = std::vector<DeconQueue_ptr>())
            : Entity(id), dataset_id(dataset_id), models(models),
              p_decon_queue(p_decon_queue), aux_decon_queues(aux_decon_queues)
    {
        if (dataset_id) set_dataset_id(dataset_id);
    }

    void set_dataset_id(const bigint dataset_id_);

    bigint get_dataset_id() const
    {
        return dataset_id;
    }

//    std::vector<SVRParameters_ptr>& get_vec_svr_parameters()
//    {
//        return vec_svr_parameters;
//    }

//    const std::vector<SVRParameters_ptr>& get_vec_svr_parameters() const
//    {
//        return vec_svr_parameters;
//    }

//    void set_vec_svr_parameters(const std::vector<SVRParameters_ptr> &vec_svr_parameters)
//    {
//        this->vec_svr_parameters = vec_svr_parameters;
//        for (SVRParameters_ptr p_svr_parameters : vec_svr_parameters) {
//            p_svr_parameters->set_ensemble_id(get_id());
//        }
//    }

    std::vector<Model_ptr> &get_models()
    {
        return models;
    }

    Model_ptr &get_model(const size_t ix)
    {
        return models[ix];
    }

    const std::vector<Model_ptr> &get_models() const
    {
        return models;
    }

    std::vector<bigint> get_models_ids()
    {
        std::vector<bigint> res;
        res.reserve(models.size());
        for (Model_ptr model: models) res.emplace_back(model->get_id());
        return res;
    }

    void set_models(const std::vector<Model_ptr> &models_)
    {
        models = models_;
    }

    DeconQueue_ptr &get_decon_queue()
    {
        return p_decon_queue;
    }

    void set_decon_queue(const DeconQueue_ptr &p_decon_queue_)
    {
        p_decon_queue = p_decon_queue_;
    }

    std::vector<DeconQueue_ptr> &get_aux_decon_queues()
    {
        return aux_decon_queues;
    }

    DeconQueue_ptr &get_aux_decon_queue(const size_t index = 0)
    {
        if (index >= aux_decon_queues.size()) LOG4_THROW("Illegal index " << index);
        return aux_decon_queues[index];
    }

    std::vector<data_row_container_ptr> get_aux_decon_datas()
    {
        std::vector<data_row_container_ptr> result;
        for (const auto &aux_decon_queue: aux_decon_queues)
            if (aux_decon_queue)
                result.emplace_back(shared_observer_ptr<data_row_container>(aux_decon_queue->get_data()));
        return result;
    }

    const std::vector<DeconQueue_ptr> &get_aux_decon_queues() const
    {
        return aux_decon_queues;
    }

    std::vector<std::string> get_aux_decon_table_names() const
    {
        std::vector<std::string> res;
        res.reserve(aux_decon_queues.size());
        std::transform(aux_decon_queues.begin(), aux_decon_queues.end(), std::back_inserter(res),
                       [](const DeconQueue_ptr &decon) { return decon->get_table_name(); });
        return res;
    }

    void set_aux_decon_queues(const std::vector<DeconQueue_ptr> &aux_decon_queues_)
    {
        aux_decon_queues = aux_decon_queues_;
    }

    std::pair<std::string, std::string> get_key_pair() const
    {
        return {p_decon_queue->get_input_queue_table_name(), p_decon_queue->get_input_queue_column_name()};
    }

    std::string to_string() const override
    {
        std::stringstream ss;
        ss << "Ensemble ID " << id << ", dataset ID " << dataset_id << ", models:\n";
        for (const auto &p_model: models)
            if (p_model) ss << "\t" << p_model->to_string() << "\n";
        if (p_decon_queue)
            ss << "Main decon queue:\n" << p_decon_queue->to_string() << "\n, aux decon queues:\n";
        for (const auto &p_aux_decon_queue: aux_decon_queues)
            if (p_aux_decon_queue)
                ss << "\t" << p_aux_decon_queue->to_string() << "\n";
        return ss.str();
    }

};


}
}

using Ensemble_ptr = std::shared_ptr<svr::datamodel::Ensemble>;
