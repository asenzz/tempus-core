#pragma once

#include "util/PerformanceUtils.hpp"
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
//    std::deque<datamodel::SVRParameters_ptr> vec_svr_parameters;
    std::deque<datamodel::Model_ptr> models;
    datamodel::DeconQueue_ptr p_decon_queue;
    std::deque<datamodel::DeconQueue_ptr> aux_decon_queues;

public:
    bool operator==(Ensemble const &o)
    {
        return dataset_id == o.dataset_id && models == o.models && p_decon_queue == o.p_decon_queue && aux_decon_queues == o.aux_decon_queues;
    }

    Ensemble() : Entity() {}

    Ensemble(const bigint id, const bigint dataset_id, const std::string &decon_queue_table_name, const std::deque<std::string> &aux_decon_queue_table_names, const bool load_decon_data = false);

    Ensemble(const bigint id, const bigint dataset_id, const std::deque<datamodel::Model_ptr> &models,
             const datamodel::DeconQueue_ptr &p_decon_queue, const std::deque<datamodel::DeconQueue_ptr> &aux_decon_queues = {})
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

//    std::deque<datamodel::SVRParameters_ptr>& get_vec_svr_parameters()
//    {
//        return vec_svr_parameters;
//    }

//    const std::deque<datamodel::SVRParameters_ptr>& get_vec_svr_parameters() const
//    {
//        return vec_svr_parameters;
//    }

//    void set_vec_svr_parameters(const std::deque<datamodel::SVRParameters_ptr> &vec_svr_parameters)
//    {
//        this->vec_svr_parameters = vec_svr_parameters;
//        for (datamodel::SVRParameters_ptr p_param_set : vec_svr_parameters) {
//            p_param_set->set_ensemble_id(get_id());
//        }
//    }

    std::deque<datamodel::Model_ptr> &get_models()
    {
        return models;
    }

    datamodel::Model_ptr get_model(const size_t levix = 0);

    const std::deque<datamodel::Model_ptr> &get_models() const
    {
        return models;
    }

    std::deque<bigint> get_models_ids()
    {
        std::deque<bigint> res;
        for (const auto &m: models) res.emplace_back(m->get_id());
        return res;
    }

    void set_models(const std::deque<datamodel::Model_ptr> &m)
    {
        models = m;
    }

    // Each ensemble is associated with one column of the main input queue of the dataset, this is it's decon queue
    datamodel::DeconQueue_ptr get_decon_queue() const
    {
        return p_decon_queue;
    }

    datamodel::DeconQueue_ptr &get_decon_queue()
    {
        return p_decon_queue;
    }

    void set_decon_queue(const datamodel::DeconQueue_ptr &p_decon_queue_)
    {
        p_decon_queue = p_decon_queue_;
    }

    std::deque<datamodel::DeconQueue_ptr> &get_aux_decon_queues()
    {
        return aux_decon_queues;
    }

    datamodel::DeconQueue_ptr &get_aux_decon_queue(const size_t i = 0)
    {
        if (i >= aux_decon_queues.size()) LOG4_THROW("Illegal i " << i);
        return aux_decon_queues[i];
    }

    datamodel::DeconQueue_ptr get_aux_decon_queue(const std::string &column_name)
    {
        const auto res = std::find_if(std::execution::par_unseq,
                aux_decon_queues.begin(), aux_decon_queues.end(),
                [&column_name](const datamodel::DeconQueue_ptr &d) { return d->get_input_queue_column_name() == column_name; });
        if (res == aux_decon_queues.end())
            return nullptr;
        return *res;
    }

    std::deque<data_row_container_ptr> get_aux_decon_datas()
    {
        std::deque<data_row_container_ptr> result;
        for (const auto &aux_decon_queue: aux_decon_queues)
            if (aux_decon_queue)
                result.emplace_back(shared_observer_ptr<data_row_container>(aux_decon_queue->get_data()));
        return result;
    }

    const std::deque<datamodel::DeconQueue_ptr> &get_aux_decon_queues() const
    {
        return aux_decon_queues;
    }

    std::deque<std::string> get_aux_decon_table_names() const
    {
        std::deque<std::string> res;
        std::transform(aux_decon_queues.begin(), aux_decon_queues.end(), std::back_inserter(res),
                       [](const datamodel::DeconQueue_ptr &decon) { return decon->get_table_name(); });
        return res;
    }

    void set_aux_decon_queues(const std::deque<datamodel::DeconQueue_ptr> &aux_decon_queues_)
    {
        aux_decon_queues = aux_decon_queues_;
    }

    std::string to_string() const override
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
};

using Ensemble_ptr = std::shared_ptr<Ensemble>;

}
}
