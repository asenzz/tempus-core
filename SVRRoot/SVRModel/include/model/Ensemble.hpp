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

    bigint get_dataset_id() const;

    std::string get_column_name() const;

    std::deque<datamodel::Model_ptr> &get_models();

    datamodel::Model_ptr get_model(const size_t levix = 0);

    const std::deque<datamodel::Model_ptr> &get_models() const;

    std::deque<bigint> get_models_ids();

    void set_models(const std::deque<datamodel::Model_ptr> &m);

    // Each ensemble is associated with one column of the main input queue of the dataset, this is it's decon queue
    datamodel::DeconQueue_ptr get_decon_queue() const;

    datamodel::DeconQueue_ptr &get_decon_queue();

    ALIAS_TEMPLATE_FUNCTION(get_label_decon, get_decon_queue)

    datamodel::DeconQueue_ptr get_label_aux_decon() const;

    void set_decon_queue(const datamodel::DeconQueue_ptr &p_decon_queue_);

    std::deque<datamodel::DeconQueue_ptr> &get_aux_decon_queues();

    datamodel::DeconQueue_ptr &get_aux_decon_queue(const size_t i = 0);

    datamodel::DeconQueue_ptr get_aux_decon_queue(const std::string &column_name);

    std::deque<data_row_container_ptr> get_aux_decon_datas();

    const std::deque<datamodel::DeconQueue_ptr> &get_aux_decon_queues() const;

    std::deque<std::string> get_aux_decon_table_names() const;

    void set_aux_decon_queues(const std::deque<datamodel::DeconQueue_ptr> &aux_decon_queues_);

    std::string to_string() const override;
};

using Ensemble_ptr = std::shared_ptr<Ensemble>;

}
}
