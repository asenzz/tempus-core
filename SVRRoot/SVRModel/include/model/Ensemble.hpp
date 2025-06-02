#pragma once

#include <boost/unordered/unordered_flat_map.hpp>
#include "util/PerformanceUtils.hpp"
#include "model/Entity.hpp"
#include "model/Model.hpp"
#include "model/DeconQueue.hpp"

namespace svr {
namespace datamodel {

struct t_level_predict_features
{
    data_row_container times;
    mat_ptr p;
};
using t_predict_features = boost::unordered_flat_map<std::tuple<size_t /* level */, size_t /* step */>, t_level_predict_features, common::hash_tuple>;

struct t_training_data
{
    boost::unordered_flat_map<std::tuple<size_t /* level */, size_t /* step */>, mat_ptr> labels;
    boost::unordered_flat_map<std::tuple<size_t /* level */, size_t /* step */>, mat_ptr> features;
    boost::unordered_flat_map<size_t /* level */, vec_ptr> last_knowns;
    bpt::ptime last_row_time = bpt::not_a_date_time;
};

class Ensemble : public Entity
{
    bigint dataset_id = 0; /* TODO Replace with pointer to a dataset */
    std::deque<datamodel::Model_ptr> models;
    datamodel::DeconQueue_ptr p_decon_queue;
    std::deque<datamodel::DeconQueue_ptr> aux_decon_queues;

public:
    bool operator==(Ensemble const &o) const;

    Ensemble() : Entity() {}

    Ensemble(const bigint id, const bigint dataset_id, const std::string &decon_queue_table_name,
             const std::deque<std::string> &aux_decon_queue_table_names, const bool load_decon_data = false);

    Ensemble(const bigint id, const bigint dataset_id, const std::deque<datamodel::Model_ptr> &models,
             const datamodel::DeconQueue_ptr &p_decon_queue, const std::deque<datamodel::DeconQueue_ptr> &aux_decon_queues = {});

    virtual void init_id() override;

    void set_dataset_id(const bigint dataset_id_);

    bigint get_dataset_id() const;

    std::string get_column_name() const;

    datamodel::Model_ptr get_model(const size_t levix, const size_t stepix) const;

    std::deque<datamodel::Model_ptr> &get_models();

    const std::deque<datamodel::Model_ptr> &get_models() const;

    std::deque<bigint> get_models_ids();

    void set_models(const std::deque<datamodel::Model_ptr> &p_new_model, const bool overwrite);

    // Each ensemble is associated with one column of the main input queue of the dataset, this is it's decon queue
    datamodel::DeconQueue_ptr get_decon_queue() const;

    datamodel::DeconQueue_ptr &get_decon_queue();

    ALIAS_TEMPLATE_FUNCTION(get_label_decon, get_decon_queue)

    datamodel::DeconQueue_ptr get_label_aux_decon() const;

    void set_decon_queue(const datamodel::DeconQueue_ptr &p_decon_queue_);

    std::deque<datamodel::DeconQueue_ptr> &get_aux_decon_queues();

    datamodel::DeconQueue_ptr get_aux_decon_queue(const size_t i = 0) const;

    datamodel::DeconQueue_ptr get_aux_decon_queue(const std::string &column_name) const;

    std::deque<data_row_container_ptr> get_aux_decon_datas();

    const std::deque<datamodel::DeconQueue_ptr> &get_aux_decon_queues() const;

    std::deque<std::string> get_aux_decon_table_names() const;

    void set_aux_decon_queues(const std::deque<datamodel::DeconQueue_ptr> &aux_decon_queues_);

    std::string to_string() const override;

    uint16_t get_level_ct() const;

    uint16_t get_model_ct() const;
};

using Ensemble_ptr = std::shared_ptr<Ensemble>;

}

}
