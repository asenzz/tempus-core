#pragma once

#include <unordered_map>
#include <mutex>

#include "common/constants.hpp"
#include "model/Entity.hpp"
#include "model/Priority.hpp"
#include "model/InputQueue.hpp"
#include "model/IQScalingFactor.hpp"
#include "model/DQScalingFactor.hpp"
#include "relations/iq_relation.hpp"
#include "spectral_transform.hpp"
#include "calc_cache.hpp"
#include "fast_cvmd.hpp"
#include "online_emd.hpp"


namespace svr {
namespace oemd {
class online_emd;
}
namespace business {
class DatasetService;
}

namespace datamodel {

class Ensemble;

using Ensemble_ptr = std::shared_ptr<Ensemble>;

class Dataset : public Entity
{
    friend svr::business::DatasetService;
    business::calc_cache ccache;

    void init_transform();

    bool initialized = false;

    std::string dataset_name_; // Dataset name
    std::string user_name_; // Owner of the dataset
    iq_relation input_queue_; // The input queue that is predicted
    std::deque<iq_relation> aux_input_queues_; // Aux input queues are decomposed and used for model features and labels
    Priority priority_ = Priority::Normal;
    std::string description_; // Textual description of the dataset
    uint16_t gradients_ = common::C_default_gradient_count; // Gradients per model, zero gradient is the base model operating on the original input data
    uint32_t max_chunk_size_; // Chunks are specific to SVR models, the chunk size specifies if the model training data should be divided in chunks, this value should be less than decrement distance
    uint16_t multistep_ = common::C_default_multistep_len; // Number of samples to predict for the future time interval as defined by input queue resolution, eg. a multistep of 4 will predict 4 samples of 15 minutes if the input queue has a resolution of 1 hour

    std::unique_ptr<svr::oemd::online_emd> p_oemd_transformer_fat;
    std::unique_ptr<svr::vmd::fast_cvmd> p_cvmd_transformer;
    uint16_t spectrum_levels_ = common::C_default_level_count; // Number of spectral components to extract from every input queue column
    std::string transformation_name_ = "cvmd"; // Deconstruction type
    bpt::time_duration max_lookback_time_gap_ = common::C_default_features_max_time_gap; // Maximum time gap between feature points after which the whole row is discarded from the learning process

    std::deque<datamodel::Ensemble_ptr> ensembles_; // Number of ensembles equals number of columns in the main input queue
    std::mutex ensembles_mx;
    bool is_active_ = false; // Enable or disable processing of the dataset

    std::deque<IQScalingFactor_ptr> iq_scaling_factors_;

    virtual void on_set_id() override;

public:
    Dataset();

    Dataset(
            bigint id,
            const std::string &dataset_name,
            const std::string &user_name,
            datamodel::InputQueue_ptr p_input_queue,
            const std::deque<datamodel::InputQueue_ptr> &aux_input_queues,
            const Priority &priority = Priority::Normal,
            const std::string &description = "",
            const uint16_t gradients = common::C_default_gradient_count,
            const uint32_t chunk_size = common::AppConfig::C_default_kernel_length,
            const uint16_t multistep = common::C_default_multistep_len,
            const uint16_t transformation_levels = common::C_default_level_count,
            const std::string &transformation_name = "cvmd",
            const bpt::time_duration &max_lookback_time_gap = common::C_default_features_max_time_gap,
            const std::deque<datamodel::Ensemble_ptr> &ensembles = {},
            bool is_active = false,
            const std::deque<IQScalingFactor_ptr> iq_scaling_factors = {});

    Dataset(
            bigint id,
            const std::string &dataset_name_,
            const std::string &user_name_,
            const std::string &input_queue_table_name,
            const std::deque<std::string> &aux_input_queues_table_names,
            const Priority &priority = Priority::Normal,
            const std::string &description = "",
            const uint16_t gradients = common::C_default_gradient_count,
            const uint32_t chunk_size = common::AppConfig::C_default_kernel_length,
            const uint16_t multistep = common::C_default_multistep_len,
            const uint16_t transformation_levels = common::C_default_level_count,
            const std::string &transformation_name = "cvmd",
            const bpt::time_duration &max_lookback_time_gap_ = common::C_default_features_max_time_gap,
            const std::deque<datamodel::Ensemble_ptr> &ensembles_ = {},
            bool is_active_ = false,
            const std::deque<IQScalingFactor_ptr> iq_scaling_factors = {});

    Dataset(Dataset const &dataset);

    bool operator==(const Dataset &o) const;

    bool operator^=(const Dataset &other) const; /* functionally equivalent */

    svr::vmd::fast_cvmd &get_cvmd_transformer();

    svr::oemd::online_emd &get_oemd_transformer();

    bool get_initialized();

    const std::string &get_dataset_name() const;

    void set_dataset_name(const std::string &dataset_name);

    const std::string &get_user_name() const;

    void set_user_name(const std::string &user_name);

    datamodel::InputQueue_ptr get_input_queue() const;

    void set_input_queue(const datamodel::InputQueue_ptr &p_input_queue);

    Priority const &get_priority() const;

    void set_priority(Priority const &priority);

    const std::string &get_description() const;

    void set_description(const std::string &description);

    uint16_t get_spectral_levels() const;

    uint16_t get_trans_levix() const;

    uint16_t get_spectrum_levels_cvmd() const;

    uint16_t get_spectrum_levels_oemd() const noexcept;

    uint16_t get_model_count() const;

    uint16_t get_gradient_count() const;

    uint32_t get_max_chunk_size() const;

    uint16_t get_multistep() const;

    void set_spectrum_levels(const uint16_t spectrum_levels);

    void set_gradients(const uint16_t grads);

    void set_chunk_size(const uint32_t chunk_size);

    void set_multistep(const uint16_t multistep);

    const std::string &get_transformation_name() const noexcept;

    void set_transformation_name(const std::string &transformation_name);

    static bool validate_transformation_name(const std::string &transformation_name) noexcept;

    const bpt::time_duration &get_max_lookback_time_gap() const noexcept;

    void set_max_lookback_time_gap(const bpt::time_duration &max_lookback_time_gap);

    std::deque<datamodel::Ensemble_ptr> &get_ensembles() noexcept;

    const std::deque<datamodel::Ensemble_ptr> &get_ensembles() const noexcept;

    datamodel::Ensemble_ptr get_ensemble(const std::string &column_name) noexcept;

    datamodel::Ensemble_ptr get_ensemble(const std::string &table_name, const std::string &column_name) noexcept;

    datamodel::Ensemble_ptr get_ensemble(const uint16_t idx = 0);

    uint16_t get_ensemble_count() const;

    void set_ensembles(const std::deque<datamodel::Ensemble_ptr> &new_ensembles, const bool overwrite);

    datamodel::DeconQueue_ptr get_decon_queue(const datamodel::InputQueue &input_queue, const std::string &column_name) const;

    datamodel::DeconQueue_ptr get_decon_queue(const std::string &table_name, const std::string &column_name) const;

    boost::unordered_flat_map<std::pair<std::string, std::string>, datamodel::DeconQueue_ptr> get_decon_queues() const;

    void set_decon_queue(const datamodel::DeconQueue_ptr &p_decon_queue);

    void clear_data();

    bool get_is_active() const noexcept;

    void set_is_active(const bool is_active) noexcept;

    IQScalingFactor_ptr get_iq_scaling_factor(const std::string &input_queue_table_name, const std::string &input_queue_column_name) const;

    std::deque<IQScalingFactor_ptr> &get_iq_scaling_factors() noexcept;

    std::deque<IQScalingFactor_ptr> get_iq_scaling_factors(const InputQueue &input_queue) const;

    void set_iq_scaling_factors(const std::deque<datamodel::IQScalingFactor_ptr> &new_iq_scaling_factors, const bool overwrite);

    std::deque<datamodel::InputQueue_ptr> get_aux_input_queues() const;

    datamodel::InputQueue_ptr get_aux_input_queue(const uint16_t idx = 0) const;

    datamodel::InputQueue_ptr get_aux_input_queue(const std::string &table_name) const;

    std::deque<std::string> get_aux_input_table_names() const;

    std::string to_string() const override;

    uint32_t get_max_lag_count() const;

    uint32_t get_max_decrement() const;

    uint32_t get_max_quantise() const;

    uint32_t get_max_residuals_length() const;

    uint32_t get_max_possible_residuals_length() const;

    uint32_t get_residuals_length(const std::string &decon_queue_table_name = {}) const;

    static uint32_t get_residuals_length(const uint16_t levels);

    bpt::ptime get_last_modeled_time() const;

    business::calc_cache &get_calc_cache();

    void init_id() override;

    PROPERTY(bpt::ptime, last_self_request, bpt::min_date_time); // Last time the dataset created self-request for prediction
};

template<typename T>
std::basic_ostream<T> &operator<<(std::basic_ostream<T> &s, const Dataset &d)
{
    return s << d.to_string();
}

using Dataset_ptr = std::shared_ptr<Dataset>;

}
}

