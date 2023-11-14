#pragma once

#include <algorithm>
#include <unordered_map>
#include <mutex>

#include "model/Entity.hpp"
#include "model/Priority.hpp"
#include "model/Ensemble.hpp"
#include "model/InputQueue.hpp"
#include "model/IQScalingFactor.hpp"
#include "model/DQScalingFactor.hpp"
#include "util/string_utils.hpp"
#include "relations/iq_relation.hpp"
#include "spectral_transform.hpp"
#include "fast_cvmd.hpp"
#include "online_emd.hpp"

namespace svr {
namespace business {
    class DatasetService;
}

namespace datamodel {

static const std::vector<std::string> transformation_names {
        // 0 - 14
        "db1", "db2", "db3", "db4", "db5", "db6", "db7", "db8", "db9", "db10", "db11", "db12", "db13", "db14", "db15",
        // 15 - 29
        "bior1.1", "bior1.3", "bior1.5", "bior2.2", "bior2.4", "bior2.6", "bior2.8",
        "bior3.1", "bior3.3", "bior3.5", "bior3.7", "bior3.9", "bior4.4", "bior5.5", "bior6.8",
        // 30 - 34
        "coif1", "coif2", "coif3", "coif4", "coif5",
        // 35 - 44
        "sym1", "sym2", "sym3", "sym4", "sym5", "sym6", "sym7", "sym8", "sym9", "sym10",
        // 45 - 55
        "sym11", "sym12", "sym13", "sym14", "sym15", "sym16", "sym17", "sym18", "sym19", "sym20",
        "stft", "oemd", "cvmd"
};

class Dataset : public Entity
{
    friend svr::business::DatasetService;

    void init_transform();

    size_t max_lag_count_cache_ = 0;
    size_t max_decremental_distance_cache_ = 0;
    std::string dataset_name_;
    std::string user_name_;
    iq_relation input_queue_;
    std::vector<iq_relation> aux_input_queues_;
    Priority priority_;
    std::string description_;
    size_t transformation_levels_;
    std::string transformation_name_;
    bpt::time_duration max_lookback_time_gap_;
    std::vector<Ensemble_ptr> ensembles_;
    bool is_active_;
    std::vector<IQScalingFactor_ptr> iq_scaling_factors_;
    svr::datamodel::dq_scaling_factor_container_t dq_scaling_factors_;
    ensemble_svr_parameters_t ensemble_svr_parameters_;

    std::mutex dq_scaling_factors_mutex;
    std::mutex mutable svr_params_mutex;

    void on_set_id() override;
public:
    Dataset() : Entity()
    {
        is_active_ = false;
        transformation_levels_ = 0;
    }

    Dataset(
        bigint id,
        const std::string &dataset_name,
        const std::string &user_name,
        InputQueue_ptr p_input_queue,
        const std::vector<InputQueue_ptr> &aux_input_queues,
        const Priority &priority,
        const std::string &description,
        const size_t transformation_levels,
        const std::string &transformation_name,
        const bpt::time_duration &max_lookback_time_gap = DEFAULT_FEATURES_MAX_TIME_GAP,
        const std::vector<Ensemble_ptr> &ensembles = {},
        bool is_active = false,
        const std::vector<IQScalingFactor_ptr> iq_scaling_factors = {},
        const dq_scaling_factor_container_t dq_scaling_factors = {}
    );

    Dataset(
            bigint id,
            const std::string &dataset_name_,
            const std::string &user_name_,
            const std::string &input_queue_table_name,
            const std::vector<std::string> &aux_input_queues_table_names,
            const Priority &priority_,
            const std::string &description_,
            const size_t transformation_levels,
            const std::string &transformation_name,
            const bpt::time_duration &max_lookback_time_gap_ = DEFAULT_FEATURES_MAX_TIME_GAP,
            const std::vector<Ensemble_ptr> &ensembles_ = {},
            bool is_active_ = false,
            const std::vector<IQScalingFactor_ptr> iq_scaling_factors = {},
            const dq_scaling_factor_container_t dq_scaling_factors = {}
    );

    std::unique_ptr<svr::online_emd> p_oemd_transformer_fat;
    std::unique_ptr<svr::fast_cvmd> p_cvmd_transformer;

    Dataset(Dataset const &dataset);

    bool operator==(const Dataset &other) const;

    std::string get_dataset_name() const;
    void set_dataset_name(const std::string &dataset_name);

    std::string get_user_name() const;
    void set_user_name(const std::string &user_name);

    InputQueue_ptr get_input_queue() ;
    void set_input_queue(const InputQueue_ptr &p_input_queue) ;

    Priority const &get_priority() const;
    void set_priority(Priority const &priority);

    std::string get_description() const;
    void set_description(const std::string &description);

    size_t get_transformation_levels() const;
    size_t get_transformation_levels_cvmd() const;
    size_t get_transformation_levels_oemd() const;
    void set_transformation_levels(const size_t transformation_levels);

    std::string get_transformation_name() const;
    void set_transformation_name(const std::string& transformation_name);
    bool validate_transformation_name(const std::string& transformation_name) const;

    const bpt::time_duration& get_max_lookback_time_gap() const;
    void set_max_lookback_time_gap(const bpt::time_duration& max_lookback_time_gap);

    std::vector<Ensemble_ptr>& get_ensembles();
    Ensemble_ptr &get_ensemble(const std::string &column_name);
    Ensemble_ptr &get_ensemble(const std::string &table_name, const std::string &column_name);
    Ensemble_ptr &get_ensemble(const size_t idx = 0);
    void set_ensembles(const std::vector<Ensemble_ptr> &ensembles);

    DeconQueue_ptr &get_decon_queue(const InputQueue_ptr &p_input_queue, const std::string &column_name);
    std::map<std::pair<std::string, std::string>, DeconQueue_ptr> get_decon_queues() const;
    void set_decon_queue(const DeconQueue_ptr &p_decon_queue);
    void clear_data();

    bool get_is_active() const;
    void set_is_active(const bool is_active);

    std::vector<IQScalingFactor_ptr> get_iq_scaling_factors();
    void set_iq_scaling_factors(const std::vector<IQScalingFactor_ptr>& iq_scaling_factors);

    dq_scaling_factor_container_t get_dq_scaling_factors() ;
    void set_dq_scaling_factors(const dq_scaling_factor_container_t& dq_scaling_factors);
    void add_dq_scaling_factors(const dq_scaling_factor_container_t& new_dq_scaling_factors);
    double get_dq_scaling_factor_features(const std::string &input_queue_table_name, const std::string &input_queue_column_name, const size_t decon_level);
    double get_dq_scaling_factor_labels(const std::string &input_queue_table_name, const std::string &input_queue_column_name, const size_t decon_level);

    ensemble_svr_parameters_t get_ensemble_svr_parameters() const;
    SVRParameters_ptr get_svr_parameters(
            const std::string &table_name, const std::string &column_name, const size_t level_number) const;
    void set_ensemble_svr_parameters(const ensemble_svr_parameters_t &ensemble_svr_parameters);
    void set_ensemble_svr_parameters_deep(const ensemble_svr_parameters_t &ensemble_svr_parameters);

    std::vector<InputQueue_ptr> get_aux_input_queues() const;
    InputQueue_ptr get_aux_input_queue(const size_t idx) const;
    std::vector<std::string> get_aux_input_table_names() const;

    virtual std::string to_string() const override;

    std::string parameters_to_string() const;

    size_t get_max_lag_count();
    size_t get_max_decrement();
    size_t get_max_residuals_count() const;
    size_t get_maxpos_residuals_count() const;
    size_t get_residuals_count(const std::string &decon_queue_table_name = {}) const;
};


}
}

using Dataset_ptr = std::shared_ptr<svr::datamodel::Dataset>;
