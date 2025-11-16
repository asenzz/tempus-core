#pragma once

#include "common/constants.hpp"
#include "relations/ensemble_relation.hpp"
#include "Entity.hpp"
#include "SVRParameters.hpp"
#include "DataRow.hpp"

namespace svr {
namespace datamodel {

class DataRow;
using DataRow_ptr = std::shared_ptr<DataRow>;

using data_row_container = std::deque<DataRow_ptr>;

using t_model_train_data = std::tuple<mat_ptr, mat_ptr, vec_ptr, mat_ptr, datamodel::data_row_container_ptr>;

class Model : public Entity
{
    ensemble_relation ensemble;
    uint16_t decon_level = C_default_svrparam_decon_level; // This model's prediction level
    uint16_t step = C_default_svrparam_step;
    uint16_t outputs = common::C_default_outputs;
    uint16_t gradient_ct = common::C_default_gradient_count;
    uint32_t max_chunk_size = common::AppConfig::C_default_kernel_length;
    std::deque<OnlineSVR_ptr> svr_models; // one model per gradient
    bpt::ptime last_modified = bpt::not_a_date_time; // model modified (system) time
    bpt::ptime last_modeled_value_time = bpt::min_date_time; // last input queue modeled value time
    std::pair<SVRParameters_ptr, SVRParameters_ptr> parameters;
    PROPERTY(arma::mat, features)
    PROPERTY(arma::mat, labels)
    PROPERTY(arma::vec, last_knowns)
    PROPERTY(data_row_container, times)

    void init_id() override;

public:
    Model() = default;

    Model(bigint id, bigint ensemble_id, uint16_t decon_level, uint16_t step, uint16_t outputs_, uint16_t gradient_ct, uint32_t chunk_size,
          std::deque<OnlineSVR_ptr> svr_models = {}, const bpt::ptime &last_modified = bpt::min_date_time, const bpt::ptime &last_modeled_value_time = bpt::min_date_time);

    OnlineSVR_ptr get_gradient(uint16_t i = 0) const;

    std::deque<OnlineSVR_ptr> &get_gradients();

    std::deque<OnlineSVR_ptr> get_gradients() const;

    void set_gradient(const OnlineSVR_ptr &m);

    void set_gradients(const std::deque<OnlineSVR_ptr> &_svr_models, const bool overwrite);

    uint16_t get_gradient_count() const;

    uint32_t get_max_chunk_size() const;

    void set_max_chunk_size(uint32_t chunk_size);

    uint16_t get_outputs() const;

    std::pair<SVRParameters_ptr, SVRParameters_ptr> &get_head_params();

    const std::pair<SVRParameters_ptr, SVRParameters_ptr> &get_head_params() const;

    void set_head_params(const std::pair<SVRParameters_ptr, SVRParameters_ptr> &params);

    void adjust_gradient_decrement(uint32_t distance) const;

    bool operator==(const Model &o) const;

    void reset();

    bigint get_ensemble_id() const;

    void set_ensemble_id(bigint ensemble_id);

    uint16_t get_decon_level() const;

    void set_decon_level(uint16_t _decon_level);

    uint16_t get_step() const;

    void set_step(uint16_t step);

    bpt::ptime const &get_last_modified() const;

    void set_last_modified(bpt::ptime const &_last_modified);

    bpt::ptime const &get_last_modeled_value_time() const;

    void set_last_modeled_value_time(bpt::ptime const &_last_modeled_value_time);

    virtual std::string to_string() const override;

    static constexpr uint16_t C_paramid_left = 0xbeef;
    static constexpr uint16_t C_paramid_right = 0xdead;
};

using Model_ptr = std::shared_ptr<Model>;

datamodel::Model_ptr find_model(const std::deque<datamodel::Model_ptr> &models, uint16_t levix);

template<typename T> std::basic_ostream<T> &operator<<(std::basic_ostream<T> &s, const datamodel::Model &m)
{
    return s << m.to_string();
}

}
}
