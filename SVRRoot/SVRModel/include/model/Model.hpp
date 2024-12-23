#pragma once

#include <set>
#include "model/Entity.hpp"
#include "common/constants.hpp"
#include "relations/ensemble_relation.hpp"
#include "SVRParametersService.hpp"
#include "SVRParameters.hpp"


namespace svr {
namespace datamodel {


class Model : public Entity
{
    ensemble_relation ensemble;
    unsigned decon_level = C_default_svrparam_decon_level; // This model's prediction level
    unsigned step = C_default_svrparam_step;
    unsigned multiout = common::C_default_multiout;
    unsigned gradient_ct = common::C_default_gradient_count;
    unsigned max_chunk_size = common::C_default_gradient_count;
    std::deque<OnlineMIMOSVR_ptr> svr_models; // one model per gradient
    bpt::ptime last_modified = bpt::not_a_date_time; // model modified (system) time
    bpt::ptime last_modeled_value_time = bpt::min_date_time; // last input queue modeled value time

    virtual void init_id() override;
public:
    Model() = default;

    Model(const bigint id, const bigint ensemble_id, const unsigned decon_level, const unsigned step, const unsigned multiout_, const unsigned gradient_ct, const unsigned chunk_size,
          std::deque<OnlineMIMOSVR_ptr> svr_models = {}, const bpt::ptime &last_modified = bpt::min_date_time, const bpt::ptime &last_modeled_value_time = bpt::min_date_time);

    OnlineMIMOSVR_ptr get_gradient(const unsigned i = 0) const;

    std::deque<OnlineMIMOSVR_ptr> &get_gradients();

    std::deque<OnlineMIMOSVR_ptr> get_gradients() const;

    void set_gradient(const OnlineMIMOSVR_ptr &m);

    void set_gradients(const std::deque<OnlineMIMOSVR_ptr> &_svr_models, const bool overwrite);

    unsigned get_gradient_count() const;

    unsigned get_max_chunk_size() const;

    void set_max_chunk_size(const unsigned chunk_size);

    unsigned get_multiout() const;

    datamodel::SVRParameters_ptr get_head_params() const;

    bool operator==(const Model &o) const;

    void reset();

    bigint get_ensemble_id() const;

    void set_ensemble_id(const bigint ensemble_id);

    unsigned get_decon_level() const;

    void set_decon_level(const unsigned _decon_level);

    unsigned get_step() const;

    void set_step(const unsigned step);

    bpt::ptime const &get_last_modified() const;

    void set_last_modified(bpt::ptime const &_last_modified);

    bpt::ptime const &get_last_modeled_value_time() const;

    void set_last_modeled_value_time(bpt::ptime const &_last_modeled_value_time);

    virtual std::string to_string() const override;
};

using Model_ptr = std::shared_ptr<Model>;

datamodel::Model_ptr
find_model(const std::deque<datamodel::Model_ptr> &models, const unsigned levix);

template<typename T> std::basic_ostream<T> &
operator<<(std::basic_ostream<T> &s, const datamodel::Model &m)
{ return s << m.to_string(); }

}
}
