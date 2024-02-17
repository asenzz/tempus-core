#pragma once

#include <set>
#include <model/Entity.hpp>
#include "SVRParameters.hpp"
#include "relations/ensemble_relation.hpp"
#include "SVRParametersService.hpp"


namespace svr {

class OnlineMIMOSVR;
typedef std::shared_ptr<svr::OnlineMIMOSVR> OnlineMIMOSVR_ptr;

namespace datamodel {


class Model : public Entity
{
    ensemble_relation ensemble;
    size_t decon_level; // This model's prediction level
    size_t multiout_;
    size_t gradient_ct;
    size_t chunk_size;
    std::deque<OnlineMIMOSVR_ptr> svr_models; // one model per gradient
    bpt::ptime last_modified {bpt::max_date_time}; // model modified (system) time
    bpt::ptime last_modeled_value_time {bpt::min_date_time}; // last input queue modeled value time
    t_param_set_ptr p_param_set; // parameters used by SVR models for every gradient, chunk and manifold of this level
public:
    Model() = default;

    Model(const bigint id, const bigint ensemble_id, const size_t decon_level, const size_t multiout_, const size_t gradient_ct, const size_t chunk_size,
          std::deque<OnlineMIMOSVR_ptr> svr_models = {},
          const bpt::ptime &last_modified = bpt::min_date_time, const bpt::ptime &last_modeled_value_time = bpt::min_date_time, t_param_set_ptr p_param_set = {});

    OnlineMIMOSVR_ptr &get_gradient(const size_t i = 0);
    OnlineMIMOSVR_ptr get_gradient(const size_t i = 0) const;
    std::deque<OnlineMIMOSVR_ptr> &get_gradients();
    std::deque<OnlineMIMOSVR_ptr> get_gradients() const;
    void set_gradient(const size_t i, const OnlineMIMOSVR_ptr &m);
    void set_gradients(const std::deque<OnlineMIMOSVR_ptr> &_svr_models);
    size_t get_gradient_count() const { return gradient_ct; }
    size_t get_chunk_size() const { return chunk_size; }
    size_t get_multiout() const { return multiout_; }
    datamodel::SVRParameters &get_params(const size_t chunk_ix = std::numeric_limits<size_t>::max(), const size_t grad_ix = std::numeric_limits<size_t>::max());
    datamodel::SVRParameters get_params(const size_t chunk_ix = std::numeric_limits<size_t>::max(), const size_t grad_ix = std::numeric_limits<size_t>::max()) const;
    datamodel::SVRParameters_ptr get_params_ptr(const size_t chunk_ix = std::numeric_limits<size_t>::max(), const size_t grad_ix = std::numeric_limits<size_t>::max()) const;
    datamodel::t_param_set_ptr get_param_set(const size_t chunk_ix = std::numeric_limits<size_t>::max(), const size_t grad_ix = std::numeric_limits<size_t>::max()) const;

    void set_params(const datamodel::SVRParameters_ptr &p) {
        datamodel::SVRParameters_ptr existing_p;
        if ((existing_p = business::SVRParametersService::find(*p_param_set, p->get_chunk_ix(), p->get_grad_level())))
            *existing_p = *p;
        else
            p_param_set->emplace(p);
    }

    t_param_set &get_param_set() { return *p_param_set; }
    void set_param_set(const t_param_set &ps) { *p_param_set = ps; }
    t_param_set_ptr &get_param_set_ptr() { return p_param_set; }
    void set_param_set_ptr(const t_param_set_ptr &pps) { p_param_set = pps; }

    bool operator==(const Model &o) const;

    void reset();

    bigint get_ensemble_id() const;

    void set_ensemble_id(const bigint ensemble_id);

    size_t get_decon_level() const;

    void set_decon_level(const size_t _decon_level);

    bpt::ptime const &get_last_modified() const;

    void set_last_modified(bpt::ptime const &_last_modified);

    bpt::ptime const &get_last_modeled_value_time() const;

    void set_last_modeled_value_time(bpt::ptime const &_last_modeled_value_time);

    virtual std::string to_string() const override;
};

using Model_ptr = std::shared_ptr<Model>;

datamodel::Model_ptr
find_model(const std::deque<datamodel::Model_ptr> &models, const size_t levix);

template<typename T> std::basic_ostream<T> &
operator<<(std::basic_ostream<T> &s, const datamodel::Model &m) { return s << m.to_string(); }

}
}
