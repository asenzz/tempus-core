#include <model/Model.hpp>
#include <model/Ensemble.hpp>
#include "onlinesvr.hpp"

#include <appcontext.hpp>
#include <utility>

namespace svr {
namespace datamodel {

bool Model::operator==(const Model &o) const
{
    return ensemble.get_id() == o.ensemble.get_id() && decon_level == o.decon_level
           && learning_levels == o.learning_levels && last_modified == o.last_modified
           && last_modeled_value_time == o.last_modeled_value_time && svr_models == o.svr_models;
}

Model::Model(const bigint id, const bigint ensemble_id, const size_t decon_level,
                const size_t multiout_, const size_t gradient_ct, const size_t chunk_size,
                const std::set<size_t> &learning_levels,
             std::deque<OnlineMIMOSVR_ptr> svr_model, const bpt::ptime &last_modified,
             const bpt::ptime &last_modeled_value_time, t_param_set_ptr p_param_set
)
        : Entity(id),
          ensemble(ensemble_id),
          decon_level(decon_level),
          multiout_(multiout_),
          gradient_ct(gradient_ct),
          chunk_size(chunk_size),
          learning_levels(learning_levels),
          svr_models(std::move(svr_model)),
          last_modified(last_modified),
          last_modeled_value_time(last_modeled_value_time),
          p_param_set(std::move(p_param_set))
{
}

void Model::reset()
{
    last_modeled_value_time = bpt::min_date_time;
    last_modified = bpt::min_date_time;
    svr_models.clear();
    ensemble = ensemble_relation();
}

datamodel::SVRParameters &Model::get_params(const size_t chunk_ix, const size_t grad_ix)
{
    return *business::SVRParametersService::find(*p_param_set, chunk_ix, grad_ix);
}

datamodel::SVRParameters Model::get_params(const size_t chunk_ix, const size_t grad_ix) const
{
    return *business::SVRParametersService::find(*p_param_set, chunk_ix, grad_ix);
}

datamodel::SVRParameters_ptr Model::get_params_ptr(const size_t chunk_ix, const size_t grad_ix)
{
    return business::SVRParametersService::find(*p_param_set, chunk_ix, grad_ix);
}

datamodel::SVRParameters_ptr Model::get_params_ptr(const size_t chunk_ix, const size_t grad_ix) const
{
    return business::SVRParametersService::find(*p_param_set, chunk_ix, grad_ix);
}

/** Get ensemble database ID this model is part of */
bigint Model::get_ensemble_id() const
{
    return ensemble.get_id();
}

/** Change the ensemble this model is part of
 * \param ensemble_id Database ensemble ID this model is to become part of.
 */
void Model::set_ensemble_id(const bigint ensemble_id)
{
    this->ensemble.set_id(ensemble_id);
}

/** Get the wavelet deconstruction level this model is predicting. */
size_t Model::get_decon_level() const
{
    return decon_level;
}

/** Set the decon level this model is predicting */
void Model::set_decon_level(const size_t _decon_level)
{
    this->decon_level = _decon_level;
}

/** Get training wavelet deconstruction level indexes this model is trained against. */
std::set<size_t> Model::get_learning_levels() const
{
    return learning_levels;
}

/** Set learning wavelet deconstruction level indexes this model is trained against. */
void Model::set_learning_levels(const std::set<size_t> &_learning_levels)
{
    this->learning_levels = _learning_levels;
}

/** Get pointer to an OnlineSVR model instance */
OnlineMIMOSVR_ptr &Model::get_gradient(const size_t i)
{
    return svr_models[i];
}

OnlineMIMOSVR_ptr Model::get_gradient(const size_t i) const
{
    return svr_models[i];
}

std::deque<OnlineMIMOSVR_ptr> &Model::get_gradients()
{
    return svr_models;
}

std::deque<OnlineMIMOSVR_ptr> Model::get_gradients() const
{
    return svr_models;
}

void Model::set_gradient(const size_t i, const OnlineMIMOSVR_ptr &m)
{
    if (svr_models.size() < i + 1) svr_models.resize(i + 1);
    svr_models[i] = m;
}

/** Set member svr model point to a OnlineSVR instance
 * \param _svr_models new OnlineSVR instance
 */
void Model::set_gradients(const std::deque<OnlineMIMOSVR_ptr> &_svr_models)
{
    svr_models = _svr_models;
}

/** Get last time model was updated */
bpt::ptime const &Model::get_last_modified() const
{
    return last_modified;
}

/** Set time model was last updated
 * \param _last_modified time model was last modified
 */
void Model::set_last_modified(bpt::ptime const &_last_modified)
{
    this->last_modified = _last_modified;
}

/** Get value time of the latest row this model was trained against. */
bpt::ptime const &Model::get_last_modeled_value_time() const
{
    return last_modeled_value_time;
}

/** Set value time of the latest row this model was trained against. */
void Model::set_last_modeled_value_time(const bpt::ptime &_last_modeled_value_time)
{
    this->last_modeled_value_time = _last_modeled_value_time;
}

std::string Model::to_string() const
{
    std::stringstream s;
    s << "Model ID " << get_id()
       << ", ensemble ID " << get_ensemble_id()
       << ", decon level " << get_decon_level()
       << ", learning levels " << common::to_string(learning_levels)
       << ", last modified time " << bpt::to_simple_string(get_last_modified())
       << ", last modeled value time " << bpt::to_simple_string(get_last_modeled_value_time());
    return s.str();
}


}
}
