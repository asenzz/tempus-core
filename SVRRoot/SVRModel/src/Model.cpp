#include <utility>
#include <atomic>
#include "appcontext.hpp"
#include "model/Model.hpp"
#include "model/Ensemble.hpp"
#include "onlinesvr.hpp"

namespace svr {
namespace datamodel {

bool Model::operator==(const Model &o) const
{
    return ensemble.get_id() == o.ensemble.get_id() && decon_level == o.decon_level
           && last_modified == o.last_modified && last_modeled_value_time == o.last_modeled_value_time && svr_models == o.svr_models;
}

Model::Model(const bigint id, const bigint ensemble_id, const unsigned decon_level, const unsigned step,
             const unsigned multiout_, const unsigned gradient_ct, const unsigned chunk_size,
             std::deque<OnlineMIMOSVR_ptr> svr_model, const bpt::ptime &last_modified,
             const bpt::ptime &last_modeled_value_time)
        : Entity(id),
          ensemble(ensemble_id),
          decon_level(decon_level),
          step(step),
          multiout(multiout_),
          gradient_ct(gradient_ct),
          max_chunk_size(chunk_size),
          svr_models(std::move(svr_model)),
          last_modified(last_modified),
          last_modeled_value_time(last_modeled_value_time)
{
#ifdef ENTITY_INIT_ID
    init_id();
#endif
}

void Model::init_id()
{
    if (!id) {
        boost::hash_combine(id, ensemble.get_id());
        boost::hash_combine(id, decon_level);
        boost::hash_combine(id, step);
    }
}

unsigned Model::get_gradient_count() const
{ return gradient_ct; }

void Model::set_max_chunk_size(const unsigned chunk_size)
{
    max_chunk_size = chunk_size;
}

unsigned Model::get_max_chunk_size() const
{ return max_chunk_size; }

unsigned Model::get_multiout() const
{ return multiout; }

void Model::reset()
{
    last_modeled_value_time = bpt::min_date_time;
    last_modified = bpt::min_date_time;
    svr_models.clear();
    ensemble = ensemble_relation();
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
    ensemble.set_id(ensemble_id);
}

/** Get the wavelet deconstruction level this model is predicting. */
unsigned Model::get_decon_level() const
{
    return decon_level;
}

/** Set the decon level this model is predicting */
void Model::set_decon_level(const unsigned _decon_level)
{
    decon_level = _decon_level;
}

unsigned Model::get_step() const
{
    return step;
}

/** Set the decon level this model is predicting */
void Model::set_step(const unsigned _step)
{
    step = _step;
}

/** Get pointer to an OnlineSVR model instance */
OnlineMIMOSVR_ptr Model::get_gradient(const unsigned i) const
{
    const auto svr_model_iter = std::find_if(C_default_exec_policy, svr_models.begin(), svr_models.end(), [&](const auto &p_svr_model) {
        return p_svr_model->get_gradient_level() == i;
    });
    if (svr_model_iter != svr_models.end()) return *svr_model_iter;
    LOG4_WARN("Gradient " << i << " not found!");
    return nullptr;
}

datamodel::SVRParameters_ptr Model::get_head_params() const
{
    return (svr_models.empty() || (**svr_models.cbegin()).get_param_set().empty()) ? nullptr : *(**svr_models.cbegin()).get_param_set().cbegin();
}

std::deque<OnlineMIMOSVR_ptr> &Model::get_gradients()
{
    return svr_models;
}

std::deque<OnlineMIMOSVR_ptr> Model::get_gradients() const
{
    return svr_models;
}


void Model::set_gradient(const OnlineMIMOSVR_ptr &m)
{
    std::atomic<bool> found{false};
#pragma omp parallel for num_threads(adj_threads(svr_models.size()))
    for (unsigned g = 0; g < svr_models.size(); ++g) {
        if (m->get_gradient_level() == svr_models[g]->get_gradient_level()) {
            svr_models[g] = m;
            svr_models[g]->set_model_id(id);
            found.store(true, std::memory_order_relaxed);
        }
    }
    if (found.load()) return;
    svr_models.emplace_back(m);
    svr_models.back()->set_model_id(id);
}

/** Set member svr model point to a OnlineSVR instance
 * \param new_svr_models new OnlineSVR instance
 */
void Model::set_gradients(const std::deque<OnlineMIMOSVR_ptr> &new_svr_models, const bool overwrite)
{
    const unsigned prev_size = svr_models.size();
    for (const auto &new_m: new_svr_models) {
        std::atomic<bool> found = false;
#pragma omp parallel for num_threads(adj_threads(prev_size))
        for (unsigned i = 0; i < prev_size; ++i)
            if (new_m->get_gradient_level() == svr_models[i]->get_gradient_level() && new_m->get_decon_level() == svr_models[i]->get_decon_level()) {
                if (overwrite) {
                    svr_models[i] = new_m;
                    svr_models[i]->set_model_id(id);
                }
                found.store(true, std::memory_order_relaxed);
            }
        if (found) continue;
        svr_models.emplace_back(new_m);
        svr_models.back()->set_model_id(id);
    }
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
    last_modified = _last_modified;
}

/** Get value time of the latest row this model was trained against. */
bpt::ptime const &Model::get_last_modeled_value_time() const
{
    return last_modeled_value_time;
}

/** Set value time of the latest row this model was trained against. */
void Model::set_last_modeled_value_time(const bpt::ptime &_last_modeled_value_time)
{
    last_modeled_value_time = _last_modeled_value_time;
}

std::string Model::to_string() const
{
    std::stringstream s;
    s << "Model ID " << id
      << ", ensemble ID " << ensemble.get_id()
      << ", decon level " << decon_level
      << ", gradients " << gradient_ct
      << ", outputs " << multiout
      << ", chunk size " << max_chunk_size
      << ", last modified time " << last_modified
      << ", last modeled value time " << last_modeled_value_time;
    return s.str();
}


}
}

