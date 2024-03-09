#include "model/Dataset.hpp"
#include "onlinesvr.hpp"

#include "appcontext.hpp"

#include "SVRParametersService.hpp"

#include "online_emd.hpp"
#include "short_term_fourier_transform.hpp"
#include "spectral_transform.hpp"
#include "fast_cvmd.hpp"
#include "DQScalingFactorService.hpp"
#include "calc_cache.hpp"
#include "common/compatibility.hpp"

namespace svr {
namespace datamodel {

void Dataset::init_transform()
{
    if (transformation_levels_ < 8) return;
    p_oemd_transformer_fat = std::make_unique<svr::oemd::online_emd>(transformation_levels_ / 4, OEMD_STRETCH_COEF);
    p_cvmd_transformer = std::make_unique<svr::vmd::fast_cvmd>(transformation_levels_ / 2);
}

Dataset::Dataset() :
        Entity(),
        ccache(*this),
        gradients_(common::C_default_gradient_count),
        max_chunk_size_(common::C_kernel_default_max_chunk_size),
        multiout_(common::C_default_multistep_len),
        transformation_levels_(common::C_default_level_count),
        is_active_(false)
{
    init_transform();
#ifdef ENTITY_INIT_ID
    init_id();
#endif
}

Dataset::Dataset(
        bigint id,
        const std::string &dataset_name,
        const std::string &user_name,
        datamodel::InputQueue_ptr p_input_queue,
        const std::deque<datamodel::InputQueue_ptr> &aux_input_queues,
        const Priority &priority,
        const std::string &description,
        const size_t gradients,
        const size_t chunk_size,
        const size_t multiout,
        const size_t transformation_levels,
        const std::string &transformation_name,
        const bpt::time_duration &max_lookback_time_gap,
        const std::deque<datamodel::Ensemble_ptr> &ensembles,
        bool is_active,
        const std::deque<datamodel::IQScalingFactor_ptr> iq_scaling_factors,
        const dq_scaling_factor_container_t dq_scaling_factors
)
        : Entity(id),
          ccache(*this),
          dataset_name_(dataset_name),
          user_name_(user_name),
          priority_(priority),
          description_(description),
          gradients_(gradients),
          max_chunk_size_(chunk_size),
          multiout_(multiout),
          transformation_levels_(transformation_levels),
          transformation_name_(transformation_name),
          max_lookback_time_gap_(max_lookback_time_gap),
          ensembles_(svr::common::clone_shared_ptr_elements(ensembles)),
          is_active_(is_active),
          iq_scaling_factors_(svr::common::clone_shared_ptr_elements(iq_scaling_factors)),
          dq_scaling_factors_(svr::common::clone_shared_ptr_elements(dq_scaling_factors))
{
    if (!p_input_queue) THROW_EX_FS(std::logic_error, "Input queue cannot be null.");

    input_queue_.set_obj(p_input_queue);

    for (const auto &p_aux_input_queue: aux_input_queues)
        aux_input_queues_.emplace_back(p_aux_input_queue);

    init_transform();
#ifdef ENTITY_INIT_ID
    init_id();
#endif
}

void Dataset::init_id()
{
    if (!id) {
        if (input_queue_.get_obj() && input_queue_.get_obj()->get_table_name().size()) boost::hash_combine(id, input_queue_.get_obj()->get_table_name());
        else if (dataset_name_.size()) boost::hash_combine(id, dataset_name_);
        else if (description_.size()) boost::hash_combine(id, description_);
        else if (user_name_.size()) boost::hash_combine(id, user_name_);
        else {
            LOG4_WARN("Dataset id is nil!");
            return;
        }
        on_set_id();
    }
}

Dataset::Dataset(
        bigint id,
        const std::string &dataset_name,
        const std::string &user_name,
        const std::string &input_queue_table_name,
        const std::deque<std::string> &aux_input_queue_table_names,
        const Priority &priority,
        const std::string &description, /* This is description */
        const size_t gradients,
        const size_t chunk_size,
        const size_t multiout,
        const size_t transformation_levels,
        const std::string &transformation_name,
        const bpt::time_duration &max_lookback_time_gap,
        const std::deque<datamodel::Ensemble_ptr> &ensembles,
        bool is_active,
        const std::deque<datamodel::IQScalingFactor_ptr> iq_scaling_factors,
        const dq_scaling_factor_container_t dq_scaling_factors
)
        : Entity(id),
          ccache(*this),
          dataset_name_(dataset_name),
          user_name_(user_name),
          priority_(priority),
          description_(description),
          gradients_(gradients),
          max_chunk_size_(chunk_size),
          multiout_(multiout),
          transformation_levels_(transformation_levels),
          transformation_name_(transformation_name),
          max_lookback_time_gap_(max_lookback_time_gap),
          ensembles_(ensembles),
          is_active_(is_active),
          iq_scaling_factors_(iq_scaling_factors),
          dq_scaling_factors_(dq_scaling_factors)
{
    if (input_queue_table_name.empty()) THROW_EX_FS(std::logic_error, "Input queue table name cannot be empty");

    input_queue_.set_id(input_queue_table_name);

    for (const auto &aux_input_queue_table_name: aux_input_queue_table_names)
        aux_input_queues_.emplace_back(aux_input_queue_table_name);

    init_transform();
#ifdef ENTITY_INIT_ID
    init_id();
#endif
}

Dataset::Dataset(Dataset const &dataset) :
        Dataset(dataset.get_id(),
                dataset.dataset_name_,
                dataset.user_name_,
                dataset.input_queue_.get_obj(),
                dataset.get_aux_input_queues(),
                dataset.priority_,
                dataset.description_,
                dataset.gradients_,
                dataset.max_chunk_size_,
                dataset.multiout_,
                dataset.transformation_levels_,
                dataset.transformation_name_,
                dataset.max_lookback_time_gap_,
                dataset.ensembles_,
                dataset.is_active_,
                dataset.iq_scaling_factors_,
                dataset.dq_scaling_factors_)
{
    p_cvmd_transformer = std::make_unique<vmd::fast_cvmd>(*dataset.p_cvmd_transformer);
    p_oemd_transformer_fat = std::make_unique<oemd::online_emd>(*dataset.p_oemd_transformer_fat);
#ifdef ENTITY_INIT_ID
    init_id();
#endif
}


bool Dataset::operator==(const Dataset &o) const
{
    return (*this ^= o)
           && ensembles_.size() == o.ensembles_.size()
           && std::equal(ensembles_.begin(), ensembles_.end(), o.ensembles_.begin());
}

bool Dataset::operator^=(const Dataset &o) const
{
    bool res = id == o.id
               && dataset_name_ == o.dataset_name_
               && user_name_ == o.user_name_
               && input_queue_.get_id() == o.input_queue_.get_id()
               && priority_ == o.priority_
               && gradients_ == o.gradients_
               && max_chunk_size_ == o.max_chunk_size_
               && multiout_ == o.multiout_
               && transformation_levels_ == o.transformation_levels_
               && transformation_name_ == o.transformation_name_
               && max_lookback_time_gap_ == o.max_lookback_time_gap_
               && is_active_ == o.is_active_;


    for (const auto &iq: aux_input_queues_) {
        bool found = false;
        for (const auto &oiq: o.aux_input_queues_) {
            if (iq.get_id() == oiq.get_id()) {
                found = true;
                break;
            }
        }
        if (!found) res = false;
    }
    return res;
}

void Dataset::on_set_id()
{
    for (datamodel::Ensemble_ptr &p_ensemble: ensembles_)
        p_ensemble->set_dataset_id(get_id());
}

business::calc_cache &Dataset::get_calc_cache()
{
    return ccache;
}

vmd::fast_cvmd &Dataset::get_cvmd_transformer()
{ return *p_cvmd_transformer; }

oemd::online_emd &Dataset::get_oemd_transformer()
{ return *p_oemd_transformer_fat; }

bool Dataset::get_initialized()
{ return initialized; }

size_t Dataset::get_gradient_count() const
{ return gradients_; };

size_t Dataset::get_max_chunk_size() const
{ return max_chunk_size_; };

size_t Dataset::get_multiout() const
{ return multiout_; };

void Dataset::set_gradients(const size_t grads)
{ gradients_ = grads; }

void Dataset::set_chunk_size(const size_t chunk_size)
{ max_chunk_size_ = chunk_size; }

void Dataset::set_multiout(const size_t multiout)
{ multiout_ = multiout; }

std::string Dataset::get_dataset_name() const
{ return dataset_name_; }

void Dataset::set_dataset_name(const std::string &dataset_name)
{ Dataset::dataset_name_ = dataset_name; }

std::string Dataset::get_user_name() const
{ return user_name_; }

void Dataset::set_user_name(const std::string &user_name)
{ user_name_ = user_name; }

// Main input queue of which all columns are being predicted
datamodel::InputQueue_ptr Dataset::get_input_queue() const
{ return input_queue_.get_obj(); }

void Dataset::set_input_queue(const datamodel::InputQueue_ptr &p_input_queue)
{ input_queue_.set_obj(p_input_queue); }

/* Aux input queues columns should be all unique among each other and at least one identical to the main input queue column that is being predicted, the rest are being
 * used as features.
 */
std::deque<datamodel::InputQueue_ptr> Dataset::get_aux_input_queues() const
{
    std::deque<datamodel::InputQueue_ptr> result;
    for (const auto &relation_aux_input_queue: aux_input_queues_)
        result.emplace_back(relation_aux_input_queue.get_obj());
    return result;
}


datamodel::InputQueue_ptr Dataset::get_aux_input_queue(const size_t idx) const
{
    return aux_input_queues_[idx].get_obj();
}


std::deque<std::string> Dataset::get_aux_input_table_names() const
{
    std::deque<std::string> res;
    std::transform(aux_input_queues_.begin(), aux_input_queues_.end(), std::back_inserter(res),
                   [](const iq_relation &inque_rel) { return inque_rel.get_obj()->get_table_name(); });
    return res;
}

Priority const &Dataset::get_priority() const
{
    return priority_;
}

void Dataset::set_priority(Priority const &priority)
{
    priority_ = priority;
}

std::string Dataset::get_description() const
{
    return description_;
}

void Dataset::set_description(const std::string &description)
{
    description_ = description;
}

size_t Dataset::get_transformation_levels() const
{
    return transformation_levels_ >= MIN_LEVEL_COUNT ? transformation_levels_ : 1;
}

size_t Dataset::get_half_levct() const
{
    return transformation_levels_ >= MIN_LEVEL_COUNT ? transformation_levels_ / 2 : std::numeric_limits<size_t>::max();
}

size_t Dataset::get_model_count() const
{
    return business::ModelService::to_model_ct(transformation_levels_);
}

size_t Dataset::get_transformation_levels_cvmd() const
{
    return transformation_levels_ >= MIN_LEVEL_COUNT ? std::max<ssize_t>(1, transformation_levels_ / 2) : 0;
}

size_t Dataset::get_transformation_levels_oemd() const
{
    return transformation_levels_ >= MIN_LEVEL_COUNT ? std::max<ssize_t>(1, transformation_levels_ / 4) : 0;
}

void Dataset::set_transformation_levels(const size_t transformation_levels)
{ transformation_levels_ = transformation_levels; }

std::string Dataset::get_transformation_name() const
{ return transformation_name_; }

void Dataset::set_transformation_name(const std::string &transformation_name)
{
    assert(validate_transformation_name(transformation_name));
    transformation_name_ = transformation_name;
}

bool Dataset::validate_transformation_name(const std::string &transformation_name) const
{
    return std::find(transformation_names.begin(), transformation_names.end(), transformation_name) != transformation_names.end();
}

const bpt::time_duration &Dataset::get_max_lookback_time_gap() const
{ return max_lookback_time_gap_; }


void Dataset::set_max_lookback_time_gap(const bpt::time_duration &max_lookback_time_gap)
{ max_lookback_time_gap_ = max_lookback_time_gap; }


std::deque<datamodel::Ensemble_ptr> &Dataset::get_ensembles()
{ return ensembles_; }


datamodel::Ensemble_ptr Dataset::get_ensemble(const std::string &column_name)
{
    return get_ensemble(input_queue_.get_obj()->get_table_name(), column_name);
}


datamodel::Ensemble_ptr Dataset::get_ensemble(const std::string &table_name, const std::string &column_name)
{
    const auto res = std::find_if(std::execution::par_unseq, ensembles_.begin(), ensembles_.end(), [&](const auto &p_ensemble) {
        return p_ensemble->get_decon_queue()->get_input_queue_column_name() == column_name &&
               p_ensemble->get_decon_queue()->get_input_queue_table_name() == table_name;
    });
    if (res != ensembles_.end()) return *res;
    LOG4_ERROR("Ensemble for table " << table_name << ", column " << column_name << " not found!");
    return nullptr;
}

void Dataset::set_decon_queue(const datamodel::DeconQueue_ptr &p_decon_queue)
{
    datamodel::Ensemble_ptr p_ensemble = get_ensemble(p_decon_queue->get_input_queue_table_name(), p_decon_queue->get_input_queue_column_name());
    if (!p_ensemble) LOG4_THROW("Ensemble not found!");
    if (!p_ensemble->get_decon_queue() || (p_ensemble->get_decon_queue()->get_input_queue_column_name() == p_decon_queue->get_input_queue_column_name() &&
                                           p_ensemble->get_decon_queue()->get_input_queue_table_name() == p_decon_queue->get_input_queue_table_name()))
        p_ensemble->set_decon_queue(p_decon_queue);
    auto &aux_decon_queues = p_ensemble->get_aux_decon_queues();
    for (auto &p_aux_decon_queue: aux_decon_queues)
        if (p_aux_decon_queue->get_input_queue_column_name() == p_decon_queue->get_input_queue_column_name() &&
            p_aux_decon_queue->get_input_queue_table_name() == p_decon_queue->get_input_queue_table_name())
            p_aux_decon_queue = p_decon_queue;
    LOG4_THROW("Decon queue " << p_decon_queue->to_string() << " not found in dataset!");
}

std::unordered_map<std::pair<std::string, std::string>, datamodel::DeconQueue_ptr>
Dataset::get_decon_queues() const
{
    std::unordered_map<std::pair<std::string, std::string>, datamodel::DeconQueue_ptr> result;
    for (const auto &p_ensemble: ensembles_) {
        result[{p_ensemble->get_decon_queue()->get_input_queue_table_name(),
                p_ensemble->get_decon_queue()->get_input_queue_column_name()}] = p_ensemble->get_decon_queue();
        for (const auto &p_decon_queue: p_ensemble->get_aux_decon_queues())
            result[{p_decon_queue->get_input_queue_table_name(), p_decon_queue->get_input_queue_column_name()}] = p_decon_queue;
    }
    return result;
}

void Dataset::clear_data()
{
    for (auto &p_decon_queue: get_decon_queues())
        p_decon_queue.second->get_data().clear();
    input_queue_.get_obj()->get_data().clear();
    for (auto &p_input_queue: aux_input_queues_)
        p_input_queue.get_obj()->get_data().clear();
}

datamodel::DeconQueue_ptr Dataset::get_decon_queue(const datamodel::InputQueue_ptr &p_input_queue, const std::string &column_name)
{
    return get_decon_queue(p_input_queue->get_table_name(), column_name);
}

datamodel::DeconQueue_ptr Dataset::get_decon_queue(const std::string &table_name, const std::string &column_name)
{
    for (const auto &p_ensemble: ensembles_) {
        if (p_ensemble->get_decon_queue()->get_input_queue_table_name() == table_name and p_ensemble->get_decon_queue()->get_input_queue_column_name() == column_name)
            return p_ensemble->get_decon_queue();
        for (auto &p_decon_queue: p_ensemble->get_aux_decon_queues())
            if (p_decon_queue->get_input_queue_table_name() == table_name and p_decon_queue->get_input_queue_column_name() == column_name)
                return p_decon_queue;
    }

    LOG4_ERROR("Decon queue for input table " << table_name << " and input column name " << column_name << " not found!");

    return nullptr;
}

datamodel::Ensemble_ptr Dataset::get_ensemble(const size_t idx)
{
    return ensembles_[idx];
}

size_t Dataset::get_ensemble_count() const
{
    return input_queue_.get_obj()->get_value_columns().size();
}

void Dataset::set_ensembles(const std::deque<datamodel::Ensemble_ptr> &new_ensembles, const bool overwrite)
{
    const auto prev_size = ensembles_.size();
    for (const auto &e: new_ensembles) {
        std::atomic<bool> found = false;
#pragma omp parallel for num_threads(adj_threads(prev_size))
        for (size_t i = 0; i < prev_size; ++i)
            if (e->get_column_name() == ensembles_[i]->get_column_name()) {
                found.store(true, std::memory_order_relaxed);
                if (overwrite) {
                    ensembles_[i] = e;
                    ensembles_[i]->set_dataset_id(id);
                }
            }
        if (found) continue;
        ensembles_.emplace_back(e);
        ensembles_.back()->set_dataset_id(id);
    }
}

bool Dataset::get_is_active() const
{ return is_active_; }

void Dataset::set_is_active(const bool is_active)
{ is_active_ = is_active; }

std::deque<datamodel::IQScalingFactor_ptr> &Dataset::get_iq_scaling_factors()
{ return iq_scaling_factors_; }

std::deque<datamodel::IQScalingFactor_ptr> Dataset::get_iq_scaling_factors(const InputQueue &input_queue) const
{
    std::deque<datamodel::IQScalingFactor_ptr> res;
    for (const auto &iqsf: iq_scaling_factors_)
        if (iqsf->get_input_queue_table_name() == input_queue.get_table_name())
            res.emplace_back(iqsf);
    return res;
}

datamodel::IQScalingFactor_ptr Dataset::get_iq_scaling_factor(const std::string &input_queue_table_name, const std::string &input_queue_column_name) const
{
    for (const auto &iqsf: iq_scaling_factors_)
        if (iqsf->get_input_queue_table_name() == input_queue_table_name && iqsf->get_input_queue_column_name() == input_queue_column_name)
            return iqsf;
    return nullptr;
}

void Dataset::set_iq_scaling_factors(const std::deque<datamodel::IQScalingFactor_ptr> &new_iq_scaling_factors, const bool overwrite)
{
    const auto prev_size = iq_scaling_factors_.size();
    for (const auto &new_iqsf: new_iq_scaling_factors) {
        std::atomic<bool> found = false;
#pragma omp parallel for num_threads(adj_threads(prev_size))
        for (size_t i = 0; i < prev_size; ++i) {
            auto &old_iqsf = iq_scaling_factors_[i];
            if (new_iqsf->get_input_queue_table_name() == old_iqsf->get_input_queue_table_name() &&
                new_iqsf->get_input_queue_column_name() == old_iqsf->get_input_queue_column_name()) {
                if (overwrite || !std::isnormal(old_iqsf->get_scaling_factor())) old_iqsf->set_scaling_factor(new_iqsf->get_scaling_factor());
                found.store(true, std::memory_order_relaxed);
            }
        }
        if (found) continue;
        iq_scaling_factors_.emplace_back(new_iqsf);
        iq_scaling_factors_.back()->set_dataset_id(id);
    }
}

svr::datamodel::dq_scaling_factor_container_t Dataset::get_dq_scaling_factors() const
{
    return dq_scaling_factors_;
}

svr::datamodel::dq_scaling_factor_container_t &Dataset::get_dq_scaling_factors()
{
    return dq_scaling_factors_;
}

void Dataset::set_dq_scaling_factors(const dq_scaling_factor_container_t &new_dq_scaling_factors)
{
    const std::scoped_lock l(dq_scaling_factors_mutex);
    business::DQScalingFactorService::add(dq_scaling_factors_, new_dq_scaling_factors);
}

std::shared_ptr<DQScalingFactor> Dataset::get_dq_scaling_factor(
        const std::string &input_queue_table_name, const std::string &input_queue_column_name, const size_t level)
{
    const std::scoped_lock l(dq_scaling_factors_mutex);
    const auto res = business::DQScalingFactorService::slice(dq_scaling_factors_, id, input_queue_column_name, std::set{level});
    if (res.empty()) LOG4_ERROR("Could not find scaling factor for " << input_queue_table_name << ", column " << input_queue_column_name << ", level " << level);
    return *res.begin();
}


size_t Dataset::get_max_lag_count()
{
    if (max_lag_count_cache_) return max_lag_count_cache_;

    max_lag_count_cache_ = 0;
    OMP_LOCK(max_lag_l)
#pragma omp parallel for num_threads(adj_threads(ensembles_.size()))
    for (const auto &e: ensembles_)
#pragma omp parallel for num_threads(adj_threads(e->get_models().size()))
        for (const auto &m: e->get_models())
            for (const auto &svr: m->get_gradients())
                for (const auto &p: svr->get_param_set())
                    if (p) {
                        omp_set_lock(&max_lag_l);
                        max_lag_count_cache_ = std::max(max_lag_count_cache_, p->get_lag_count());
                        omp_unset_lock(&max_lag_l);
                    }
    LOG4_DEBUG("Returning non-cached value " << max_lag_count_cache_);
    return max_lag_count_cache_;
}

size_t Dataset::get_max_decrement()
{
    if (max_decremental_distance_cache_) return max_decremental_distance_cache_;
    max_decremental_distance_cache_ = 0;
    OMP_LOCK(max_decrement_l)
#pragma omp parallel for num_threads(adj_threads(ensembles_.size()))
    for (const auto &e: ensembles_)
#pragma omp parallel for num_threads(adj_threads(e->get_models().size()))
        for (const auto &m: e->get_models())
            for (const auto &svr: m->get_gradients())
                for (const auto &p: svr->get_param_set())
                    if (p) {
                        omp_set_lock(&max_decrement_l);
                        max_decremental_distance_cache_ = std::max(max_decremental_distance_cache_, p->get_svr_decremental_distance());
                        omp_unset_lock(&max_decrement_l);
                    }

    LOG4_DEBUG("Returning non-cached value " << max_decremental_distance_cache_);
    return max_decremental_distance_cache_;
}

size_t Dataset::get_max_residuals_length() const
{
    if (ensembles_.empty()) LOG4_THROW("EVMD needs ensembles initialized to calculate residuals count.");

    size_t result = 0;
    OMP_LOCK(max_residuals_l)
#pragma omp parallel for num_threads(adj_threads(ensembles_.size())) schedule(static, 1)
    for (const auto &p_ensemble: ensembles_) {
        const size_t res_count = get_residuals_length(p_ensemble->get_decon_queue()->get_table_name());
        omp_set_lock(&max_residuals_l);
        result = std::max(result, res_count);
        omp_unset_lock(&max_residuals_l);
#pragma omp parallel for num_threads(adj_threads(p_ensemble->get_aux_decon_queues().size())) schedule(static, 1)
        for (const auto &p_decon: p_ensemble->get_aux_decon_queues()) {
            const size_t res_count_aux = get_residuals_length(p_decon->get_table_name());
            omp_set_lock(&max_residuals_l);
            result = std::max(result, res_count_aux);
            omp_unset_lock(&max_residuals_l);
        }
    }
    return result;
}

size_t Dataset::get_max_possible_residuals_length() const
{
    return get_residuals_length(common::gen_random(8));
}

size_t Dataset::get_residuals_length(const std::string &decon_queue_table_name) const
{
    return std::max(
            p_cvmd_transformer ? p_cvmd_transformer->get_residuals_length(decon_queue_table_name) : 0,
            p_oemd_transformer_fat ? p_oemd_transformer_fat->get_residuals_length(decon_queue_table_name) : 0);
}

bpt::ptime Dataset::get_last_modeled_time() const
{
    bpt::ptime res = bpt::min_date_time;
    for (const auto &p_ensemble: ensembles_)
        for (const auto &p_model: p_ensemble->get_models())
            if (res < p_model->get_last_modeled_value_time())
                res = p_model->get_last_modeled_value_time();
    return res;
}

std::string Dataset::to_string() const
{
    std::stringstream s;

    s << "Id " << get_id()
      << ", name " << get_dataset_name()
      << ", input queue table name " << input_queue_.get_id()
      << ", user name " << get_user_name()
      << ", priority " << svr::datamodel::to_string(get_priority())
      << ", description " << get_description()
      << ", transformation levels " << transformation_levels_
      << ", transformation name " << transformation_name_
      << ", gradients " << gradients_
      << ", chunk size " << max_chunk_size_
      << ", multi out " << multiout_
      << ", max lookback time gap " << bpt::to_simple_string(get_max_lookback_time_gap())
      << ", is active " << get_is_active();

    return s.str();
}

}   //end of datamodel namespace
}   //end of svr
