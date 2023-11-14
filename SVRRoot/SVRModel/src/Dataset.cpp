#include "model/Dataset.hpp"
#include "onlinesvr.hpp"

#include "appcontext.hpp"

#include "SVRParametersService.hpp"

#include "online_emd.hpp"
#include "short_term_fourier_transform.hpp"
#include "spectral_transform.hpp"
#include "fast_cvmd.hpp"


namespace svr {
namespace datamodel {

void Dataset::init_transform()
{
    p_oemd_transformer_fat = std::unique_ptr<svr::online_emd>(new svr::online_emd(transformation_levels_ / 4, OEMD_STRETCH_COEF, PROPS.get_oemd_find_fir_coefficients()));
    p_cvmd_transformer = std::unique_ptr<svr::fast_cvmd>(new svr::fast_cvmd(transformation_levels_ / 2));
}

Dataset::Dataset(
        bigint id,
        const std::string &dataset_name,
        const std::string &user_name,
        InputQueue_ptr p_input_queue,
        const std::vector<InputQueue_ptr> &aux_input_queues,
        const Priority &priority,
        const std::string &description,
        const size_t transformation_levels,
        const std::string &transformation_name,
        const bpt::time_duration &max_lookback_time_gap,
        const std::vector<Ensemble_ptr> &ensembles,
        bool is_active,
        const std::vector<IQScalingFactor_ptr> iq_scaling_factors,
        const dq_scaling_factor_container_t dq_scaling_factors
)
        : Entity(id),
          dataset_name_(dataset_name),
          user_name_(user_name),
          priority_(priority),
          description_(description),
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
        aux_input_queues_.push_back(p_aux_input_queue);

    init_transform();
}

Dataset::Dataset(
        bigint id,
        const std::string &dataset_name,
        const std::string &user_name,
        const std::string &input_queue_table_name,
        const std::vector<std::string> &aux_input_queue_table_names,
        const Priority &priority,
        const std::string &description, /* This is description */
        const size_t transformation_levels,
        const std::string &transformation_name,
        const bpt::time_duration &max_lookback_time_gap,
        const std::vector<Ensemble_ptr> &ensembles,
        bool is_active,
        const std::vector<IQScalingFactor_ptr> iq_scaling_factors,
        const dq_scaling_factor_container_t dq_scaling_factors
)
        : Entity(id),
          dataset_name_(dataset_name),
          user_name_(user_name),
          priority_(priority),
          description_(description),
          transformation_levels_(transformation_levels),
          transformation_name_(transformation_name),
          max_lookback_time_gap_(max_lookback_time_gap),
          ensembles_(ensembles),
          is_active_(is_active),
          iq_scaling_factors_(svr::common::clone_shared_ptr_elements(iq_scaling_factors)),
          dq_scaling_factors_(svr::common::clone_shared_ptr_elements(dq_scaling_factors))
{
    if (input_queue_table_name.empty()) THROW_EX_FS(std::logic_error, "Input queue table name cannot be empty");

    input_queue_.set_id(input_queue_table_name);

    for (const auto &aux_input_queue_table_name: aux_input_queue_table_names)
        aux_input_queues_.push_back(aux_input_queue_table_name);

    init_transform();
}

Dataset::Dataset(Dataset const &dataset) :
        Dataset(dataset.get_id(),
                dataset.dataset_name_,
                dataset.user_name_,
                dataset.input_queue_.get_obj(),
                dataset.get_aux_input_queues(),
                dataset.priority_,
                dataset.description_,
                dataset.transformation_levels_,
                dataset.transformation_name_,
                dataset.max_lookback_time_gap_,
                dataset.ensembles_,
                dataset.is_active_,
                dataset.iq_scaling_factors_,
                dataset.dq_scaling_factors_)
{
    set_ensemble_svr_parameters_deep(dataset.get_ensemble_svr_parameters());
}


bool Dataset::operator==(const Dataset &other) const
{
    return get_id() == other.get_id()
           && get_dataset_name() == other.get_dataset_name()
           && get_user_name() == other.get_user_name()
           && input_queue_.get_id() == other.input_queue_.get_id()
           && get_priority() == other.get_priority()
           && get_transformation_levels() == other.get_transformation_levels()
           && get_transformation_name() == other.get_transformation_name()
           && get_max_lookback_time_gap() == other.get_max_lookback_time_gap()
           //          && ensembles_.size() == other.get_ensembles().size()
           //          && std::equal(ensembles.begin(), ensembles.end(),
           //             other.get_ensembles().begin())
           && get_is_active() == other.get_is_active();
}


void Dataset::on_set_id()
{
    for (Ensemble_ptr &p_ensemble : ensembles_)
        p_ensemble->set_dataset_id(get_id());

    for (auto &vec_svr_parameters : ensemble_svr_parameters_)
        for (auto svr_parameters : vec_svr_parameters.second)
            svr_parameters->set_dataset_id(get_id());
}


std::string Dataset::get_dataset_name() const
{ return dataset_name_; }

void Dataset::set_dataset_name(const std::string &dataset_name)
{ Dataset::dataset_name_ = dataset_name; }

std::string Dataset::get_user_name() const
{ return user_name_; }

void Dataset::set_user_name(const std::string &user_name)
{ this->user_name_ = user_name; }

//const InputQueue_ptr &Dataset::get_input_queue() const { return input_queue_.get_obj(); }

InputQueue_ptr Dataset::get_input_queue()
{ return input_queue_.get_obj(); }


void Dataset::set_input_queue(const InputQueue_ptr &p_input_queue) { input_queue_.set_obj(p_input_queue); }


std::vector<InputQueue_ptr> Dataset::get_aux_input_queues() const
{
    std::vector<InputQueue_ptr> result;
    for (const auto &relation_aux_input_queue: aux_input_queues_)
        result.push_back(relation_aux_input_queue.get_obj());
    return result;
}


InputQueue_ptr Dataset::get_aux_input_queue(const size_t idx) const
{
    return aux_input_queues_[idx].get_obj();
}


std::vector<std::string> Dataset::get_aux_input_table_names() const
{
    std::vector<std::string> res;
    res.reserve(aux_input_queues_.size());
    std::transform(aux_input_queues_.begin(), aux_input_queues_.end(),
                   std::back_inserter(res),
                   [](const iq_relation &inque_rel) { return inque_rel.get_obj()->get_table_name(); });
    return res;
}

Priority const &Dataset::get_priority() const
{ return priority_; }

void Dataset::set_priority(Priority const &priority)
{ this->priority_ = priority; }

std::string Dataset::get_description() const
{ return description_; }

void Dataset::set_description(const std::string &description)
{ this->description_ = description; }

size_t Dataset::get_transformation_levels() const
{ return transformation_levels_; }

size_t Dataset::get_transformation_levels_cvmd() const
{ return transformation_levels_ / 2; }

size_t Dataset::get_transformation_levels_oemd() const
{ return transformation_levels_ / 4; }

void Dataset::set_transformation_levels(const size_t transformation_levels)
{ this->transformation_levels_ = transformation_levels; }

std::string Dataset::get_transformation_name() const
{ return this->transformation_name_; }

void Dataset::set_transformation_name(const std::string &transformation_name)
{
    assert(validate_transformation_name(transformation_name));
    this->transformation_name_ = transformation_name;
}

bool Dataset::validate_transformation_name(const std::string &transformation_name) const
{
    return std::find(transformation_names.begin(), transformation_names.end(), transformation_name) !=
           transformation_names.end();
}

const bpt::time_duration &Dataset::get_max_lookback_time_gap() const
{ return max_lookback_time_gap_; }

void Dataset::set_max_lookback_time_gap(const bpt::time_duration &max_lookback_time_gap)
{ this->max_lookback_time_gap_ = max_lookback_time_gap; }

std::vector<Ensemble_ptr> &Dataset::get_ensembles()
{ return ensembles_; }

Ensemble_ptr &Dataset::get_ensemble(const std::string &column_name)
{
    for (auto &p_ensemble: ensembles_) {
        if (p_ensemble->get_decon_queue()->get_input_queue_column_name() != column_name) continue;
        if (p_ensemble->get_decon_queue()->get_input_queue_table_name() == input_queue_.get_obj()->get_table_name())
            return p_ensemble;
        for (const auto &p_aux_input_queue: aux_input_queues_) {
            if (p_ensemble->get_decon_queue()->get_input_queue_table_name() == p_aux_input_queue.get_obj()->get_table_name())
                return p_ensemble;
            for (const auto &p_aux_decon_queue: p_ensemble->get_aux_decon_queues())
                if (p_aux_decon_queue->get_input_queue_table_name() == p_aux_input_queue.get_obj()->get_table_name())
                    return p_ensemble;
        }
    }
    LOG4_ERROR("Ensemble for column " << column_name << " not found!");
    static Ensemble_ptr fail;
    return fail;
}


Ensemble_ptr &Dataset::get_ensemble(const std::string &table_name, const std::string &column_name)
{
    for (auto &p_ensemble: ensembles_) {
        if (p_ensemble->get_decon_queue()->get_input_queue_column_name() == column_name && p_ensemble->get_decon_queue()->get_input_queue_table_name() == table_name)
            return p_ensemble;
        for (const auto &p_aux_decon_queue: p_ensemble->get_aux_decon_queues())
            if (p_aux_decon_queue->get_input_queue_column_name() == column_name && p_aux_decon_queue->get_input_queue_table_name() == table_name)
                return p_ensemble;
    }
    LOG4_ERROR("Ensemble for column " << column_name << " not found!");
    static Ensemble_ptr fail;
    return fail;
}

void Dataset::set_decon_queue(const DeconQueue_ptr &p_decon_queue)
{
    Ensemble_ptr p_ensemble = get_ensemble(p_decon_queue->get_input_queue_table_name(), p_decon_queue->get_input_queue_column_name());
    if (!p_ensemble) LOG4_THROW("Ensemble not found!");
    if (!p_ensemble->get_decon_queue() || (p_ensemble->get_decon_queue()->get_input_queue_column_name() == p_decon_queue->get_input_queue_column_name() && p_ensemble->get_decon_queue()->get_input_queue_table_name() == p_decon_queue->get_input_queue_table_name()))
        p_ensemble->set_decon_queue(p_decon_queue);
    auto &aux_decon_queues = p_ensemble->get_aux_decon_queues();
    for (auto &p_aux_decon_queue: aux_decon_queues)
        if (p_aux_decon_queue->get_input_queue_column_name() == p_decon_queue->get_input_queue_column_name() && p_aux_decon_queue->get_input_queue_table_name() == p_decon_queue->get_input_queue_table_name())
            p_aux_decon_queue = p_decon_queue;
    LOG4_THROW("Decon queue " << p_decon_queue->to_string() << " not found in dataset!");
}

std::map<std::pair<std::string, std::string>, DeconQueue_ptr>
Dataset::get_decon_queues() const
{
    std::map<std::pair<std::string, std::string>, DeconQueue_ptr> result;
    for (const auto &p_ensemble: ensembles_) {
        result[{p_ensemble->get_decon_queue()->get_input_queue_table_name(), p_ensemble->get_decon_queue()->get_input_queue_column_name()}] = p_ensemble->get_decon_queue();
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

DeconQueue_ptr &Dataset::get_decon_queue(const InputQueue_ptr &p_input_queue, const std::string &column_name)
{
    for (const auto &p_ensemble: ensembles_) {
        if (p_ensemble->get_decon_queue()->get_input_queue_table_name() == p_input_queue->get_table_name() and p_ensemble->get_decon_queue()->get_input_queue_column_name() == column_name)
            return p_ensemble->get_decon_queue();
        for (auto &p_decon_queue: p_ensemble->get_aux_decon_queues())
            if (p_decon_queue->get_input_queue_table_name() == p_input_queue->get_table_name() and p_decon_queue->get_input_queue_column_name() == column_name)
                return p_decon_queue;
    }

    LOG4_ERROR("Decon queue for input table " << p_input_queue->get_table_name() << " and input column name " << column_name << " not found!");

    static DeconQueue_ptr fail{nullptr};
    return fail;
}

Ensemble_ptr &Dataset::get_ensemble(const size_t idx)
{
    return ensembles_[idx];
}

void Dataset::set_ensembles(const std::vector<Ensemble_ptr> &ensembles)
{
    ensembles_ = ensembles;
    for (const Ensemble_ptr &p_ensemble: ensembles_) p_ensemble->set_dataset_id(get_id());
}

bool Dataset::get_is_active() const { return is_active_; }

void Dataset::set_is_active(const bool is_active) { is_active_ = is_active; }

std::vector<IQScalingFactor_ptr> Dataset::get_iq_scaling_factors() { return iq_scaling_factors_; }

void Dataset::set_iq_scaling_factors(const std::vector<IQScalingFactor_ptr> &iq_scaling_factors) { iq_scaling_factors_ = iq_scaling_factors; }

svr::datamodel::dq_scaling_factor_container_t Dataset::get_dq_scaling_factors()
{
    const std::scoped_lock l(dq_scaling_factors_mutex);
    return dq_scaling_factors_;
}

void Dataset::set_dq_scaling_factors(const svr::datamodel::dq_scaling_factor_container_t &dq_scaling_factors)
{
    const std::scoped_lock l(dq_scaling_factors_mutex);
    dq_scaling_factors_ = dq_scaling_factors;
}

void Dataset::add_dq_scaling_factors(const dq_scaling_factor_container_t& new_dq_scaling_factors)
{
    const std::scoped_lock l(dq_scaling_factors_mutex);
    for (const auto &nf: new_dq_scaling_factors) {
        bool fac_set = false;
        for (auto &of: dq_scaling_factors_) {
            if (nf->get_dataset_id() == of->get_dataset_id() &&
                nf->get_decon_level() == of->get_decon_level() &&
                nf->get_input_queue_table_name() == of->get_input_queue_table_name() &&
                nf->get_input_queue_column_name() == of->get_input_queue_column_name()) {
                if (common::isnormalz(nf->get_labels_factor()))
                    of->set_labels_factor(nf->get_labels_factor());
                if (common::isnormalz(nf->get_features_factor()))
                    of->set_features_factor(nf->get_features_factor());
                fac_set = true;
                break;
            }
        }
        if (!fac_set) dq_scaling_factors_.emplace(nf);
    }
}

double Dataset::get_dq_scaling_factor_features(
        const std::string &input_queue_table_name, const std::string &input_queue_column_name, const size_t decon_level)
{
    const std::scoped_lock l(dq_scaling_factors_mutex);
    for (const auto &p_dq_scaling_factor: dq_scaling_factors_)
        if (p_dq_scaling_factor->get_input_queue_table_name() == input_queue_table_name and
            p_dq_scaling_factor->get_input_queue_column_name() == input_queue_column_name and
                p_dq_scaling_factor->get_decon_level() == decon_level)
            return p_dq_scaling_factor->get_features_factor();
    LOG4_ERROR("Features factor for " << input_queue_table_name << " " << input_queue_column_name << " " << decon_level << " not found.");
    return std::numeric_limits<double>::signaling_NaN();
}

double
Dataset::get_dq_scaling_factor_labels(
        const std::string &input_queue_table_name, const std::string &input_queue_column_name, const size_t decon_level)
{
    const std::scoped_lock l(dq_scaling_factors_mutex);
    for (const auto &p_dq_scaling_factor: dq_scaling_factors_)
        if (p_dq_scaling_factor->get_input_queue_table_name() == input_queue_table_name and
            p_dq_scaling_factor->get_input_queue_column_name() == input_queue_column_name and
                p_dq_scaling_factor->get_decon_level() == decon_level)
            return p_dq_scaling_factor->get_labels_factor();
    LOG4_ERROR("Labels factor for " << input_queue_table_name << " " << input_queue_column_name << " " << decon_level << " not found.");
    return std::numeric_limits<double>::signaling_NaN();
}


ensemble_svr_parameters_t Dataset::get_ensemble_svr_parameters() const
{
    const std::scoped_lock lck(svr_params_mutex);
    return ensemble_svr_parameters_;
}


SVRParameters_ptr
Dataset::get_svr_parameters(
        const std::string &table_name,
        const std::string &column_name,
        const size_t level_number) const
{
    const std::scoped_lock l(svr_params_mutex);
    return ensemble_svr_parameters_.at({table_name, column_name}).at(level_number);
}


void Dataset::set_ensemble_svr_parameters(const ensemble_svr_parameters_t &ensemble_svr_parameters)
{
    const std::scoped_lock lck(svr_params_mutex);
    max_decremental_distance_cache_ = 0;
    max_lag_count_cache_ = 0;
    ensemble_svr_parameters_ = ensemble_svr_parameters;
}


void Dataset::set_ensemble_svr_parameters_deep(const ensemble_svr_parameters_t &ensemble_svr_parameters)
{
    const std::scoped_lock lck(svr_params_mutex);
    max_decremental_distance_cache_ = 0;
    max_lag_count_cache_ = 0;
    for (const auto &new_param_set: ensemble_svr_parameters) {
        auto it_old_set = ensemble_svr_parameters_.find(new_param_set.first);
        if (it_old_set == ensemble_svr_parameters_.end()) {
            auto ret = ensemble_svr_parameters_.insert({new_param_set.first, {}});
            if (!ret.second) {
                LOG4_ERROR("Failed creating " << new_param_set.first.first << " " << new_param_set.first.second);
                continue;
            }
            it_old_set = ret.first;
        }
        it_old_set->second.insert(it_old_set->second.begin(), new_param_set.second.begin(), new_param_set.second.end());
    }
}


/*
 *  TODO for this, and every other parameter  
    Possibly have a boolean flag for IS_GIVEN (but not tuned)  
    that will read the parameter, instead of leaving it at the default value,  
    but do not use it for empty iterations of optimization.  
    The current solution of checking for nan is unsafe.  
*/

#define CHECK_PARAM_SET(INDEX, PARAM_NAME) \
    if (bounds[model_number].is_tuned.PARAM_NAME)  \
    {  \
        if (parameter_values[(INDEX)] < bounds[model_number].min_bounds.get_##PARAM_NAME())  \
            p_svr_params->set_##PARAM_NAME(bounds[model_number].min_bounds.get_##PARAM_NAME());  \
        else if (parameter_values[(INDEX)] > bounds[model_number].max_bounds.get_##PARAM_NAME())  \
            p_svr_params->set_##PARAM_NAME(bounds[model_number].max_bounds.get_##PARAM_NAME());  \
        else  \
            p_svr_params->set_##PARAM_NAME(parameter_values[(INDEX)]);  \
    } else {  \
        if (!std::isnan(parameter_values[(INDEX)]))  \
            p_svr_params->set_##PARAM_NAME(parameter_values[(INDEX)]);  \
            else  \
                LOG4_WARN(  \
                    #PARAM_NAME " parameter for " << column_name << " level " << model_number <<  \
                                      " left at default value " << p_svr_params->get_##PARAM_NAME());  \
    }

size_t Dataset::get_max_lag_count()
{
    if (max_lag_count_cache_) return max_lag_count_cache_;
    std::scoped_lock scoped_lock(svr_params_mutex);
    if (ensemble_svr_parameters_.empty()) LOG4_ERROR("Called with empty SVR parameters.");

    max_lag_count_cache_ = 0;
    for (auto const &ens_svr_params : ensemble_svr_parameters_)
        for (auto const &p_svr_param : ens_svr_params.second)
            if (p_svr_param)
                max_lag_count_cache_ = std::max(max_lag_count_cache_, p_svr_param->get_lag_count());
    LOG4_DEBUG("Returning non-cached value " << max_lag_count_cache_);
    return max_lag_count_cache_;
}

size_t Dataset::get_max_decrement()
{
    if (max_decremental_distance_cache_) return max_decremental_distance_cache_;
    std::scoped_lock scoped_lock(svr_params_mutex);
    if (ensemble_svr_parameters_.empty()) LOG4_ERROR("Get max decrement called with empty SVR parameters.");
    max_decremental_distance_cache_ = 0;
    for (auto const &ens_svr_params : ensemble_svr_parameters_)
        for (auto const &p_svr_param : ens_svr_params.second)
            if (p_svr_param)
                max_decremental_distance_cache_ = std::max(max_decremental_distance_cache_, p_svr_param->get_svr_decremental_distance());
    LOG4_DEBUG("Returning non-cached value " << max_decremental_distance_cache_);
    return max_decremental_distance_cache_;
}

size_t Dataset::get_max_residuals_count() const
{
    if (ensembles_.empty()) LOG4_THROW("EVMD needs ensembles initialized to calculate residuals count.");

    size_t result = 0;
    for (const auto &p_ensemble: ensembles_) {
        const size_t res_count = get_residuals_count(p_ensemble->get_decon_queue()->get_table_name());
        if (res_count > result) result = res_count;
    }
    return result;
}

size_t Dataset::get_maxpos_residuals_count() const
{
    return get_residuals_count(common::gen_random(8));
}

size_t Dataset::get_residuals_count(const std::string &decon_queue_table_name) const
{
    return std::max(
            p_cvmd_transformer->get_residuals_length(decon_queue_table_name),
            p_oemd_transformer_fat->get_residuals_length());
}

std::string Dataset::to_string() const
{
    std::stringstream ss;

    ss << "Id " << get_id()
       << ", name " << get_dataset_name()
       << ", input queue table name " << input_queue_.get_id()
       << ", user name " << get_user_name()
       << ", priority " << svr::datamodel::to_string(get_priority())
       << ", description " << get_description()
       << ", transformation levels " << get_transformation_levels()
       << ", transformation name " << get_transformation_name()
       << ", max lookback time gap " << bpt::to_simple_string(get_max_lookback_time_gap())
       << ", is active " << get_is_active();

    return ss.str();
}

std::string Dataset::parameters_to_string() const
{
    std::scoped_lock scoped_lock(svr_params_mutex);

    std::string s{"{"};

    s += "\"transformation_levels\":\"" + std::to_string(transformation_levels_) + "\",";
    s += "\"transformation_name\":\"" + transformation_name_ + "\",";

    // ensembles[0] is used since each ensemble has the same parameters
    const std::vector<SVRParameters_ptr> &vec_parameters{ensemble_svr_parameters_.begin()->second};
    size_t model_number = 0;

    for (auto &parameters : vec_parameters)
        s += parameters->to_options_string(model_number++) + ",";

    // remove last character, i.e. ","
    s.pop_back();

    return s + "}";
}

}   //end of datamodel namespace
}   //end of svr
