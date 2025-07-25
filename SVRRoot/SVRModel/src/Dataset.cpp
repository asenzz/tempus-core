#include "onlinesvr.hpp"
#include "appcontext.hpp"
#include "model/Dataset.hpp"
#include "model/Ensemble.hpp"
#include "SVRParametersService.hpp"
#include "ModelService.hpp"
#include "online_emd.hpp"
#include "spectral_transform.hpp"
#include "fast_cvmd.hpp"
#include "DQScalingFactorService.hpp"
#include "calc_cache.hpp"
#include "common/compatibility.hpp"

namespace svr {
namespace datamodel {

void Dataset::init_transform()
{
    if (spectrum_levels_ >= MIN_LEVEL_COUNT) {
#ifdef VMD_ONLY
        p_cvmd_transformer = std::make_unique<svr::vmd::fast_cvmd>(spectrum_levels_);
#elif defined(EMD_ONLY)
        p_oemd_transformer_fat = std::make_unique<svr::oemd::online_emd>(spectrum_levels_);
#else
        p_oemd_transformer_fat = std::make_unique<svr::oemd::online_emd>(spectrum_levels_ / 4);
        p_cvmd_transformer = std::make_unique<svr::vmd::fast_cvmd>(spectrum_levels_ / 2);
#endif
    }
}

Dataset::Dataset() :
        Entity(),
        ccache(),
        gradients_(common::C_default_gradient_count),
        max_chunk_size_(common::AppConfig::C_default_kernel_length),
        multistep_(common::C_default_multistep_len),
        spectrum_levels_(common::C_default_level_count),
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
        const uint16_t gradients,
        const uint32_t chunk_size,
        const uint16_t multistep,
        const uint16_t spectrum_levels,
        const std::string &transformation_name,
        const bpt::time_duration &max_lookback_time_gap,
        const std::deque<datamodel::Ensemble_ptr> &ensembles,
        bool is_active,
        const std::deque<datamodel::IQScalingFactor_ptr> iq_scaling_factors
)
        : Entity(id),
          ccache(),
          dataset_name_(dataset_name),
          user_name_(user_name),
          priority_(priority),
          description_(description),
          gradients_(gradients),
          max_chunk_size_(chunk_size),
          multistep_(multistep),
          spectrum_levels_(spectrum_levels),
          transformation_name_(transformation_name),
          max_lookback_time_gap_(max_lookback_time_gap),
          ensembles_(svr::common::clone_shared_ptr_elements(ensembles)),
          is_active_(is_active),
          iq_scaling_factors_(svr::common::clone_shared_ptr_elements(iq_scaling_factors))
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

Dataset::Dataset(
        bigint id,
        const std::string &dataset_name,
        const std::string &user_name,
        const std::string &input_queue_table_name,
        const std::deque<std::string> &aux_input_queue_table_names,
        const Priority &priority,
        const std::string &description,
        const uint16_t gradients,
        const uint32_t chunk_size,
        const uint16_t multistep,
        const uint16_t spectrum_levels,
        const std::string &transformation_name,
        const bpt::time_duration &max_lookback_time_gap,
        const std::deque<datamodel::Ensemble_ptr> &ensembles,
        bool is_active,
        const std::deque<datamodel::IQScalingFactor_ptr> iq_scaling_factors)
        : Entity(id),
          ccache(),
          dataset_name_(dataset_name),
          user_name_(user_name),
          priority_(priority),
          description_(description),
          gradients_(gradients),
          max_chunk_size_(chunk_size),
          multistep_(multistep),
          spectrum_levels_(spectrum_levels),
          transformation_name_(transformation_name),
          max_lookback_time_gap_(max_lookback_time_gap),
          ensembles_(ensembles),
          is_active_(is_active),
          iq_scaling_factors_(iq_scaling_factors)
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
                dataset.multistep_,
                dataset.spectrum_levels_,
                dataset.transformation_name_,
                dataset.max_lookback_time_gap_,
                dataset.ensembles_,
                dataset.is_active_,
                dataset.iq_scaling_factors_)
{
    if (dataset.p_cvmd_transformer) p_cvmd_transformer = std::make_unique<vmd::fast_cvmd>(*dataset.p_cvmd_transformer);
    if (dataset.p_oemd_transformer_fat) p_oemd_transformer_fat = std::make_unique<oemd::online_emd>(*dataset.p_oemd_transformer_fat);
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


bool Dataset::operator==(const Dataset &o) const
{
    return (*this ^= o)
           && ensembles_.size() == o.ensembles_.size()
           && std::equal(ensembles_.begin(), ensembles_.end(), o.ensembles_.begin());
}

bool Dataset::operator^=(const Dataset &o) const
{
    std::atomic<bool> res = id == o.id
                            && dataset_name_ == o.dataset_name_
                            && user_name_ == o.user_name_
                            && input_queue_.get_id() == o.input_queue_.get_id()
                            && priority_ == o.priority_
                            && gradients_ == o.gradients_
                            && max_chunk_size_ == o.max_chunk_size_
                            && multistep_ == o.multistep_
                            && spectrum_levels_ == o.spectrum_levels_
                            && transformation_name_ == o.transformation_name_
                            && max_lookback_time_gap_ == o.max_lookback_time_gap_
                            && is_active_ == o.is_active_;

    OMP_FOR(aux_input_queues_.size())
    for (const auto &iq: aux_input_queues_)
        res &= std::any_of(o.aux_input_queues_.begin(), o.aux_input_queues_.end(), [&](const auto &oiq) { return iq.get_id() == oiq.get_id(); });

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

uint16_t Dataset::get_gradient_count() const
{ return gradients_; };

uint32_t Dataset::get_max_chunk_size() const
{ return max_chunk_size_; };

uint16_t Dataset::get_multistep() const
{ return multistep_; };

void Dataset::set_gradients(const uint16_t grads)
{ gradients_ = grads; }

void Dataset::set_chunk_size(const uint32_t chunk_size)
{ max_chunk_size_ = chunk_size; }

void Dataset::set_multistep(const uint16_t multistep)
{ multistep_ = multistep; }

const std::string &Dataset::get_dataset_name() const
{ return dataset_name_; }

void Dataset::set_dataset_name(const std::string &dataset_name)
{ Dataset::dataset_name_ = dataset_name; }

const std::string &Dataset::get_user_name() const
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


datamodel::InputQueue_ptr Dataset::get_aux_input_queue(const uint16_t idx) const
{
    return aux_input_queues_[idx].get_obj();
}

datamodel::InputQueue_ptr Dataset::get_aux_input_queue(const std::string &table_name) const
{
    const auto res = std::find_if(aux_input_queues_.cbegin(), aux_input_queues_.cend(), [&](const iq_relation &inque_rel) {
        return inque_rel.get_obj()->get_table_name() == table_name;
    });
    if (res != aux_input_queues_.cend()) return res->get_obj();
    LOG4_THROW("Aux input queue for table " << table_name << " not found!");
    return res->get_obj();
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

const std::string &Dataset::get_description() const
{
    return description_;
}

void Dataset::set_description(const std::string &description)
{
    description_ = description;
}

uint16_t Dataset::get_spectral_levels() const
{
    return spectrum_levels_ >= MIN_LEVEL_COUNT ? spectrum_levels_ : 1;
}

uint16_t Dataset::get_trans_levix() const
{
    return business::SVRParametersService::get_trans_levix(spectrum_levels_);
}

uint16_t Dataset::get_model_count() const
{
    return business::ModelService::to_model_ct(spectrum_levels_);
}

uint16_t Dataset::get_spectrum_levels_cvmd() const
{
#ifdef VMD_ONLY
    return spectrum_levels_ / 2;
#elif defined(EMD_ONLY)
    return 0;
#else
    return spectrum_levels_ >= MIN_LEVEL_COUNT ? std::max<uint16_t>(1, spectrum_levels_ / 2) : 0;
#endif
}

uint16_t Dataset::get_spectrum_levels_oemd() const noexcept
{
#ifdef VMD_ONLY
    return 0;
#elif defined(EMD_ONLY)
    return spectrum_levels_;
#else
    return spectrum_levels_ >= MIN_LEVEL_COUNT ? std::max<uint16_t>(1, spectrum_levels_ / 4) : 0;
#endif
}

void Dataset::set_spectrum_levels(const uint16_t spectrum_levels)
{
    assert(spectrum_levels >= 1);
    spectrum_levels_ = spectrum_levels;
}

const std::string &Dataset::get_transformation_name() const noexcept
{
    return transformation_name_;
}

void Dataset::set_transformation_name(const std::string &transformation_name)
{
    assert(validate_transformation_name(transformation_name));
    transformation_name_ = transformation_name;
}

bool Dataset::validate_transformation_name(const std::string &transformation_name) noexcept
{
    return std::find(C_default_exec_policy, transformation_names.cbegin(), transformation_names.cend(), transformation_name) != transformation_names.cend();
}

const bpt::time_duration &Dataset::get_max_lookback_time_gap() const noexcept
{
    return max_lookback_time_gap_;
}


void Dataset::set_max_lookback_time_gap(const bpt::time_duration &max_lookback_time_gap)
{
    assert(!max_lookback_time_gap_.is_neg_infinity());
    max_lookback_time_gap_ = max_lookback_time_gap;
}

std::deque<datamodel::Ensemble_ptr> &Dataset::get_ensembles() noexcept
{ return ensembles_; }


const std::deque<datamodel::Ensemble_ptr> &Dataset::get_ensembles() const noexcept
{ return ensembles_; }


datamodel::Ensemble_ptr Dataset::get_ensemble(const std::string &column_name) noexcept
{
    return get_ensemble(input_queue_.get_obj()->get_table_name(), column_name);
}


datamodel::Ensemble_ptr Dataset::get_ensemble(const std::string &table_name, const std::string &column_name) noexcept
{
    const auto res = std::find_if(C_default_exec_policy, ensembles_.cbegin(), ensembles_.cend(), [&](const auto &p_ensemble) {
        return p_ensemble->get_decon_queue()->get_input_queue_column_name() == column_name &&
               p_ensemble->get_decon_queue()->get_input_queue_table_name() == table_name;
    });
    if (res != ensembles_.cend()) return *res;
    LOG4_ERROR("Ensemble for table " << table_name << ", column " << column_name << " not found!");
    return nullptr;
}

void Dataset::set_decon_queue(const datamodel::DeconQueue_ptr &p_decon_queue)
{
    assert(p_decon_queue);
    datamodel::Ensemble_ptr p_ensemble = get_ensemble(p_decon_queue->get_input_queue_table_name(), p_decon_queue->get_input_queue_column_name());
    if (!p_ensemble) LOG4_THROW("Ensemble not found!");
    if (!p_ensemble->get_decon_queue() || (p_ensemble->get_decon_queue()->get_input_queue_column_name() == p_decon_queue->get_input_queue_column_name() &&
                                           p_ensemble->get_decon_queue()->get_input_queue_table_name() == p_decon_queue->get_input_queue_table_name()))
        p_ensemble->set_decon_queue(p_decon_queue);
    auto &aux_decon_queues = p_ensemble->get_aux_decon_queues();
    OMP_FOR(aux_decon_queues.size())
    for (auto &p_aux_decon_queue: aux_decon_queues)
        if (p_aux_decon_queue->get_input_queue_column_name() == p_decon_queue->get_input_queue_column_name() &&
            p_aux_decon_queue->get_input_queue_table_name() == p_decon_queue->get_input_queue_table_name())
            p_aux_decon_queue = p_decon_queue;
    LOG4_THROW("Decon queue " << p_decon_queue->to_string() << " not found in dataset!");
}


boost::unordered_flat_map<std::pair<std::string, std::string>, datamodel::DeconQueue_ptr>
Dataset::get_decon_queues() const
{
    boost::unordered_flat_map<std::pair<std::string, std::string>, datamodel::DeconQueue_ptr> result;
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
    OMP_FOR(aux_input_queues_.size())
    for (auto &p_input_queue: aux_input_queues_)
        p_input_queue.get_obj()->get_data().clear();
}

datamodel::DeconQueue_ptr Dataset::get_decon_queue(const datamodel::InputQueue &input_queue, const std::string &column_name) const
{
    return get_decon_queue(input_queue.get_table_name(), column_name);
}

datamodel::DeconQueue_ptr Dataset::get_decon_queue(const std::string &table_name, const std::string &column_name) const
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

datamodel::Ensemble_ptr Dataset::get_ensemble(const uint16_t idx)
{
    return ensembles_[idx];
}

uint16_t Dataset::get_ensemble_count() const
{
    return input_queue_.get_obj()->get_value_columns().size();
}

void Dataset::set_ensembles(const std::deque<datamodel::Ensemble_ptr> &new_ensembles, const bool overwrite)
{
    const auto prev_size = ensembles_.size();

#pragma omp parallel ADJ_THREADS(ensembles_.size() + new_ensembles.size())
#pragma omp single
    {
    	OMP_TASKLOOP_1(untied)
        for (const auto &e: new_ensembles) {
            std::atomic<bool> found = false;
            OMP_TASKLOOP_1(firstprivate(prev_size) untied)
            for (DTYPE(prev_size) i = 0; i < prev_size; ++i)
                if (e->get_column_name() == ensembles_[i]->get_column_name()) {
                    found.store(true, std::memory_order_relaxed);
                    if (!overwrite) continue;
                    const std::scoped_lock lk(ensembles_mx);
                    ensembles_[i] = e;
                    ensembles_[i]->set_dataset_id(id);
                }
            if (found) continue;
            const std::scoped_lock lk(ensembles_mx);
            ensembles_.emplace_back(e);
            ensembles_.back()->set_dataset_id(id);
        }
    }
}

bool Dataset::get_is_active() const noexcept
{ return is_active_; }

void Dataset::set_is_active(const bool is_active) noexcept
{ is_active_ = is_active; }

std::deque<datamodel::IQScalingFactor_ptr> &Dataset::get_iq_scaling_factors() noexcept
{ return iq_scaling_factors_; }

std::deque<datamodel::IQScalingFactor_ptr> Dataset::get_iq_scaling_factors(const InputQueue &input_queue) const
{
    std::deque<datamodel::IQScalingFactor_ptr> res;
    std::copy_if(C_default_exec_policy, iq_scaling_factors_.cbegin(), iq_scaling_factors_.cend(), std::back_inserter(res),
                 [&](const auto &iqsf) { return iqsf->get_input_queue_table_name() == input_queue.get_table_name(); });
    return res;
}

datamodel::IQScalingFactor_ptr Dataset::get_iq_scaling_factor(const std::string &input_queue_table_name, const std::string &input_queue_column_name) const
{
    const auto res = std::find_if(C_default_exec_policy, iq_scaling_factors_.cbegin(), iq_scaling_factors_.cend(),
                                  [&](const auto &iqsf) {
                                      return iqsf->get_input_queue_table_name() == input_queue_table_name && iqsf->get_input_queue_column_name() == input_queue_column_name;
                                  });
    return res == iq_scaling_factors_.cend() ? nullptr : *res;
}

void Dataset::set_iq_scaling_factors(const std::deque<datamodel::IQScalingFactor_ptr> &new_iq_scaling_factors, const bool overwrite)
{
    const auto prev_size = iq_scaling_factors_.size();
    tbb::mutex iq_scaling_factors_l;
#pragma omp parallel ADJ_THREADS(new_iq_scaling_factors.size() * prev_size)
#pragma omp single
    {
    	OMP_TASKLOOP_(new_iq_scaling_factors.size(),)
        for (const auto &new_iqsf: new_iq_scaling_factors) {
            std::atomic<bool> found = false;
            for (DTYPE(prev_size) i = 0; i < prev_size; ++i) {
                tbb::mutex::scoped_lock lk(iq_scaling_factors_l);
                auto &old_iqsf = iq_scaling_factors_[i];
                lk.release();
                if (new_iqsf->get_input_queue_table_name() == old_iqsf->get_input_queue_table_name() &&
                    new_iqsf->get_input_queue_column_name() == old_iqsf->get_input_queue_column_name()) {
                    if (overwrite || !std::isnormal(old_iqsf->get_scaling_factor())) {
                        const tbb::mutex::scoped_lock l2(iq_scaling_factors_l);
                        old_iqsf->set_scaling_factor(new_iqsf->get_scaling_factor());
                    }
                    if (overwrite || !common::isnormalz(old_iqsf->get_dc_offset())) {
                        const tbb::mutex::scoped_lock l2(iq_scaling_factors_l);
                        old_iqsf->set_dc_offset(new_iqsf->get_dc_offset());
                    }
                    found = true;
                }
            }
            if (!found) {
                const tbb::mutex::scoped_lock l2(iq_scaling_factors_l);
                iq_scaling_factors_.emplace_back(new_iqsf);
                iq_scaling_factors_.back()->set_dataset_id(id);
            }
        }
    }
}

uint32_t Dataset::get_max_lag_count() const
{
    uint32_t max_lag_count = 0;
    for (const auto &e: ensembles_)
        for (const auto &m: e->get_models())
            for (const auto &svr: m->get_gradients())
                for (const auto &p: svr->get_param_set())
                    if (p && p->get_lag_count() > max_lag_count)
                        max_lag_count = p->get_lag_count();

    return max_lag_count;
}

uint32_t Dataset::get_max_quantise() const
{
    uint32_t res = 0;
    for (const auto &e: ensembles_)
        for (const auto &m: e->get_models())
            for (const auto &svr: m->get_gradients())
                for (const auto &p: svr->get_param_set())
                    if (p && !p->get_feature_mechanics().needs_tuning())
                        res = std::max(res, p->get_feature_mechanics().quantization.max());
    if (!res) res = business::ModelService::get_max_quantisation();
    LOG4_DEBUG("Returning  " << res);
    return res;
}

uint32_t Dataset::get_max_decrement() const
{
    uint32_t res = 0;
    for (const auto &e: ensembles_)
        for (const auto &m: e->get_models())
            for (const auto &svr: m->get_gradients())
                for (const auto &p: svr->get_param_set())
                    if (p) MAXAS(res, p->get_svr_decremental_distance());

    LOG4_DEBUG("Returning  " << res);
    return res;
}

uint32_t Dataset::get_max_residuals_length() const
{
    if (ensembles_.empty()) LOG4_THROW("EVMD needs ensembles initialized to calculate residuals count.");

    uint32_t result = 0;
    tbb::mutex max_residuals_l;
#pragma omp parallel ADJ_THREADS(2 * ensembles_.size())
#pragma omp single
    {
        OMP_TASKLOOP_(ensembles_.size(), untied)
        for (const auto &p_ensemble: ensembles_) {
            const auto res_count = get_residuals_length(p_ensemble->get_decon_queue()->get_table_name());
            tbb::mutex::scoped_lock lk(max_residuals_l);
            MAXAS(result, res_count);
            lk.release();
            OMP_TASKLOOP_(p_ensemble->get_aux_decon_queues().size(), untied)
            for (const auto &p_decon: p_ensemble->get_aux_decon_queues()) {
                const auto res_count_aux = get_residuals_length(p_decon->get_table_name());
                tbb::mutex::scoped_lock l2(max_residuals_l);
                MAXAS(result, res_count_aux);
            }
        }
    }
    return result;
}

uint32_t Dataset::get_max_possible_residuals_length() const
{
    return get_residuals_length(common::gen_random(8));
}

uint32_t Dataset::get_residuals_length(const std::string &decon_queue_table_name) const
{
    return std::max<uint32_t>(
            p_cvmd_transformer ? p_cvmd_transformer->get_residuals_length(decon_queue_table_name) : 0,
            p_oemd_transformer_fat ? p_oemd_transformer_fat->get_residuals_length(decon_queue_table_name) : 0);
}

uint32_t Dataset::get_residuals_length(const uint16_t levels)
{
    if (levels < MIN_LEVEL_COUNT) return 0;

#ifdef VMD_ONLY
    return vmd::fast_cvmd::get_residuals_length(levels);
#elif defined(EMD_ONLY)
    return oemd::online_emd::get_residuals_length();
#else
    return std::max<size_t>(vmd::fast_cvmd::get_residuals_length(levels / 2), oemd::online_emd::get_residuals_length());
#endif
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

    s << "Id " << id
      << ", name " << dataset_name_
      << ", input queue table name " << input_queue_.get_id()
      << ", user name " << user_name_
      << ", priority " << datamodel::to_string(priority_)
      << ", description " << description_
      << ", transformation levels " << spectrum_levels_
      << ", transformation name " << transformation_name_
      << ", gradients " << gradients_
      << ", chunk size " << max_chunk_size_
      << ", multi out " << multistep_
      << ", max lookback time gap " << get_max_lookback_time_gap()
      << ", is active " << get_is_active();

    return s.str();
}

}   //end of datamodel namespace
}   //end of svr
