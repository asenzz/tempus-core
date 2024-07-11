//
// Created by zarko on 1/16/24.
//
#include "model/DQScalingFactor.hpp"
#include "util/math_utils.hpp"
#include "DQScalingFactorService.hpp"

namespace svr {
namespace datamodel {

void DQScalingFactor::init_id()
{
    if (!id) {
        boost::hash_combine(id, decon_level_);
        boost::hash_combine(id, step_);
        boost::hash_combine(id, grad_depth_);
        boost::hash_combine(id, chunk_ix_);
        boost::hash_combine(id, model_id_);
    }
}

std::string DQScalingFactor::to_string() const
{
    std::stringstream str;
    str << std::setprecision(std::numeric_limits<double>::max_digits10) << "Decon queue scaling factor ID " << id <<
        ", model ID " << model_id_ <<
        ", level " << decon_level_ <<
        ", step " << step_ <<
        ", gradient " << grad_depth_ <<
        ", chunk " << chunk_ix_ <<
        ", labels factor " << scaling_factor_labels <<
        ", features factor " << scaling_factor_features <<
        ", labels dc offset " << dc_offset_labels <<
        ", features dc offset " << dc_offset_features;
    return str.str();
}

bool DQScalingFactor::operator==(const DQScalingFactor &o) const
{
    return (o ^= *this)
           && o.scaling_factor_labels == scaling_factor_labels
           && o.scaling_factor_features == scaling_factor_features
           && o.dc_offset_labels == dc_offset_labels
           && o.dc_offset_features == dc_offset_features;
}

bool DQScalingFactor::operator^=(const DQScalingFactor &o) const
{
    return (!o.model_id_ || !model_id_ || o.model_id_ == model_id_)
           && o.decon_level_ == decon_level_
           && o.step_ == step_
           && o.grad_depth_ == grad_depth_
           && o.chunk_ix_ == chunk_ix_;
}

bool DQScalingFactor::operator<(const DQScalingFactor &o) const
{
    if (model_id_ < o.model_id_) return true;
    if (model_id_ > o.model_id_) return false;

    if (chunk_ix_ < o.chunk_ix_) return true;
    if (chunk_ix_ > o.chunk_ix_) return false;

    if (grad_depth_ < o.grad_depth_) return true;
    if (grad_depth_ > o.grad_depth_) return false;

    if (step_ < o.step_) return true;
    if (step_ > o.step_) return false;

    return decon_level_ < o.decon_level_;
}

bool DQScalingFactor::in(const dq_scaling_factor_container_t &c)
{
    return std::any_of(std::execution::par_unseq, c.begin(), c.end(), [&](const auto &p) { return *p ^= *this; });
}


bool operator<(const DQScalingFactor_ptr &lhs, const DQScalingFactor_ptr &rhs)
{
    return *lhs < *rhs;
}

DQScalingFactor::DQScalingFactor(
        const bigint id, const bigint model_id,
        const size_t decon_level, const size_t step, const size_t grad_depth, const size_t chunk_index,
        const double scale_feat, const double scale_labels, const double dc_offset_feat, const double dc_offset_labels) :
        Entity(id),
        model_id_(model_id),
        decon_level_(decon_level),
        step_(step),
        grad_depth_(grad_depth),
        chunk_ix_(chunk_index),
        scaling_factor_features(scale_feat),
        scaling_factor_labels(scale_labels),
        dc_offset_features(dc_offset_feat),
        dc_offset_labels(dc_offset_labels)
{
#ifdef ENTITY_INIT_ID
    init_id();
#endif
}

bigint DQScalingFactor::get_model_id() const
{
    return model_id_;
}

void DQScalingFactor::set_model_id(const bigint model_id)
{
    model_id_ = model_id;
}

size_t DQScalingFactor::get_decon_level() const
{
    return decon_level_;
}

void DQScalingFactor::set_decon_level(const size_t decon_level)
{
    decon_level_ = decon_level;
}

size_t DQScalingFactor::get_step() const
{
    return step_;
}

void DQScalingFactor::set_step(const size_t step)
{
    step_ = step;
}

size_t DQScalingFactor::get_grad_depth() const
{
    return grad_depth_;
}

void DQScalingFactor::set_grad_depth(const size_t grad_level)
{
    grad_depth_ = grad_level;
}

size_t DQScalingFactor::get_chunk_index() const
{
    return chunk_ix_;
}

void DQScalingFactor::set_chunk_index(const size_t chunk_index)
{
    chunk_ix_ = chunk_index;
}

double DQScalingFactor::get_features_factor() const
{
    return scaling_factor_features;
}

void DQScalingFactor::set_features_factor(const double scaling_factor)
{
    scaling_factor_features = scaling_factor;
}

double DQScalingFactor::get_labels_factor() const
{
    return scaling_factor_labels;
}

void DQScalingFactor::set_labels_factor(const double label_factor)
{
    scaling_factor_labels = label_factor;
}

double DQScalingFactor::get_dc_offset_features() const
{
    return dc_offset_features;
}

void DQScalingFactor::set_dc_offset_features(const double dc_offset_features_)
{
    dc_offset_features = dc_offset_features_;
}

double DQScalingFactor::get_dc_offset_labels() const
{
    return dc_offset_labels;
}

void DQScalingFactor::set_dc_offset_labels(const double dc_offset_labels_)
{
    dc_offset_labels = dc_offset_labels_;
}


}
}