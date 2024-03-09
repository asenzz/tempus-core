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
        boost::hash_combine(id, dataset_id_);
        boost::hash_combine(id, input_queue_table_name_);
        boost::hash_combine(id, input_queue_column_name_);
    }
}

std::string DQScalingFactor::to_string() const
{
    std::stringstream str;
    str << std::setprecision(std::numeric_limits<double>::max_digits10) << "Decon queue scaling factor ID " << id << 
        ", dataset ID " << dataset_id_ <<
        ", input queue table name " << input_queue_table_name_ << 
        ", input queue column name " << input_queue_column_name_ << 
        ", level " << decon_level_ << 
        ", labels factor " << scaling_factor_labels << 
        ", features factor " << scaling_factor_features << 
        ", labels dc offset " << dc_offset_labels << 
        ", features dc offset " << dc_offset_features;
    return str.str();
}

bool DQScalingFactor::operator==(const DQScalingFactor &other) const
{
    return other.id == id &&
           other.dataset_id_ == dataset_id_ &&
           other.input_queue_table_name_ == input_queue_table_name_ &&
           other.input_queue_column_name_ == input_queue_column_name_ &&
           other.decon_level_ == decon_level_ &&
           other.scaling_factor_labels == scaling_factor_labels &&
           other.scaling_factor_features == scaling_factor_features &&
           other.dc_offset_labels == dc_offset_labels &&
           other.dc_offset_labels == dc_offset_labels;
}

bool DQScalingFactor::operator<(const DQScalingFactor &o) const
{
    if (dataset_id_ < o.dataset_id_) return true;
    if (dataset_id_ > o.dataset_id_) return false;

    if (input_queue_table_name_ < o.input_queue_table_name_) return true;
    if (input_queue_table_name_ > o.input_queue_table_name_) return false;

    if (input_queue_column_name_ < o.input_queue_column_name_) return true;
    if (input_queue_column_name_ > o.input_queue_column_name_) return false;

    return decon_level_ < o.decon_level_;
}

bool DQScalingFactor::in(const dq_scaling_factor_container_t &c)
{
    return std::any_of(std::execution::par_unseq, c.begin(), c.end(), [&](const auto &p) {
        return decon_level_ == p->decon_level_ &&
               dataset_id_ == p->dataset_id_ &&
               input_queue_column_name_ == p->input_queue_column_name_ &&
               input_queue_table_name_ == p->input_queue_table_name_;
    });
}


bool operator<(const DQScalingFactor_ptr &lhs, const DQScalingFactor_ptr &rhs)
{
    return *lhs < *rhs;
}

DQScalingFactor::DQScalingFactor(
        const bigint id, const bigint dataset_id, const std::string &input_queue_table_name, const std::string &input_queue_column_name,
        const size_t decon_level,
        const double scale_feat,
        const double scale_labels,
        const double dc_offset_feat,
        const double dc_offset_labels) :
        Entity(id),
        dataset_id_(dataset_id),
        input_queue_table_name_(input_queue_table_name),
        input_queue_column_name_(input_queue_column_name),
        decon_level_(decon_level),
        scaling_factor_features(scale_feat),
        scaling_factor_labels(scale_labels),
        dc_offset_features(dc_offset_feat),
        dc_offset_labels(dc_offset_labels)
{
#ifdef ENTITY_INIT_ID
    init_id();
#endif
}

bigint DQScalingFactor::get_dataset_id() const
{
    return dataset_id_;
}

void DQScalingFactor::set_dataset_id(const bigint dataset_id)
{
    dataset_id_ = dataset_id;
}

std::string DQScalingFactor::get_input_queue_table_name() const
{
    return input_queue_table_name_;
}

void DQScalingFactor::set_input_queue_table_name(const std::string &input_queue_table_name)
{
    input_queue_table_name_ = input_queue_table_name;
}

std::string DQScalingFactor::get_input_queue_column_name() const
{
    return input_queue_column_name_;
}

void DQScalingFactor::set_input_queue_column_name(const std::string &input_queue_column_name)
{
    input_queue_column_name_ = input_queue_column_name;
}

size_t DQScalingFactor::get_decon_level() const
{
    return decon_level_;
}

void DQScalingFactor::set_decon_level(const size_t decon_level)
{
    decon_level_ = decon_level;
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