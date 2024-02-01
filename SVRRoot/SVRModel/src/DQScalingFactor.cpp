//
// Created by zarko on 1/16/24.
//
#include "model/DQScalingFactor.hpp"
#include "util/math_utils.hpp"

namespace svr {
namespace datamodel {


std::string DQScalingFactor::to_string() const
{
    std::stringstream str;
    str << std::setprecision(std::numeric_limits<double>::max_digits10) << "Decon queue scaling factor ID " << id << ", " <<
        "dataset ID " << dataset_id_ << ", " <<
        "input queue table name " << input_queue_table_name_ << ", " <<
        "input queue column name " << input_queue_column_name_ << ", " <<
        "level " << decon_level_ << ", " <<
        "labels factor " << scaling_factor_labels << ", " <<
        "features factor " << scaling_factor_features;
    return str.str();
}

bool DQScalingFactor::operator==(const DQScalingFactor &other) const
{
    return other.get_id() == get_id() &&
           other.get_dataset_id() == get_dataset_id() &&
           other.get_input_queue_table_name() == get_input_queue_table_name() &&
           other.get_input_queue_column_name() == get_input_queue_column_name() &&
           other.get_decon_level() == get_decon_level() &&
           other.get_features_factor() == get_features_factor() &&
           other.get_labels_factor() == get_labels_factor();
}

bool DQScalingFactor::operator<(const DQScalingFactor &o) const
{
    if (this->dataset_id_ < o.dataset_id_)
        return true;
    if (this->dataset_id_ > o.dataset_id_)
        return false;

    if (this->input_queue_table_name_ < o.input_queue_table_name_)
        return true;
    if (this->input_queue_table_name_ > o.input_queue_table_name_)
        return false;

    if (this->input_queue_column_name_ < o.input_queue_column_name_)
        return true;
    if (this->input_queue_column_name_ > o.input_queue_column_name_)
        return false;

    return this->decon_level_ < o.decon_level_;
}

bool DQScalingFactor::in(const dq_scaling_factor_container_t &c)
{
    return std::any_of(std::execution::par_unseq, c.begin(), c.end(), [&](const DQScalingFactor_ptr &p) -> bool {
        return this->decon_level_ == p->decon_level_ &&
               this->dataset_id_ == p->dataset_id_ &&
               this->input_queue_column_name_ == p->input_queue_column_name_ &&
               this->input_queue_table_name_ == p->input_queue_table_name_;
    });
}


void DQScalingFactor::add(dq_scaling_factor_container_t &sf, const bool overwrite)
{
    if (std::none_of(std::execution::par_unseq, sf.begin(), sf.end(), [&](auto &of) {
        if (this->get_dataset_id() != of->get_dataset_id() ||
            this->get_decon_level() != of->get_decon_level() || // TODO implement grad level and manifold depth matching
            this->get_input_queue_table_name() != of->get_input_queue_table_name() ||
            this->get_input_queue_column_name() != of->get_input_queue_column_name())
            return false;

        if ((common::isnormalz(this->get_labels_factor()) || overwrite) || !common::isnormalz(of->get_labels_factor()))
            of->set_labels_factor(this->get_labels_factor());
        if ((common::isnormalz(this->get_features_factor()) || overwrite) || !common::isnormalz(of->get_features_factor()))
            of->set_features_factor(this->get_features_factor());

        return true;
    }))
        sf.emplace(std::make_shared<DQScalingFactor>(*this));
}


void DQScalingFactor::add(dq_scaling_factor_container_t &sf, const dq_scaling_factor_container_t &new_sf, const bool overwrite)
{
    for_each (std::execution::par_unseq, new_sf.begin(), new_sf.end(), [&](const auto nf) {
        bool factors_found = false;
        for (auto &of: sf) {
            if (nf->get_dataset_id() == of->get_dataset_id() &&
                nf->get_decon_level() == of->get_decon_level() && // TODO implement grad level and manifold depth
                nf->get_input_queue_table_name() == of->get_input_queue_table_name() &&
                nf->get_input_queue_column_name() == of->get_input_queue_column_name()) {
                if ((common::isnormalz(nf->get_labels_factor()) || overwrite) || !common::isnormalz(of->get_labels_factor()))
                    of->set_labels_factor(nf->get_labels_factor());
                if ((common::isnormalz(nf->get_features_factor()) || overwrite) || !common::isnormalz(of->get_features_factor()))
                    of->set_features_factor(nf->get_features_factor());
                factors_found = true;
                break;
            }
        }
        if (!factors_found) sf.emplace(nf);
    });
}


bool operator<(const DQScalingFactor_ptr &lhs, const DQScalingFactor_ptr &rhs)
{
    return *lhs < *rhs;
}


}
}