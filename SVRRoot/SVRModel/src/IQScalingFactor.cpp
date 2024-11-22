//
// Created by zarko on 16/11/2024.
//

#include <sstream>
#include "model/IQScalingFactor.hpp"

namespace svr {
namespace datamodel {

IQScalingFactor::IQScalingFactor(const bigint id, const bigint dataset_id,
                const std::string &input_queue_table_name,
                const std::string &input_queue_column_name,
                const double scaling_factor,
                const double dc_offset) :
        Entity(id),
        dataset_id_(dataset_id),
        input_queue_table_name_(input_queue_table_name),
        input_queue_column_name_(input_queue_column_name),
        scaling_factor_(scaling_factor),
        dc_offset_(dc_offset)
{
#ifdef ENTITY_INIT_ID
    init_id();
#endif
}

void IQScalingFactor::init_id()
{
    if (!id) {
        boost::hash_combine(id, dataset_id_);
        boost::hash_combine(id, input_queue_table_name_);
        boost::hash_combine(id, input_queue_column_name_);
    }
}

bool IQScalingFactor::operator==(const IQScalingFactor &other) const
{
    return other.id == id &&
           other.dataset_id_ == dataset_id_ &&
           other.input_queue_table_name_ == input_queue_table_name_ &&
           other.input_queue_column_name_ == input_queue_column_name_ &&
           other.scaling_factor_ == scaling_factor_;
}

bigint IQScalingFactor::get_dataset_id() const
{
    return dataset_id_;
}

void IQScalingFactor::set_dataset_id(const bigint dataset_id)
{
    dataset_id_ = dataset_id;
}

std::string IQScalingFactor::get_input_queue_table_name() const
{
    return input_queue_table_name_;
}

void IQScalingFactor::set_input_queue_table_name(const std::string &input_queue_table_name)
{
    input_queue_table_name_ = input_queue_table_name;
}


std::string IQScalingFactor::get_input_queue_column_name() const
{
    return input_queue_column_name_;
}

void IQScalingFactor::set_input_queue_column_name(const std::string &input_queue_column_name)
{
    input_queue_column_name_ = input_queue_column_name;
}

double IQScalingFactor::get_scaling_factor() const
{
    return scaling_factor_;
}

void IQScalingFactor::set_scaling_factor(const double scaling_factor)
{
    scaling_factor_ = scaling_factor;
}

double IQScalingFactor::get_dc_offset() const
{
    return dc_offset_;
}

void IQScalingFactor::set_dc_offset(const double dc_offset)
{
    dc_offset_ = dc_offset;
}

std::string IQScalingFactor::to_string() const
{
    std::stringstream s;
    s << "Scaling task id " << id <<
       ", dataset id " << dataset_id_ <<
       ", input queue table name " << input_queue_table_name_ <<
       ", input queue column name " << input_queue_column_name_ <<
       ", scaling factor " << scaling_factor_ <<
       ", dc offset " << dc_offset_;
    return s.str();
}


} // namespace datamodel
} // namespace svr
