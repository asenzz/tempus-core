#pragma once

#include "Entity.hpp"

namespace svr {
namespace datamodel {

class IQScalingFactor : public Entity
{
private:
    bigint dataset_id_;

    std::string input_queue_table_name_;
    double scaling_factor_;

public:
    IQScalingFactor(const bigint id, const bigint dataset_id,
                const std::string& input_queue_table_name,
                const double scaling_factor = 1.0) :
        Entity(id),
        dataset_id_(dataset_id),
        input_queue_table_name_(input_queue_table_name),
        scaling_factor_(scaling_factor)
    {}

    bool operator==(const IQScalingFactor& other) const
    {
        return other.get_id() == get_id() &&
               other.get_dataset_id() == get_dataset_id() &&
               other.get_input_queue_table_name() == get_input_queue_table_name() &&
               other.get_scaling_factor() == get_scaling_factor();
    }

    bigint get_dataset_id() const
    {
        return dataset_id_;
    }

    void set_dataset_id(const bigint dataset_id)
    {
        dataset_id_ = dataset_id;
    }

    std::string get_input_queue_table_name() const
    {
        return input_queue_table_name_;
    }

    void set_input_queue_table_name(const std::string& input_queue_table_name)
    {
        input_queue_table_name_ = input_queue_table_name;
    }

    double get_scaling_factor() const
    {
        return scaling_factor_;
    }

    void set_scaling_factor(const double scaling_factor)
    {
        scaling_factor_ = scaling_factor;
    }


    virtual std::string to_string() const override
    {
        std::stringstream ss;
        std::string sep {", "};

        ss << "Scaling task id: " << get_id() << sep <<
              "Dataset id: " << get_dataset_id() << sep <<
              "Input queue table name: " << get_input_queue_table_name() << sep <<
              "Scaling factor: " << get_scaling_factor();

        return ss.str();
    }
};

} // namespace datamodel
} // namespace svr

using IQScalingFactor_ptr = std::shared_ptr<svr::datamodel::IQScalingFactor>;
