#pragma once

#include "Entity.hpp"
#include <sstream>

namespace svr {
namespace datamodel {

class IQScalingFactor : public Entity
{
private:
    bigint dataset_id_ = 0;

    std::string input_queue_table_name_;
    std::string input_queue_column_name_;
    double scaling_factor_ = std::numeric_limits<double>::quiet_NaN();
    double dc_offset_ = std::numeric_limits<double>::quiet_NaN();

public:
    IQScalingFactor(const bigint id, const bigint dataset_id,
                    const std::string &input_queue_table_name,
                    const std::string &input_queue_column_name,
                    const double scaling_factor = std::numeric_limits<double>::quiet_NaN(),
                    const double dc_offset = std::numeric_limits<double>::quiet_NaN()) :
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

    virtual void init_id() override
    {
        if (!id) {
            boost::hash_combine(id, dataset_id_);
            boost::hash_combine(id, input_queue_table_name_);
            boost::hash_combine(id, input_queue_column_name_);
        }
    }

    bool operator==(const IQScalingFactor &other) const
    {
        return other.id == id &&
               other.dataset_id_ == dataset_id_ &&
               other.input_queue_table_name_ == input_queue_table_name_ &&
               other.input_queue_column_name_ == input_queue_column_name_ &&
               other.scaling_factor_ == scaling_factor_;
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

    void set_input_queue_table_name(const std::string &input_queue_table_name)
    {
        input_queue_table_name_ = input_queue_table_name;
    }


    std::string get_input_queue_column_name() const
    {
        return input_queue_column_name_;
    }

    void set_input_queue_column_name(const std::string &input_queue_column_name)
    {
        input_queue_column_name_ = input_queue_column_name;
    }

    double get_scaling_factor() const
    {
        return scaling_factor_;
    }

    void set_scaling_factor(const double scaling_factor)
    {
        scaling_factor_ = scaling_factor;
    }

    double get_dc_offset() const
    {
        return dc_offset_;
    }

    void set_dc_offset(const double dc_offset)
    {
        dc_offset_ = dc_offset;
    }

    virtual std::string to_string() const override
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
};


template<typename T> inline std::basic_ostream<T> &
operator<<(std::basic_ostream<T> &s, const IQScalingFactor &i)
{
    return s << i.to_string();
}

using IQScalingFactor_ptr = std::shared_ptr<svr::datamodel::IQScalingFactor>;

} // namespace datamodel
} // namespace svr
