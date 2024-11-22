#pragma once

#include "Entity.hpp"

namespace svr {
namespace datamodel {

class IQScalingFactor : public Entity {
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
                    const double dc_offset = std::numeric_limits<double>::quiet_NaN());

    virtual void init_id() override;

    bool operator==(const IQScalingFactor &other) const;

    bigint get_dataset_id() const;

    void set_dataset_id(const bigint dataset_id);

    std::string get_input_queue_table_name() const;

    void set_input_queue_table_name(const std::string &input_queue_table_name);

    std::string get_input_queue_column_name() const;

    void set_input_queue_column_name(const std::string &input_queue_column_name);

    double get_scaling_factor() const;

    void set_scaling_factor(const double scaling_factor);

    double get_dc_offset() const;

    void set_dc_offset(const double dc_offset);

    virtual std::string to_string() const override;
};


template<typename T> inline std::basic_ostream<T> &
operator<<(std::basic_ostream<T> &s, const IQScalingFactor &i)
{
    return s << i.to_string();
}

using IQScalingFactor_ptr = std::shared_ptr<svr::datamodel::IQScalingFactor>;

} // namespace datamodel
} // namespace svr
