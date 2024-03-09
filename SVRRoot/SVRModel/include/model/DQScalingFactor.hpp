#pragma once

#include <execution>
#include </opt/intel/oneapi/tbb/latest/include/tbb/concurrent_set.h>
#include "model/Entity.hpp"
#include "util/string_utils.hpp"


namespace svr {
namespace datamodel {
class DQScalingFactor;

using DQScalingFactor_ptr = std::shared_ptr<DQScalingFactor>;
}
}

namespace svr {
namespace datamodel {

struct DQScalingFactorsLess;
typedef tbb::concurrent_set<DQScalingFactor_ptr, DQScalingFactorsLess> dq_scaling_factor_container_t;

class DQScalingFactor : public Entity
{
private:
    bigint dataset_id_ = 0;  // TODO Replace with pointer to Dataset

    std::string input_queue_table_name_; // TODO Replace with pointer to Input Queue
    std::string input_queue_column_name_; // TODO Replace with pointer to Input Queue
    size_t decon_level_ = DEFAULT_SVRPARAM_DECON_LEVEL;

    double scaling_factor_features = std::numeric_limits<double>::quiet_NaN();
    double scaling_factor_labels = std::numeric_limits<double>::quiet_NaN();
    double dc_offset_features = std::numeric_limits<double>::quiet_NaN();
    double dc_offset_labels = std::numeric_limits<double>::quiet_NaN();

public:
    DQScalingFactor(
            const bigint id, const bigint dataset_id, const std::string &input_queue_table_name, const std::string &input_queue_column_name,
            const size_t decon_level,
            const double scale_feat = std::numeric_limits<double>::quiet_NaN(),
            const double scale_labels = std::numeric_limits<double>::quiet_NaN(),
            const double dc_offset_feat = std::numeric_limits<double>::quiet_NaN(),
            const double dc_offset_labels = std::numeric_limits<double>::quiet_NaN());

    bool operator==(const DQScalingFactor &other) const;

    bool operator<(const DQScalingFactor &o) const;

    virtual void init_id() override;

    bigint get_dataset_id() const;

    void set_dataset_id(const bigint dataset_id);

    std::string get_input_queue_table_name() const;

    void set_input_queue_table_name(const std::string &input_queue_table_name);

    std::string get_input_queue_column_name() const;

    void set_input_queue_column_name(const std::string &input_queue_column_name);

    size_t get_decon_level() const;

    void set_decon_level(const size_t decon_level);

    double get_features_factor() const;

    void set_features_factor(const double scaling_factor);

    double get_labels_factor() const;

    void set_labels_factor(const double label_factor);

    double get_dc_offset_features() const;

    void set_dc_offset_features(const double dc_offset_features_);

    double get_dc_offset_labels() const;

    void set_dc_offset_labels(const double dc_offset_labels_);

    std::string to_string() const override;

    bool in(const dq_scaling_factor_container_t &c);
};

template<typename T> inline std::basic_ostream<T> &
operator<<(std::basic_ostream<T> &s, const DQScalingFactor &d)
{
    return s << d.to_string();
}


struct DQScalingFactorsLess
{
    bool operator()(const DQScalingFactor_ptr &lhs, const DQScalingFactor_ptr &rhs) const
    { return *lhs < *rhs; }
};

bool operator<(const DQScalingFactor_ptr &lhs, const DQScalingFactor_ptr &rhs);

template<typename T> inline std::basic_ostream<T> &
operator<<(std::basic_ostream<T> &s, const dq_scaling_factor_container_t &c)
{
    return s << common::to_string(c);
}


} // namespace datamodel
} // namespace svr
