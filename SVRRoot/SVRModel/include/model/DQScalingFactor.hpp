#pragma once

#include <set>
#include <execution>
#include <algorithm>
#include </opt/intel/oneapi/tbb/latest/include/tbb/concurrent_set.h>
#include "model/Entity.hpp"
#include "util/string_utils.hpp"


namespace svr { namespace datamodel { class DQScalingFactor; }}
using DQScalingFactor_ptr = std::shared_ptr<svr::datamodel::DQScalingFactor>;


namespace svr {
namespace datamodel {

struct DQScalingFactorsLess;
typedef tbb::concurrent_set<DQScalingFactor_ptr, DQScalingFactorsLess> dq_scaling_factor_container_t;

class DQScalingFactor: public Entity
{
private:
    bigint dataset_id_;  // TODO Replace with pointer to Dataset

    std::string input_queue_table_name_; // TODO Replace with pointer to Input Queue
    std::string input_queue_column_name_; // TODO Replace with pointer to Input Queue
    size_t decon_level_;

    double scaling_factor_features;
    double scaling_factor_labels;
public:
    DQScalingFactor(
            const bigint id, const bigint dataset_id, const std::string& input_queue_table_name, const std::string& input_queue_column_name,
                const size_t wavelet_level, const double scale_feat = 1.0, const double scale_labels = 1.0):
            Entity(id),
            dataset_id_(dataset_id),
            input_queue_table_name_(input_queue_table_name),
            input_queue_column_name_(input_queue_column_name),
            decon_level_(wavelet_level),
            scaling_factor_features(scale_feat),
            scaling_factor_labels(scale_labels)
    {}

    bool operator==(const DQScalingFactor& other) const;
    bool operator<(const DQScalingFactor &o) const;

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

    std::string get_input_queue_column_name() const
    {
        return input_queue_column_name_;
    }

    void set_input_queue_column_name(const std::string& input_queue_column_name)
    {
        input_queue_column_name_ = input_queue_column_name;
    }

    size_t get_decon_level() const
    {
        return decon_level_;
    }

    void set_decon_level(const size_t decon_level)
    {
        decon_level_ = decon_level;
    }

    double get_features_factor() const
    {
        return scaling_factor_features;
    }

    void set_features_factor(const double scaling_factor)
    {
        scaling_factor_features = scaling_factor;
    }

    double get_labels_factor() const
    {
        return scaling_factor_labels;
    }

    void set_labels_factor(const double label_factor)
    {
        scaling_factor_labels = label_factor;
    }

    std::string to_string() const override;

    bool in(const dq_scaling_factor_container_t &c);

    void add(dq_scaling_factor_container_t &sf, const bool overwrite = false);
    static void add(dq_scaling_factor_container_t &sf, const dq_scaling_factor_container_t &new_sf, const bool overwrite = false);
};

template<typename T> inline std::basic_ostream<T> &
operator<< (std::basic_ostream<T> &s, const DQScalingFactor &d)
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
