#pragma once

#include "model/Entity.hpp"
#include <set>


namespace svr { namespace datamodel { class DQScalingFactor; }}
using DQScalingFactor_ptr = std::shared_ptr<svr::datamodel::DQScalingFactor>;


namespace svr {
namespace datamodel {

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

    bool operator==(const DQScalingFactor& other) const
    {
        return other.get_id() == get_id() &&
                other.get_dataset_id() == get_dataset_id() &&
                other.get_input_queue_table_name() == get_input_queue_table_name() &&
                other.get_input_queue_column_name() == get_input_queue_column_name() &&
                other.get_decon_level() == get_decon_level() &&
                other.get_features_factor() == get_features_factor() &&
                other.get_labels_factor() == get_labels_factor() ;
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

    std::string to_string() const override
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

};

static inline std::stringstream &operator<< (std::stringstream &str, DQScalingFactor &dqsf)
{
    str << dqsf.to_string();
    return str;
}

struct DQScalingFactorsLess : public std::binary_function<DQScalingFactor_ptr, DQScalingFactor_ptr, bool>
{
    bool operator()(const DQScalingFactor_ptr &lhs, const DQScalingFactor_ptr &rhs) const
    {
        if( lhs->get_dataset_id() < rhs->get_dataset_id() )
            return true;
        if( lhs->get_dataset_id() > rhs->get_dataset_id() )
            return false;

        if( lhs->get_input_queue_table_name() < rhs->get_input_queue_table_name() )
            return true;
        if( lhs->get_input_queue_table_name() > rhs->get_input_queue_table_name() )
            return false;

        if( lhs->get_input_queue_column_name() < rhs->get_input_queue_column_name() )
            return true;
        if( lhs->get_input_queue_column_name() > rhs->get_input_queue_column_name() )
            return false;

        return lhs->get_decon_level() < rhs->get_decon_level();
    }
};

typedef std::set<DQScalingFactor_ptr, DQScalingFactorsLess> dq_scaling_factor_container_t;

} // namespace datamodel
} // namespace svr
