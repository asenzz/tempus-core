#pragma once

#include <util/string_utils.hpp>
#include "model/Entity.hpp"

namespace svr {
namespace datamodel {

class DecrementTask : public Entity
{
private:
    bigint dataset_id_;

    bpt::ptime start_task_time_;
    bpt::ptime end_task_time_;

    bpt::ptime start_train_time_;
    bpt::ptime end_train_time_;
    bpt::ptime start_validation_time_;
    bpt::ptime end_validation_time_;

    std::string parameters_; // json
    size_t status_; // enum
    std::string decrement_step_;

    size_t vp_sliding_direction_;
    size_t vp_slide_count_;
    bpt::seconds vp_slide_period_sec_;

    std::string values_; // json
    std::string suggested_value_;

public:
    DecrementTask() :
            Entity(),
            vp_slide_period_sec_(bpt::seconds(0))
    {}

    DecrementTask(
            const bigint id,
            const bigint dataset_id,
            const bpt::ptime &start_task_time,
            const bpt::ptime &end_task_time,
            const bpt::ptime &start_train_time,
            const bpt::ptime &end_train_time,
            const bpt::ptime &start_validation_time,
            const bpt::ptime &end_validation_time,
            const std::string &parameters,
            const size_t status,
            const std::string &decrement_step,
            size_t vp_sliding_direction,
            size_t vp_slide_count,
            const bpt::seconds &vp_slide_period_sec,
            const std::string &values = "{}",
            const std::string &suggested_value = "{}") :
            Entity(id),
            dataset_id_(dataset_id),
            start_task_time_(start_task_time),
            end_task_time_(end_task_time),
            start_train_time_(start_train_time),
            end_train_time_(end_train_time),
            start_validation_time_(start_validation_time),
            end_validation_time_(end_validation_time),
            parameters_(parameters),
            status_(status),
            decrement_step_(decrement_step),
            vp_sliding_direction_(vp_sliding_direction),
            vp_slide_count_(vp_slide_count),
            vp_slide_period_sec_(vp_slide_period_sec),
            values_(values),
            suggested_value_(suggested_value)
    {}

    bool operator==(const DecrementTask &other) const
    {
        return other.get_id() == get_id() &&
               other.get_dataset_id() == get_dataset_id() &&
               other.get_start_task_time() == get_start_task_time() &&
               other.get_end_task_time() == get_end_task_time() &&
               other.get_start_train_time() == get_start_train_time() &&
               other.get_end_train_time() == get_end_train_time() &&
               other.get_start_validation_time() == get_start_validation_time() &&
               other.get_end_validation_time() == get_end_validation_time() &&
               other.get_parameters() == get_parameters() &&
               other.get_status() == get_status() &&
               other.get_decrement_step() == get_decrement_step() &&
               other.get_values() == get_values() &&
               other.get_suggested_value() == get_suggested_value();
    }

    bigint get_dataset_id() const
    {
        return dataset_id_;
    }

    void set_dataset_id(const bigint dataset_id)
    {
        dataset_id_ = dataset_id;
    }

    // task time
    bpt::ptime get_start_task_time() const
    {
        return start_task_time_;
    }

    void set_start_task_time(const bpt::ptime &start_task_time)
    {
        start_task_time_ = start_task_time;
    }

    bpt::ptime get_end_task_time() const
    {
        return end_task_time_;
    }

    void set_end_task_time(const bpt::ptime &end_task_time)
    {
        end_task_time_ = end_task_time;
    }

    // train time
    bpt::ptime get_start_train_time() const
    {
        return start_train_time_;
    }

    void set_start_train_time(const bpt::ptime &start_train_time)
    {
        start_train_time_ = start_train_time;
    }

    bpt::ptime get_end_train_time() const
    {
        return end_train_time_;
    }

    void set_end_train_time(const bpt::ptime &end_train_time)
    {
        end_train_time_ = end_train_time;
    }

    // validation time
    bpt::ptime get_start_validation_time() const
    {
        return start_validation_time_;
    }

    void set_start_validation_time(const bpt::ptime &start_validation_time)
    {
        start_validation_time_ = start_validation_time;
    }

    bpt::ptime get_end_validation_time() const
    {
        return end_validation_time_;
    }

    void set_end_validation_time(const bpt::ptime &end_validation_time)
    {
        end_validation_time_ = end_validation_time;
    }

    // parameters, status, values
    std::string get_parameters() const
    {
        return parameters_;
    }

    void set_parameters(const std::string parameters)
    {
        parameters_ = parameters;
    }

    size_t get_status() const
    {
        return status_;
    }

    void set_status(const size_t status)
    {
        status_ = status;
    }

    std::string get_decrement_step() const
    {
        return decrement_step_;
    }

    void set_decrement_step(const std::string &decrement_step)
    {
        decrement_step_ = decrement_step;
    }

    size_t get_vp_slide_count() const
    {
        return vp_slide_count_;
    }

    void set_vp_slide_count(size_t vp_slide_count)
    {
        vp_slide_count_ = vp_slide_count;
    }

    bpt::seconds get_vp_slide_period_sec() const
    {
        return vp_slide_period_sec_;
    }

    void set_vp_slide_period_sec(const bpt::seconds &vp_slide_period_sec)
    {
        vp_slide_period_sec_ = vp_slide_period_sec;
    }

    size_t get_vp_sliding_direction() const
    {
        return vp_sliding_direction_;
    }

    void set_vp_sliding_direction(size_t vp_sliding_direction)
    {
        vp_sliding_direction_ = vp_sliding_direction;
    }

    std::string get_values() const
    {
        return values_;
    }

    void set_values(const std::string values)
    {
        values_ = values;
    }

    std::string get_suggested_value() const
    {
        return suggested_value_;
    }

    void set_suggested_value(const std::string &suggested_value)
    {
        suggested_value_ = suggested_value;
    }

    virtual std::string to_string() const override
    {
        std::stringstream ss;
        ss.precision(std::numeric_limits<double>::max_digits10);
        std::string sep{", "};

        ss << "Decrement task id: " << get_id() << sep <<
           "Dataset id: " << get_dataset_id() << sep <<
           "Start task time: " << get_start_task_time() << sep <<
           "End task time: " << get_end_task_time() << sep <<
           "Start train time: " << get_start_train_time() << sep <<
           "End train time: " << get_end_train_time() << sep <<
           "Start validation time: " << get_start_validation_time() << sep <<
           "End validation time: " << get_end_validation_time() << sep <<
           "Parameters: " << get_parameters() << sep <<
           "Status: " << get_status() << sep <<
           "Decrement step: " << get_decrement_step() << sep <<
           "Values: " << get_values() <<
           "Suggested value: " << get_suggested_value();

        return ss.str();
    }

    std::string values_to_string(const std::vector<double> &values, const std::string &separator = ", ") const
    {
        std::string result;

        for (auto value:values)
            result.append(common::to_string_with_precision(value) + separator);

        // remove last delimiter
        if (!result.empty())
            result.resize(result.size() - separator.size());

        return result;
    }

    template<typename T>
    std::string values_to_string(const std::vector<T> &values, const std::string &separator = ", ") const
    {
        std::string result;

        for (auto value:values)
            result.append(std::to_string(value) + separator);

        // remove last delimiter
        if (!result.empty())
            result.resize(result.size() - separator.size());

        return result;
    }

};

} // namespace datamodel
} // namespace svr

using DecrementTask_ptr = std::shared_ptr<svr::datamodel::DecrementTask>;
