#pragma once

#include <string>
#include <memory>
#include "DataRow.hpp"
#include "common/parallelism.hpp"

#define DISPLAY_ROWS_LIMIT (5)


namespace svr {
namespace datamodel {

class Queue
{
protected:
    std::string table_name_;
    data_row_container data_; // TODO Replace with pointer to data_row_container

public:
    Queue() = default;

    Queue(const Queue &rhs);

    explicit Queue(const std::string &table_name, const data_row_container &data = data_row_container());

    inline const std::string &get_table_name() const
    { return table_name_; }

    void set_table_name(const std::string &tableName);

    virtual std::string to_string() const;

    virtual std::string data_to_string() const;

    virtual std::string metadata_to_string() const = 0;

    virtual void update_data(const DataRow::container &new_data, const bool overwrite = true);

    virtual const DataRow::container &get_data() const
    { return data_; }

    virtual DataRow::container &get_data()
    { return data_; }

    inline void set_data(const DataRow::container &data)
    { this->data_ = data; }


    inline DataRow::container get_data(const size_t row_count, const bpt::ptime &time_to) const
    {
        LOG4_BEGIN();
        DataRow::container new_data;
        auto end_iter = lower_bound(this->data_, time_to);
        auto start_iter = end_iter;
        const auto distance_lag = std::distance(data_.begin(), start_iter);
        if (distance_lag < (decltype(distance_lag)) row_count)
            THROW_EX_FS(std::runtime_error, "Missing data, distance from begin is " << distance_lag);
        std::advance(start_iter, -row_count);
        for (;start_iter != end_iter; ++start_iter) new_data.emplace_back(*start_iter);
        LOG4_END();
        return new_data;
    }

    inline DataRow::container get_data(const size_t tail_length, const bpt::time_period &range) const
    {
        LOG4_BEGIN();
        DataRow::container new_data;
        auto end_iter = upper_bound(data_, range.end());
        auto start_iter = lower_bound(data_, range.begin());
        const auto distance_lag = std::distance(data_.begin(), start_iter);
        if (distance_lag < (decltype(distance_lag)) tail_length)
            THROW_EX_FS(std::runtime_error, "Missing data, distance from begin is " << distance_lag);
        std::advance(start_iter, -tail_length);
        for (;start_iter != end_iter; ++start_iter) new_data.emplace_back(*start_iter);
        LOG4_END();
        return new_data;
    }

    virtual DataRow::container get_data(const bpt::time_period &range) const
    {
        if (data_.empty()) {
            LOG4_DEBUG("Empty data!");
            return {};
        }
        LOG4_DEBUG("Looking for range " << range << " in data from " << data_.begin()->get()->get_value_time() << " until " << data_.rbegin()->get()->get_value_time());
        DataRow::container new_data;
        auto start_iter = lower_bound(data_, range.begin());
        if (start_iter == data_.end())
            THROW_EX_F(std::runtime_error, "Could not find " << range.begin() << " in data from " << data_.begin()->get()->get_value_time() << " until " << data_.rbegin()->get()->get_value_time());
        const auto end_iter = upper_bound(data_, range.end());
        for (;start_iter != end_iter; ++start_iter)
            new_data.emplace_back(*start_iter);
        LOG4_DEBUG("For range " << range << " found data from " << new_data.begin()->get()->get_value_time() << " until " << new_data.rbegin()->get()->get_value_time());
        return new_data;
    }

    std::deque<double>
    get_column_values(
            const size_t column_index,
            const boost::posix_time::ptime start_time = boost::posix_time::min_date_time,
            const boost::posix_time::ptime end_time = boost::posix_time::max_date_time) const
    {
        std::deque<double> output_values;
        if (data_.empty() or data_.begin()->get()->get_values().size() <= column_index)
            THROW_EX_FS(std::invalid_argument, "No data for column index " << column_index);

        auto row_iter = lower_bound(data_, start_time);
        if (row_iter == data_.end()) THROW_EX_F(std::runtime_error, "Row for time " << start_time << " not found.");
        const auto end_row_iter = upper_bound(data_, end_time);
        for (; row_iter != end_row_iter; ++row_iter) {
            const auto value = row_iter->get()->get_value(column_index);
            output_values.emplace_back(value);
        }
        return output_values;
    }

    std::deque<double> get_column_values(
            const size_t &column_index,
            const size_t start_pos = 0,
            const size_t count = std::numeric_limits<size_t>::max()) const
    {
        if (data_.empty() or data_.begin()->get()->get_values().size() <= column_index)
            THROW_EX_FS(std::invalid_argument, "No data for column index " << column_index);

        auto start_iter = std::next(data_.begin(), start_pos);
        const size_t dist_start_end = std::distance(start_iter, data_.end());
        const size_t count_limited = count < dist_start_end ? count : dist_start_end;
        std::deque<double> output_values(count_limited);
        __par_iter(start_iter, count_limited,
            output_values[_IX] = _ITER->get()->get_value(column_index));
        return output_values;
    }

    bool get_column_values(
            const size_t column_index,
            std::deque<double> &output_values,
            const bpt::time_period &range,
            const bpt::time_duration &resolution)
    {
        if (data_.empty() or data_.begin()->get()->get_values().size() <= column_index) {
            LOG4_WARN("No data for column index " << column_index);
            return false;
        }

        const auto iter_start = lower_bound(data_, range.begin());
        if (iter_start == this->data_.end()) {
            LOG4_DEBUG("Couldn't find range " << range);
            return false;
        }
        auto row_iter = lower_bound(data_, range.begin());
        if (row_iter == data_.end()) THROW_EX_F(std::runtime_error, "Row for time " << range.begin() << " not found.");
        const auto end_row_iter = upper_bound(data_, range.end());
        for (; row_iter != end_row_iter; ++row_iter) {
            const auto value = row_iter->get()->get_value(column_index);
            output_values.emplace_back(value);
        }
        return true;
    }

    // TODO Override in DeconQueue and reset anchor properly
    inline void trim(const bpt::ptime &start_time, const bpt::ptime &end_time = bpt::max_date_time)
    {
        auto from_iter = lower_bound(data_, start_time);
        data_.erase(data_.begin(), from_iter);
        data_.erase(upper_bound(data_, end_time), data_.end());
    }

    void trim(const ssize_t lag, const bpt::ptime &start_time, const bpt::ptime &end_time = bpt::max_date_time)
    {
        auto from_iter = lower_bound(data_, start_time);
        std::advance(from_iter, -lag);
#ifdef ALL_VALUES_DELTA
	    while (from_iter->second->is_anchor()) ++from_iter;
	    --from_iter;
#endif
        data_.erase(data_.begin(), from_iter);
        data_.erase(upper_bound(data_, end_time), data_.end());
    }

    inline void trim(const bpt::time_period &time_range)
    {
        trim(time_range.begin(), time_range.end());
    }

    inline std::deque<double> get_tick_volume() const
    {
        std::deque<double> result;
        for (const auto &row: data_) result.emplace_back(row.get()->get_tick_volume());
        return result;
    }

};

} /* namespace model */
} /* namespace svr */
