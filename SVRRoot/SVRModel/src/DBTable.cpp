#include <algorithm>
#include "common/parallelism.hpp"
#include "model/DBTable.hpp"
#include "util/math_utils.hpp"


namespace svr {
namespace datamodel {


Queue::Queue(const Queue &rhs) : Entity(rhs), table_name_(rhs.table_name_), data_(rhs.data_)
{
#ifdef ENTITY_INIT_ID
    init_id();
#endif
}


Queue::Queue(const std::string &table_name, const data_row_container &data) :
        table_name_(table_name), data_(data)
{
#ifdef ENTITY_INIT_ID
    init_id();
#endif
}

void Queue::init_id()
{
    if (!id) boost::hash_combine(id, table_name_);
}

std::string Queue::get_table_name() const
{
    return table_name_;
}

void Queue::set_table_name(const std::string &table_name)
{
    table_name_ = table_name;
    id = 0;
    boost::hash_combine(id, table_name_);
}

const DataRow::container &Queue::get_data() const
{
    return data_;
}

DataRow::container &Queue::get_data()
{
    return data_;
}

Queue::operator datarow_range()
{
    return datarow_range(data_);
}

Queue::operator datarow_crange() const
{
    return datarow_crange(data_);
}

Queue::operator data_row_container &()
{
    return data_;
}

Queue::operator const data_row_container &() const
{
    return data_;
}

void Queue::set_data(const DataRow::container &data)
{
    data_ = data;
}

size_t Queue::size() const
{
    return data_.size();
}

bool Queue::empty() const
{
    return data_.empty();
}

datamodel::DataRow_ptr Queue::operator[](const size_t i) const
{
    return data_.at(i);
}

datamodel::DataRow_ptr &Queue::operator[](const size_t i)
{
    return data_[i];
}

datamodel::DataRow_ptr Queue::at(const size_t i) const
{
    return data_.at(i);
}

datamodel::DataRow_ptr &Queue::at(const size_t i)
{
    return data_[i];
}

std::string Queue::to_string() const
{
    return "Table name " + table_name_ + ", " + metadata_to_string() + ", " + data_to_string();
}

datamodel::DataRow_ptr Queue::front() const
{
    return data_.front();
}

datamodel::DataRow_ptr &Queue::front()
{
    return data_.front();
}

datamodel::DataRow_ptr Queue::back() const
{
    return data_.back();
}

datamodel::DataRow_ptr &Queue::back()
{
    return data_.back();
}

DataRow::container::const_iterator Queue::cbegin() const
{
    return data_.cbegin();
}

DataRow::container::iterator Queue::begin()
{
    return data_.begin();
}

DataRow::container::const_iterator Queue::cend() const
{
    return data_.cend();
}

DataRow::container::iterator Queue::end()
{
    return data_.end();
}

DataRow::container Queue::get_data(const size_t row_count, const bpt::ptime &time_to) const
{
    LOG4_BEGIN();
    DataRow::container new_data;
    auto end_iter = lower_bound(this->data_, time_to);
    auto start_iter = end_iter;
    const auto distance_lag = std::distance(data_.begin(), start_iter);
    if (distance_lag < (dtype(distance_lag)) row_count)
        THROW_EX_FS(std::runtime_error, "Missing data, distance from begin is " << distance_lag);
    std::advance(start_iter, -row_count);
    for (; start_iter != end_iter; ++start_iter) new_data.emplace_back(*start_iter);
    LOG4_END();
    return new_data;
}

DataRow::container Queue::get_data(const size_t tail_length, const bpt::time_period &range) const
{
    LOG4_BEGIN();
    DataRow::container new_data;
    auto end_iter = upper_bound(data_, range.end());
    auto start_iter = lower_bound(data_, range.begin());
    const auto distance_lag = std::distance(data_.begin(), start_iter);
    if (distance_lag < (dtype(distance_lag)) tail_length)
        THROW_EX_FS(std::runtime_error, "Missing data, distance from begin is " << distance_lag);
    std::advance(start_iter, -tail_length);
    for (; start_iter != end_iter; ++start_iter) new_data.emplace_back(*start_iter);
    LOG4_END();
    return new_data;
}

DataRow::container Queue::get_data(const bpt::time_period &range) const
{
    if (data_.empty()) {
        LOG4_DEBUG("Empty data!");
        return {};
    }
    LOG4_DEBUG("Looking for range " << range << " in data from " << data_.front()->get_value_time() << " until " << data_.back()->get_value_time());
    DataRow::container new_data;
    auto iter = lower_bound(data_, range.begin());
    if (iter == data_.cend())
        THROW_EX_F(std::runtime_error,
                   "Could not find " << range.begin() << " in data from " << data_.front()->get_value_time() << " until " << data_.back()->get_value_time());
    const auto end_iter = upper_bound(data_, range.end());
UNROLL()
    for (; iter != end_iter; ++iter) new_data.emplace_back(*iter);
    LOG4_DEBUG("For range " << range << " found data from " << new_data.front()->get_value_time() << " until " << new_data.back()->get_value_time());
    return new_data;
}

std::deque<double>
Queue::get_column_values(
        const size_t column_index,
        const boost::posix_time::ptime start_time,
        const boost::posix_time::ptime end_time) const
{
    std::deque<double> output_values;
    if (data_.empty() or data_.front()->get_values().size() <= column_index)
        THROW_EX_FS(std::invalid_argument, "No data for column index " << column_index);

    auto iter = lower_bound(data_, start_time);
    if (iter == data_.cend()) THROW_EX_F(std::runtime_error, "Row for time " << start_time << " not found.");
    const auto end_row_iter = upper_bound(data_, end_time);
UNROLL()
    for (; iter != end_row_iter; ++iter) output_values.emplace_back((**iter).get_value(column_index));
    return output_values;
}

std::deque<double> Queue::get_column_values(
        const size_t column_index,
        const size_t start_pos,
        const size_t count) const
{
    if (data_.empty() || (**data_.cbegin()).size() <= column_index || start_pos >= data_.size())
        THROW_EX_FS(std::invalid_argument, "No data for column index " << column_index << " or start position " << start_pos);

    const auto start_iter = data_.cbegin() + start_pos;
    const size_t dist_start_end = data_.cend() - start_iter;
    const auto count_limited = std::min<size_t>(count, dist_start_end);
    std::deque<double> res(count_limited);
    __par_iter(start_iter, count_limited, res[_IX] = (**_ITER)[column_index]; if (!std::isnormal(res[_IX])) res[_IX] = 0.0; );
    return res;
}

bool Queue::get_column_values(
        const size_t column_index,
        std::deque<double> &output_values,
        const bpt::time_period &range,
        const bpt::time_duration &resolution)
{
    if (data_.empty() or data_.front()->get_values().size() <= column_index) {
        LOG4_WARN("No data for column index " << column_index);
        return false;
    }

    const auto iter_start = lower_bound(data_, range.begin());
    if (iter_start == data_.cend()) {
        LOG4_DEBUG("Couldn't find range " << range);
        return false;
    }
    auto row_iter = lower_bound(data_, range.begin());
    if (row_iter == data_.cend()) THROW_EX_F(std::runtime_error, "Row for time " << range.begin() << " not found.");
    const auto end_row_iter = upper_bound(data_, range.end());
UNROLL()
    for (; row_iter != end_row_iter; ++row_iter) output_values.emplace_back((**row_iter).get_value(column_index));
    return true;
}

void Queue::trim(const bpt::ptime &start_time, const bpt::ptime &end_time)
{
    const auto from_iter = lower_bound(data_, start_time);
    data_.erase(data_.cbegin(), from_iter);
    data_.erase(upper_bound(data_, end_time), data_.cend());
}

void Queue::trim(const ssize_t lag, const bpt::ptime &start_time, const bpt::ptime &end_time)
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

void Queue::trim(const bpt::time_period &time_range)
{
    trim(time_range.begin(), time_range.end());
}

std::deque<double> Queue::get_tick_volume() const
{
    std::deque<double> result(data_.size());
    OMP_FOR(data_.size())
    for (size_t i = 0; i < data_.size(); ++i) result[i] = data_[i]->get_tick_volume();
    return result;
}

void Queue::update_data(const DataRow::container &new_data, const bool overwrite /* default true */)
{
    LOG4_BEGIN();

    if (!data_.empty()
        && !new_data.empty()
        && new_data.front()->get_value_time() < data_.back()->get_value_time())
        data_.erase(lower_bound_back(data_, new_data.front()->get_value_time()), data_.cend());

    data_.insert(data_.cend(), new_data.cbegin(), new_data.cend());

    LOG4_END();
}


std::string Queue::data_to_string() const
{
    std::stringstream s;
UNROLL()
    for (size_t i = 0; i < std::min<size_t>(DISPLAY_ROWS_LIMIT, data_.size()); ++i) {
        if (i == DISPLAY_ROWS_LIMIT - 1) {
            s << ". . . " << (size() - DISPLAY_ROWS_LIMIT) << " more" << std::endl;
        } else
            s << *data_[i] << "\n\t";
    }
    return s.str();
}


}
}
