#include <model/InputQueue.hpp>
#include "InputQueueService.hpp"
#include "util/TimeUtils.hpp"

namespace svr {
namespace datamodel {


InputQueue::InputQueue(
        const std::string &table_name,
        const std::string &logical_name,
        const std::string &owner_user_name,
        const std::string &description,
        const bpt::time_duration &resolution,
        const bpt::time_duration &legal_time_deviation,
        const std::string &time_zone,
        const std::deque<std::string> &value_columns,
        const bool uses_fix_connection,
        const data_row_container &rows)
        :
        Queue(business::InputQueueService::make_queue_table_name(owner_user_name, logical_name, resolution), rows),
        logical_name_(logical_name),
        owner_user_name_(owner_user_name),
        description_(description),
        resolution_(resolution),
        legal_time_deviation_(legal_time_deviation),
        time_zone_(time_zone),
        value_columns_(value_columns),
        uses_fix_connection(uses_fix_connection)
{
}


InputQueue InputQueue::get_copy_metadata() const
{
    return InputQueue(
            get_table_name(), logical_name_, owner_user_name_, description_, resolution_,
            legal_time_deviation_, time_zone_, value_columns_, uses_fix_connection);
}


datamodel::InputQueue_ptr InputQueue::clone_empty() const
{
    return std::make_shared<InputQueue>(
            get_table_name(), logical_name_, owner_user_name_, description_, resolution_,
            legal_time_deviation_, time_zone_, value_columns_);
}

const std::string &InputQueue::get_description() const
{ return description_; }

void InputQueue::set_description(const std::string &description)
{ this->description_ = description; }

const bpt::time_duration &InputQueue::get_legal_time_deviation() const
{ return legal_time_deviation_; }

void InputQueue::set_legal_time_deviation(const bpt::time_duration &legalTimeDeviation)
{ legal_time_deviation_ = legalTimeDeviation; }

const std::string &InputQueue::get_logical_name() const
{ return logical_name_; }

const std::string &InputQueue::get_owner_user_name() const
{ return owner_user_name_; }

const bpt::time_duration &InputQueue::get_resolution() const
{ return resolution_; }

const std::string &InputQueue::get_time_zone() const
{ return time_zone_; }

void InputQueue::set_time_zone(const std::string &time_zone)
{ time_zone_ = time_zone; }

const std::string InputQueue::get_value_column(const size_t i) const
{ return value_columns_[i]; }

const std::deque<std::string> &InputQueue::get_value_columns() const
{ return value_columns_; }

void InputQueue::set_value_columns(const std::deque<std::string> &value_columns)
{ value_columns_ = value_columns; }

const bool InputQueue::is_tick_queue() const
{ return resolution_ < onesec; }

bpt::time_duration const &InputQueue::get_missing_hours_retention() const
{
    static auto const weeks2 = bpt::hours(24 * 14);
    return weeks2;
}

size_t InputQueue::get_value_column_index(const std::string &column_name) const
{
    const auto pos = find(value_columns_.begin(), value_columns_.end(), column_name);
    if (pos == value_columns_.end())
        THROW_EX_FS(
                std::invalid_argument, "Column " << column_name << " is not part of input queue " << table_name_);
    return std::distance(value_columns_.begin(), pos);
}

std::deque<double> InputQueue::get_column_values(const std::string &column_name, const size_t start_pos, const size_t count) const
{
    return Queue::get_column_values(get_value_column_index(column_name), start_pos, count);
}

void InputQueue::set_logical_name(const std::string &logical_name)
{
    logical_name_ = logical_name;
    reinit_table_name();
}


void InputQueue::set_owner_user_name(const std::string &owner_user_name)
{
    owner_user_name_ = owner_user_name;
    reinit_table_name();
}


void InputQueue::set_resolution(const bpt::time_duration &resolution)
{
    this->resolution_ = resolution;
    reinit_table_name();
}


void InputQueue::reinit_table_name()
{
    set_table_name(business::InputQueueService::make_queue_table_name(get_owner_user_name(), get_logical_name(), get_resolution()));
}


std::string InputQueue::metadata_to_string() const
{
    std::stringstream ss;
    ss << "Table name " << this->table_name_
       << " logical name " << this->logical_name_
       << " owner user name " << this->owner_user_name_
       << " description " << this->description_
       << " resolution " << this->resolution_
       << " legal time deviation " << this->legal_time_deviation_
       << " time zone " << this->time_zone_
       << " columns ";

    for (const std::string &column: this->value_columns_)
        ss << ", " << column;

    return ss.str();
}

bool InputQueue::get_uses_fix_connection() const
{
    return uses_fix_connection;
}

void InputQueue::set_uses_fix_connection(bool value)
{
    uses_fix_connection = value;
}


} // namespace svr
} //namespace datamodel
