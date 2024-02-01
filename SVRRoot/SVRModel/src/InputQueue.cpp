#include <model/InputQueue.hpp>

namespace svr {
namespace datamodel {


InputQueue::InputQueue(
        const std::string& table_name,
        const std::string& logical_name,
        const std::string& owner_user_name,
        const std::string& description,
        const bpt::time_duration& resolution,
        const bpt::time_duration& legal_time_deviation,
        const std::string &time_zone,
        const std::deque<std::string> &value_columns,
        const bool uses_fix_connection,
        const data_row_container &rows)
    :
    Queue( make_queue_table_name( owner_user_name, logical_name, resolution ), rows),
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


void InputQueue::set_logical_name(const std::string& logical_name)
{
    logical_name_ = logical_name;
    reinit_table_name();
}


void InputQueue::set_owner_user_name(const std::string& owner_user_name)
{
    owner_user_name_ = owner_user_name;
    reinit_table_name();
}


void InputQueue::set_resolution(const bpt::time_duration& resolution)
{
    this->resolution_ = resolution;
    reinit_table_name();
}


void InputQueue::reinit_table_name()
{
    set_table_name(make_queue_table_name(get_owner_user_name(), get_logical_name(), get_resolution()));
}


std::string InputQueue::metadata_to_string() const {
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

std::string InputQueue::make_queue_table_name(const std::string &user_name, const std::string &logical_name, const bpt::time_duration &resolution)
{
    std::string result = svr::common::sanitize_db_table_name(
            svr::common::C_input_queue_table_name_prefix + "_" + user_name + "_" + logical_name + "_" + std::to_string(resolution.total_seconds()));
    std::transform(result.begin(), result.end(), result.begin(), ::tolower);
    LOG4_DEBUG("Returning " << result);
    return result;
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
