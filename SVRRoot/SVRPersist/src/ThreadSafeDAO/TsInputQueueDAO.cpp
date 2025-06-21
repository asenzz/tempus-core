
#include "TsInputQueueDAO.hpp"
#include <boost/optional.hpp>

namespace svr {
namespace dao {


DEFINE_THREADSAFE_DAO_CONSTRUCTOR (TsInputQueueDAO, InputQueueDAO)
{}

std::deque<datamodel::InputQueue_ptr> TsInputQueueDAO::get_all_user_queues(const std::string &user_name)
{
    return ts_call<std::deque<datamodel::InputQueue_ptr>>(&InputQueueDAO::get_all_user_queues, user_name);
}


std::deque<datamodel::InputQueue_ptr> TsInputQueueDAO::get_all_queues_with_sign(bool uses_fix_connection)
{
    return ts_call<std::deque<datamodel::InputQueue_ptr>>(&InputQueueDAO::get_all_queues_with_sign, uses_fix_connection);
}


datamodel::InputQueue_ptr TsInputQueueDAO::get_queue_metadata(const std::string &user_name, const std::string &logical_name,
                                                              const bpt::time_duration &resolution)
{
    const std::scoped_lock scope_guard(mutex);
    return dao->get_queue_metadata(user_name, logical_name, resolution);
}


datamodel::InputQueue_ptr TsInputQueueDAO::get_queue_metadata(const std::string &table_name)
{
    const std::scoped_lock scope_guard(mutex);
    return dao->get_queue_metadata(table_name);
}


std::deque<datamodel::DataRow_ptr> TsInputQueueDAO::get_queue_data_by_table_name(
        const std::string &table_name, const bpt::ptime &time_from, const bpt::ptime &time_to, const size_t limit)
{
    return ts_call<std::deque<datamodel::DataRow_ptr>>(&InputQueueDAO::get_queue_data_by_table_name, table_name, time_from, time_to, limit);
}


std::deque<datamodel::DataRow_ptr> TsInputQueueDAO::get_latest_queue_data_by_table_name(const std::string &table_name, const size_t limit, const bpt::ptime &last_time)
{
    return ts_call<std::deque<datamodel::DataRow_ptr>>(&InputQueueDAO::get_latest_queue_data_by_table_name, table_name, limit, last_time);
}

datamodel::DataRow_ptr TsInputQueueDAO::get_nth_last_row(const std::string &table_name, const size_t position, const bpt::ptime target_time)
{
    return ts_call<datamodel::DataRow_ptr>(&InputQueueDAO::get_nth_last_row, table_name, position, target_time);
}

size_t TsInputQueueDAO::get_count_from_start(const std::string &table_name, const bpt::ptime target_time)
{
    return ts_call<size_t>(&InputQueueDAO::get_count_from_start, table_name, target_time);
}

size_t TsInputQueueDAO::save(const datamodel::InputQueue_ptr &input_queue, const bpt::ptime &start_time)
{
    return ts_call<size_t>(&InputQueueDAO::save, input_queue, start_time);
}


size_t TsInputQueueDAO::save_data(const datamodel::InputQueue_ptr &input_queue, const bpt::ptime &start_time)
{
    return ts_call<size_t>(&InputQueueDAO::save_data, input_queue, start_time);
}


size_t TsInputQueueDAO::save_metadata(const datamodel::InputQueue_ptr &input_queue)
{
    return ts_call<size_t>(&InputQueueDAO::save_metadata, input_queue);
}


size_t TsInputQueueDAO::update_metadata(const datamodel::InputQueue_ptr &input_queue)
{
    return ts_call<size_t>(&InputQueueDAO::update_metadata, input_queue);
}


size_t TsInputQueueDAO::remove(const datamodel::InputQueue_ptr &input_queue)
{
    return ts_call<size_t>(&InputQueueDAO::remove, input_queue);
}


size_t TsInputQueueDAO::clear(const datamodel::InputQueue_ptr &input_queue)
{
    return ts_call<size_t>(&InputQueueDAO::clear, input_queue);
}

bool TsInputQueueDAO::exists(const std::string &table_name)
{
    const std::scoped_lock scope_guard(mutex);
    return dao->exists(table_name);
}


bool TsInputQueueDAO::exists(const datamodel::InputQueue_ptr &input_queue)
{
    const std::scoped_lock scope_guard(mutex);
    return dao->exists(input_queue);
}

void TsInputQueueDAO::upsert_row_str(CRPTR(char) table_name, CRPTR(char) value_time, CRPTR(char) update_time, CRPTR(char) volume, CRPTR(char *) values, const uint16_t n_values)
{
    ts_call<void>(&InputQueueDAO::upsert_row_str, table_name, value_time, update_time, volume, values, n_values);
}

bool TsInputQueueDAO::exists(const std::string &user_name, const std::string &logical_name, const bpt::time_duration &resolution)
{
    const std::scoped_lock scope_guard(mutex);
    return dao->exists(user_name, logical_name, resolution);
}

datamodel::DataRow_ptr TsInputQueueDAO::find_oldest_record(const datamodel::InputQueue_ptr &queue)
{
    return ts_call<datamodel::DataRow_ptr>(&InputQueueDAO::find_oldest_record, queue);
}


datamodel::DataRow_ptr TsInputQueueDAO::find_newest_record(const datamodel::InputQueue_ptr &queue)
{
    return ts_call<datamodel::DataRow_ptr>(&InputQueueDAO::find_newest_record, queue);
}


std::deque<std::shared_ptr<std::string>> TsInputQueueDAO::get_db_table_column_names(const datamodel::InputQueue_ptr &queue)
{
    return ts_call<std::deque<std::shared_ptr<std::string>>>(&InputQueueDAO::get_db_table_column_names, queue);
}


OptionalTimeRange TsInputQueueDAO::get_missing_hours(datamodel::InputQueue_ptr const &input_queue, TimeRange const &time_range)
{
    return ts_call<OptionalTimeRange>(&InputQueueDAO::get_missing_hours, input_queue, time_range);
}


void TsInputQueueDAO::purge_missing_hours(datamodel::InputQueue_ptr const &queue)
{
    return ts_call<void>(&InputQueueDAO::purge_missing_hours, queue);
}


}
}
