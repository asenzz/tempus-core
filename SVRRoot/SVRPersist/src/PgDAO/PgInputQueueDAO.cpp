#include "PgInputQueueDAO.hpp"
#include "InputQueueService.hpp"

#include <DAO/InputQueueRowRowMapper.hpp>
#include <DAO/DataRowRowMapper.hpp>
#include <model/InputQueue.hpp>
#include <DAO/InputQueueRowMapper.hpp>
#include <DAO/DataSource.hpp>

namespace svr {
namespace dao {

PgInputQueueDAO::PgInputQueueDAO(svr::common::PropertiesFileReader &sql_properties, svr::dao::DataSource &data_source)
        : InputQueueDAO(sql_properties, data_source)
{
}

datamodel::InputQueue_ptr PgInputQueueDAO::get_queue_metadata(const std::string &user_name, const std::string &logical_name, const bpt::time_duration &resolution)
{

    InputQueueRowMapper row_mapper;
    std::string sql = AbstractDAO::get_sql("get_queue_metadata");
    return data_source.query_for_object(&row_mapper, sql, user_name, logical_name, resolution);
}

datamodel::InputQueue_ptr PgInputQueueDAO::get_queue_metadata(const std::string &table_name)
{
    InputQueueRowMapper row_mapper;
    return data_source.query_for_object(&row_mapper, get_sql("get_queue_metadata_by_table_name"), table_name);
}

std::deque<datamodel::DataRow_ptr> PgInputQueueDAO::get_queue_data_by_table_name(
        const std::string &table_name,
        const bpt::ptime &time_from,
        const bpt::ptime &time_to,
        size_t limit)
{
    LOG4_DEBUG("Reading data from queue " << table_name);// << " from " << time_from << " to " << time_to);
    DataRowRowMapper row_mapper;

    std::string sql = (boost::format(AbstractDAO::get_sql("get_queue_data_by_table_name")) % table_name).str();

    if (limit) sql += " LIMIT " + pqxx::to_string(limit);

    return data_source.query_for_deque(row_mapper, sql, time_from, time_to, limit);
}

std::deque<datamodel::DataRow_ptr> PgInputQueueDAO::get_latest_queue_data_by_table_name(
        const std::string &table_name,
        const size_t limit,
        const bpt::ptime &last_time)
{
    LOG4_DEBUG("Reading data from queue " << table_name);
    DataRowRowMapper row_mapper;

    std::string sql = (boost::format(AbstractDAO::get_sql("get_latest_queue_data_by_table_name")) % table_name).str();
    return data_source.query_for_deque(row_mapper, sql, last_time, limit);
}

datamodel::DataRow_ptr PgInputQueueDAO::get_nth_last_row(
        const std::string &table_name,
        const size_t position,
        const bpt::ptime target_time)
{
    LOG4_DEBUG("Reading data from queue " << table_name);
    DataRowRowMapper row_mapper;

    std::string sql = (boost::format(AbstractDAO::get_sql("load_nth_last_row")) % table_name).str();

    return data_source.query_for_object(&row_mapper, sql, target_time, position);
}

size_t PgInputQueueDAO::get_count_from_start(const std::string &table_name, const bpt::ptime target_time)
{
    LOG4_DEBUG("Reading data from queue " << table_name);
    std::string sql = (boost::format(AbstractDAO::get_sql("get_count_from_start")) % table_name).str();
    return data_source.query_for_type<long>(sql, target_time);
}

size_t PgInputQueueDAO::save_metadata(const datamodel::InputQueue_ptr &p_input_queue)
{
    std::string sql = AbstractDAO::get_sql("create_queue_table");
    std::string table_name = p_input_queue->get_table_name();
    p_input_queue->set_table_name(table_name);

    LOG4_DEBUG("Saving InputQueue with table name: " << table_name);
    std::string value_column_sql;

    for (const std::string &columnName: p_input_queue->get_value_columns())
        value_column_sql += "\"" + columnName + "\" double precision DEFAULT 0, ";

    boost::format sql_format(sql);
    sql_format % table_name % value_column_sql % table_name;

    sql = sql_format.str();
    LOG4_TRACE("CREATE TABLE SQL: " << sql);

    int ret;
    { // transaction
        scoped_transaction_guard_ptr trx = data_source.open_transaction();

        data_source.update(sql);

        p_input_queue->set_table_name(table_name);

        sql = AbstractDAO::get_sql("save_metadata");

        ret = data_source.update(sql,
                                 table_name,
                                 p_input_queue->get_logical_name(),
                                 p_input_queue->get_owner_user_name(),
                                 p_input_queue->get_description(),
                                 p_input_queue->get_resolution(),
                                 p_input_queue->get_legal_time_deviation(),
                                 p_input_queue->get_time_zone(),
                                 p_input_queue->get_value_columns(),
                                 p_input_queue->get_missing_hours_retention(),
                                 p_input_queue->get_uses_fix_connection()
        );
    }

    return ret;
}

size_t PgInputQueueDAO::save(const datamodel::InputQueue_ptr &p_input_queue, const bpt::ptime &start_time)
{
    long ret = 0;

    bool exist = exists(p_input_queue);
    if (exist)
        ret = update_metadata(p_input_queue);
    else
        ret = save_metadata(p_input_queue);

    if (p_input_queue->size() > 0) ret = save_data(p_input_queue, start_time);

    return ret;
}

size_t PgInputQueueDAO::save_data(const datamodel::InputQueue_ptr &p_input_queue, const bpt::ptime &start_time)
{
    data_source.cleanup_queue_table(p_input_queue->get_table_name(), p_input_queue->get_data(), start_time);
    return data_source.batch_update(p_input_queue->get_table_name(), p_input_queue->get_data(), start_time);
}

size_t PgInputQueueDAO::update_metadata(const datamodel::InputQueue_ptr &inputQueue)
{

    LOG4_DEBUG("Updating InputQueue table metadata: " << inputQueue->get_table_name());
    std::string sql = AbstractDAO::get_sql("update_metadata");

    return data_source.update(sql,
                              inputQueue->get_logical_name(),
                              inputQueue->get_description(),
                              inputQueue->get_legal_time_deviation(),
                              inputQueue->get_time_zone(),
                              inputQueue->get_uses_fix_connection(),
                              inputQueue->get_table_name()
    );
}

bool PgInputQueueDAO::exists(const std::string &table_name)
{
    return 1 == data_source.query_for_type<long>(AbstractDAO::get_sql("exists"), table_name);
}

bool PgInputQueueDAO::exists(
        const std::string &user_name,
        const std::string &logical_name,
        const bpt::time_duration &resolution)
{
    return exists(business::InputQueueService::make_queue_table_name(user_name, logical_name, resolution));
}

bool PgInputQueueDAO::exists(const datamodel::InputQueue_ptr &p_input_queue)
{
    if (p_input_queue == nullptr) return false;
    std::string table_name = p_input_queue->get_table_name();
    p_input_queue->set_table_name(table_name);
    return exists(table_name);
}

size_t PgInputQueueDAO::remove(const datamodel::InputQueue_ptr &p_input_queue)
{
    std::string table_name = business::InputQueueService::make_queue_table_name(
            p_input_queue->get_owner_user_name(), p_input_queue->get_logical_name(), p_input_queue->get_resolution());

    LOG4_DEBUG("Removing InputQueue with table name " << table_name);

    std::string sql = get_sql("remove_queue_table");

    boost::format sql_format(sql);
    sql_format % table_name;
    sql = sql_format.str();

    data_source.update(sql, table_name);
    return data_source.update(get_sql("remove_queue_metadata"), table_name);

}

size_t PgInputQueueDAO::clear(const datamodel::InputQueue_ptr &p_input_queue)
{
    std::string table_name = business::InputQueueService::make_queue_table_name(
            p_input_queue->get_owner_user_name(), p_input_queue->get_logical_name(), p_input_queue->get_resolution());

    LOG4_DEBUG("Clear InputQueue with table name " << table_name);

    std::string sql = get_sql("clear_queue_table");

    boost::format sql_format(sql);
    sql_format % table_name;
    sql = sql_format.str();

    return data_source.update(sql, table_name);
}

datamodel::DataRow_ptr PgInputQueueDAO::find_oldest_record(const datamodel::InputQueue_ptr &p_input_queue)
{
    std::string sql_format = get_sql("find_oldest_record");
    InputQueueRowRowMapper row_mapper;

    boost::format sql(sql_format);
    sql % p_input_queue->get_table_name();

    return data_source.query_for_object(&row_mapper, sql.str());
}

datamodel::DataRow_ptr PgInputQueueDAO::find_newest_record(const datamodel::InputQueue_ptr &queue)
{
    std::string sql_format = get_sql("find_newest_record");
    InputQueueRowRowMapper row_mapper;

    boost::format sql(sql_format);
    sql % queue->get_table_name();

    return data_source.query_for_object(&row_mapper, sql.str());
}

std::deque<std::shared_ptr<std::string>> PgInputQueueDAO::get_db_table_column_names(const datamodel::InputQueue_ptr &queue)
{
    std::string query = get_sql("get_db_table_column_names");

    InputQueueDbTableColumnsMapper row_mapper;
    return data_source.query_for_deque(row_mapper, query, queue->get_table_name());
}

std::deque<datamodel::InputQueue_ptr> PgInputQueueDAO::get_all_user_queues(const std::string &user_name)
{
    InputQueueRowMapper row_mapper;
    return data_source.query_for_deque(row_mapper, get_sql("get_all_user_queues"), user_name);
}


std::deque<datamodel::InputQueue_ptr> PgInputQueueDAO::get_all_queues_with_sign(bool uses_fix_connection)
{
    InputQueueRowMapper rowMapper;
    return data_source.query_for_deque(rowMapper, get_sql("get_all_queues_with_sign"), uses_fix_connection);
}


bool PgInputQueueDAO::row_exists(const std::string &tableName, const bpt::ptime &valueTime)
{
    std::string sql_format = get_sql("row_exists");
    boost::format sql(sql_format);
    sql % tableName;

    return data_source.query_for_type<bool>(sql.str(), valueTime);
}

/******************************************************************************/
/******************************************************************************/
/******************************************************************************/

namespace {

struct MissedHoursrow_mapper : public IRowMapper<bpt::ptime>
{

    std::shared_ptr<bpt::ptime> mapRow(const pqxx_tuple &rowSet) const override
    {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
        if (rowSet.empty())
#pragma GCC diagnostic pop
            return std::make_shared<bpt::ptime>();
        return std::make_shared<bpt::ptime>(rowSet[0].as<bpt::ptime>(bpt::ptime{}));
    }
};

struct TimeRange_mapper : public IRowMapper<TimeRange>
{

    std::shared_ptr<TimeRange> mapRow(const pqxx_tuple &rowSet) const override
    {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
        if (rowSet.empty() || rowSet[0].is_null() || rowSet[1].is_null())
            return std::shared_ptr<TimeRange>();
#pragma GCC diagnostic pop

        return std::make_shared<TimeRange>(rowSet[0].as<bpt::ptime>({}), rowSet[1].as<bpt::ptime>(bpt::ptime{}));
    }
};


OptionalTimeRange get_reconciled_interval_impl(InputQueueDAO &dao, DataSource &data_source, std::string const &iq_name)
{
    std::string format = dao.get_sql("get_reconciled_interval");
    boost::format recon_interval_sql(format);
    recon_interval_sql % iq_name;

    static TimeRange_mapper row_mapper;

    auto range = data_source.query_for_object(&row_mapper, recon_interval_sql.str());
    if (range)
        return OptionalTimeRange(*range);
    return OptionalTimeRange();
}


OptionalTimeRange
get_missing_hours_impl(InputQueueDAO &dao, DataSource &data_source, std::string const &iq_name, bpt::ptime const &end, bpt::ptime const &start, size_t const &res_seconds)
{
    std::string sql_format = dao.get_sql("get_missing_hours_start");
    boost::format sqlStart(sql_format);
    sqlStart % iq_name;

    static MissedHoursrow_mapper row_mapper;

    std::shared_ptr<bpt::ptime> missing_start = data_source.query_for_object(&row_mapper, sqlStart.str(), end, start, res_seconds);

    if (!missing_start)
        return OptionalTimeRange();

    sql_format = dao.get_sql("get_missing_hours_end");
    boost::format sqlEnd(sql_format);
    sqlEnd % iq_name;

    std::shared_ptr<bpt::ptime> missing_end = data_source.query_for_object(&row_mapper, sqlEnd.str(), *missing_start, start, res_seconds);
    if (!missing_end) return OptionalTimeRange({start, *missing_start + bpt::seconds(res_seconds)});

    return OptionalTimeRange({*missing_end + bpt::seconds(res_seconds), *missing_start + bpt::seconds(res_seconds)});
}


void mark_interval_reconciled_impl(InputQueueDAO &dao, DataSource &data_source, std::string const &iq_name, bpt::ptime const &start, bpt::ptime const &end)
{
    std::string sql_format = dao.get_sql("mark_interval_reconciled");
    data_source.update(sql_format, iq_name, start, end);
}

}

OptionalTimeRange PgInputQueueDAO::get_missing_hours(datamodel::InputQueue_ptr const &queue, TimeRange const &from_range)
{
    size_t const sec = queue->get_resolution().total_seconds();
    bpt::time_duration one_month = bpt::hours(24 * 30);

    OptionalTimeRange reconciled_interval = get_reconciled_interval_impl(*this, data_source, queue->get_table_name());

    if (!reconciled_interval)
        reconciled_interval.reset({from_range.second, from_range.second});

    if (from_range.second >= reconciled_interval->second + bpt::seconds(10 * sec)) {
        OptionalTimeRange misses_in_front = get_missing_hours_impl(*this, data_source, queue->get_table_name(), from_range.second, reconciled_interval->second, sec);

        mark_interval_reconciled_impl(*this, data_source, queue->get_table_name(), reconciled_interval->second, from_range.second);

        if (misses_in_front)
            return misses_in_front;
    }


    if (from_range.first < reconciled_interval->first) {
        auto one_month_to_reconcile = std::max(reconciled_interval->first - one_month, from_range.first);

        OptionalTimeRange misses_in_back = get_missing_hours_impl(*this, data_source, queue->get_table_name(), reconciled_interval->first - queue->get_resolution(),
                                                                  one_month_to_reconcile, sec);

        mark_interval_reconciled_impl(*this, data_source, queue->get_table_name(), one_month_to_reconcile, reconciled_interval->first);

        reconciled_interval->first = std::max(one_month_to_reconcile, from_range.first);

        if (misses_in_back)
            return misses_in_back;

        // This is to initiate the next round of reconciliation
        return OptionalTimeRange({reconciled_interval->first, reconciled_interval->first});
    }

    //Reconciliation can stop now
    return OptionalTimeRange();
}

void PgInputQueueDAO::purge_missing_hours(datamodel::InputQueue_ptr const &queue)
{
    std::string sql_format = get_sql("purge_missing_hours");
    data_source.update(sql_format, queue->get_table_name());
}

}
}
