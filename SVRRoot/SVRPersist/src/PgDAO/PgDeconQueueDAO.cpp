#include "appcontext.hpp"
#include "PgDeconQueueDAO.hpp"
#include "common/constants.hpp"
#include "util/validation_utils.hpp"
#include "util/string_utils.hpp"
#include "DAO/DataRowRowMapper.hpp"
#include "DAO/DeconQueueRowMapper.hpp"
#include "DAO/DataSource.hpp"

namespace svr {
namespace dao {

PgDeconQueueDAO::PgDeconQueueDAO(common::PropertiesReader &tempus_config, dao::DataSource &data_source)
        : DeconQueueDAO(tempus_config, data_source)
{}

datamodel::DeconQueue_ptr PgDeconQueueDAO::get_decon_queue_by_table_name(const std::string &table_name)
{
    DeconQueueRowMapper row_mapper;
    return data_source.query_for_object(&row_mapper, AbstractDAO::get_sql("get_decon_queue_by_table_name"), table_name);
}

std::deque<datamodel::DataRow_ptr> PgDeconQueueDAO::get_data(const std::string &deconQueueTableName, bpt::ptime const &timeFrom, bpt::ptime const &timeTo, const size_t limit)
{
    REJECT_EMPTY(deconQueueTableName);
    boost::format sqlFormat(AbstractDAO::get_sql("get_data"));
    sqlFormat % deconQueueTableName;
    auto sql = sqlFormat.str();
    if (limit != 0) sql += " LIMIT " + std::to_string(limit);
    DataRowRowMapper row_mapper;
    return data_source.query_for_deque(row_mapper, sql, timeFrom, timeTo);
}

std::deque<datamodel::DataRow_ptr> PgDeconQueueDAO::get_latest_data(const std::string &deconQueueTableName, bpt::ptime const &timeTo, const size_t limit)
{
    REJECT_EMPTY(deconQueueTableName);

    boost::format sqlFormat(AbstractDAO::get_sql("get_latest_data"));
    sqlFormat % deconQueueTableName;

    const auto sql = sqlFormat.str();
    DataRowRowMapper row_mapper;
    return data_source.query_for_deque(row_mapper, sql, timeTo, limit);
}

bool PgDeconQueueDAO::exists(const datamodel::DeconQueue_ptr &deconQueue)
{
    return exists(deconQueue->get_table_name());
}

bool PgDeconQueueDAO::exists(const std::string &table_name)
{
    if (table_name.empty()) return false;
    return data_source.query_for_type<int>(AbstractDAO::get_sql("decon_queue_exists"), table_name) == 1 &&
           data_source.query_for_type<int>(AbstractDAO::get_sql("table_exists"), table_name) == 1;
}

void PgDeconQueueDAO::save(datamodel::DeconQueue_ptr const &p_decon_queue, const boost::posix_time::ptime &start_time)
{
    const auto res = save_metadata(p_decon_queue);
    save_data(p_decon_queue, res == 100 ? bpt::min_date_time : start_time);
}

bool PgDeconQueueDAO::decon_table_needs_recreation(datamodel::DeconQueue_ptr const &existing_queue, datamodel::DeconQueue_ptr const &new_queue)
{
    return existing_queue->get_input_queue_column_name() != new_queue->get_input_queue_column_name()
           || (existing_queue->get_dataset_id() != 0 && new_queue->get_dataset_id() != 0 && (existing_queue->get_dataset_id() != new_queue->get_dataset_id()))
           || get_level_count(existing_queue) != get_level_count(new_queue);
}

int PgDeconQueueDAO::save_metadata(datamodel::DeconQueue_ptr const &p_decon_queue)
{
    int ret = 0;
    const auto existing = get_decon_queue_by_table_name(p_decon_queue->get_table_name());
    if (existing) {
        scoped_transaction_guard_ptr tx = data_source.open_transaction();
        if (decon_table_needs_recreation(p_decon_queue, existing)) {
            LOG4_WARN("Recreating " << existing->to_string() << " with " << p_decon_queue->to_string());
            remove_decon_table_no_trx(p_decon_queue);
            (void) update_metadata_no_trx(p_decon_queue);
            create_decon_table_no_trx(p_decon_queue);
            ret = 100;
        } else if (existing->get_input_queue_table_name() != p_decon_queue->get_input_queue_table_name() or
                   existing->get_input_queue_column_name() != p_decon_queue->get_input_queue_column_name() or
                   existing->get_dataset_id() != p_decon_queue->get_dataset_id() or
                   existing->get_table_name() != p_decon_queue->get_table_name())
            ret = update_metadata_no_trx(p_decon_queue);
    } else {
        scoped_transaction_guard_ptr tx = data_source.open_transaction();
        ret = data_source.update(AbstractDAO::get_sql("save_metadata"),
                                 p_decon_queue->get_table_name(),
                                 p_decon_queue->get_input_queue_table_name(),
                                 p_decon_queue->get_input_queue_column_name(),
                                 p_decon_queue->get_dataset_id()
        );
        create_decon_table_no_trx(p_decon_queue);
    }
    return ret;
}

int PgDeconQueueDAO::remove(datamodel::DeconQueue_ptr const &deconQueue)
{
    scoped_transaction_guard_ptr trx = data_source.open_transaction();
    auto dropTableSql = AbstractDAO::get_sql("remove_decon_queue_table");
    boost::format sqlFormat(dropTableSql);
    sqlFormat % deconQueue->get_table_name();
    data_source.update(sqlFormat.str());
    return data_source.update(AbstractDAO::get_sql("remove_decon_queue"), deconQueue->get_table_name());
}

long PgDeconQueueDAO::save_data(const datamodel::DeconQueue_ptr &p_decon_queue, const boost::posix_time::ptime &start_time)
{
    if (p_decon_queue->size() < 1) return 0;
    if (count(p_decon_queue) > 0) data_source.cleanup_queue_table(p_decon_queue->get_table_name(), p_decon_queue->get_data(), start_time);
    return data_source.batch_update(p_decon_queue->get_table_name(), p_decon_queue->get_data(), start_time);
}

std::deque<datamodel::DataRow_ptr>
PgDeconQueueDAO::get_data_having_update_time_greater_than(const std::string &deconQueueTableName, const bpt::ptime &updateTime, const size_t limit)
{
    DataRowRowMapper row_mapper;
    auto sql = get_sql("get_data_having_update_time_greater_than");
    boost::format sqlFormat(sql);
    sqlFormat % deconQueueTableName;

    return data_source.query_for_deque(row_mapper, sqlFormat.str(), updateTime, limit);
}

int PgDeconQueueDAO::clear(const datamodel::DeconQueue_ptr &deconQueue)
{

    boost::format sqlFormat(get_sql("clear_table"));
    sqlFormat % deconQueue->get_table_name();
    return data_source.update(sqlFormat.str());
}

long PgDeconQueueDAO::count(const datamodel::DeconQueue_ptr &deconQueue)
{
    boost::format sqlFormat(get_sql("count"));
    sqlFormat % deconQueue->get_table_name();
    return data_source.query_for_type<long>(sqlFormat.str());
}

void PgDeconQueueDAO::create_decon_table_no_trx(const datamodel::DeconQueue_ptr &decon_queue)
{
    auto create_table_sql = AbstractDAO::get_sql("create_decon_table");
    LOG4_DEBUG("Creating new DeconQueue table with name: " << decon_queue->get_table_name());
    std::string level_columns_sql;
    for (size_t level = 0; level < decon_queue->get_decon_level_number(); ++level)
        level_columns_sql += common::C_decon_queue_column_level_prefix + std::to_string(level) + " double precision NOT NULL DEFAULT 0, ";


    boost::format sql_format(create_table_sql);
    sql_format % decon_queue->get_table_name() % level_columns_sql % decon_queue->get_table_name();

    LOG4_TRACE("CREATE DECON TABLE SQL: " << sql_format.str());

    data_source.update(sql_format.str());
}

void PgDeconQueueDAO::remove_decon_table_no_trx(datamodel::DeconQueue_ptr const &decon_queue)
{
    auto dropTableSql = AbstractDAO::get_sql("remove_decon_queue_table");
    boost::format sqlFormat(dropTableSql);
    sqlFormat % decon_queue->get_table_name();

    data_source.update(sqlFormat.str());
}

int PgDeconQueueDAO::update_metadata_no_trx(datamodel::DeconQueue_ptr const &decon_queue)
{
    return data_source.update(AbstractDAO::get_sql("update_metadata"),
                              decon_queue->get_input_queue_table_name(),
                              decon_queue->get_input_queue_column_name(),
                              decon_queue->get_dataset_id(),
                              decon_queue->get_table_name()
    );
}

size_t PgDeconQueueDAO::get_level_count(const datamodel::DeconQueue_ptr &p_decon_queue)
{
    datamodel::Dataset_ptr p_dataset;
    size_t result = 0;

    if (p_decon_queue->get_decon_level_number() > 0) {
        result = p_decon_queue->get_decon_level_number();
        goto __bail;
    }

    if (!p_decon_queue->get_data().empty() && !p_decon_queue->front()->get_values().empty())
        result = p_decon_queue->front()->get_values().size();

    if (result != 0) goto __bail;

    result = get_level_count_db(p_decon_queue);

    if (result != 0) goto __bail;

    result = APP.dataset_service.get_level_count(p_decon_queue->get_dataset_id());

    __bail:
    LOG4_DEBUG("Returning " << result << " for " << p_decon_queue->get_table_name());
    return result;
}


size_t PgDeconQueueDAO::get_level_count_db(datamodel::DeconQueue_ptr const &queue)
{
    auto ret = data_source.query_for_type<int>(AbstractDAO::get_sql("get_db_column_number"), queue->get_table_name());
    if (ret < 3) return 0;
    return ret - 3;
}


}
}
