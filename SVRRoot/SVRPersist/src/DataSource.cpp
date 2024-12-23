#include <memory>
#include <thread>

#include "DAO/DataSource.hpp"

#define CLEANUP_BATCH_SIZE 10000

namespace svr {
namespace dao {


DataSource::~DataSource()
{
}

DataSource::DataSource(const std::string &connection_string, const bool commit_on_scope_exit) :
        connection_string(connection_string)
{
    LOG4_DEBUG("Opening connection using connection string " << connection_string);
    try {
        statement_preparer_template = std::make_unique<StatementPreparerDBTemplate>(connection_string);
    } catch (const std::exception &e) {
        LOG4_FATAL(e.what());
        throw e;
    }
}

#if 0
void DataSource::reopen_connection()
{
    const size_t inc_sleep_seconds = 30;
    const size_t max_attempt_numb = 30;
    size_t sleep_seconds = 1;
    size_t attempt_numb = 0;
    LOG4_DEBUG("Opening connection for " << connection_string);
    do {
        LOG4_INFO("Attempt number to create new transaction : " << attempt_numb);
        try {
            p_connection.reset();
            p_connection = ptr<pqxx::connection>(connection_string);
        } catch (const std::exception &e) {
            LOG4_ERROR("Can't open transaction for thread " << std::this_thread::get_id() << ", due error " << e.what() << ", sleeping " << sleep_seconds << " secs.");
            sleep(sleep_seconds);
            // increment counters
            if (attempt_numb % 10 == 0) sleep_seconds += inc_sleep_seconds;
            ++attempt_numb;
        }
    } while (!p_connection->is_open() and attempt_numb < max_attempt_numb);
}
#endif

scoped_transaction_guard_ptr DataSource::open_transaction()
{
    return ptr<scoped_transaction_guard>(connection_string, *this);
}

long DataSource::batch_update(const std::string &table_name, const datamodel::DataRow::container &data, const bpt::ptime &start_time)
{
    if (data.empty()) {
        LOG4_ERROR("No data to save!");
        return 0;
    }
    if (table_name.empty()) THROW_EX_FS(std::invalid_argument, "Invalid table name!");
    LOG4_DEBUG("Updating " << table_name << " with up to " << data.size() << " rows.");
    scoped_transaction_guard_ptr trx = open_transaction();
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    try {
        size_t ret = 0;
        pqxx::stream_to tw(*trx->get_pqxx_work(), table_name);
        for (const auto &row: data) {
            if (row->get_value_time() < start_time) continue;
            tw << row->to_tuple();
            ++ret;
        }
        tw.complete();
        LOG4_DEBUG("Saved " << ret << " rows.");
        return ret;
    } catch (const pqxx::failure &e) {
        LOG4_FATAL(e.what());
        throw;
    }
#pragma GCC diagnostic pop
}

void DataSource::upsert_row(CRPTR(char) table_name, CRPTR(char *) row_fields, const uint16_t n_fields)
{
    assert(n > 3);
    assert(table_name);
    assert(row_fields);
    assert(*row_fields);
    LOG4_TRACE("Upserting row into " << table_name);
    scoped_transaction_guard_ptr trx = open_transaction();
    std::ostringstream ostr;
    ostr << "SELECT upsert_row('" << table_name << "','{";
    for (uint16_t i = 0; i < n_fields - 1; ++i) ostr << row_fields[i] << ',';
    ostr << row_fields[n_fields - 1] << "}');";
    try {
        trx->exec(ostr.str());
    } catch (const std::exception &e) {
        LOG4_THROW(e.what());
    }
    LOG4_END();
}

void DataSource::cleanup_queue_table(const std::string &table_name, const datamodel::DataRow::container &data, const bpt::ptime &start_time)
{
    if (data.empty()) return;
    LOG4_DEBUG(
            "Cleaning queue of size " << data.size() << " rows, starting " << std::max(start_time, data.front()->get_value_time()) << " until " << data.back()->get_value_time());
    scoped_transaction_guard_ptr trx = open_transaction();
    const auto start_iter = lower_bound_back(data, start_time);
    auto start_ix = start_iter - data.cbegin();
    if (start_ix < 0 or start_ix >= CAST2(start_ix) data.size()) start_ix = 0;
    for (DTYPE(data.size()) i = start_ix; i < data.size(); i += CLEANUP_BATCH_SIZE) {
        std::ostringstream ostr;
        ostr << "SELECT cleanup_queue('" << table_name << "', '{";
        auto ivt = data.cbegin() + i;
        ostr << (**ivt).get_value_time();
        ++ivt;
        uint32_t j = 0;
        for (; ivt != data.cend() && j < CLEANUP_BATCH_SIZE; ++ivt, ++j)
            ostr << ',' << (**ivt).get_value_time();
        ostr << "}'::timestamp[])";
        try {
            trx->exec(ostr.str());
        } catch (const std::exception &e) {
            LOG4_THROW(e.what());
        }
    }

    LOG4_END();
}

}
}

