#include <thread>

#include "common/logging.hpp"
#include "DAO/ScopedTransaction.hpp"
#include "DAO/DataSource.hpp"


namespace svr::dao {

scoped_transaction_guard::scoped_transaction_guard(const std::string &connection_string, DataSource &data_source)
        : connection_pg(pqxx::connection(connection_string)), data_source(data_source)
{
    trx = new pqxx::work(connection_pg);
    LOG4_TRACE("Opening new transaction " << trx);
}


scoped_transaction_guard::scoped_transaction_guard(scoped_transaction_guard &&other) : trx(other.trx), data_source(other.data_source)
{
    other.trx = nullptr;
}


scoped_transaction_guard::~scoped_transaction_guard()
{
    if (trx == nullptr) return;

    if (commit_on_destroy) {
        LOG4_TRACE("Finishing transaction " << trx << " with autocommit enabled from thread " << std::this_thread::get_id());
        try {
            trx->commit();
        } catch (const std::exception &e) {
            LOG4_ERROR("Cannot commit transaction " << trx << ": " << e.what() << " thread " << std::this_thread::get_id());
        } catch (...) {
            LOG4_FATAL("Unrecoverable error occurred: Cannot commit transaction!");
        }
    } else {
        LOG4_TRACE("Finishing transaction " << trx << " without commiting anything from thread " << std::this_thread::get_id());
    }

    delete trx;
}


pqxx::result scoped_transaction_guard::exec(const std::string &query) const
{
    return trx->exec(query);
}


pqxx::work *scoped_transaction_guard::get_pqxx_work() const
{
    return trx;
}

scoped_file_guard::scoped_file_guard(const std::string &path, DataSource &data_source) : data_source(data_source), wait(PROPS.get_db_wait())
{
    int dd_err, retries = 0;
    while ((dd_err = duckdb_open(path.c_str(), &db)) == DuckDBError && retries < PROPS.get_db_retries()) {
        LOG4_DEBUG("Retry " << ++retries << " opening database at " << path << ", last error " << dd_err);
        std::this_thread::sleep_for(wait);
    }
    if (dd_err != DuckDBSuccess) LOG4_THROW("Error " << dd_err << " opening database at " << path);
    retries = 0;
    while ((dd_err = duckdb_connect(db, &con)) == DuckDBError && retries < PROPS.get_db_retries()) {
        LOG4_DEBUG("Retry " << ++retries << " connecting to database at " << path << ", last error " << dd_err);
        std::this_thread::sleep_for(wait);
    }
    if (dd_err != DuckDBSuccess) LOG4_THROW("Error " << dd_err << " connecting to database at " << path);
}

scoped_file_guard::~scoped_file_guard()
{
    duckdb_disconnect(&con);
    duckdb_close(&db);
}

duckdb_result scoped_file_guard::exec(const std::string &query) const
{
    duckdb_result res;
    int dd_err, retries = 0;
    while ((dd_err = duckdb_query(con, query.c_str(), &res)) == DuckDBError && retries < PROPS.get_db_retries()) {
        LOG4_DEBUG("Retry " << ++retries << " querying " << query << " database, last error " << dd_err);
        std::this_thread::sleep_for(wait);
    }
    if (dd_err != DuckDBSuccess) LOG4_THROW("Error " << dd_err << " querying " << query);
    return res; // Don't forget to call duckdb_destroy_result()
}

}