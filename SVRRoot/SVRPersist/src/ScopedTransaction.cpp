#include <thread>

#include "common/logging.hpp"
#include "DAO/ScopedTransaction.hpp"
#include "DAO/DataSource.hpp"


namespace svr::dao {

scoped_transaction_guard::scoped_transaction_guard(const std::string &connection_string, DataSource &data_source)
        : commit_on_destroy(true), data_source(data_source), connection(pqxx::connection(connection_string))
{
    trx = new pqxx::work(connection);
    LOG4_TRACE("Opening new transaction " << trx);
}


scoped_transaction_guard::scoped_transaction_guard(scoped_transaction_guard &&other)
        : commit_on_destroy(other.commit_on_destroy), trx(other.trx), data_source(other.data_source)
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


pqxx::result scoped_transaction_guard::exec(const std::string &query)
{
    return trx->exec(query);
}


pqxx::work *scoped_transaction_guard::get_pqxx_work()
{
    return trx;
}


}