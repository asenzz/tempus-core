#pragma once

#include "common.hpp"
#include "duckdb.h"

namespace svr{
namespace dao{

class DataSource;

class scoped_transaction_guard
{
    static constexpr bool commit_on_destroy = true;
    pqxx::work *trx;
    pqxx::connection connection_pg;
    DataSource &data_source;
public:
    scoped_transaction_guard(const std::string &connection_string, DataSource &data_source);
    ~scoped_transaction_guard();
    scoped_transaction_guard(scoped_transaction_guard && other);

    scoped_transaction_guard(const scoped_transaction_guard&) = delete;
    void operator= (const scoped_transaction_guard& ) = delete;

    pqxx::result exec(std::string const & query) const;
    pqxx::work* get_pqxx_work() const;
};

typedef std::shared_ptr<svr::dao::scoped_transaction_guard> scoped_transaction_guard_ptr;

class scoped_file_guard
{
    duckdb_database db;
    duckdb_connection con;
    DataSource &data_source;
    const std::chrono::milliseconds wait;
public:
    scoped_file_guard(const std::string &path, DataSource &data_source);
    ~scoped_file_guard();
    duckdb_result exec(const std::string &query) const;
};

typedef std::shared_ptr<svr::dao::scoped_file_guard> scoped_file_guard_ptr;

}
}

