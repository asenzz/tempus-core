#pragma once

#include "common.hpp"

namespace svr{
namespace dao{

class DataSource;

class scoped_transaction_guard
{
    bool commit_on_destroy = true;
    pqxx::work* trx;
    DataSource &data_source;
    pqxx::connection connection;
public:
    scoped_transaction_guard(const std::string &connection_string, DataSource &data_source);
    ~scoped_transaction_guard();
    scoped_transaction_guard(scoped_transaction_guard && other);

    scoped_transaction_guard(const scoped_transaction_guard&) = delete;
    void operator= (const scoped_transaction_guard& ) = delete;

    pqxx::result exec(std::string const & query);
    pqxx::work* get_pqxx_work();

};

}}

typedef std::shared_ptr<svr::dao::scoped_transaction_guard> scoped_transaction_guard_ptr;
