#pragma once


#include <mutex>
#include "common.hpp"
#include "DAO/IRowMapper.hpp"
#include "DAO/StatementPreparerDBTemplate.hpp"
#include "DAO/ScopedTransaction.hpp"
#include "DAO/DummyRowMapper.hpp"
#include "common/logging.hpp"
#include "util/math_utils.hpp"


namespace svr::dao {


class DataSource
{
    const std::string C_tempus_cursor_name = "tempuscursor";
private:
    const std::string connection_string;
    std::shared_ptr<StatementPreparerDBTemplate> statementPreparerTemplate;

public:
    explicit DataSource(const std::string &connection_string, const bool commit_on_scope_exit = true);

    virtual ~DataSource();

    scoped_transaction_guard_ptr open_transaction();

    template<typename T, typename ...Args> std::shared_ptr<T>
    query_for_object(IRowMapper<T> *rowMapper, const std::string &sql, Args &&... args);

    template<typename T, class ...Args> T
    query_for_type(const std::string &sql, Args &&... args);

    template<typename M, typename T, template<typename, typename> typename Container, class ...Args>  Container<T, std::allocator<T>>
    query_for_type_array(const IRowMapper<M> &row_mapper, const std::string &sql, Args &&... args);

    template<typename T, template<typename, typename> typename Container, class ...Args>  Container<T, std::allocator<T>>
    query_for_type_array(const IRowMapper<T> &row_mapper, const std::string &sql, Args &&... args);

    template<typename T, template<typename, typename> typename Container, class ...Args>  Container<T, std::allocator<T>>
    query_for_type_array(const std::string &sql, Args &&... args);

    template<typename T, typename ...Args> std::vector<std::shared_ptr<T>, std::allocator<std::shared_ptr<T>>>
    query_for_array(const IRowMapper<T> &row_mapper, const std::string &sql, Args &&...args);

    template<typename T, typename ...Args> std::deque<std::shared_ptr<T>, std::allocator<std::shared_ptr<T>>>
    query_for_deque(const IRowMapper<T> &row_mapper, const std::string &sql, Args &&...args);

    template<typename ...Args> int
    update(const std::string &sql, Args &&... args);

    long batch_update(const std::string &table_name, const datamodel::DataRow::container &data, const bpt::ptime &start_time = bpt::min_date_time); // Data will be cleaned as added

    void cleanup_queue_table(const std::string &table_name, const datamodel::DataRow::container &data, const bpt::ptime &start_time = bpt::min_date_time);
};


template<typename T, typename ...Args> std::shared_ptr<T>
DataSource::query_for_object(IRowMapper<T> *rowMapper, const std::string &sql, Args &&... args)
{
    using namespace pqxx;

    if (sql.empty()) THROW_EX_FS(std::invalid_argument, "Invalid SQL query passed, cannot be null!");
    std::shared_ptr<T> object(nullptr);
    std::string query;
    try {
        query = statementPreparerTemplate->prepareStatement(sql, args...);
        scoped_transaction_guard_ptr trx = open_transaction();
        pqxx::result result = trx->exec(query);
        if (!result.empty()) object = rowMapper->mapRow(result.at(0));
    } catch (const pqxx::broken_connection &ex) {
        LOG4_ERROR("Broken connection, " << ex.what() << ", while executing " << query);
    } catch (const std::exception &ex) {
        LOG4_ERROR("Error " << ex.what() << ", while executing " << query);
        throw;
    }
    return object;
}

template<typename T, class ...Args>
T DataSource::query_for_type(const std::string &sql, Args &&... args)
{
    using namespace pqxx;
    if (sql.empty()) THROW_EX_FS(std::invalid_argument, "Invalid SQL query passed, cannot be null!");

    LOG4_DEBUG("Query for " << svr::common::demangle(typeid(T).name()) << " [" << sql << "]");
    std::string query;
    try {
        scoped_transaction_guard_ptr trx = open_transaction();
        query = statementPreparerTemplate->prepareStatement(sql, args...);
        pqxx::result result;
        result = trx->exec(query);
        if (result.empty())
            return T();
        else
            return result.at(0).at(0).as<T>();

    } catch (const std::exception &ex) {
        LOG4_ERROR("Error " << ex.what() << ", while executing " << query);
        throw;
    }
}


template<typename M, typename T, template<typename, typename> typename Container, typename ...Args>  Container<T, std::allocator<T>>
DataSource::query_for_type_array(const IRowMapper<M> &row_mapper, const std::string &sql, Args &&... args)
{
    using namespace pqxx;
    if (sql.empty()) THROW_EX_FS(std::invalid_argument, "Invalid SQL query passed, cannot be null!");

    LOG4_DEBUG("Query for " << svr::common::demangle(typeid(Container<T, std::allocator<T>>).name()) << " of " << svr::common::demangle(typeid(T).name()) << ", " << sql);
    std::string query;
    Container<T, std::allocator<T>> res;
    try {
        query = statementPreparerTemplate->prepareStatement(sql, args...);
        scoped_transaction_guard_ptr trx = open_transaction();
        const std::string count_query = "SELECT COUNT(*) FROM (" + query + ") AS SUBQ";
        const auto result = trx->exec(count_query);
        if (result.empty()) LOG4_THROW("Failed getting result size, using query " << count_query);
        const auto result_size = result[0][0].as<size_t>(0);
        trx.reset();
        const size_t num_cursors = std::min<size_t>(common::C_cursors_per_query, result_size / common::C_min_cursor_rows + 1);
        const size_t cursor_size = result_size / num_cursors;
        LOG4_DEBUG("Getting up to " << result_size << " rows for " << query);
        res.resize(result_size);
#pragma omp parallel num_threads(C_n_cpu)
#pragma omp single
        {
#pragma omp taskloop grainsize(1) mergeable default(shared) untied firstprivate(num_cursors, result_size)
            for (size_t cur_ix = 0; cur_ix < num_cursors; ++cur_ix) {
                const size_t start_ix = cur_ix * cursor_size;
                if (start_ix >= result_size) continue;
                const auto end_ix = cur_ix == num_cursors - 1 ? result_size : start_ix + cursor_size;
                scoped_transaction_guard_ptr l_trx = open_transaction();
                pqxx::stateless_cursor<pqxx::cursor_base::read_only, pqxx::cursor_base::owned> l_cursor(*l_trx->get_pqxx_work(), query,
                                                                                                        to_string(cur_ix) + C_tempus_cursor_name, false);
                const auto l_result = l_cursor.retrieve(start_ix, end_ix);
                if (l_result.size() < 1 || size_t(l_result.size()) != end_ix - start_ix)
                    LOG4_ERROR("Cursor didn't return expected size " << end_ix - start_ix << ", got " << l_result.size() << " instead.");
                else
                    LOG4_DEBUG("Got " << l_result.size() << " rows for cursor " << cur_ix << " range " << start_ix << " - " << end_ix);
#pragma omp taskloop simd grainsize(cdiv(l_result.size(), C_n_cpu)) mergeable default(shared) untied firstprivate(start_ix)
                for (size_t r = 0; r < size_t(l_result.size()); ++r) {
                    const auto this_res = row_mapper.mapRow(l_result[r]);
                    res[r + start_ix] = this_res;
                    if (!res[r + start_ix] || !this_res)
                        LOG4_ERROR("Result for " << (r + start_ix) << " is empty, row string " << l_result[r][0]);
                }
            }
        }
        return res;
    } catch (const std::exception &ex) {
        LOG4_ERROR("Error " << ex.what() << ", while executing " << query);
        throw;
    }
}

template<typename T, template<typename, typename> typename Container, class ...Args>  Container<T, std::allocator<T>>
DataSource::query_for_type_array(const IRowMapper<T> &row_mapper, const std::string &sql, Args &&... args)
{
    return query_for_type_array<T, std::shared_ptr<T>, Container>(row_mapper, sql, args...);
}

template<typename T, template<typename, typename> typename Container, typename ...Args>  Container<T, std::allocator<T>>
DataSource::query_for_type_array(const std::string &sql, Args &&... args)
{
    return query_for_type_array<T, Container<T, std::allocator<T>>>(DummyRowMapper<T>{}, sql, args...);
}

template<typename T, typename ...Args> std::vector<std::shared_ptr<T>, std::allocator<std::shared_ptr<T>>>
DataSource::query_for_array(const IRowMapper<T> &row_mapper, const std::string &sql, Args &&...args)
{
    return query_for_type_array<T, std::shared_ptr<T>, std::vector>(row_mapper, sql, args...);
}

template<typename T, typename ...Args> std::deque<std::shared_ptr<T>>
DataSource::query_for_deque(const IRowMapper<T> &row_mapper, const std::string &sql, Args &&...args)
{
    return query_for_type_array<T, std::shared_ptr<T>, std::deque>(row_mapper, sql, args...);
}


template<typename ...Args>
int DataSource::update(const std::string &sql, Args &&... args)
{
    using namespace pqxx;
    if (sql.empty()) THROW_EX_FS(std::invalid_argument, "Invalid SQL query passed! Query cannot be null!");

    LOG4_DEBUG("Update [" << sql << "]");
    std::string query;
    try {
        scoped_transaction_guard_ptr trx = open_transaction();
        query = statementPreparerTemplate->prepareStatement(sql, args...);
        pqxx::result result;
        result = trx->exec(query);
        return result.affected_rows();

    } catch (const pqxx::failure &ex) {
        LOG4_ERROR("Error " << ex.what() << ", while executing " << query);
        throw;
    }
}

} /* namespace svr::dao */

