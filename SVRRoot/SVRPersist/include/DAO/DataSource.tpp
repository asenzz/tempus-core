#ifndef DATASOURCE_TPP
#define DATASOURCE_TPP

#ifndef DATASOURCE_HPP
#include "DataSource.hpp"
#endif
#include "DAO/DummyRowMapper.hpp"


namespace svr {
namespace dao {

template<typename T, typename ...Args> std::shared_ptr<T> DataSource::query_for_object(IRowMapper<T> *row_mapper, const std::string &sql, Args &&... args)
{
    if (sql.empty()) THROW_EX_FS(std::invalid_argument, "Invalid SQL query passed, cannot be null!");
    std::shared_ptr<T> p_object;
    std::string query;
    try {
        query = statement_preparer_template->prepare_statement(sql, args...);
#ifdef USE_DUCKDB
        if (PROPS.is_duck()) {
            const auto trx = open_file();
            auto res = trx->exec(query);
            if (duckdb_row_count(&res) && duckdb_column_count(&res)) p_object = row_mapper->map_row(res, 1, 0);
            duckdb_destroy_result(&res);
        } else {
#endif
            const auto trx = open_transaction();
            if (const auto result = trx->exec(query); !result.empty()) p_object = row_mapper->map_row(result.at(0));
#ifdef USE_DUCKDB
        }
#endif
    } catch (const pqxx::broken_connection &ex) {
        LOG4_ERROR("Broken connection, " << ex.what() << ", while executing " << query);
    } catch (const std::exception &ex) {
        LOG4_ERROR("Error " << ex.what() << ", while executing " << query);
        throw;
    }
    if (!p_object) LOG4_DEBUG("No data returned for " << query);
    return p_object;
}

template<typename T, class ...Args> T DataSource::query_for_type(const std::string &sql, Args &&... args)
{
    if (sql.empty()) THROW_EX_FS(std::invalid_argument, "Invalid SQL query passed, cannot be null!");

    LOG4_DEBUG("Query for " << common::demangle(typeid(T).name()) << ", " << sql);
    std::string query;
    try {
        query = statement_preparer_template->prepare_statement(sql, args...);
#ifdef USE_DUCKDB
        if (PROPS.is_duck()) { // DuckDB
            auto res = open_file()->exec(query);
            T ret;
            const auto column_count = duckdb_column_count(&res);
            bool no_data = false;
            if (duckdb_row_count(&res) && column_count) ret = common::dd_get_value(res, 0, column_count, "", ret);
            else no_data = true;
            duckdb_destroy_result(&res);
            if (no_data) LOG4_THROW("No data returned for " << query);
            return ret;
        }
#endif // USE_DUCKDB
        // Postgres
        const auto res = open_transaction()->exec(query);
        if (res.empty()) LOG4_THROW("No data returned for " << query);
        return res.at(0, 0).as<T>();
    } catch (const std::exception &ex) {
        LOG4_ERROR("Error " << ex.what() << ", while executing " << query);
        throw;
    }
}

// TODO A separately optimized function for querying datarow container tables using duckdb chunks
template<typename M, typename T, template<typename, typename> typename Container, typename ...Args>  Container<T, std::allocator<T>>
DataSource::query_for_type_array(const IRowMapper<M> &row_mapper, const std::string &sql, Args &&... args)
{
    if (sql.empty()) THROW_EX_FS(std::invalid_argument, "Invalid SQL query passed, cannot be null!");

    LOG4_DEBUG("Query for " << common::demangle(typeid(Container<T, std::allocator<T>>).name()) << " of " << common::demangle(typeid(T).name()) << ", " << sql);
    std::string query;
    Container<T, std::allocator<T>> res;
    try {
        query = statement_preparer_template->prepare_statement(sql, args...);
#ifdef USE_DUCKDB
        if (PROPS.is_duck()) {
            auto dbres = open_file()->exec(query);
            const auto row_count = duckdb_row_count(&dbres);
            const auto column_count = duckdb_column_count(&dbres);
            if (row_count < 1 || column_count < 1) {
                LOG4_DEBUG("No data returned for " << query);
                return {};
            }
            res.resize(row_count);
            OMP_FOR_i(row_count) res[i] = row_mapper.map_row(dbres, column_count, i);
            duckdb_destroy_result(&dbres);
            return res;
        }
#endif
        // Postgres
        query = statement_preparer_template->prepare_statement(sql, args...);
        scoped_transaction_guard_ptr trx = open_transaction();
        const std::string count_query = "SELECT COUNT(*) FROM (" + query + ") AS SUBQ";
        const auto result = trx->exec(count_query);
        if (result.empty()) LOG4_THROW("Failed getting result size, using query " << count_query);
        const auto result_size = result[0][0].as<size_t>(0);
        trx.reset();
        const auto num_cursors = std::min<uint32_t>(PROPS.get_db_num_threads(), result_size / common::C_min_cursor_rows + 1);
        const auto cursor_size = result_size / num_cursors;
        LOG4_DEBUG("Getting up to " << result_size << " rows for " << query);
        res.resize(result_size, nullptr);
#pragma omp parallel ADJ_THREADS(result_size)
#pragma omp single
        {
            OMP_TASKLOOP_1(untied firstprivate(num_cursors, result_size))
            for (DTYPE(num_cursors) cur_ix = 0; cur_ix < num_cursors; ++cur_ix) {
                const auto start_ix = cur_ix * cursor_size;
                if (start_ix >= result_size) continue;
                const auto end_ix = cur_ix == num_cursors - 1 ? result_size : start_ix + cursor_size;
                scoped_transaction_guard_ptr l_trx = open_transaction();
                pqxx::stateless_cursor<pqxx::cursor_base::read_only, pqxx::cursor_base::owned> l_cursor(
                    *l_trx->get_pqxx_work(), query, std::to_string(cur_ix) + C_tempus_cursor_name, false);
                const auto l_result = l_cursor.retrieve(start_ix, end_ix);
                if (l_result.size() < 1 || size_t(l_result.size()) != end_ix - start_ix)
                    LOG4_ERROR("Cursor didn't return expected size " << end_ix - start_ix << ", got " << l_result.size() << " instead.");
                else
                    LOG4_DEBUG("Got " << l_result.size() << " rows for cursor " << cur_ix << " range " << start_ix << " - " << end_ix);
                // OMP_TASKLOOP_(l_result.size(), untied firstprivate(start_ix)) // TODO OMP bug, doesn't respect barrier at end of nested taskloops if non collapsed
                for (size_t r = 0; r < size_t(l_result.size()); ++r) {
                    res[r + start_ix] = row_mapper.map_row(l_result[r]);
                    if (res[r + start_ix]) continue;
                    LOG4_ERROR("Result for " << r + start_ix << " is empty, row string " << l_result[r][0]);
                }
            }
        }
        for (auto it_res = res.begin(); it_res != res.end();)
            if (*it_res) ++it_res; else it_res = res.erase(it_res);
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

template<typename ...Args> int DataSource::update(const std::string &sql, Args &&... args)
{
    if (sql.empty()) THROW_EX_FS(std::invalid_argument, "Invalid SQL query passed! Query cannot be null!");

    LOG4_DEBUG("Update, " << sql);
    std::string query;
    try {
        query = statement_preparer_template->prepare_statement(sql, args...);
#ifdef USE_DUCKDB
        if (PROPS.is_duck()) {
            auto res = open_file()->exec(query);
            const auto ret = duckdb_rows_changed(&res);
            duckdb_destroy_result(&res);
            return ret;
        }
#endif // USE_DUCKDB
        return open_transaction()->exec(query).affected_rows();
    } catch (const pqxx::failure &ex) {
        LOG4_ERROR("Error " << ex.what() << ", while executing " << query);
        throw;
    }
}

}
}

#endif // DATASOURCE_TPP
