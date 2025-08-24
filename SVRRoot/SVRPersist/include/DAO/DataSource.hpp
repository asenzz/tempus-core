#ifndef DATASOURCE_HPP
#define DATASOURCE_HPP

#include <mutex>
#include "appcontext.hpp"
#include "common.hpp"
#include "DAO/IRowMapper.hpp"
#include "DAO/StatementPreparerDBTemplate.hpp"
#include "DAO/ScopedTransaction.hpp"
#include "common/logging.hpp"
#include "util/math_utils.hpp"

namespace svr::dao {

class DataSource
{
    static constexpr std::string C_tempus_cursor_name = "tempuscursor";
    const std::string connection_string;
    std::shared_ptr<StatementPreparerDBTemplate> statement_preparer_template;

public:
    explicit DataSource(const std::string &connection_string);

    virtual ~DataSource();

    scoped_transaction_guard_ptr open_transaction();

    scoped_file_guard_ptr open_file();

    template<typename T, typename ...Args> std::shared_ptr<T>
    query_for_object(IRowMapper<T> *row_mapper, const std::string &sql, Args &&... args);

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

    void upsert_row(CRPTR(char) table_name, CRPTR(char *) row_fields, const uint16_t n_fields);
};

} /* namespace svr::dao */

#include "DataSource.tpp"

#endif