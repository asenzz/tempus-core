#pragma once

#include "common.hpp"
#include <util/string_utils.hpp>
#include "IRowMapper.hpp"
#include <model/InputQueue.hpp>

namespace svr {
namespace dao {

class InputQueueRowMapper : public IRowMapper<datamodel::InputQueue>
{
public:
    InputQueueRowMapper()
    {}

    virtual ~InputQueueRowMapper()
    {}

    datamodel::InputQueue_ptr map_row(const pqxx_tuple &row_set) const override
    {
        if (row_set["table_name"].is_null()) LOG4_THROW("Cannot map a row with empty table_name");
        datamodel::InputQueue_ptr result = ptr<datamodel::InputQueue>(
                row_set["table_name"].as<std::string>(""),
                row_set["logical_name"].as<std::string>(""),
                row_set["user_name"].as<std::string>(""),
                row_set["description"].as<std::string>(""),
                row_set["resolution"].as<bpt::time_duration>(common::C_default_resolution),
                row_set["legal_time_deviation"].as<bpt::time_duration>(common::C_default_legal_time_deviation),
                row_set["timezone"].as<std::string>(""),
                common::from_sql_array(row_set["value_columns"].as<std::string>("")), // Should be in order of appearance
                row_set["uses_fix_connection"].as<bool>(false)
        );
        return result;
    }

    datamodel::InputQueue_ptr map_row(duckdb_result& row_set, const size_t col_count, const size_t row) const override
    {
        return ptr<datamodel::InputQueue>(
            common::dd_get_value(row_set, row, col_count, "table_name", std::string()),
            common::dd_get_value(row_set, row, col_count, "logical_name", std::string()),
            common::dd_get_value(row_set, row, col_count, "user_name", std::string()),
            common::dd_get_value(row_set, row, col_count, "description", std::string()),
            common::dd_get_value(row_set, row, col_count, "resolution", common::C_default_resolution),
            common::dd_get_value(row_set, row, col_count, "legal_time_deviation", common::C_default_legal_time_deviation),
            common::dd_get_value(row_set, row, col_count, "timezone", std::string()),
            common::from_sql_array(common::dd_get_value(row_set, row, col_count, "value_columns", std::string())), // Should be in order of appearance
            common::dd_get_value(row_set, row, col_count, "uses_fix_connection", false)
        );
    }
};


class InputQueueDbTableColumnsMapper : public IRowMapper<std::string>
{
public:
    InputQueueDbTableColumnsMapper()
    {}

    virtual ~InputQueueDbTableColumnsMapper()
    {}

    std::shared_ptr<std::string> map_row(const pqxx_tuple &row_set) const override
    {
        return ptr<std::string>(row_set[0].as<std::string>());
    }

    datamodel::InputQueue_ptr map_row(duckdb_result& row_set, const size_t col_count, const size_t row) const override
    {
        return ptr<std::string>(common::dd_get_value<std::string>(row_set, 0, 0));
    }

};

} /* namespace dao */
} /* namespace svr */

