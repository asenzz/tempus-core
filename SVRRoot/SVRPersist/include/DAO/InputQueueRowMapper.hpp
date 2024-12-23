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
                row_set["resolution"].as<bpt::time_duration>({}),
                row_set["legal_time_deviation"].as<bpt::time_duration>({}),
                row_set["timezone"].as<std::string>(""),
                svr::common::from_sql_array(row_set["value_columns"].as<std::string>("")), // Should be in order of appearance
                row_set["uses_fix_connection"].as<bool>(false)
        );
        return result;
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
        return otr<std::string>(row_set[0].as<std::string>());
    }
};

} /* namespace dao */
} /* namespace svr */

