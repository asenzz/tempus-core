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

    datamodel::InputQueue_ptr mapRow(const pqxx_tuple &rowSet) const override
    {
        if (rowSet["table_name"].is_null()) LOG4_THROW("Cannot map a row with empty table_name");
        datamodel::InputQueue_ptr result = std::make_shared<datamodel::InputQueue>(
                rowSet["table_name"].as<std::string>(),
                rowSet["logical_name"].as<std::string>(""),
                rowSet["user_name"].as<std::string>(""),
                rowSet["description"].as<std::string>(""),
                rowSet["resolution"].as<bpt::time_duration>({}),
                rowSet["legal_time_deviation"].as<bpt::time_duration>({}),
                rowSet["timezone"].as<std::string>(""),
                svr::common::from_sql_array(rowSet["value_columns"].as<std::string>("")), // Should be in order of appearance
                rowSet["uses_fix_connection"].as<bool>(false)
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

    std::shared_ptr<std::string> mapRow(const pqxx_tuple &rowSet) const override
    {
        return std::make_shared<std::string>(rowSet[0].as<std::string>(""));
    }
};

} /* namespace dao */
} /* namespace svr */

