#pragma once

#include "DAO/IRowMapper.hpp"
#include "model/DeconQueue.hpp"

namespace svr {
namespace dao {

class DeconQueueRowMapper : public IRowMapper<datamodel::DeconQueue>
{
public:
    datamodel::DeconQueue_ptr map_row(const pqxx_tuple& row_set) const override
    {
        return ptr<datamodel::DeconQueue>(
            row_set["table_name"].as<std::string>(),
            row_set["input_queue_table_name"].as<std::string>(),
            row_set["input_queue_column_name"].as<std::string>(),
            row_set["dataset_id"].as<bigint>(0),
            row_set["levels"].as<uint16_t>(0)
        );
    }
    
    datamodel::DeconQueue_ptr map_row(duckdb_result& row_set, const size_t col_count, const size_t row) const override
    {
        return ptr<datamodel::DeconQueue>(
            common::dd_get_value(row_set, row, col_count, "table_name", std::string()),
            common::dd_get_value(row_set, row, col_count, "dataset_name", std::string()),
            common::dd_get_value(row_set, row, col_count, "input_queue_column_name", std::string()),
            common::dd_get_value(row_set, row, col_count, "dataset_id", bigint(0)),
            common::dd_get_value(row_set, row, col_count, "levels", uint16_t(0)),
        );
    };
}

}
}