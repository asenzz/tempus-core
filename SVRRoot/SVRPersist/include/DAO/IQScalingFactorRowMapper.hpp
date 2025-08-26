#pragma once

#include "DAO/IRowMapper.hpp"
#include "model/IQScalingFactor.hpp"

namespace svr {
namespace dao {

class IQScalingFactorRowMapper : public IRowMapper<datamodel::IQScalingFactor>
{
public:
    datamodel::IQScalingFactor_ptr map_row(const pqxx_tuple &row_set) const override
    {
        return ptr<datamodel::IQScalingFactor>(
                row_set["id"].as<bigint>(0),
                row_set["dataset_id"].as<bigint>(0),
                row_set["input_queue_table_name"].as<std::string>(),
                row_set["input_queue_column_name"].as<std::string>(),
                row_set["scaling_factor"].as<double>(1),
                row_set["dc_offset"].as<double>(0)
        );
    }
#ifdef USE_DUCKDB
    datamodel::IQScalingFactor_ptr map_row(duckdb_result& row_set, const size_t col_count, const size_t row) const override
    {
        return ptr<datamodel::IQScalingFactor>(
            common::dd_get_value(row_set, row, col_count, "id", bigint(0)),
            common::dd_get_value(row_set, row, col_count, "dataset_id", bigint(0)),
            common::dd_get_value(row_set, row, col_count, "input_queue_table_name", std::string()),
            common::dd_get_value(row_set, row, col_count, "input_queue_column_name", std::string()),
            common::dd_get_value(row_set, row, col_count, "scaling_factor", double(1)),
            common::dd_get_value(row_set, row, col_count, "dc_offset", double(0)));
    }
#endif
};

}
}
