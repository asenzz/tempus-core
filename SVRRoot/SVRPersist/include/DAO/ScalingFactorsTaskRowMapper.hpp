#pragma once

#include "DAO/IRowMapper.hpp"
#include "model/ScalingFactorsTask.hpp"

namespace svr{
namespace dao{

class ScalingFactorsTaskRowMapper : public IRowMapper<datamodel::ScalingFactorsTask>{
public:
    datamodel::ScalingFactorsTask_ptr map_row(const pqxx_tuple& row_set) const override 
    {
        return ptr<datamodel::ScalingFactorsTask>(
                row_set["id"].as<bigint>(0),
                row_set["dataset_id"].as<bigint>(0),
                row_set["status"].as<int>(0),
                row_set["mse"].as<double>(0)
        );
    }
#ifdef USE_DUCKDB
    datamodel::ScalingFactorsTask_ptr map_row(duckdb_result& row_set, const size_t col_count, const size_t row) const override
    {
        return ptr<datamodel::ScalingFactorsTask>(
                common::dd_get_value(row_set, row, col_count, "id", bigint(0)),
                common::dd_get_value(row_set, row, col_count, "dataset_id", bigint(0)),
                common::dd_get_value(row_set, row, col_count, "status", int(0)),
                common::dd_get_value(row_set, row, col_count, "mse", double(0))
        );
    }
#endif
};
}
}
