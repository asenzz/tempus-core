#pragma once

#include "DAO/IRowMapper.hpp"
#include "model/PredictionTask.hpp"

namespace svr{
namespace dao{

class PredictionTaskRowMapper : public IRowMapper<datamodel::PredictionTask>{
public:
    datamodel::PredictionTask_ptr map_row(const pqxx_tuple& row_set) const override 
    {
        return ptr<datamodel::PredictionTask>(
                row_set["id"].as<bigint>(0),
                row_set["dataset_id"].as<bigint>(0),
                row_set["start_time"].as<bpt::ptime>(bpt::not_a_date_time),
                row_set["end_time"].as<bpt::ptime>(bpt::not_a_date_time),
                row_set["start_prediction_time"].as<bpt::ptime>(bpt::not_a_date_time),
                row_set["end_prediction_time"].as<bpt::ptime>(bpt::not_a_date_time),
                row_set["status"].as<int>(0),
                row_set["mse"].as<double>(0)
        );
    }

#ifdef USE_DUCKDB
    datamodel::PredictionTask_ptr map_row(duckdb_result& row_set, const size_t col_count, const size_t row) const override
    {
        return ptr<datamodel::PredictionTask>(
            common::dd_get_value(row_set, row, col_count, "id", bigint(0)),
            common::dd_get_value(row_set, row, col_count, "dataset_id", bigint(0)),
            common::dd_get_value<bpt::ptime>(row_set, row, col_count, "start_time", bpt::not_a_date_time),
            common::dd_get_value<bpt::ptime>(row_set, row, col_count, "end_time", bpt::not_a_date_time),
            common::dd_get_value<bpt::ptime>(row_set, row, col_count, "start_prediction_time", bpt::not_a_date_time),
            common::dd_get_value<bpt::ptime>(row_set, row, col_count, "end_prediction_time", bpt::not_a_date_time),
            common::dd_get_value(row_set, row, col_count, "status", int(0)),
            common::dd_get_value(row_set, row, col_count, "mse", double(0))
        );
    }
#endif

};
}
}
