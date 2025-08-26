#pragma once

#include "DAO/IRowMapper.hpp"
#include "model/DecrementTask.hpp"

namespace svr {
namespace dao {

class DecrementTaskRowMapper : public IRowMapper<datamodel::DecrementTask>
{
private:
    // empty

public:
    datamodel::DecrementTask_ptr map_row(const pqxx_tuple& row_set) const override
    {
        return ptr<datamodel::DecrementTask>(
                    row_set["id"].as<bigint>(0),
                    row_set["dataset_id"].as<bigint>(0),
                    row_set["start_task_time"].as<bpt::ptime>(bpt::not_a_date_time),
                    row_set["end_task_time"].as<bpt::ptime>(bpt::not_a_date_time),
                    row_set["start_train_time"].as<bpt::ptime>(bpt::not_a_date_time),
                    row_set["end_train_time"].as<bpt::ptime>(bpt::not_a_date_time),
                    row_set["start_validation_time"].as<bpt::ptime>(bpt::not_a_date_time),
                    row_set["end_validation_time"].as<bpt::ptime>(bpt::not_a_date_time),
                    row_set["parameters"].as<std::string>(),
                    row_set["status"].as<int>(0),
                    row_set["decrement_step"].as<std::string>(),
                    row_set["vp_sliding_direction"].as<uint32_t>(0),
                    row_set["vp_slide_count"].as<uint32_t>(0),
                    bpt::seconds(row_set["vp_slide_period_sec"].as<uint32_t>(0)),
                    row_set["values"].as<std::string>(""),
                    row_set["suggested_value"].as<std::string>("")
                );
    }

#ifdef USE_DUCKDB

    datamodel::DecrementTask_ptr map_row(duckdb_result& row_set, const size_t col_count, const size_t row) const override
    {
        return ptr<datamodel::DecrementTask>(
            common::dd_get_value(row_set, row, col_count, "id", bigint(0)),
            common::dd_get_value(row_set, row, col_count, "dataset_id", bigint(0)),
            common::dd_get_value<bpt::ptime>(row_set, row, col_count, "start_task_time", bpt::not_a_date_time),
            common::dd_get_value<bpt::ptime>(row_set, row, col_count, "end_task_time", bpt::not_a_date_time),
            common::dd_get_value<bpt::ptime>(row_set, row, col_count, "start_train_time", bpt::not_a_date_time),
            common::dd_get_value<bpt::ptime>(row_set, row, col_count, "end_train_time", bpt::not_a_date_time),
            common::dd_get_value<bpt::ptime>(row_set, row, col_count, "start_validation_time", bpt::not_a_date_time),
            common::dd_get_value<bpt::ptime>(row_set, row, col_count, "end_validation_time", bpt::not_a_date_time),
            common::dd_get_value(row_set, row, col_count, "parameters", std::string()),
            common::dd_get_value(row_set, row, col_count, "status", int(0)),
            common::dd_get_value(row_set, row, col_count, "decrement_step", std::string()),
            common::dd_get_value(row_set, row, col_count, "vp_sliding_direction", uint32_t(0)),
            common::dd_get_value(row_set, row, col_count, "vp_slide_count", uint32_t(0)),
            bpt::seconds(common::dd_get_value(row_set, row, col_count, "vp_slide_period_sec", uint32_t(0))),
            common::dd_get_value(row_set, row, col_count, "values", std::string()),
            common::dd_get_value(row_set, row, col_count, "suggested_value", std::string())
        );
    }

#endif

};

}
}
