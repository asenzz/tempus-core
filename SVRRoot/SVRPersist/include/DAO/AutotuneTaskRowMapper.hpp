#pragma once

#include "DAO/IRowMapper.hpp"
#include "model/AutotuneTask.hpp"

namespace svr {
namespace dao {

class AutotuneTaskRowMapper : public IRowMapper<datamodel::AutotuneTask>
{
public:
    AutotuneTask_ptr map_row(const pqxx_tuple& row_set) const override
    {
        return ptr<datamodel::AutotuneTask>(
            row_set["id"].as<bigint>(0),
            row_set["dataset_id"].as<bigint>(0),
            row_set["result_dataset_id"].as<bigint>(0),
            row_set["creation_time"].as<bpt::ptime>(bpt::not_a_date_time),
            row_set["done_time"].as<bpt::ptime>(bpt::not_a_date_time),
            row_set["parameters"].as<std::string>(),
            row_set["start_train_time"].as<bpt::ptime>(bpt::not_a_date_time),
            row_set["end_train_time"].as<bpt::ptime>(bpt::not_a_date_time),
            row_set["start_tuning_time"].as<bpt::ptime>(bpt::not_a_date_time),
            row_set["end_tuning_time"].as<bpt::ptime>(bpt::not_a_date_time),
            row_set["vp_sliding_direction"].as<size_t>(0),
            row_set["vp_slide_count"].as<size_t>(0),
            bpt::seconds(row_set["vp_slide_period_sec"].as<uint32_t>(0)),
            row_set["pso_best_points_counter"].as<size_t>(0),
            row_set["pso_iteration_number"].as<size_t>(0),
            row_set["pso_particles_number"].as<size_t>(0),
            row_set["pso_topology"].as<size_t>(0),
            row_set["nm_max_iteration_number"].as<size_t>(0),
            row_set["nm_tolerance"].as<double>(0),
            row_set["status"].as<int>(0),
            row_set["mse"].as<double>(-1)
        );
    }

    AutotuneTask_ptr map_row(duckdb_result& row_set, const size_t col_count, const size_t row) const override
    {
        return ptr<datamodel::AutotuneTask>(
            common::dd_get_value(row_set, row, col_count, "id", bigint(0)),
            common::dd_get_value(row_set, row, col_count, "dataset_id", bigint(0)),
            common::dd_get_value(row_set, row, col_count, "result_dataset_id", bigint(0)),
            common::dd_get_value<bpt::ptime>(row_set, row, col_count, "creation_time", bpt::not_a_date_time),
            common::dd_get_value<bpt::ptime>(row_set, row, col_count, "done_time", bpt::not_a_date_time),
            common::dd_get_value(row_set, row, col_count, "parameters", std::string()),
            common::dd_get_value<bpt::ptime>(row_set, row, col_count, "start_train_time", bpt::not_a_date_time),
            common::dd_get_value<bpt::ptime>(row_set, row, col_count, "end_train_time", bpt::not_a_date_time),
            common::dd_get_value<bpt::ptime>(row_set, row, col_count, "start_tuning_time", bpt::not_a_date_time),
            common::dd_get_value<bpt::ptime>(row_set, row, col_count, "end_tuning_time", bpt::not_a_date_time),
            common::dd_get_value(row_set, row, col_count, "vp_sliding_direction", size_t(0)),
            common::dd_get_value(row_set, row, col_count, "vp_slide_count", size_t(0)),
            common::dd_get_value(row_set, row, col_count, "vp_slide_period_sec", bpt::seconds(0)),
            common::dd_get_value(row_set, row, col_count, "pso_best_points_counter", size_t(0)),
            common::dd_get_value(row_set, row, col_count, "pso_iteration_number", size_t(0)),
            common::dd_get_value(row_set, row, col_count, "pso_particles_number", size_t(0)),
            common::dd_get_value(row_set, row, col_count, "pso_topology", size_t(0)),
            common::dd_get_value(row_set, row, col_count, "nm_max_iteration_number", size_t(0)),
            common::dd_get_value(row_set, row, col_count, "nm_tolerance", double(0)),
            common::dd_get_value(row_set, row, col_count, "status", int(0)),
            common::dd_get_value(row_set, row, col_count, "mse", double(-1))
        );
    }
};

}
}