#pragma once

#include "DAO/IRowMapper.hpp"
#include "model/AutotuneTask.hpp"

namespace svr{
namespace dao{

class AutotuneTaskRowMapper : public IRowMapper<svr::datamodel::AutotuneTask>{
public:
    AutotuneTask_ptr map_row(const pqxx_tuple& row_set) const override {

        return ptr<svr::datamodel::AutotuneTask>(
                row_set["id"].as<bigint>(0),
                row_set["dataset_id"].as<bigint>(0),
                row_set["result_dataset_id"].as<bigint>(0),
                row_set["creation_time"].as<bpt::ptime>(bpt::not_a_date_time),
                row_set["done_time"].as<bpt::ptime>(bpt::not_a_date_time),
                row_set["parameters"].as<std::string>(""),
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
};
}
}
