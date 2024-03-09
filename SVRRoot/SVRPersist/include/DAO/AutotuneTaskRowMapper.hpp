#pragma once

#include "DAO/IRowMapper.hpp"
#include "model/AutotuneTask.hpp"

namespace svr{
namespace dao{

class AutotuneTaskRowMapper : public IRowMapper<svr::datamodel::AutotuneTask>{
public:
    AutotuneTask_ptr mapRow(const pqxx_tuple& rowSet) const override {

        return ptr<svr::datamodel::AutotuneTask>(
                rowSet["id"].as<bigint>(0),
                rowSet["dataset_id"].as<bigint>(0),
                rowSet["result_dataset_id"].as<bigint>(0),
                rowSet["creation_time"].is_null() ? bpt::ptime() : bpt::time_from_string(rowSet["creation_time"].as<std::string>("")),
                rowSet["done_time"].is_null() ? bpt::ptime() : bpt::time_from_string(rowSet["done_time"].as<std::string>("")),
                rowSet["parameters"].as<std::string>(""),
                rowSet["start_train_time"].is_null() ? bpt::ptime() : bpt::time_from_string(rowSet["start_train_time"].as<std::string>("")),
                rowSet["end_train_time"].is_null() ? bpt::ptime() : bpt::time_from_string(rowSet["end_train_time"].as<std::string>("")),
                rowSet["start_tuning_time"].is_null() ? bpt::ptime() : bpt::time_from_string(rowSet["start_tuning_time"].as<std::string>("")),
                rowSet["end_tuning_time"].is_null() ? bpt::ptime() : bpt::time_from_string(rowSet["end_tuning_time"].as<std::string>("")),
                rowSet["vp_sliding_direction"].as<size_t>(0),
                rowSet["vp_slide_count"].as<size_t>(0),
                rowSet["vp_slide_period_sec"].is_null() ? bpt::seconds(0) : bpt::seconds(rowSet["vp_slide_period_sec"].as<long>(0)),
                rowSet["pso_best_points_counter"].as<size_t>(0),
                rowSet["pso_iteration_number"].as<size_t>(0),
                rowSet["pso_particles_number"].as<size_t>(0),
                rowSet["pso_topology"].as<size_t>(0),
                rowSet["nm_max_iteration_number"].as<size_t>(0),
                rowSet["nm_tolerance"].as<double>(0),
                rowSet["status"].as<int>(0),
                rowSet["mse"].as<double>(-1)
        );
    }
};
}
}
