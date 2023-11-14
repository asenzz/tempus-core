#pragma once

#include "DAO/IRowMapper.hpp"
#include "model/DecrementTask.hpp"

namespace svr {
namespace dao {

class DecrementTaskRowMapper : public IRowMapper<svr::datamodel::DecrementTask>
{
private:
    // empty

public:
    DecrementTask_ptr mapRow(const pqxx_tuple& rowSet) const override
    {
        return std::make_shared<svr::datamodel::DecrementTask>(
                    rowSet["id"].as<bigint>(),
                    rowSet["dataset_id"].is_null() ? 0 : rowSet["dataset_id"].as<bigint>(),
                    rowSet["start_task_time"].is_null() ? bpt::ptime() : bpt::time_from_string(rowSet["start_task_time"].as<std::string>()),
                    rowSet["end_task_time"].is_null() ? bpt::ptime() : bpt::time_from_string(rowSet["end_task_time"].as<std::string>()),
                    rowSet["start_train_time"].is_null() ? bpt::ptime() : bpt::time_from_string(rowSet["start_train_time"].as<std::string>()),
                    rowSet["end_train_time"].is_null() ? bpt::ptime() : bpt::time_from_string(rowSet["end_train_time"].as<std::string>()),
                    rowSet["start_validation_time"].is_null() ? bpt::ptime() : bpt::time_from_string(rowSet["start_validation_time"].as<std::string>()),
                    rowSet["end_validation_time"].is_null() ? bpt::ptime() : bpt::time_from_string(rowSet["end_validation_time"].as<std::string>()),
                    rowSet["parameters"].is_null() ? "" : rowSet["parameters"].as<std::string>(),
                    rowSet["status"].is_null() ? 0 : rowSet["status"].as<int>(),
                    rowSet["decrement_step"].is_null() ? "" : rowSet["decrement_step"].as<std::string>(),
                    rowSet["vp_sliding_direction"].is_null() ? 0 : rowSet["vp_sliding_direction"].as<size_t>(),
                    rowSet["vp_slide_count"].is_null() ? 0 : rowSet["vp_slide_count"].as<size_t>(),
                    rowSet["vp_slide_period_sec"].is_null() ? bpt::seconds(0) : bpt::seconds(rowSet["vp_slide_period_sec"].as<long>()),
                    rowSet["values"].is_null() ? "" : rowSet["values"].as<std::string>(),
                    rowSet["suggested_value"].is_null() ? "" : rowSet["suggested_value"].as<std::string>()
                );
    }
};

}
}
