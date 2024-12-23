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
    DecrementTask_ptr map_row(const pqxx_tuple& row_set) const override
    {
        return ptr<svr::datamodel::DecrementTask>(
                    row_set["id"].as<bigint>(0),
                    row_set["dataset_id"].is_null() ? 0 : row_set["dataset_id"].as<bigint>(),
                    row_set["start_task_time"].as<bpt::ptime>(bpt::not_a_date_time),
                    row_set["end_task_time"].as<bpt::ptime>(bpt::not_a_date_time),
                    row_set["start_train_time"].as<bpt::ptime>(bpt::not_a_date_time),
                    row_set["end_train_time"].as<bpt::ptime>(bpt::not_a_date_time),
                    row_set["start_validation_time"].as<bpt::ptime>(bpt::not_a_date_time),
                    row_set["end_validation_time"].as<bpt::ptime>(bpt::not_a_date_time),
                    row_set["parameters"].as<std::string>(""),
                    row_set["status"].as<int>(0),
                    row_set["decrement_step"].as<std::string>(""),
                    row_set["vp_sliding_direction"].as<size_t>(0),
                    row_set["vp_slide_count"].as<size_t>(0),
                    bpt::seconds(row_set["vp_slide_period_sec"].as<long>(0)),
                    row_set["values"].as<std::string>(""),
                    row_set["suggested_value"].as<std::string>("")
                );
    }
};

}
}
