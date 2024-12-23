#pragma once

#include "DAO/IRowMapper.hpp"
#include "model/PredictionTask.hpp"

namespace svr{
namespace dao{

class PredictionTaskRowMapper : public IRowMapper<svr::datamodel::PredictionTask>{
public:
    PredictionTask_ptr map_row(const pqxx_tuple& row_set) const override {

        return ptr<svr::datamodel::PredictionTask>(
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
};
}
}
