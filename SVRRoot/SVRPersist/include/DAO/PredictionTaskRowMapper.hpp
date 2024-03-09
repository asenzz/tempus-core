#pragma once

#include "DAO/IRowMapper.hpp"
#include "model/PredictionTask.hpp"

namespace svr{
namespace dao{

class PredictionTaskRowMapper : public IRowMapper<svr::datamodel::PredictionTask>{
public:
    PredictionTask_ptr mapRow(const pqxx_tuple& rowSet) const override {

        return ptr<svr::datamodel::PredictionTask>(
                rowSet["id"].as<bigint>(0),
                rowSet["dataset_id"].as<bigint>(0),
                rowSet["start_time"].is_null() ? bpt::ptime() : bpt::time_from_string(rowSet["start_time"].as<std::string>("")),
                rowSet["end_time"].is_null() ? bpt::ptime() : bpt::time_from_string(rowSet["end_time"].as<std::string>("")),
                rowSet["start_prediction_time"].is_null() ? bpt::ptime() : bpt::time_from_string(rowSet["start_prediction_time"].as<std::string>("")),
                rowSet["end_prediction_time"].is_null() ? bpt::ptime() : bpt::time_from_string(rowSet["end_prediction_time"].as<std::string>("")),
                rowSet["status"].as<int>(0),
                rowSet["mse"].as<double>(0)
        );
    }
};
}
}
