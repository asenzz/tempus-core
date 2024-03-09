#pragma once

#include "DAO/IRowMapper.hpp"
#include "model/ScalingFactorsTask.hpp"

namespace svr{
namespace dao{

class ScalingFactorsTaskRowMapper : public IRowMapper<svr::datamodel::ScalingFactorsTask>{
public:
    ScalingFactorsTask_ptr mapRow(const pqxx_tuple& rowSet) const override {

        return ptr<svr::datamodel::ScalingFactorsTask>(
                rowSet["id"].as<bigint>(0),
                rowSet["dataset_id"].as<bigint>(0),
                rowSet["status"].as<int>(0),
                rowSet["mse"].as<double>(0)
        );
    }
};
}
}
