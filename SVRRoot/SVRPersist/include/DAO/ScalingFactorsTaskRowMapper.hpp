#pragma once

#include "DAO/IRowMapper.hpp"
#include "model/ScalingFactorsTask.hpp"

namespace svr{
namespace dao{

class ScalingFactorsTaskRowMapper : public IRowMapper<svr::datamodel::ScalingFactorsTask>{
public:
    ScalingFactorsTask_ptr map_row(const pqxx_tuple& row_set) const override {

        return ptr<svr::datamodel::ScalingFactorsTask>(
                row_set["id"].as<bigint>(0),
                row_set["dataset_id"].as<bigint>(0),
                row_set["status"].as<int>(0),
                row_set["mse"].as<double>(0)
        );
    }
};
}
}
