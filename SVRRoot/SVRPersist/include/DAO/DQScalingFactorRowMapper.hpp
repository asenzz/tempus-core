#pragma once

#include "DAO/IRowMapper.hpp"
#include "model/DQScalingFactor.hpp"

namespace svr {
namespace dao {

class DQScalingFactorRowMapper : public IRowMapper<svr::datamodel::DQScalingFactor>
{
private:
    //empty

public:
    datamodel::DQScalingFactor_ptr map_row(const pqxx_tuple& row_set) const override
    {
        return ptr<svr::datamodel::DQScalingFactor>(
                    row_set["id"].as<bigint>(0),
                    row_set["model_id"].as<bigint>(0),
                    row_set["level"].as<size_t>(0),
                    row_set["step"].as<size_t>(0),
                    row_set["scaling_factor_features"].as<double>(1),
                    row_set["scaling_factor_labels"].as<double>(1),
                    row_set["dc_offset_features"].as<double>(0),
                    row_set["dc_offset_labels"].as<double>(0),
                    row_set["gradient"].as<size_t>(0),
                    row_set["chunk"].as<size_t>(0)
                );
    }

};

}
}
