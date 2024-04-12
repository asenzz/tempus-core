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
    datamodel::DQScalingFactor_ptr mapRow(const pqxx_tuple& rowSet) const override
    {
        return ptr<svr::datamodel::DQScalingFactor>(
                    rowSet["id"].as<bigint>(0),
                    rowSet["model_id"].as<bigint>(0),
                    rowSet["level"].as<size_t>(0),
                    rowSet["scaling_factor_features"].as<double>(1),
                    rowSet["scaling_factor_labels"].as<double>(1),
                    rowSet["dc_offset_features"].as<double>(0),
                    rowSet["dc_offset_labels"].as<double>(0),
                    rowSet["gradient"].as<size_t>(0),
                    rowSet["chunk"].as<size_t>(0)
                );
    }

};

}
}
