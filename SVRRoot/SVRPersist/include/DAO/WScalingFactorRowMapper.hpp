#pragma once

#include "DAO/IRowMapper.hpp"
#include "model/WScalingFactor.hpp"

namespace svr {
namespace dao {

class WScalingFactorRowMapper : public IRowMapper<svr::datamodel::WScalingFactor>
{
public:
    datamodel::WScalingFactor_ptr map_row(const pqxx_tuple &row_set) const override
    {
        return ptr<svr::datamodel::WScalingFactor>(
                row_set["id"].as<bigint>(0),
                row_set["dataset_id"].as<bigint>(0),
                row_set["step"].as<uint16_t>(0),
                row_set["scaling_factor"].as<double>(1),
                row_set["dc_offset"].as<double>(0)
        );
    }
};

}
}
