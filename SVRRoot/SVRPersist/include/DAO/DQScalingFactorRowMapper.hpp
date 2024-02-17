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
        return std::make_shared<svr::datamodel::DQScalingFactor>(
                    rowSet["id"].as<bigint>(0),
                    rowSet["dataset_id"].as<bigint>(0),
                    rowSet["input_queue_table_name"].as<std::string>(""),
                    rowSet["input_queue_column_name"].as<std::string>(""),
                    rowSet["level"].as<size_t>(0),
                    rowSet["scaling_factor_features"].as<double>(1),
                    rowSet["scaling_factor_features"].as<double>(1),
                    rowSet["dc_offset_features"].as<double>(0),
                    rowSet["dc_offset_labels"].as<double>(0)
                );
    }

};

}
}
