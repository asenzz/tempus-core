#pragma once

#include "DAO/IRowMapper.hpp"
#include "model/IQScalingFactor.hpp"

namespace svr {
namespace dao {

class IQScalingFactorRowMapper : public IRowMapper<svr::datamodel::IQScalingFactor>
{
private:
    //empty

public:
    IQScalingFactor_ptr mapRow(const pqxx_tuple& rowSet) const override
    {
        return std::make_shared<svr::datamodel::IQScalingFactor>(
                    rowSet["id"].as<bigint>(0),
                    rowSet["dataset_id"].as<bigint>(0),
                    rowSet["input_queue_table_name"].as<std::string>(""),
                    rowSet["scaling_factor"].as<double>(1)
                );
    }

};

}
}
