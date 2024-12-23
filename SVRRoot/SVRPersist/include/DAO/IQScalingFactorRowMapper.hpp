#pragma once

#include "DAO/IRowMapper.hpp"
#include "model/IQScalingFactor.hpp"

namespace svr {
namespace dao {

class IQScalingFactorRowMapper : public IRowMapper<svr::datamodel::IQScalingFactor>
{
public:
    datamodel::IQScalingFactor_ptr map_row(const pqxx_tuple &row_set) const override
    {
        return ptr<svr::datamodel::IQScalingFactor>(
                row_set["id"].as<bigint>(0),
                row_set["dataset_id"].as<bigint>(0),
                row_set["input_queue_table_name"].as<std::string>(""),
                row_set["input_queue_column_name"].as<std::string>(""),
                row_set["scaling_factor"].as<double>(1),
                row_set["dc_offset"].as<double>(0)
        );
    }
};

}
}
