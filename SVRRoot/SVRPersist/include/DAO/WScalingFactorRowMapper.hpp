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

    datamodel::WScalingFactor_ptr map_row(duckdb_result& row_set, const size_t col_count, const size_t row) const override
    {
        return ptr<datamodel::WScalingFactor>(
                common::dd_get_value(row_set, row, col_count, "id", bigint(0)),
                common::dd_get_value(row_set, row, col_count, "dataset_id", bigint(0)),
                common::dd_get_value(row_set, row, col_count, "step" uint16_t(0)),
                common::dd_get_value(row_set, row, col_count, "scaling_factor", double(1)),
                common::dd_get_value(row_set, row, col_count, "dc_offset", double(0))
        );
    }

};

}
}
