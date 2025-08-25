#pragma once

#include "DAO/IRowMapper.hpp"
#include "model/DQScalingFactor.hpp"

namespace svr {
namespace dao {

class DQScalingFactorRowMapper : public IRowMapper<datamodel::DQScalingFactor>
{
public:
    datamodel::DQScalingFactor_ptr map_row(const pqxx_tuple& row_set) const override
    {
        return ptr<datamodel::DQScalingFactor>(
                    row_set["id"].as<bigint>(0),
                    row_set["model_id"].as<bigint>(0),
                    row_set["level"].as<uint16_t>(0),
                    row_set["step"].as<uint16_t>(0),
                    row_set["scaling_factor_features"].as<double>(1),
                    row_set["scaling_factor_labels"].as<double>(1),
                    row_set["dc_offset_features"].as<double>(0),
                    row_set["dc_offset_labels"].as<double>(0),
                    row_set["gradient"].as<uint16_t>(0),
                    row_set["chunk"].as<uint16_t>(0)
                );
    }

    datamodel::DQScalingFactor_ptr map_row(duckdb_result& row_set, const size_t col_count, const size_t row) const override
    {
        return ptr<datamodel::DQScalingFactor>(
            common::dd_get_value(row_set, row, col_count, "id", bigint(0)),
            common::dd_get_value(row_set, row, col_count, "model_id", bigint(0)),
            common::dd_get_value(row_set, row, col_count, "level", uint16_t(0)),
            common::dd_get_value(row_set, row, col_count, "step", uint16_t(0)),
            common::dd_get_value(row_set, row, col_count, "scaling_factor_features", double(1)),
            common::dd_get_value(row_set, row, col_count, "scaling_factor_labels", double(1)),
            common::dd_get_value(row_set, row, col_count, "dc_offset_features", double(0)),
            common::dd_get_value(row_set, row, col_count, "dc_offset_labels", double(0)),
            common::dd_get_value(row_set, row, col_count, "gradient", uint16_t(0)),
            common::dd_get_value(row_set, row, col_count, "chunk", uint16_t(0)));
    }
};

}
}
