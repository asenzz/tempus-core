#pragma once

#include "DAO/IRowMapper.hpp"
#include "model/SVRParameters.hpp"

namespace svr {
namespace dao {

class SVRParametersRowMapper : public IRowMapper<svr::datamodel::SVRParameters>{
public:
    datamodel::SVRParameters_ptr mapRow(const pqxx_tuple& rowSet) const override {
        return std::make_shared<svr::datamodel::SVRParameters>(
                rowSet["id"].as<bigint>(0),
                rowSet["dataset_id"].as<bigint>(0),
                rowSet["input_queue_table_name"].as<std::string>(""),
                rowSet["input_queue_column_name"].as<std::string>(""),
                rowSet["decon_level"].as<size_t>(0),
                rowSet["chunk_ix"].as<size_t>(0),
                rowSet["grad_level"].as<size_t>(0),
                mimo_type_e(rowSet["mimo_type"].as<size_t>(1)),
                rowSet["svr_c"].as<double>(std::numeric_limits<double>::quiet_NaN()),
                rowSet["svr_epsilon"].as<double>(std::numeric_limits<double>::quiet_NaN()),
                rowSet["svr_kernel_param"].as<double>(std::numeric_limits<double>::quiet_NaN()),
                rowSet["svr_kernel_param2"].as<double>(std::numeric_limits<double>::quiet_NaN()),
                rowSet["svr_decremental_distance"].as<bigint>(1),
                rowSet["svr_adjacent_levels_ratio"].as<double>(std::numeric_limits<double>::quiet_NaN()),
                static_cast<datamodel::kernel_type_e>(rowSet["svr_kernel_type"].as<size_t>(0)),
                rowSet["lag_count"].as<size_t>(0)
        );
    }
};
}
}
