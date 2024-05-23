#pragma once

#include "DAO/IRowMapper.hpp"
#include "model/SVRParameters.hpp"

namespace svr {
namespace dao {

class SVRParametersRowMapper : public IRowMapper<svr::datamodel::SVRParameters>{
public:
    datamodel::SVRParameters_ptr mapRow(const pqxx_tuple& rowSet) const override {
        return ptr<svr::datamodel::SVRParameters>(
                rowSet["id"].as<bigint>(0),
                rowSet["dataset_id"].as<bigint>(0),
                rowSet["input_queue_table_name"].as<std::string>(""),
                rowSet["input_queue_column_name"].as<std::string>(""),
                rowSet["levels"].as<size_t>(DEFAULT_SVRPARAM_DECON_LEVEL + 1),
                rowSet["decon_level"].as<size_t>(DEFAULT_SVRPARAM_DECON_LEVEL),
                rowSet["chunk_ix"].as<size_t>(DEFAULT_SVRPARAM_CHUNK_IX),
                rowSet["grad_level"].as<size_t>(DEFAULT_SVRPARAM_GRAD_LEVEL),
                rowSet["svr_c"].as<double>(DEFAULT_SVRPARAM_SVR_COST),
                rowSet["svr_epsilon"].as<double>(DEFAULT_SVRPARAM_SVR_EPSILON),
                rowSet["svr_kernel_param"].as<double>(DEFAULT_SVRPARAM_KERNEL_PARAM_1),
                rowSet["svr_kernel_param2"].as<double>(DEFAULT_SVRPARAM_KERNEL_PARAM_2),
                rowSet["svr_decremental_distance"].as<bigint>(DEFAULT_SVRPARAM_DECREMENT_DISTANCE),
                rowSet["svr_adjacent_levels_ratio"].as<double>(DEFAULT_SVRPARAM_ADJACENT_LEVELS_RATIO),
                static_cast<datamodel::kernel_type_e>(rowSet["svr_kernel_type"].as<size_t>(DEFAULT_SVRPARAM_KERNEL_TYPE_UINT)),
                rowSet["lag_count"].as<size_t>(DEFAULT_SVRPARAM_LAG_COUNT),
                rowSet["quantize"].as<double>(QUANTIZE_FIXED)
        );
    }
};
}
}
