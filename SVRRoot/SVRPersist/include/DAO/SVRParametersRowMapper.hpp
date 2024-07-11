#pragma once

#include "DAO/IRowMapper.hpp"
#include "model/SVRParameters.hpp"
#include "appcontext.hpp"

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
                rowSet["levels"].as<size_t>(datamodel::C_default_svrparam_decon_level + 1),
                rowSet["decon_level"].as<size_t>(datamodel::C_default_svrparam_decon_level),
                rowSet["step"].as<size_t>(datamodel::C_default_svrparam_step),
                rowSet["chunk_ix"].as<size_t>(datamodel::C_default_svrparam_chunk_ix),
                rowSet["grad_level"].as<size_t>(datamodel::C_default_svrparam_grad_level),
                rowSet["svr_c"].as<double>(datamodel::C_default_svrparam_svr_cost),
                rowSet["svr_epsilon"].as<double>(datamodel::C_default_svrparam_svr_epsilon),
                rowSet["svr_kernel_param"].as<double>(datamodel::C_default_svrparam_kernel_param1),
                rowSet["svr_kernel_param2"].as<double>(datamodel::C_default_svrparam_kernel_param2),
                rowSet["svr_decremental_distance"].as<bigint>(datamodel::C_default_svrparam_decrement_distance),
                rowSet["svr_adjacent_levels_ratio"].as<double>(datamodel::C_default_svrparam_adjacent_levels_ratio),
                static_cast<datamodel::kernel_type_e>(rowSet["svr_kernel_type"].as<size_t>(datamodel::C_default_svrparam_kernel_type_uint)),
                rowSet["lag_count"].as<size_t>(datamodel::C_default_svrparam_lag_count)
        );
    }
};
}
}
