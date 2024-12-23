#pragma once

#include "DAO/IRowMapper.hpp"
#include "model/SVRParameters.hpp"
#include "appcontext.hpp"

namespace svr {
namespace dao {

class SVRParametersRowMapper : public IRowMapper<svr::datamodel::SVRParameters>{
public:
    datamodel::SVRParameters_ptr map_row(const pqxx_tuple& row_set) const override {
        return ptr<svr::datamodel::SVRParameters>(
                row_set["id"].as<bigint>(0),
                row_set["dataset_id"].as<bigint>(0),
                row_set["input_queue_table_name"].as<std::string>(""),
                row_set["input_queue_column_name"].as<std::string>(""),
                row_set["levels"].as<size_t>(datamodel::C_default_svrparam_decon_level + 1),
                row_set["decon_level"].as<size_t>(datamodel::C_default_svrparam_decon_level),
                row_set["step"].as<size_t>(datamodel::C_default_svrparam_step),
                row_set["chunk_ix"].as<size_t>(datamodel::C_default_svrparam_chunk_ix),
                row_set["grad_level"].as<size_t>(datamodel::C_default_svrparam_grad_level),
                row_set["svr_c"].as<double>(datamodel::C_default_svrparam_svr_cost),
                row_set["svr_epsilon"].as<double>(datamodel::C_default_svrparam_svr_epsilon),
                row_set["svr_kernel_param"].as<double>(datamodel::C_default_svrparam_kernel_param1),
                row_set["svr_kernel_param2"].as<double>(datamodel::C_default_svrparam_kernel_param2),
                row_set["svr_decremental_distance"].as<bigint>(datamodel::C_default_svrparam_decrement_distance),
                row_set["svr_adjacent_levels_ratio"].as<double>(datamodel::C_default_svrparam_adjacent_levels_ratio),
                static_cast<datamodel::e_kernel_type>(row_set["svr_kernel_type"].as<size_t>(datamodel::C_default_svrparam_kernel_type_uint)),
                row_set["lag_count"].as<size_t>(datamodel::C_default_svrparam_lag_count)
        );
    }
};
}
}
