#pragma once

#include "DAO/IRowMapper.hpp"
#include "model/SVRParameters.hpp"
#include "appcontext.hpp"

namespace svr {
namespace dao {

class SVRParametersRowMapper : public IRowMapper<datamodel::SVRParameters>{
public:
    datamodel::SVRParameters_ptr map_row(const pqxx_tuple& row_set) const override {
        return ptr<datamodel::SVRParameters>(
                row_set["id"].as<bigint>(0),
                row_set["dataset_id"].as<bigint>(0),
                row_set["input_queue_table_name"].as<std::string>(""),
                row_set["input_queue_column_name"].as<std::string>(""),
                row_set["levels"].as<uint16_t>(datamodel::C_default_svrparam_decon_level + 1),
                row_set["decon_level"].as<uint16_t>(datamodel::C_default_svrparam_decon_level),
                row_set["step"].as<uint16_t>(datamodel::C_default_svrparam_step),
                row_set["chunk_ix"].as<uint16_t>(datamodel::C_default_svrparam_chunk_ix),
                row_set["grad_level"].as<uint16_t>(datamodel::C_default_svrparam_grad_level),
                row_set["svr_c"].as<double>(datamodel::C_default_svrparam_svr_cost),
                row_set["svr_epsilon"].as<double>(datamodel::C_default_svrparam_svr_epsilon),
                row_set["svr_kernel_param"].as<double>(datamodel::C_default_svrparam_kernel_param1),
                row_set["svr_kernel_param2"].as<double>(datamodel::C_default_svrparam_kernel_param2),
                row_set["svr_kernel_param3"].as<double>(datamodel::C_default_svrparam_kernel_param_tau),
                row_set["svr_decremental_distance"].as<uint32_t>(datamodel::C_default_svrparam_decrement_distance),
                row_set["svr_adjacent_levels_ratio"].as<double>(datamodel::C_default_svrparam_adjacent_levels_ratio),
                datamodel::e_kernel_type(row_set["svr_kernel_type"].as<uint16_t>(datamodel::C_default_svrparam_kernel_type_uint)),
                row_set["lag_count"].as<uint16_t>(datamodel::C_default_svrparam_lag_count),
                std::set<uint16_t>{},
                datamodel::t_feature_mechanics::load(row_set["feature_mechanics"].as<std::string>(""))
        );
    }

    datamodel::ScalingFactorsTask_ptr map_row(duckdb_result& row_set, const size_t col_count, const size_t row) const override
    {
        return ptr<datamodel::ScalingFactorsTask>(
                common::dd_get_value(row_set, row, col_count, "id", bigint(0)),
                common::dd_get_value(row_set, row, col_count, "dataset_id", bigint(0)),
                common::dd_get_value(row_set, row, col_count, "intel_queue_table_name", std::string()),
                common::dd_get_value(row_set, row, col_count, "input_queue_column_name", std::string()),
                common::dd_get_value(row_set, row, col_count, "levels", uint16_t(0)),
                common::dd_get_value(row_set, row, col_count, "decon_level", uint16_t(0)),
                common::dd_get_value(row_set, row, col_count, "step", uint16_t(0)),
                common::dd_get_value(row_set, row, col_count, "chunk_ix", uint16_t(0)),
                common::dd_get_value(row_set, row, col_count, "grad_level", uint16_t(0)),
                common::dd_get_value(row_set, row, col_count, "svr_c", double(0)),
                common::dd_get_value(row_set, row, col_count, "svr_epsilon", double(0)),
                common::dd_get_value(row_set, row, col_count, "svr_kernel_param", double(0)),
                common::dd_get_value(row_set, row, col_count, "svr_kernel_param2", double(0)),
                common::dd_get_value(row_set, row, col_count, "svr_kernel_param3", double(0)),
                common::dd_get_value(row_set, row, col_count, "svr_decremental_distance", uint32_t(0)),
                common::dd_get_value(row_set, row, col_count, "svr_adjacent_levels_ratio", double(0)),
                datamodel::e_kernel_type(common::dd_get_value(row_set, row, col_count, "svr_kernel_type", datamodel::C_default_svrparam_kernel_type_uint)),
                common::dd_get_value(row_set, row, col_count, "lag_count", uint16_t(0)),
                std::set<uint16_t>{},
                datamodel::t_feature_mechanics::load(common::dd_get_value(row_set, row, col_count, "feature_mechanics", std::string()))
            );
    }
};

}
}
