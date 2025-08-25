#pragma once

#include <cstddef>
#include <sstream>
#include "DAO/IRowMapper.hpp"
#include "model/Model.hpp"
#include <pqxx/binarystring>
#include "util/CompressionUtils.hpp"
#include "util/string_utils.hpp"
#include "onlinesvr_persist.tpp"
#include "appcontext.hpp"
#include "common/constants.hpp"

namespace svr {
namespace dao {

// TODO Not fully working, implement model persistence to binary blob and test!
class ModelRowMapper : public IRowMapper<datamodel::Model>
{
public:
    datamodel::Model_ptr map_row(const pqxx_tuple &row_set) const override
    {
        return ptr<datamodel::Model>(
                row_set["id"].as<bigint>(0),
                row_set["ensemble_id"].as<bigint>(0),
                row_set["decon_level"].as(datamodel::C_default_svrparam_decon_level),
                row_set["step"].as(datamodel::C_default_svrparam_step),
                PROPS.get_multiout(),
                row_set["gradients"].as(datamodel::C_default_svrparam_grad_level + 1),
                common::AppConfig::C_default_kernel_length,
                std::deque<datamodel::OnlineSVR_ptr>{},
                row_set["last_modified_time"].as(bpt::second_clock::local_time()),
                row_set["last_modeled_value_time"].as<bpt::ptime>(bpt::not_a_date_time)
        );
    }

    datamodel::Model_ptr map_row(duckdb_result& row_set, const size_t col_count, const size_t row) const override
    {
        return ptr<datamodel::Model>(
            common::dd_get_value(row_set, row, col_count, "id", bigint(0)),
            common::dd_get_value(row_set, row, col_count, "ensemble_id", bigint(0)),
            common::dd_get_value(row_set, row, col_count, "decon_level", datamodel::C_default_svrparam_decon_level),
            common::dd_get_value(row_set, row, col_count, "step", datamodel::C_default_svrparam_step),
            PROPS.get_multiout(),
            common::dd_get_value(row_set, row, col_count, "gradients", datamodel::C_default_svrparam_grad_level + 1),
            common::AppConfig::C_default_kernel_length,
            std::deque<datamodel::OnlineSVR_ptr>(),
            common::dd_get_value(row_set, row, col_count, "last_modified_time", bpt::second_clock::local_time()),
            common::dd_get_value<bpt::ptime>(row_set, row, col_count, "last_modeled_value_time", bpt::not_a_date_time)
        );
    }

};

class SVRModelRowMapper : public IRowMapper<datamodel::OnlineSVR>
{
public:
    datamodel::OnlineSVR_ptr map_row(const pqxx_tuple &row_set) const override
    {
        std::stringstream model_bin;
        const auto &binstr = row_set["model_binary"].as<std::basic_string<std::byte>>();
        model_bin.write((const char *)binstr.c_str(), binstr.size());
        return ptr<datamodel::OnlineSVR>(row_set["id"].as<bigint>(0), row_set["model_id"].as<bigint>(0), model_bin);
    }

    datamodel::OnlineSVR_ptr map_row(duckdb_result& row_set, const size_t col_count, const size_t row) const override
    {
        std::stringstream model_bin;
        const auto binstr = common::dd_get_value(row_set, row, col_count, "model_binary", std::vector<uint8_t>());
        model_bin.write((const char *)binstr.data(), binstr.size());
        return ptr<datamodel::OnlineSVR>(
            common::dd_get_value(row_set, row, col_count, "id", bigint(0)), 
            common::dd_get_value(row_set, row, col_count, "model_id", bigint(0)), 
            model_bin);
    }
};


}
}
