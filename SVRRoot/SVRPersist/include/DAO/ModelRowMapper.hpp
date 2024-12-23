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

// TODO Not working, implement and test!
class ModelRowMapper : public IRowMapper<datamodel::Model>
{
public:
    datamodel::Model_ptr map_row(const pqxx_tuple &row_set) const override
    {
        return ptr<datamodel::Model>(
                row_set["id"].as<bigint>(0),
                row_set["ensemble_id"].as<bigint>(0),
                row_set["decon_level"].as<size_t>(datamodel::C_default_svrparam_decon_level),
                row_set["step"].as<size_t>(datamodel::C_default_svrparam_step),
                PROPS.get_multiout(),
                row_set["gradients"].as<size_t>(datamodel::C_default_svrparam_grad_level + 1),
                common::C_default_kernel_max_chunk_len,
                std::deque<datamodel::OnlineMIMOSVR_ptr>{},
                row_set["last_modified_time"].is_null() ? bpt::not_a_date_time : bpt::time_from_string(
                        row_set["last_modified_time"].as<std::string>("")),
                row_set["last_modeled_value_time"].is_null() ? bpt::not_a_date_time : bpt::time_from_string(
                        row_set["last_modeled_value_time"].as<std::string>())
        );
    }
};

class SVRModelRowMapper : public IRowMapper<datamodel::OnlineMIMOSVR>
{
public:
    datamodel::OnlineMIMOSVR_ptr map_row(const pqxx_tuple &row_set) const override
    {
        std::stringstream model_bin;
        const auto &binstr = row_set["model_binary"].as<std::basic_string<std::byte>>();
        model_bin.write((const char *)binstr.c_str(), binstr.size());
        return ptr<datamodel::OnlineMIMOSVR>(row_set["id"].as<bigint>(0), row_set["model_id"].as<bigint>(0), model_bin);
    }
};


}
}
