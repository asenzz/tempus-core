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

namespace svr {
namespace dao {

// TODO Not working, implement and test!
class ModelRowMapper : public IRowMapper<datamodel::Model>
{
public:
    datamodel::Model_ptr mapRow(const pqxx_tuple &rowSet) const override
    {
        return std::make_shared<datamodel::Model>(
                rowSet["id"].as<bigint>(0),
                rowSet["ensemble_id"].as<bigint>(0),
                rowSet["decon_level"].as<size_t>(DEFAULT_SVRPARAM_DECON_LEVEL),
                PROPS.get_multistep_len(),
                rowSet["gradients"].as<size_t>(DEFAULT_SVRPARAM_GRAD_LEVEL + 1),
                C_kernel_default_max_chunk_size,
                std::deque<OnlineMIMOSVR_ptr>{},
                rowSet["last_modified_time"].is_null() ? bpt::not_a_date_time : bpt::time_from_string(
                        rowSet["last_modified_time"].as<std::string>()),
                rowSet["last_modeled_value_time"].is_null() ? bpt::not_a_date_time : bpt::time_from_string(
                        rowSet["last_modeled_value_time"].as<std::string>())
        );
    }
};

class SVRModelRowMapper : public IRowMapper<OnlineMIMOSVR>
{
public:
    std::shared_ptr<OnlineMIMOSVR> mapRow(const pqxx_tuple &rowSet) const override
    {
        if (!rowSet["model_binary"].is_null()) {
            std::stringstream model_bin;
            const auto &binstr = rowSet["model_binary"].as<std::basic_string<std::byte>>();
            model_bin.write((const char *)binstr.c_str(), binstr.size());
            return std::make_shared<OnlineMIMOSVR>(model_bin);
        } else
            return std::make_shared<OnlineMIMOSVR>();
    }
};


}
}
