#pragma once

#include <sstream>
#include "DAO/IRowMapper.hpp"
#include "model/Model.hpp"
#include "onlinesvr_persist.hpp"
#include <pqxx/binarystring>
#include <util/CompressionUtils.hpp>
#include <util/string_utils.hpp>

namespace svr {
namespace dao {

class ModelRowMapper : public IRowMapper<svr::datamodel::Model>
{
public:
    Model_ptr mapRow(const pqxx_tuple &rowSet) const override
    {
        std::shared_ptr<svr::OnlineMIMOSVR> model = nullptr;

        bool skip_loading_model = false;
        try { // TODO Rewrite this check
            (void) rowSet.column_number("model_binary");
        } catch (...) {
            skip_loading_model = true;
        }
        if (!skip_loading_model && !rowSet["model_binary"].is_null()) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
            pqxx::binarystring binary_model(rowSet["model_binary"]);
            std::stringstream ss;
            const char *pBinaryCompressed = reinterpret_cast<const char *>(binary_model.data());
            std::string decompressed = svr::common::decompress(pBinaryCompressed, binary_model.size());
            ss.rdbuf()->pubsetbuf(const_cast<char *>(decompressed.c_str()), decompressed.size());
            model = std::make_shared<svr::OnlineMIMOSVR>(ss);
#pragma GCC diagnostic pop
        }
        std::set<size_t> learning_levels;
        if (!rowSet["learning_levels"].is_null()) {
            std::vector<std::string> learning_levels_str = svr::common::from_sql_array(
                    rowSet["learning_levels"].as<std::string>());
            for (const auto &learning_level: learning_levels_str)
                learning_levels.insert(std::stoi(learning_level));
        }

        return std::make_shared<svr::datamodel::Model>(
                rowSet["id"].as<bigint>(),
                rowSet["ensemble_id"].is_null() ? 0 : rowSet["ensemble_id"].as<bigint>(),
                rowSet["decon_level"].is_null() ? 0 : rowSet["decon_level"].as<bigint>(),
                learning_levels,
                model,
                rowSet["last_modified_time"].is_null() ? bpt::not_a_date_time : bpt::time_from_string(
                        rowSet["last_modified_time"].as<std::string>()),
                rowSet["last_modeled_value_time"].is_null() ? bpt::not_a_date_time : bpt::time_from_string(
                        rowSet["last_modeled_value_time"].as<std::string>())
        );
    }
};
}
}
