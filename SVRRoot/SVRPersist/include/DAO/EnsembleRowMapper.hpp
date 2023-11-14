#pragma once

#include "DAO/IRowMapper.hpp"
#include "model/Ensemble.hpp"

namespace svr {
namespace dao {

class EnsembleRowMapper : public IRowMapper<svr::datamodel::Ensemble> {
public:
    Ensemble_ptr mapRow(const pqxx_tuple &rowSet) const override {
        return std::make_shared<svr::datamodel::Ensemble>(
                rowSet["id"].as<bigint>(),
                rowSet["dataset_id"].as<bigint>(),
                rowSet["decon_queue_table_name"].is_null() ? "" : rowSet["decon_queue_table_name"].as<std::string>(),
                svr::common::from_sql_array(rowSet["aux_decon_queues_table_names"].as<std::string>())
        );
    }
};
}
}
