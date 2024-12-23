#pragma once

#include "DAO/IRowMapper.hpp"
#include "model/Ensemble.hpp"

namespace svr {
namespace dao {

class EnsembleRowMapper : public IRowMapper<svr::datamodel::Ensemble> {
public:
    datamodel::Ensemble_ptr map_row(const pqxx_tuple &row_set) const override {
        return ptr<svr::datamodel::Ensemble>(
                row_set["id"].as<bigint>(0),
                row_set["dataset_id"].as<bigint>(0),
                row_set["decon_queue_table_name"].is_null() ? "" : row_set["decon_queue_table_name"].as<std::string>(),
                svr::common::from_sql_array(row_set["aux_decon_queues_table_names"].as<std::string>() )
        );
    }
};
}
}
