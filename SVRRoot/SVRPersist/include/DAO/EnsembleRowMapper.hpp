#pragma once

#include "DAO/IRowMapper.hpp"
#include "model/Ensemble.hpp"

namespace svr {
namespace dao {

class EnsembleRowMapper : public IRowMapper<svr::datamodel::Ensemble> {
public:
    datamodel::Ensemble_ptr map_row(const pqxx_tuple &row_set) const override {
        return ptr<datamodel::Ensemble>(
                row_set["id"].as<bigint>(0),
                row_set["dataset_id"].as<bigint>(0),
                row_set["decon_queue_table_name"].as<std::string>(),
                common::from_sql_array(row_set["aux_decon_queues_table_names"].as<std::string>())
        );
    }

    datamodel::Ensemble_ptr map_row(duckdb_result& row_set, const size_t col_count, const size_t row) const override
    {
        return ptr<datamodel::Ensemble>(
            common::dd_get_value(row_set, row, col_count, "id", bigint(0)),
            common::dd_get_value(row_set, row, col_count, "dataset_id", bigint(0)),
            common::dd_get_value(row_set, row, col_count, "decon_queue_table_name", std::string()),
            common::from_sql_array(common::dd_get_value(row_set, row, col_count, "decon_queue_table_name", std::string()))
        );
    }
};
}
}
