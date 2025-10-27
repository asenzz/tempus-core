#pragma once

#include "DAO/IRowMapper.hpp"
#include "model/Dataset.hpp"
#include "util/time_utils.hpp"

namespace svr{
namespace dao{

class DatasetRowMapper : public IRowMapper<datamodel::Dataset>{
public:
    datamodel::Dataset_ptr map_row(const pqxx_tuple& row_set) const override
    {
        return ptr<datamodel::Dataset>(
                row_set["id"].as<bigint>(0),
                row_set["dataset_name"].as<std::string>(),
                row_set["user_name"].as<std::string>(""),
                row_set["main_input_queue_table_name"].as<std::string>(),
                common::from_sql_array(row_set["aux_input_queues_table_names"].as<std::string>()),
                static_cast<datamodel::Priority>(row_set["priority"].as<int>((int)datamodel::Priority::Normal)),
                row_set["description"].as<std::string>(),
                row_set["gradients"].as<size_t>(0),
                row_set["max_chunk_size"].as<size_t>(0),
                row_set["steps"].as<size_t>(0),
                row_set["levels"].as<size_t>(0),
                row_set["deconstruction"].as<std::string>(),
                row_set["max_gap"].as<bpt::time_duration>(common::C_default_features_max_time_gap),
                std::deque<datamodel::Ensemble_ptr>(),
                row_set["is_active"].as<bool>(false)
        );
    }

#ifdef USE_DUCKDB
    datamodel::Dataset_ptr map_row(duckdb_result& row_set, const size_t col_count, const size_t row) const override
    {
        return ptr<datamodel::Dataset>(
            common::dd_get_value(row_set, row, col_count, "id", bigint(0)),
            common::dd_get_value(row_set, row, col_count, "dataset_name", std::string()),
            common::dd_get_value(row_set, row, col_count, "user_name", std::string()),
            common::dd_get_value(row_set, row, col_count, "main_input_queue_table_name", std::string()),
            common::from_sql_array(common::dd_get_value(row_set, row, col_count, "aux_input_queues_table_names", std::string())),
            static_cast<datamodel::Priority>(common::dd_get_value(row_set, row, col_count, "priority", (int) datamodel::Priority::Normal)),
            common::dd_get_value(row_set, row, col_count, "description", std::string()),
            common::dd_get_value(row_set, row, col_count, "gradients", uint16_t(0)),
            common::dd_get_value(row_set, row, col_count, "max_chunk_size", uint32_t(0)),
            common::dd_get_value(row_set, row, col_count, "steps", uint16_t(0)),
            common::dd_get_value(row_set, row, col_count, "levels", uint16_t(0)),
            common::dd_get_value(row_set, row, col_count, "deconstruction", std::string()),
            common::dd_get_value(row_set, row, col_count, "max_gap", common::C_default_features_max_time_gap),
            std::deque<datamodel::Ensemble_ptr>{},
            common::dd_get_value(row_set, row, col_count, "is_active", false)
        );
    }
#endif

};

class UserDatasetRowMapper : public IRowMapper<std::pair<std::string, datamodel::Dataset_ptr>>{
    DatasetRowMapper dataset_mapper;
public:
    std::shared_ptr<std::pair<std::string, datamodel::Dataset_ptr>> map_row(const pqxx_tuple& row_set) const override {
        return ptr<std::pair<std::string, datamodel::Dataset_ptr>>(
              row_set["linked_user_name"].as<std::string>(""), dataset_mapper.map_row(row_set)
        );
    }
#ifdef USE_DUCKDB
    std::shared_ptr<std::pair<std::string, datamodel::Dataset_ptr>> map_row(duckdb_result& row_set, const size_t col_count, const size_t row) const override
    {
        return ptr<std::pair<std::string, datamodel::Dataset_ptr>>(
            common::dd_get_value(row_set, row, col_count, "linked_user_name", std::string()),
            dataset_mapper.map_row(row_set, col_count, row)
        );
    }
#endif
};

}
}
