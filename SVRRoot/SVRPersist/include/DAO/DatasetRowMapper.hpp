#pragma once

#include "DAO/IRowMapper.hpp"
#include "model/Dataset.hpp"

namespace svr{
namespace dao{

class DatasetRowMapper : public IRowMapper<svr::datamodel::Dataset>{
public:
    datamodel::Dataset_ptr map_row(const pqxx_tuple& row_set) const override{

        using svr::common::ignore_case_equals;
        
        return ptr<svr::datamodel::Dataset>(
                row_set["id"].as<bigint>(0),
                row_set["dataset_name"].as<std::string>(""),
                row_set["user_name"].as<std::string>(""),
                row_set["main_input_queue_table_name"].as<std::string>(""),
                svr::common::from_sql_array(row_set["aux_input_queues_table_names"].as<std::string>("")),
                static_cast<svr::datamodel::Priority>(row_set["priority"].as<int>((int)svr::datamodel::Priority::Normal)),
                row_set["description"].as<std::string>(""),
                row_set["gradients"].as<size_t>(0),
                row_set["max_chunk_size"].as<size_t>(0),
                row_set["multistep"].as<size_t>(0),
                row_set["levels"].as<size_t>(0),
                row_set["deconstruction"].as<std::string>(""),
                row_set["max_gap"].as<bpt::time_duration>(onesec),
                std::deque<datamodel::Ensemble_ptr>(),
                row_set["is_active"].as<bool>(false)
        );
    }
};

class UserDatasetRowMapper : public IRowMapper<std::pair<std::string, datamodel::Dataset_ptr>>{
    DatasetRowMapper datasetRowMapper;
public:
    std::shared_ptr<std::pair<std::string, datamodel::Dataset_ptr>> map_row(const pqxx_tuple& row_set) const override {
        return ptr<std::pair<std::string, datamodel::Dataset_ptr>>
        (
              row_set["linked_user_name"].as<std::string>("")
            , datasetRowMapper.map_row(row_set)
        );
    }
};

}
}
