#pragma once

#include "DAO/IRowMapper.hpp"
#include "model/DeconQueue.hpp"

namespace svr {
namespace dao {

class DeconQueueRowMapper : public IRowMapper<svr::datamodel::DeconQueue> {
public:
    datamodel::DeconQueue_ptr map_row(const pqxx_tuple &row_set) const override {

        return ptr<svr::datamodel::DeconQueue>(
                row_set["table_name"].as<std::string>(""),
                row_set["input_queue_table_name"].as<std::string>(""),
                row_set["input_queue_column_name"].as<std::string>(""),
                row_set["dataset_id"].as<bigint>(0),
                row_set["levels"].as<size_t>(0)
        );
    }
};
}
}
