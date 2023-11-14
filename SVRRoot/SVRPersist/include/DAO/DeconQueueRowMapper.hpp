#pragma once

#include "DAO/IRowMapper.hpp"
#include "model/DeconQueue.hpp"

namespace svr {
namespace dao {

class DeconQueueRowMapper : public IRowMapper<svr::datamodel::DeconQueue> {
public:
    DeconQueue_ptr mapRow(const pqxx_tuple &rowSet) const override {

        return std::make_shared<svr::datamodel::DeconQueue>(
                rowSet["table_name"].as<std::string>(""),
                rowSet["input_queue_table_name"].as<std::string>(""),
                rowSet["input_queue_column_name"].as<std::string>(""),
                rowSet["dataset_id"].as<bigint>(0),
                rowSet["swt_levels"].as<size_t>(0)
        );
    }
};
}
}
