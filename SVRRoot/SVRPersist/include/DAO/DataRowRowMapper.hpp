#pragma once

#include "common.hpp"
#include "DAO/IRowMapper.hpp"
#include "model/DataRow.hpp"

namespace svr {
namespace dao {

class DataRowRowMapper : public IRowMapper<svr::datamodel::DataRow> {
public:
    datamodel::DataRow_ptr map_row(const pqxx_tuple &row_set) const override
    {
        if (row_set.size() < 3) {
            LOG4_ERROR("Illegal number of columns " << row_set.size());
            return nullptr;
        }
        std::vector<double> levels;
        const size_t num_levels = (row_set.size() - 3);
        for (size_t col_ix = 3; col_ix < num_levels + 3; col_ix++)
            levels.emplace_back(row_set[col_ix].as<double>(std::numeric_limits<double>::quiet_NaN()));

        const auto value_time = row_set["value_time"].as<bpt::ptime>(bpt::not_a_date_time);
        if (value_time.is_special() or value_time.date().year() <= 1900 or value_time.date().year() >= 2200)
            LOG4_ERROR("Value time not parsed correctly from string " << row_set["value_time"].as<std::string>(""));

        return ptr<svr::datamodel::DataRow>(
                value_time,
                row_set["update_time"].as<bpt::ptime>(bpt::not_a_date_time),
                row_set["tick_volume"].as<double>(std::numeric_limits<double>::quiet_NaN()),
                levels
        );
    }
};
}
}