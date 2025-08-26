#pragma once

#include "common.hpp"
#include "DAO/IRowMapper.hpp"
#include "model/DataRow.hpp"

namespace svr {
namespace dao {

class DataRowRowMapper : public IRowMapper<datamodel::DataRow> {
public:
    datamodel::DataRow_ptr map_row(const pqxx_tuple &row_set) const override
    {
        if (row_set.size() < 3) {
            LOG4_ERROR("Illegal number of columns " << row_set.size());
            return {};
        }
        std::vector<double> levels;
        const auto num_levels = row_set.size() - 3;
        for (DTYPE(num_levels) col_ix = 3; col_ix < num_levels + 3; ++col_ix)
            levels.emplace_back(row_set[col_ix].as(std::numeric_limits<double>::quiet_NaN()));

        const auto value_time = row_set["value_time"].as<bpt::ptime>(bpt::not_a_date_time);
#ifndef NDEBUG
        if (value_time.is_special() or value_time.date().year() <= 1900 or value_time.date().year() >= 2200)
            LOG4_ERROR("Value time not parsed correctly from string " << row_set["value_time"].as<std::string>("empty"));
#endif
        return ptr<datamodel::DataRow>(
                value_time,
                row_set["update_time"].as<bpt::ptime>(bpt::not_a_date_time),
                row_set["tick_volume"].as(std::numeric_limits<double>::quiet_NaN()),
                levels
        );
    }

#ifdef USE_DUCKDB
    datamodel::DataRow_ptr map_row(duckdb_result& row_set, const size_t col_count, const size_t row) const override
    {
        if (col_count < 3) {
            LOG4_ERROR("Illegal number of columns " << col_count);
            return {};
        }
        std::vector<double> levels;
        for (DTYPE(col_count) col_ix = 3; col_ix < col_count; ++col_ix)
            levels.emplace_back(duckdb_value_double(&row_set, row, col_ix));
        const auto value_time = common::dd_get_value<bpt::ptime>(row_set, row, col_count, "value_time", bpt::not_a_date_time);
#ifndef NDEBUG
        if (value_time.is_special() or value_time.date().year() <= 1900 or value_time.date().year() >= 2200)
            LOG4_ERROR("Value time not parsed correctly from string " << common::dd_get_value(row_set, row, col_count, "value_time", std::string("empty")));
#endif
        return ptr<datamodel::DataRow>(
            value_time,
            common::dd_get_value<bpt::ptime>(row_set, row, col_count, "update_time", bpt::not_a_date_time),
            common::dd_get_value(row_set, row, col_count, "tick_volume", std::numeric_limits<double>::quiet_NaN()),
            levels);
    }
#endif

};
}
}
