/*
 * InputQueueRowRowMapper.hpp
 *
 *  Created on: Aug 7, 2014
 *      Author: vg
 */

#pragma once

#include "DAO/IRowMapper.hpp"
#include "model/DataRow.hpp"

namespace svr {
namespace dao {

class InputQueueRowRowMapper: public IRowMapper<svr::datamodel::DataRow> {
public:

    datamodel::DataRow_ptr map_row(const pqxx_tuple& row_set) const override 
    {
        std::vector<double> values;
        for (auto values_iter = row_set.begin() + 4; values_iter != row_set.end(); ++values_iter)
            values.emplace_back(values_iter.as<double>(std::numeric_limits<double>::quiet_NaN()));

        return ptr<datamodel::DataRow>(
            row_set["value_time"].as<bpt::ptime>(bpt::not_a_date_time),
            row_set["update_time"].as<bpt::ptime>(bpt::not_a_date_time),
            row_set["tick_volume"].as<double>(std::numeric_limits<double>::quiet_NaN()),
            values
        );
    }
#ifdef USE_DUCKDB
    datamodel::DataRow_ptr map_row(duckdb_result& row_set, const size_t col_count, const size_t row) const override
    {
        if (col_count < 4) {
            LOG4_ERROR("Illegal number of columns " << col_count);
            return {};
        }
        std::vector<double> values;
        for (DTYPE(col_count) col_ix = 4; col_ix < col_count; ++col_ix)
            values.emplace_back(duckdb_value_double(&row_set, row, col_ix));

        return ptr<datamodel::DataRow>(
                common::dd_get_value<bpt::ptime>(row_set, row, col_count, "value_time", bpt::not_a_date_time),
                common::dd_get_value<bpt::ptime>(row_set, row, col_count, "update_time", bpt::not_a_date_time),
                common::dd_get_value(row_set, row, col_count, "tick_volume", std::numeric_limits<double>::quiet_NaN()),
                values
            );
    }
#endif
};

} /* namespace dao */
} /* namespace svr */
