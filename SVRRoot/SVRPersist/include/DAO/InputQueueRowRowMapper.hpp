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

    datamodel::DataRow_ptr map_row(const pqxx_tuple& row_set) const override {

		std::vector<double> values;
		for(auto values_iter = row_set.begin() + 4; values_iter != row_set.end(); values_iter++){
			values.push_back(svr::common::Round(values_iter.as<double>(std::numeric_limits<double>::quiet_NaN())));
		}

        const auto value_time_str = row_set["value_time"].as<std::string>();
        const auto value_time = bpt::time_from_string(value_time_str);
        if (value_time.is_not_a_date_time() or value_time.is_special() or value_time.is_infinity() or value_time.date().year() <= 1400 or value_time.date().year() >= 10000)
            LOG4_ERROR("Value time not parsed correctly from " << value_time_str);
		return ptr<svr::datamodel::DataRow>(
			value_time,
			row_set["update_time"].as<bpt::ptime>(bpt::not_a_date_time),
			svr::common::Round(row_set["tick_volume"].as<double>(std::numeric_limits<double>::quiet_NaN())),
			values
		);
	}
};

} /* namespace dao */
} /* namespace svr */
