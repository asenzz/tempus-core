#pragma once

#include "common.hpp"
#include "DAO/IRowMapper.hpp"
#include "model/DataRow.hpp"

namespace svr {
namespace dao {

class DataRowRowMapper : public IRowMapper<svr::datamodel::DataRow> {
public:
    datamodel::DataRow_ptr mapRow(const pqxx_tuple &rowSet) const override
    {
        if (rowSet.size() < 3) {
            LOG4_ERROR("Illegal number of columns " << rowSet.size());
            return nullptr;
        }
        std::vector<double> levels;
        const size_t num_levels = (rowSet.size() - 3);
        for (size_t col_ix = 3; col_ix < num_levels + 3; col_ix++)
            levels.emplace_back(rowSet[col_ix].as<double>(std::numeric_limits<double>::quiet_NaN()));

        const auto value_time = bpt::time_from_string(rowSet["value_time"].as<std::string>(""));
        if (value_time.is_not_a_date_time() or value_time.is_special() or value_time.is_infinity() or value_time.date().year() <= 1400 or value_time.date().year() >= 10000)
            LOG4_ERROR("Value time not parsed correctly from string " << rowSet["value_time"].as<std::string>(""));

        return std::make_shared<svr::datamodel::DataRow>(
                value_time,
                bpt::time_from_string(rowSet["update_time"].as<std::string>("")),
                rowSet["tick_volume"].as<double>(std::numeric_limits<double>::quiet_NaN()),
                levels
        );
    }
};
}
}