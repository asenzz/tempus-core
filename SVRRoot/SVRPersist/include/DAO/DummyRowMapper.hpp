//
// Created by zarko on 2/12/23.
//

#ifndef SVR_DUMMYROWMAPPER_HPP
#define SVR_DUMMYROWMAPPER_HPP

#include "common.hpp"
#include "DAO/IRowMapper.hpp"

namespace svr {
namespace dao {

template<typename T> class DummyRowMapper : public IRowMapper<T> {
public:
    T map_row(const pqxx_tuple &row) const override
    {
        row.at(0).as<T>();
    }

    T map_row(duckdb_result& row_set, const size_t col_count, const size_t row) const override
    {
        return common::dd_get_value<T>(row_set, row, 0);
    }
};

}
}

#endif //SVR_DUMMYROWMAPPER_HPP
