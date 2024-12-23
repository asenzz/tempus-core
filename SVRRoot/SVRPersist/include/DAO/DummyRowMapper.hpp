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
};
}
}

#endif //SVR_DUMMYROWMAPPER_HPP
