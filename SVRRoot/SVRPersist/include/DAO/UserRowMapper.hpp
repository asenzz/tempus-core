/*
 * UserRowMapper.hpp
 *
 *  Created on: Jul 27, 2014
 *      Author: vg
 */

#pragma once

#include "DAO/IRowMapper.hpp"
#include "model/User.hpp"


namespace svr {
namespace dao {

class UserRowMapper: public IRowMapper<svr::datamodel::User> {

public:
    User_ptr map_row(const pqxx_tuple& row_set) const override;
};

} /* namespace dao */
} /* namespace svr */
