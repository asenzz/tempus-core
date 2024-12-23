/*
 * IRowMapper.hpp
 *
 *  Created on: Jul 27, 2014
 *      Author: vg
 */

#pragma once

#include "common.hpp"

namespace svr {
namespace dao {

template <class T>
class IRowMapper {

public:
	virtual std::shared_ptr<T> map_row(const pqxx_tuple& row_set) const = 0;
	virtual ~IRowMapper() = 0; // make this class abstract
};

	template <typename T>
	IRowMapper<T>::~IRowMapper() {}
} /* namespace dao */
} /* namespace svr */
