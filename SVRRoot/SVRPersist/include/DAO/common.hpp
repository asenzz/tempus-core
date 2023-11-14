#pragma once
// external dependencies
#include "common/types.hpp"
#include <pqxx/pqxx>
#include "util/CompressionUtils.hpp"
//#include <pqxx/except.hxx>
#include "misc/boostPqxxConvertor.hpp"

#if PQXX_VERSION_MAJOR == 4
using pqxx_tuple = pqxx::tuple;
#elif PQXX_VERSION_MAJOR > 4
using pqxx_tuple = pqxx::row;
#endif
