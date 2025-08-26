#pragma once
#include <pqxx/pqxx>
#ifdef USE_DUCKDB
#include <duckdb.h>
#endif
#include "common/types.hpp"
#include "util/CompressionUtils.hpp"
#include "util/db_utils.hpp"

#if PQXX_VERSION_MAJOR == 4
using pqxx_tuple = pqxx::tuple;
#elif PQXX_VERSION_MAJOR > 4
using pqxx_tuple = pqxx::row;
#endif
