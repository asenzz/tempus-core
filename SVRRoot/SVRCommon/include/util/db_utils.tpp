#ifndef DBUTILS_TPP
#define DBUTILS_TPP

#include "db_utils.hpp"
#include "common/logging.hpp"

namespace svr {
namespace common {

#ifdef USE_DUCKDB
#if 0
template <typename T> T dd_get_value(duckdb_result& res, const uint32_t row, const uint32_t column_count, CRPTR(char) column_name, const T &default_value)
{
    const auto col = dd_column_name(res, column_name, column_count);
    if (col < 0) {
        LOG4_WARN("Column " << column_name << " not found");
        return default_value;
    }

    return boost::lexical_cast<T>(dd_get_string(res, row, col));
}
#endif
#endif
}
}

#endif
