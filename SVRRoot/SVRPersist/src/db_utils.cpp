//
// Created by zarko on 8/24/25.
//

#include "misc/db_utils.hpp"

namespace svr {
namespace common {

int32_t dd_name_column(duckdb_result &res, CRPTR(char) name, const uint32_t column_count)
{
    for (DTYPE(column_count) i = 0; i < column_count; ++i)
        if (strcmp(duckdb_column_name(&res, i), name) == 0) return i;
    return -1;
}

}
}