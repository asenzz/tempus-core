//
// Created by zarko on 8/24/25.
//

#include "util/db_utils.hpp"

namespace pqxx {

const char* string_traits<boost::posix_time::ptime>::name()
{
    return "boost::posix_time::ptime";
}

bool string_traits<boost::posix_time::ptime>::has_null()
{
    return false;
}

bool string_traits<boost::posix_time::ptime>::is_null(const boost::posix_time::ptime&)
{
    return false;
}

boost::posix_time::ptime string_traits<boost::posix_time::ptime>::null()
{
    return boost::posix_time::not_a_date_time;
}

void string_traits<boost::posix_time::ptime>::from_string(const char Str[], boost::posix_time::ptime& Obj)
{
    Obj = boost::posix_time::time_from_string(std::string(Str));
}

boost::posix_time::ptime string_traits<boost::posix_time::ptime>::from_string(const std::string_view& Str)
{
    return boost::posix_time::time_from_string(std::string(Str));
}

std::string string_traits<boost::posix_time::ptime>::to_string(const boost::posix_time::ptime& Obj)
{
    return boost::posix_time::to_simple_string(Obj);
}

const char* string_traits<boost::posix_time::time_duration>::name()
{
    return "boost::posix_time::time_duration";
}

bool string_traits<boost::posix_time::time_duration>::has_null()
{
    return false;
}

bool string_traits<boost::posix_time::time_duration>::is_null(const boost::posix_time::time_duration&)
{
    return false;
}

boost::posix_time::time_duration string_traits<boost::posix_time::time_duration>::null()
{
    return boost::posix_time::time_duration();
}

void string_traits<boost::posix_time::time_duration>::from_string(const char Str[], boost::posix_time::time_duration& Obj)
{
    Obj = boost::posix_time::duration_from_string(std::string(Str));
}

boost::posix_time::time_duration string_traits<boost::posix_time::time_duration>::from_string(const std::string_view& Str)
{
    return boost::posix_time::duration_from_string(std::string(Str));
}

std::string string_traits<boost::posix_time::time_duration>::to_string(const boost::posix_time::time_duration& Obj)
{
    return boost::posix_time::to_simple_string(Obj);
}

}

namespace svr {
namespace common {

#ifdef USE_DUCKDB

int32_t dd_column_name(duckdb_result &res, const char *name, const uint32_t column_count)
{
    if (strlen(name) < 1) return 0;
    for (DTYPE(column_count) i = 0; i < column_count; ++i)
        if (strcmp(duckdb_column_name(&res, i), name) == 0) return i;
    return -1;
}

std::vector<uint8_t> dd_get_blob(duckdb_result& res, const uint32_t row, const uint32_t col)
{
    const auto blob = duckdb_value_blob(&res, col, row);
    return std::vector<uint8_t>((uint8_t*)blob.data, (uint8_t*)blob.data + blob.size);
}

std::string dd_get_string(duckdb_result& res, const uint32_t row, const uint32_t col)
{
    const auto str = duckdb_value_string(&res, col, row);
    std::string r(str.data, str.size);
    duckdb_free((void*)str.data);
    return r;
}

bpt::ptime dd_get_time(duckdb_result& res, const uint32_t row, const uint32_t col)
{
    return bpt::time_from_string(dd_get_string(res, row, col));
}

bpt::time_duration dd_get_duration(duckdb_result& res, const uint32_t row, const uint32_t col) 
{
    return bpt::duration_from_string(dd_get_string(res, row, col));
}

template <> std::string dd_get_value<std::string>(duckdb_result& res, const uint32_t row, const uint32_t column_count, const char *column_name, const std::string &default_value) 
{ 
    const auto col = dd_column_name(res, column_name, column_count); 
    if (col < 0) {
        LOG4_WARN("Column " << column_name << " not found"); 
        return default_value; 
    } 
    return dd_get_string(res, row, col);
}

template <> std::vector<uint8_t> dd_get_value<std::vector<uint8_t>>(duckdb_result& res, const uint32_t row, const uint32_t column_count, const char *column_name, const std::vector<uint8_t> &default_value) 
{ 
    const auto col = dd_column_name(res, column_name, column_count); 
    if (col < 0) {
        LOG4_WARN("Column " << column_name << " not found"); 
        return default_value; 
    } 
    return dd_get_blob(res, row, col);
}

template <> bpt::time_duration dd_get_value<bpt::time_duration>(duckdb_result& res, const uint32_t row, const uint32_t column_count, const char *column_name, const bpt::time_duration &default_value) 
{ 
    const auto col = dd_column_name(res, column_name, column_count); 
    if (col < 0) {
        LOG4_WARN("Column " << column_name << " not found"); 
        return default_value; 
    } 
    return dd_get_duration(res, row, col);
}

template <> bpt::ptime dd_get_value<bpt::ptime>(duckdb_result& res, const uint32_t row, const uint32_t column_count, const char *column_name, const bpt::ptime &default_value) 
{ 
    const auto col = dd_column_name(res, column_name, column_count); 
    if (col < 0) {
        LOG4_WARN("Column " << column_name << " not found"); 
        return default_value; 
    } 
    return dd_get_time(res, row, col);
}

#define SPECIALIZE_GET(T, S) \
template <> T dd_get_value<T>(duckdb_result& res, const uint32_t row, const uint32_t column_count, const char *column_name, const T &default_value) \
{ \
    const auto col = dd_column_name(res, column_name, column_count); \
    if (col < 0) { \
        LOG4_WARN("Column " << column_name << " not found"); \
        return default_value; \
    } \
    return duckdb_value_##S (&res, row, col); \
}

SPECIALIZE_GET(bool, boolean)
SPECIALIZE_GET(int8_t, int8)
SPECIALIZE_GET(uint8_t, uint8)
SPECIALIZE_GET(int16_t, int16)
SPECIALIZE_GET(uint16_t, uint16)
SPECIALIZE_GET(int32_t, int32)
SPECIALIZE_GET(uint32_t, uint32)
SPECIALIZE_GET(int64_t, int64)
SPECIALIZE_GET(uint64_t, uint64)
SPECIALIZE_GET(float, float)
SPECIALIZE_GET(double, double)

#endif

}
}
