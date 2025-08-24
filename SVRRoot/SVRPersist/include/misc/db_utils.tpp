#ifndef DBUTILS_TPP
#define DBUTILS_TPP

#include "db_utils.hpp"
#include "common/logging.hpp"

namespace pqxx {

template <> const char* string_traits<boost::posix_time::ptime>::name()
{
    return "boost::posix_time::ptime";
}

template <> bool string_traits<boost::posix_time::ptime>::has_null()
{
    return false;
}

template <> bool string_traits<boost::posix_time::ptime>::is_null(const boost::posix_time::ptime&)
{
    return false;
}

template <> boost::posix_time::ptime string_traits<boost::posix_time::ptime>::null()
{
    return boost::posix_time::not_a_date_time;
}

template <> void string_traits<boost::posix_time::ptime>::from_string(const char Str[], boost::posix_time::ptime& Obj)
{
    Obj = boost::posix_time::time_from_string(std::string(Str));
}

template <> boost::posix_time::ptime string_traits<boost::posix_time::ptime>::from_string(const std::string_view& Str)
{
    return boost::posix_time::time_from_string(std::string(Str));
}

template <> std::string string_traits<boost::posix_time::ptime>::to_string(const boost::posix_time::ptime& Obj)
{
    return boost::posix_time::to_simple_string(Obj);
}


template <> const char* string_traits<boost::posix_time::time_duration>::name()
{
    return "boost::posix_time::time_duration";
}

template <> bool string_traits<boost::posix_time::time_duration>::has_null()
{
    return false;
}

template <> bool string_traits<boost::posix_time::time_duration>::is_null(const boost::posix_time::time_duration&)
{
    return false;
}

template <> boost::posix_time::time_duration string_traits<boost::posix_time::time_duration>::null()
{
    return boost::posix_time::time_duration();
}

template <> void string_traits<boost::posix_time::time_duration>::from_string(const char Str[], boost::posix_time::time_duration& Obj)
{
    Obj = boost::posix_time::duration_from_string(std::string(Str));
}

template <> boost::posix_time::time_duration string_traits<boost::posix_time::time_duration>::from_string(const std::string_view& Str)
{
    return boost::posix_time::duration_from_string(std::string(Str));
}

template <> std::string string_traits<boost::posix_time::time_duration>::to_string(const boost::posix_time::time_duration& Obj)
{
    return boost::posix_time::to_simple_string(Obj);
}

}

namespace svr {
namespace common {

template <typename T> T dd_get_value(duckdb_result& res, const uint32_t row, const uint32_t column_count, CRPTR(char) column_name)
{
    const auto col = dd_column_name(res, column_name, column_count);
    if (col < 0) LOG4_THROW("Column " << column_name << " not found");
    return dd_get_value(res, row, col);
}

template <typename T> T dd_get_value(duckdb_result& res, const uint32_t row, const uint32_t column_count, CRPTR(char) column_name, const T &default_value)
{
    const auto col = dd_column_name(res, column_name, column_count);
    if (col < 0) {
        LOG4_WARN("Column " << column_name << " not found");
        return default_value;
    }
    return dd_get_value(res, row, col);
}

template <typename T> T dd_get_value(duckdb_result& res, const uint32_t row, const uint32_t col)
{
    if (std::is_same_v<T, bool>) return duckdb_value_boolean(&res, col, row);
    if (std::is_integral_v<T>) {
        if (std::is_signed_v<T>) {
            switch (sizeof(T)) {
            case sizeof(int8_t): return duckdb_value_int8(&res, col, row);
            case sizeof(int16_t): return duckdb_value_int16(&res, col, row);
            case sizeof(int32_t): return duckdb_value_int32(&res, col, row);
            case sizeof(int64_t): return duckdb_value_int64(&res, col, row);
            default: return duckdb_value_hugeint(&res, col, row);
            }
        }
        switch (sizeof(T)) {
        case sizeof(uint8_t): return duckdb_value_int8(&res, col, row);
        case sizeof(uint16_t): return duckdb_value_int16(&res, col, row);
        case sizeof(uint32_t): return duckdb_value_int32(&res, col, row);
        case sizeof(uint64_t): return duckdb_value_int64(&res, col, row);
        default: return duckdb_value_uhugeint(&res, col, row);
        }
    }
    if (std::is_floating_point_v<T>) {
        switch (sizeof(T)) {
        case sizeof(float): return duckdb_value_float(&res, col, row);
        default:
        case sizeof(double): return duckdb_value_double(&res, col, row);
        }
    }
    if (std::is_same_v<T, std::vector<uint8_t>>) {
        const auto blob = duckdb_value_blob(&res, col, row);
        return std::vector<uint8_t>((uint8_t*)blob.data, (uint8_t*)blob.data + blob.size);
    }
    const auto str = duckdb_value_string(&res, col, row);
    std::string r(str.data, str.size);
    duckdb_free((void*)str.data);
    if (std::is_same_v<T, std::string>) return r;
    if (std::is_same_v<T, boost::posix_time::ptime>) return boost::posix_time::time_from_string(r);
    if (std::is_same_v<T, boost::posix_time::time_duration>) return boost::posix_time::duration_from_string(r);
    return boost::lexical_cast<T>(r);
}

}
}

#endif
