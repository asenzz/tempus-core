#ifndef DB_UTILS_HPP
#define DB_UTILS_HPP

#include <boost/date_time/posix_time/posix_time.hpp>
#ifdef USE_DUCKDB
#include <duckdb.h>
#endif
#include <pqxx/pqxx>
#include <string>
#include "common/compatibility.hpp"

namespace pqxx {

template <> struct string_traits<boost::posix_time::ptime>
{
    static const char* name();

    static bool has_null();

    static bool is_null(const boost::posix_time::ptime&);

    static boost::posix_time::ptime null();

    static void from_string(const char Str[], boost::posix_time::ptime& Obj);

    static boost::posix_time::ptime from_string(const std::string_view& Str);

    static std::string to_string(const boost::posix_time::ptime& Obj);
};

template <> struct string_traits<boost::posix_time::time_duration>
{
    static const char* name();

    static bool has_null();

    static bool is_null(const boost::posix_time::time_duration&);

    static boost::posix_time::time_duration null();

    static void from_string(const char Str[], boost::posix_time::time_duration& Obj);

    static boost::posix_time::time_duration from_string(const std::string_view& Str);

    static std::string to_string(const boost::posix_time::time_duration& Obj);
};

} // namespace pqxx

namespace svr {
namespace common {

#ifdef USE_DUCKDB

int32_t dd_column_name(duckdb_result& res, const char *name, uint32_t column_count);

template <typename T> T dd_get_value(duckdb_result& res, const uint32_t row, const uint32_t column_count, const char *column_name, const T &default_value);

std::vector<uint8_t> dd_get_blob(duckdb_result& res, const uint32_t row, const uint32_t col);

std::string dd_get_string(duckdb_result& res, const uint32_t row, const uint32_t col);

bpt::ptime dd_get_time(duckdb_result& res, const uint32_t row, const uint32_t col);

bpt::time_duration dd_get_duration(duckdb_result& res, const uint32_t row, const uint32_t col);

#endif

} // namespace common
} // namespace svr

#include "db_utils.tpp"

#endif
