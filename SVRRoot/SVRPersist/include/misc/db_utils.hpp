#ifndef DB_UTILS_HPP
#define DB_UTILS_HPP

#include <string>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <pqxx/pqxx>
#include <duckdb.h>
#include "common/compatibility.hpp"

namespace pqxx {

template <> struct string_traits<boost::posix_time::ptime>
{
    static const char* name() { return "boost::posix_time::ptime"; }
    static bool has_null() { return false; }
    static bool is_null(const boost::posix_time::ptime&) { return false; }

    static boost::posix_time::ptime null()
    {
        return boost::posix_time::not_a_date_time;
    }

    static void from_string(const char Str[], boost::posix_time::ptime& Obj)
    {
        Obj = boost::posix_time::time_from_string(std::string(Str));
    }

    static boost::posix_time::ptime from_string(const std::string_view& Str)
    {
        return boost::posix_time::time_from_string(std::string(Str));
    }

    static std::string to_string(const boost::posix_time::ptime& Obj)
    {
        return boost::posix_time::to_simple_string(Obj);
    }
};

template <> struct string_traits<boost::posix_time::time_duration>
{
    static const char* name() { return "boost::posix_time::time_duration"; }
    static bool has_null() { return false; }
    static bool is_null(const boost::posix_time::time_duration&) { return false; }

    static boost::posix_time::time_duration null()
    {
        return boost::posix_time::time_duration();
    }

    static void from_string(const char Str[], boost::posix_time::time_duration& Obj)
    {
        Obj = boost::posix_time::duration_from_string(std::string(Str));
    }

    static boost::posix_time::time_duration from_string(const std::string_view& Str)
    {
        return boost::posix_time::duration_from_string(std::string(Str));
    }

    static std::string to_string(const boost::posix_time::time_duration& Obj)
    {
        return boost::posix_time::to_simple_string(Obj);
    }
};

}

namespace svr {
namespace common {

int32_t dd_name_column(duckdb_result &res, CRPTR(char) name, uint32_t column_count);

template <typename T> T dd_get_value(duckdb_result &res, uint32_t row, CRPTR(char) column_name, uint32_t column_count);

template <typename T> T dd_get_value(duckdb_result& res, uint32_t row, uint32_t column_count, CRPTR(char) column_name, const T &default_value);

template <typename T> T dd_get_value(duckdb_result &res, uint32_t row, uint32_t col);

}
}

#endif