#pragma once

#include <string>
#include <sstream>
#include <boost/date_time/posix_time/ptime.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/date_time/posix_time/posix_time_io.hpp>
#include <pqxx/pqxx>
#include <pqxx/util>

namespace pqxx {
template<> struct string_traits<boost::posix_time::ptime>
{
	static const char *name() {return "boost::posix_time::ptime";}
	static bool has_null() {return false;}
	static bool is_null(const boost::posix_time::ptime &) {return false;}
	static boost::posix_time::ptime null()
	{
		return boost::posix_time::ptime();
	}
	static void from_string(const char Str[], boost::posix_time::ptime &Obj)
	{
		Obj = boost::posix_time::time_from_string(std::string(Str));
	}
	static boost::posix_time::ptime from_string(const std::string_view &Str)
	{
		return boost::posix_time::time_from_string(std::string(Str));
	}

	static std::string to_string(const boost::posix_time::ptime &Obj)
	{
		std::stringstream ss;
		ss << Obj;
		return ss.str();
	}
};

template<> struct string_traits<boost::posix_time::time_duration>
{
	static const char *name() {return "boost::posix_time::time_duration";}
	static bool has_null() {return false;}
	static bool is_null(const boost::posix_time::time_duration &) {return false;}
	static boost::posix_time::time_duration null()
	{
		return boost::posix_time::time_duration();
	}
	static void from_string(const char Str[], boost::posix_time::time_duration &Obj)
	{
		Obj = boost::posix_time::duration_from_string(std::string(Str));
	}
    static boost::posix_time::time_duration from_string(const std::string_view &Str)
    {
        return boost::posix_time::duration_from_string(std::string(Str));
    }
	static std::string to_string(const boost::posix_time::time_duration &Obj)
	{
		std::stringstream ss;
		ss << Obj;
		return ss.str();
	}
};

}
