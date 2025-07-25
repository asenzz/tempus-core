#pragma once

#include <boost/unordered/unordered_flat_map.hpp>
#include <string>
#include <set>
#define BOOST_BIND_GLOBAL_PLACEHOLDERS
#include <boost/date_time.hpp>

using bigint = uint64_t;
using MessageProperties = boost::unordered_flat_map<std::string, boost::unordered_flat_map<std::string, std::string>>;

namespace bpt = boost::posix_time;

typedef std::set<boost::posix_time::ptime> ptimes_set_t;

