#pragma once

#include <map>
#include <string>
#include <vector>
#include <set>
#define BOOST_BIND_GLOBAL_PLACEHOLDERS
#include <boost/date_time.hpp>

using bigint = uint64_t;
using MessageProperties = std::unordered_map<std::string, std::unordered_map<std::string, std::string>>;

namespace bpt = boost::posix_time;

typedef std::set<boost::posix_time::ptime> ptimes_set_t;

