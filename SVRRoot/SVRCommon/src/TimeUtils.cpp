#include <util/TimeUtils.hpp>
#include <sstream>

using namespace std;

namespace svr {

double
operator /(const boost::posix_time::time_duration &lhs, const boost::posix_time::time_duration &rhs)
{
    return double(lhs.total_nanoseconds()) / double(rhs.total_nanoseconds());
}

/*
const boost::posix_time::time_duration &operator /(const boost::posix_time::time_duration &lhs, const boost::posix_time::time_duration &rhs)
{
    return boost::posix_time::microseconds(lhs.total_microseconds()) / double(rhs.total_microseconds());
}
*/



namespace common {

const std::vector<size_t> sec_per_unit {24*60*60, 60*60, 60, 1}; // day, hour, minute, second


boost::posix_time::ptime round_second(const boost::posix_time::ptime &t)
{
    return bpt::ptime(t.date(), bpt::seconds(t.time_of_day().total_seconds()));
}

boost::posix_time::ptime round_millisecond(const boost::posix_time::ptime &t)
{
    return bpt::ptime(t.date(), bpt::milliseconds(t.time_of_day().total_milliseconds()));
}

boost::posix_time::ptime round_hour(const boost::posix_time::ptime &t)
{
    return bpt::ptime(t.date(), bpt::hours(t.time_of_day().hours()));
}

boost::posix_time::ptime round_minute(const boost::posix_time::ptime &t)
{
    return bpt::ptime(t.date(), bpt::minutes(t.time_of_day().minutes()));
}

bpt::seconds date_time_string_to_seconds(const string &date_time)
{
    boost::char_separator<char> separator(",: ");
    boost::tokenizer<boost::char_separator<char>> tokens(date_time, separator);

    size_t time_period  = 0;
    std::vector<size_t>::const_iterator i_sec_per_unit (sec_per_unit.cbegin());

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
    for(auto& t:tokens)
        time_period += stoll(t) * (*i_sec_per_unit++);
#pragma GCC diagnostic pop

    if(i_sec_per_unit != sec_per_unit.cend())
        throw std::runtime_error("incorrect date_time " + date_time);

    return bpt::seconds(time_period); /* TODO in future move code to milliseconds once FIX is here */
}
#pragma GCC diagnostic pop

bpt::time_period adjust_time_period_to_frame_size(const bpt::time_period & time_range,
                                                  const bpt::time_duration & resolution,
                                                  const size_t frame_size)
{
    return bpt::time_period(
                time_range.begin(),
                time_range.end() +
                bpt::minutes(frame_size -
                             time_range.length().total_seconds() / resolution.total_seconds() % frame_size));
}

} // namespace common
} // namespace svr

