#pragma once

#include <boost/date_time/posix_time/ptime.hpp>
#include <boost/date_time/posix_time/time_formatters.hpp>
#include <model/Entity.hpp>
#include <sstream>
#include <string>

#define MAX_INPUTQUEUE_RESOLUTION 86400
namespace svr{
namespace datamodel{

class MultivalRequest : public Entity{
public:

    MultivalRequest() : Entity() { resolution = bpt::not_a_date_time; }
    MultivalRequest(
            bigint request_id,
            std::string user_name,
            bigint dataset_id,
            bpt::ptime request_time,
            bpt::ptime value_time_start,
            bpt::ptime value_time_end,
            size_t resolution,
            std::string const &value_columns)
    : Entity(request_id), dataset_id(dataset_id), user_name(std::move(user_name)), request_time(request_time),
        value_time_start(value_time_start), value_time_end(value_time_end), resolution(resolution), value_columns(value_columns) {}

    bigint dataset_id;
    std::string user_name;
    bpt::ptime request_time;
    bpt::ptime value_time_start;
    bpt::ptime value_time_end;
    size_t resolution;
    std::string value_columns;

    virtual std::string to_string() const
    {
        std::stringstream ss;
        ss  << "RequestID: " << get_id()
            << ", User: " << user_name
            << ", DatasetID: " << dataset_id
            << ", Request Time: " << request_time
            << ", Value time start: " << value_time_start
            << ", Value time end: " << value_time_end
            << ", Resolution: " << resolution
            << ", value columns" << value_columns;
        return ss.str();
    }

    bool sanity_check()
    {
        if (dataset_id == 0) return false;
        if (resolution < 1 || resolution > MAX_INPUTQUEUE_RESOLUTION) return false;
        if (value_columns.empty()) return false;
        if (value_time_end == bpt::from_time_t(0) || value_time_start == bpt::from_time_t(0)) return false;
        if (value_time_end.is_special() || value_time_start.is_special()) return false;
        if (value_time_end - value_time_start > bpt::hours(10)) return false;
        return true;
    }

    bool operator==(MultivalRequest const & o) const {return dataset_id==o.dataset_id && user_name==o.user_name && request_time==o.request_time
            && value_time_start==o.value_time_start && value_time_end==o.value_time_end && resolution==o.resolution && value_columns==o.value_columns; }

};

class MultivalResponse : public Entity
{
public:
    bigint request_id;
    bpt::ptime value_time;
    std::string value_column;
    double value;

    MultivalResponse(const bigint response_id, const bigint request_id, bpt::ptime value_time, std::string value_column, double value)
            : Entity(response_id), request_id(request_id), value_time(std::move(value_time)), value_column(std::move(value_column)), value(value) {}
    MultivalResponse(): Entity() { value = std::numeric_limits<double>::quiet_NaN(); }

    virtual std::string to_string() const{
        std::stringstream ss;
        ss.precision(11);
        ss  << "ResponseID: " << get_id()
            << ", RequestID: " << request_id
            << ", ValueTime: " << bpt::to_simple_string(value_time)
            << ", ValueColumn: " << value_column
            << ", Value: " << value
        ;
        return ss.str();
    }

    bool operator==(MultivalResponse const & o) const {return request_id==o.request_id && value_time==o.value_time
            && value_column==o.value_column && value==o.value; }
};

class ValueRequest : public Entity{
public:
    bpt::ptime request_time;
    bpt::ptime value_time;
    std::string value_column;

    bool operator==(const ValueRequest &o) { return request_time == o.request_time && value_time == o.value_time && value_column == o.value_column; }

    virtual std::string to_string() const{
        std::stringstream ss;
        ss  << "ResponseID: " << get_id()
            << ", RequestTime: " << bpt::to_simple_string(request_time)
            << ", value_time: " << bpt::to_simple_string(value_time)
            << ", ValueColumn: " << value_column
        ;
        return ss.str();
    }
};

using MultivalRequest_ptr = std::shared_ptr<MultivalRequest>;
using MultivalResponse_ptr = std::shared_ptr<MultivalResponse>;
using ValueRequest_ptr = std::shared_ptr<ValueRequest>;

} // namespace datamodel
} // namespace svr

