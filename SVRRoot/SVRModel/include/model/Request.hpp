#pragma once

#include <boost/date_time/posix_time/ptime.hpp>
#include <boost/date_time/posix_time/time_formatters.hpp>
#include <sstream>
#include <string>
#include "model/Entity.hpp"
#include "util/string_utils.hpp"
#include "common/compatibility.hpp"

#define MAX_INPUTQUEUE_RESOLUTION 86400

namespace svr {
namespace datamodel {

class MultivalRequest : public Entity
{
public:

    MultivalRequest() : Entity()
    {
        resolution = bpt::not_a_date_time;
#ifdef ENTITY_INIT_ID
        init_id();
#endif
    }

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
              value_time_start(value_time_start), value_time_end(value_time_end), resolution(resolution), value_columns(value_columns)
    {
#ifdef ENTITY_INIT_ID
        init_id();
#endif
    }

    bigint dataset_id = 0;
    std::string user_name;
    bpt::ptime request_time, value_time_start, value_time_end;
    size_t resolution = 0;
    std::string value_columns;

    virtual void init_id() override
    {
        if (!id) {
            for (const auto &v: value_columns) boost::hash_combine(id, v);
            boost::hash_combine(id, dataset_id);
            boost::hash_combine(id, request_time);
            boost::hash_combine(id, value_time_start);
            boost::hash_combine(id, value_time_end);
            boost::hash_combine(id, resolution);
            boost::hash_combine(id, user_name);
        }
    }

    std::deque<std::string> get_value_columns() const
    {
        return common::from_sql_array(value_columns);
    }

    virtual std::string to_string() const override
    {
        std::stringstream s;
        s << "Request ID " << id
           << ", user " << user_name
           << ", dataset id " << dataset_id
           << ", request time " << request_time
           << ", value time start " << value_time_start
           << ", value time end " << value_time_end
           << ", resolution " << resolution
           << ", value columns " << value_columns;
        return s.str();
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

    bool operator==(MultivalRequest const &o) const
    {
        return dataset_id == o.dataset_id && user_name == o.user_name && request_time == o.request_time
               && value_time_start == o.value_time_start && value_time_end == o.value_time_end && resolution == o.resolution && value_columns == o.value_columns;
    }

};

class MultivalResponse : public Entity
{
public:
    bigint request_id = 0;
    bpt::ptime value_time;
    std::string value_column;
    double value = std::numeric_limits<double>::quiet_NaN();

    MultivalResponse(const bigint response_id, const bigint request_id, bpt::ptime value_time, std::string value_column, double value)
            : Entity(response_id), request_id(request_id), value_time(std::move(value_time)), value_column(std::move(value_column)), value(value)
    {
#ifdef ENTITY_INIT_ID
        init_id();
#endif
    }

    MultivalResponse() : Entity()
    {
        value = std::numeric_limits<double>::quiet_NaN();
#ifdef ENTITY_INIT_ID
        init_id();
#endif
    }

    virtual std::string to_string() const override
    {
        std::stringstream ss;
        ss.precision(11);
        ss << "Response ID " << id
           << ", request ID " << request_id
           << ", value time " << value_time
           << ", value column " << value_column
           << ", value " << value;
        return ss.str();
    }

    virtual void init_id() override
    {
        if (!id) {
            boost::hash_combine(id, value_column);
            boost::hash_combine(id, request_id);
            boost::hash_combine(id, value_time);
        }
    }

    bool operator==(MultivalResponse const &o) const
    {
        return request_id == o.request_id && value_time == o.value_time
               && value_column == o.value_column && value == o.value;
    }
};

class ValueRequest : public Entity
{
public:
    bpt::ptime request_time;
    bpt::ptime value_time;
    std::string value_column;

    bool operator==(const ValueRequest &o)
    { return request_time == o.request_time && value_time == o.value_time && value_column == o.value_column; }

    virtual std::string to_string() const override
    {
        std::stringstream ss;
        ss << "Response ID " << id
           << ", request time " << request_time
           << ", value time " << value_time
           << ", value column " << value_column;
        return ss.str();
    }

    virtual void init_id() override
    {
        if (!id) {
            boost::hash_combine(id, value_column);
            boost::hash_combine(id, request_time);
            boost::hash_combine(id, value_time);
        }
    }
};

using MultivalRequest_ptr = std::shared_ptr<MultivalRequest>;
using MultivalResponse_ptr = std::shared_ptr<MultivalResponse>;
using ValueRequest_ptr = std::shared_ptr<ValueRequest>;

} // namespace datamodel
} // namespace svr

