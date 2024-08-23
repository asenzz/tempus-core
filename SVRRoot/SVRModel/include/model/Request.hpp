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

struct MultivalRequest : public Entity
{
    bigint dataset_id = 0;
    std::string user_name;
    bpt::ptime request_time, value_time_start, value_time_end;
    size_t resolution = 0;
    std::string value_columns;

    MultivalRequest();

    MultivalRequest(
            bigint request_id,
            std::string user_name,
            bigint dataset_id,
            bpt::ptime request_time,
            bpt::ptime value_time_start,
            bpt::ptime value_time_end,
            size_t resolution,
            std::string const &value_columns);

    virtual void init_id() override;

    std::deque<std::string> get_value_columns() const;

    virtual std::string to_string() const override;

    bool sanity_check();

    bool operator==(MultivalRequest const &o) const;
};

struct MultivalResponse : public Entity
{
    bigint request_id = 0;
    bpt::ptime value_time;
    std::string value_column;
    double value = std::numeric_limits<double>::quiet_NaN();

    MultivalResponse(const bigint response_id, const bigint request_id, bpt::ptime value_time, std::string value_column, double value);

    MultivalResponse();

    virtual std::string to_string() const override;

    virtual void init_id() override;

    bool operator==(MultivalResponse const &o) const;
};

struct ValueRequest : public Entity
{
    bpt::ptime request_time;
    bpt::ptime value_time;
    std::string value_column;

    bool operator==(const ValueRequest &o) const;

    virtual std::string to_string() const override;

    virtual void init_id() override;
};

using MultivalRequest_ptr = std::shared_ptr<MultivalRequest>;
using MultivalResponse_ptr = std::shared_ptr<MultivalResponse>;
using ValueRequest_ptr = std::shared_ptr<ValueRequest>;

} // namespace datamodel
} // namespace svr
