//
// Created by zarko on 7/22/24.
//
#include "model/Request.hpp"

namespace svr {
namespace datamodel {

MultivalRequest::MultivalRequest() : Entity()
{
    resolution = bpt::not_a_date_time;
#ifdef ENTITY_INIT_ID
    init_id();
#endif
}

MultivalRequest::MultivalRequest(
        const bigint request_id,
        const std::string &user_name,
        const bigint dataset_id,
        const bpt::ptime &request_time,
        const bpt::ptime &value_time_start,
        const bpt::ptime &value_time_end,
        const bpt::time_duration &resolution,
        const std::string &value_columns)
        : Entity(request_id), dataset_id(dataset_id), user_name(std::move(user_name)), request_time(request_time),
          value_time_start(value_time_start), value_time_end(value_time_end), resolution(resolution), value_columns(value_columns)
{
#ifdef ENTITY_INIT_ID
    init_id();
#endif
}

void MultivalRequest::init_id()
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

std::deque<std::string> MultivalRequest::get_value_columns() const
{
    return common::from_sql_array(value_columns);
}

std::string MultivalRequest::to_string() const
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

bool MultivalRequest::sanity_check()
{
    if (dataset_id == 0) return false;
    if (resolution.is_special()) return false;
    if (value_columns.empty()) return false;
    if (value_time_end == bpt::from_time_t(0) || value_time_start == bpt::from_time_t(0)) return false;
    if (value_time_end.is_special() || value_time_start.is_special()) return false;
    return true;
}

bool MultivalRequest::operator==(MultivalRequest const &o) const
{
    return dataset_id == o.dataset_id && user_name == o.user_name && request_time == o.request_time
           && value_time_start == o.value_time_start && value_time_end == o.value_time_end && resolution == o.resolution && value_columns == o.value_columns;
}


MultivalResponse::MultivalResponse(const bigint response_id, const bigint request_id, const bpt::ptime &value_time, const std::string &value_column, const double value)
        : Entity(response_id), request_id(request_id), value_time(std::move(value_time)), value_column(std::move(value_column)), value(value)
{
#ifdef ENTITY_INIT_ID
    init_id();
#endif
}

MultivalResponse::MultivalResponse() : Entity()
{
    value = std::numeric_limits<double>::quiet_NaN();
#ifdef ENTITY_INIT_ID
    init_id();
#endif
}

std::string MultivalResponse::to_string() const
{
    std::stringstream ss;
    ss.precision(std::numeric_limits<double>::max_digits10);
    ss << "Response ID " << id
       << ", request ID " << request_id
       << ", value time " << value_time
       << ", value column " << value_column
       << ", value " << value;
    return ss.str();
}

void MultivalResponse::init_id()
{
    if (id) return;
    boost::hash_combine(id, value_column);
    boost::hash_combine(id, request_id);
    boost::hash_combine(id, value_time);
}

bool MultivalResponse::operator==(MultivalResponse const &o) const
{
    return request_id == o.request_id && value_time == o.value_time && value_column == o.value_column && value == o.value;
}


bool ValueRequest::operator==(const ValueRequest &o) const
{
    return request_time == o.request_time && value_time == o.value_time && value_column == o.value_column;
}

std::string ValueRequest::to_string() const
{
    std::stringstream ss;
    ss << "Response ID " << id
       << ", request time " << request_time
       << ", value time " << value_time
       << ", value column " << value_column;
    return ss.str();
}

void ValueRequest::init_id()
{
    if (!id) {
        boost::hash_combine(id, value_column);
        boost::hash_combine(id, request_time);
        boost::hash_combine(id, value_time);
    }
}

}
}