#pragma once

#include "common.hpp"
#include "DAO/IRowMapper.hpp"
#include "model/Request.hpp"
#include <boost/date_time/posix_time/time_parsers.hpp>
#include <boost/date_time/posix_time/time_parsers.hpp>

namespace svr {
namespace dao {

struct MultivalRequestRowMapper : public IRowMapper<svr::datamodel::MultivalRequest> {
    datamodel::MultivalRequest_ptr map_row(const pqxx_tuple &row_fields) const override
    {
        datamodel::MultivalRequest_ptr p_request = ptr<svr::datamodel::MultivalRequest>();
        p_request->set_id(row_fields["request_id"].as<bigint>(0));
        p_request->dataset_id = row_fields["dataset_id"].as<bigint>(0);
        p_request->user_name = row_fields["user_name"].as<std::string>("");
        p_request->request_time = row_fields["request_time"].as<bpt::ptime>(bpt::not_a_date_time);
        p_request->value_time_start = row_fields["value_time_start"].as<bpt::ptime>(bpt::not_a_date_time);
        p_request->value_time_end = row_fields["value_time_end"].as<bpt::ptime>(bpt::not_a_date_time);
        p_request->resolution = row_fields["resolution"].as<bpt::time_duration>(bpt::seconds(1));
        p_request->value_columns = row_fields["value_columns"].as<std::string>("");
        return p_request;
    }
};

struct MultivalResponseRowMapper : public IRowMapper<svr::datamodel::MultivalResponse> {
    datamodel::MultivalResponse_ptr map_row(const pqxx_tuple &row_fields) const override
    {
        datamodel::MultivalResponse_ptr p_response = ptr<svr::datamodel::MultivalResponse>();
        p_response->set_id(row_fields["response_id"].as<bigint>(0));
        p_response->request_id = row_fields["request_id"].as<bigint>(0);
        p_response->value_time = row_fields["value_time"].as<bpt::ptime>(bpt::not_a_date_time);
        p_response->value_column = row_fields["value_column"].as<std::string>();
        p_response->value = row_fields["value"].as<double>(std::numeric_limits<double>::quiet_NaN());
        return p_response;
    }
};

struct ValueRequestRowMapper : public IRowMapper<svr::datamodel::ValueRequest> {
    datamodel::ValueRequest_ptr map_row(const pqxx_tuple &row_fields) const override
    {
        datamodel::ValueRequest_ptr p_request = ptr<svr::datamodel::ValueRequest>();
        p_request->set_id(row_fields["request_id"].as<bigint>(0));
        p_request->request_time = row_fields["request_time"].as<bpt::ptime>(bpt::not_a_date_time);
        p_request->value_time = row_fields["value_time"].as<bpt::ptime>(bpt::not_a_date_time);
        p_request->value_column = row_fields["value_column"].as<std::string>();
        return p_request;
    }
};

}
}
