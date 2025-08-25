#pragma once

#include "common.hpp"
#include "DAO/IRowMapper.hpp"
#include "model/Request.hpp"
#include <boost/date_time/posix_time/time_parsers.hpp>
#include <boost/date_time/posix_time/time_parsers.hpp>

namespace svr {
namespace dao {

struct MultivalRequestRowMapper : public IRowMapper<datamodel::MultivalRequest> {
    datamodel::MultivalRequest_ptr map_row(const pqxx_tuple &row_fields) const override
    {
        datamodel::MultivalRequest_ptr p_request = ptr<datamodel::MultivalRequest>();
        p_request->set_id(row_fields["request_id"].as<bigint>(0));
        p_request->dataset_id = row_fields["dataset_id"].as<bigint>(0);
        p_request->user_name = row_fields["user_name"].as<std::string>();
        p_request->request_time = row_fields["request_time"].as<bpt::ptime>(bpt::not_a_date_time);
        p_request->value_time_start = row_fields["value_time_start"].as<bpt::ptime>(bpt::not_a_date_time);
        p_request->value_time_end = row_fields["value_time_end"].as<bpt::ptime>(bpt::not_a_date_time);
        p_request->resolution = row_fields["resolution"].as(common::C_default_resolution);
        p_request->value_columns = row_fields["value_columns"].as<std::string>();
        return p_request;
    }

    datamodel::MultivalRequest_ptr map_row(duckdb_result& row_set, const size_t col_count, const size_t row) const override
    {
        datamodel::MultivalRequest_ptr p_request = ptr<datamodel::MultivalRequest>();
        p_request->set_id(common::dd_get_value(row_set, row, col_count, "id", bigint(0)));
        p_request->dataset_id = common::dd_get_value(row_set, row, col_count, "dataset_id", bigint(0));
        p_request->user_name = common::dd_get_value(row_set, row, col_count, "user_name", std::string());
        p_request->request_time = common::dd_get_value<bpt::ptime>(row_set, row, col_count, "request_time", bpt::not_a_date_time);
        p_request->value_time_start = common::dd_get_value<bpt::ptime>(row_set, row, col_count, "value_time_start", bpt::not_a_date_time);
        p_request->value_time_end = common::dd_get_value<bpt::ptime>(row_set, row, col_count, "value_time_end", bpt::not_a_date_time);
        p_request->resolution = common::dd_get_value(row_set, row, col_count, "resolution", common::C_default_resolution);
        p_request->value_columns = common::dd_get_value(row_set, row, col_count, "value_columns", std::string());
        return p_request;
    }
};

struct MultivalResponseRowMapper : public IRowMapper<datamodel::MultivalResponse> {
    datamodel::MultivalResponse_ptr map_row(const pqxx_tuple &row_fields) const override
    {
        datamodel::MultivalResponse_ptr p_response = ptr<datamodel::MultivalResponse>();
        p_response->set_id(row_fields["response_id"].as<bigint>(0));
        p_response->request_id = row_fields["request_id"].as<bigint>(0);
        p_response->value_time = row_fields["value_time"].as<bpt::ptime>(bpt::not_a_date_time);
        p_response->value_column = row_fields["value_column"].as<std::string>();
        p_response->value = row_fields["value"].as(std::numeric_limits<double>::quiet_NaN());
        return p_response;
    }

    datamodel::MultivalResponse_ptr map_row(duckdb_result& row_set, const size_t col_count, const size_t row) const override
    {
        datamodel::MultivalResponse_ptr p_response = ptr<datamodel::MultivalResponse>();
        p_response->set_id(common::dd_get_value(row_set, row, col_count, "response_id", bigint(0)));
        p_response->request_id = common::dd_get_value(row_set, row, col_count, "request_id", bigint(0));
        p_response->value_time = common::dd_get_value<bpt::ptime>(row_set, row, col_count, "value_time", bpt::not_a_date_time);
        p_response->value_column = common::dd_get_value(row_set, row, col_count, "value_column", std::string());
        p_response->value = common::dd_get_value(row_set, row, col_count, "value", std::numeric_limits<double>::quiet_NaN());
        return p_response;
    }
};

struct ValueRequestRowMapper : public IRowMapper<datamodel::ValueRequest> {
    datamodel::ValueRequest_ptr map_row(const pqxx_tuple &row_fields) const override
    {
        datamodel::ValueRequest_ptr p_request = ptr<datamodel::ValueRequest>();
        p_request->set_id(row_fields["request_id"].as<bigint>(0));
        p_request->request_time = row_fields["request_time"].as<bpt::ptime>(bpt::not_a_date_time);
        p_request->value_time = row_fields["value_time"].as<bpt::ptime>(bpt::not_a_date_time);
        p_request->value_column = row_fields["value_column"].as<std::string>();
        return p_request;
    }

    datamodel::ValueRequest_ptr map_row(duckdb_result& row_set, const size_t col_count, const size_t row) const override
    {
        datamodel::ValueRequest_ptr p_request = ptr<datamodel::ValueRequest>();
        p_request->set_id(common::dd_get_value(row_set, row, col_count, "request_id", bigint(0)));
        p_request->request_time = common::dd_get_value<bpt::ptime>(row_set, row, col_count, "request_time", bpt::not_a_date_time);
        p_request->value_time = common::dd_get_value<bpt::ptime>(row_set, row, col_count, "value_time", bpt::not_a_date_time);
        p_request->value_column = common::dd_get_value(row_set, row, col_count, "value_column", std::string());
        return p_request;
    }
};

}
}
