#pragma once

#include "common.hpp"
#include "DAO/IRowMapper.hpp"
#include "model/Request.hpp"
#include <boost/date_time/posix_time/time_parsers.hpp>
#include <boost/date_time/posix_time/time_parsers.hpp>

namespace svr{
namespace dao{

class MultivalRequestRowMapper : public IRowMapper<svr::datamodel::MultivalRequest> {

public:
    MultivalRequest_ptr mapRow(const pqxx_tuple& rowSet) const override {
        MultivalRequest_ptr request = std::make_shared<svr::datamodel::MultivalRequest>();

        request->set_id(rowSet["request_id"].as<bigint>());
        request->dataset_id = rowSet["dataset_id"].as<bigint>();
        request->user_name = rowSet["user_name"].as<std::string>();
        request->request_time = bpt::time_from_string(rowSet["request_time"].as<std::string>());
        request->value_time_start = bpt::time_from_string(rowSet["value_time_start"].as<std::string>());
        request->value_time_end = bpt::time_from_string(rowSet["value_time_end"].as<std::string>());
        request->resolution = rowSet["resolution"].as<std::int32_t>();
        request->value_columns = rowSet["value_columns"].as<std::string>();
        return request;
    }

};

class MultivalResponseRowMapper : public IRowMapper<svr::datamodel::MultivalResponse> {

public:
    MultivalResponse_ptr mapRow(const pqxx_tuple& rowSet) const override
    {
        MultivalResponse_ptr response = std::make_shared<svr::datamodel::MultivalResponse>();

        response->set_id(rowSet["response_id"].as<bigint>());
        response->request_id = rowSet["request_id"].as<bigint>();
        response->value_time = bpt::time_from_string(rowSet["value_time"].as<std::string>());
        response->value_column = rowSet["value_column"].as<std::string>();
        response->value = rowSet["value"].as<double>(std::numeric_limits<double>::quiet_NaN());
        return response;
    }

};

class ValueRequestRowMapper : public IRowMapper<svr::datamodel::ValueRequest>
{

public:
    ValueRequest_ptr mapRow(const pqxx_tuple& rowSet) const override {
        ValueRequest_ptr valReq = std::make_shared<svr::datamodel::ValueRequest>();

        valReq->set_id(rowSet["request_id"].as<bigint>());
        valReq->request_time = bpt::time_from_string(rowSet["request_time"].as<std::string>());
        valReq->value_time = bpt::time_from_string(rowSet["value_time"].as<std::string>());
        valReq->value_column = rowSet["value_column"].as<std::string>();
        return valReq;
    }
};

}
}
