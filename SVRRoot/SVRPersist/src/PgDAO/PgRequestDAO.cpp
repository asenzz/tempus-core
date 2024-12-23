#include "PgRequestDAO.hpp"
#include <DAO/DataSource.hpp>
#include <model/Request.hpp>
#include <DAO/RequestRowMapper.hpp>

namespace svr {
namespace dao {

PgRequestDAO::PgRequestDAO(svr::common::PropertiesFileReader &tempus_config, svr::dao::DataSource &data_source)
        : RequestDAO(tempus_config, data_source)
{}



bigint PgRequestDAO::get_next_id()
{
    return data_source.query_for_type<bigint>(get_sql("get_next_id"));
}

bigint PgRequestDAO::get_next_result_id()
{
    return data_source.query_for_type<bigint>(get_sql("get_next_result_id"));
}

int PgRequestDAO::save(const datamodel::MultivalRequest_ptr &request)
{

    if (request->get_id() > 0) {
        return data_source.update(get_sql("multival_update"),
                                  request->dataset_id,
                                  request->request_time,
                                  request->value_time_start,
                                  request->value_time_end,
                                  request->resolution,
                                  request->value_columns,
                                  request->get_id()
        );
    }
    request->set_id(get_next_id());

    return data_source.update(get_sql("multival_save"),
                              request->get_id(),
                              request->request_time,
                              request->user_name,
                              request->dataset_id,
                              request->value_time_start,
                              request->value_time_end,
                              request->resolution,
                              request->value_columns
    );
}

bool PgRequestDAO::exists(const std::string &user, const bigint dataset_id, const boost::posix_time::ptime &start_time, const boost::posix_time::ptime &end_time,
                          const bpt::time_duration &resolution, const std::string &value_columns)
{
    return data_source.query_for_type<uint32_t>(get_sql("exists"), user, dataset_id, start_time, end_time, resolution, value_columns) > 0;
}

bigint PgRequestDAO::make_request(
            const std::string &user, const std::string &dataset_id_str, const std::string &value_time_start_str, const std::string &value_time_end_str,
            const std::string &resolution_str, const std::string &value_columns)
{
    return data_source.query_for_type<bigint>("SELECT * FROM make_request(?, ?, ?, ?, ?, ?)",
                              user, dataset_id_str, value_time_start_str, value_time_end_str, resolution_str, value_columns);
}

datamodel::MultivalRequest_ptr
PgRequestDAO::get_multival_request(const std::string &user_name, const bigint dataset_id, const bpt::ptime &value_time_start, const bpt::ptime &value_time_end)
{
    MultivalRequestRowMapper row_mapper;
    return data_source.query_for_object(&row_mapper, get_sql("get_user_multival_request"),
                                        user_name,
                                        dataset_id,
                                        value_time_start,
                                        value_time_end
    );
}

datamodel::MultivalRequest_ptr PgRequestDAO::get_multival_request(
        const std::string &user_name, const bigint dataset_id, const bpt::ptime &value_time_start, const bpt::ptime &value_time_end, const bpt::time_duration &resolution,
        const std::string &value_columns)
{
    MultivalRequestRowMapper row_mapper;
    return data_source.query_for_object(&row_mapper, get_sql("get_multival_request"),
                                        user_name,
                                        dataset_id,
                                        value_time_start,
                                        value_time_end,
                                        resolution,
                                        value_columns
    );
}

datamodel::MultivalRequest_ptr PgRequestDAO::get_latest_multival_request(const std::string &user_name, const bigint dataset_id)
{
    MultivalRequestRowMapper row_mapper;
    return data_source.query_for_object(&row_mapper, get_sql("get_latest_multival_request"), user_name, dataset_id);
}

std::deque<datamodel::MultivalRequest_ptr> PgRequestDAO::get_active_multival_requests(const std::string &user_name, const bigint dataset_id)
{
    MultivalRequestRowMapper row_mapper;
    return data_source.query_for_deque(row_mapper, get_sql("get_active_multival_requests"), user_name, dataset_id);
}

std::deque<datamodel::MultivalResponse_ptr> PgRequestDAO::get_multival_results(
        const std::string &user_name, const bigint dataset_id, const bpt::ptime &value_time_start, const bpt::ptime &value_time_end, const bpt::time_duration &resolution)
{
    MultivalResponseRowMapper row_mapper;
    return data_source.query_for_deque(row_mapper, get_sql("get_multival_results"),
            //user_name,
            //dataset_id,
                                       value_time_start,
                                       value_time_end
            //resolution
    );
}

std::deque<datamodel::MultivalResponse_ptr> PgRequestDAO::get_multival_results_str(
        const std::string &user_name, const std::string &dataset_id, const std::string &value_time_start, const std::string &value_time_end, const std::string &resolution)
{
    MultivalResponseRowMapper row_mapper;
    return data_source.query_for_deque(row_mapper, get_sql("get_multival_results"),
            //user_name,
            //dataset_id,
                                       value_time_start,
                                       value_time_end
            //resolution
    );
}

std::deque<datamodel::MultivalResponse_ptr> PgRequestDAO::get_multival_results_column(
        const std::string &user_name, const std::string &column_name, const bigint dataset_id, const bpt::ptime &value_time_start, const bpt::ptime &value_time_end,
        const bpt::time_duration &resolution)
{
    MultivalResponseRowMapper row_mapper;
    return data_source.query_for_deque(
            row_mapper, get_sql("get_multival_results_column"), user_name, column_name, dataset_id, value_time_start,
            value_time_end, resolution);
}

int PgRequestDAO::save(const datamodel::MultivalResponse_ptr &response)
{

    int result = 0;
    if (response->get_id()) {
        result = data_source.update(get_sql("value_update"),
                                    response->request_id,
                                    response->value_time,
                                    response->value_column,
                                    response->value,
                                    response->get_id()
        );
    } else {
        response->set_id(get_next_result_id());
        result = data_source.update(get_sql("value_save"),
                                    response->get_id(),
                                    response->request_id,
                                    response->value_time,
                                    response->value_column,
                                    response->value
        );
    }
    return result;
}

int PgRequestDAO::force_finalize(const datamodel::MultivalRequest_ptr &request)
{
    data_source.update(get_sql("force_finalize_request"), request->get_id());
    return 0;
}

void PgRequestDAO::prune_finalized_requests(bpt::ptime const &before)
{
    data_source.update(get_sql("prune_finalized_requests"), before);
}

}
}
