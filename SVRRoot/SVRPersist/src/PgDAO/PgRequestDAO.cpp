#include "PgRequestDAO.hpp"
#include <DAO/DataSource.hpp>
#include <model/Request.hpp>
#include <DAO/RequestRowMapper.hpp>

namespace svr {
namespace dao {

PgRequestDAO::PgRequestDAO(svr::common::PropertiesFileReader& sqlProperties, svr::dao::DataSource& dataSource)
: RequestDAO(sqlProperties, dataSource)
{}

bigint PgRequestDAO::get_next_id() {
    return data_source.query_for_type<bigint>(get_sql("get_next_id"));
}

bigint PgRequestDAO::get_next_result_id()
{
    return data_source.query_for_type<bigint>(get_sql("get_next_result_id"));
}

int PgRequestDAO::save(const MultivalRequest_ptr &request) {

    if(request->get_id() > 0){
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

MultivalRequest_ptr PgRequestDAO::get_multival_request(const std::string &user_name, bigint dataset_id, const bpt::ptime &value_time_start, const bpt::ptime &value_time_end)
{
    MultivalRequestRowMapper rowMapper;
    return data_source.query_for_object(&rowMapper, get_sql("get_user_multival_request"),
                                        user_name,
                                        dataset_id,
                                        value_time_start,
                                        value_time_end
    );
}

MultivalRequest_ptr PgRequestDAO::get_multival_request(const std::string &user_name, bigint dataset_id,
        const bpt::ptime &value_time_start, const bpt::ptime &value_time_end, size_t resolution, std::string const & value_columns)
{
    MultivalRequestRowMapper rowMapper;
    return data_source.query_for_object(&rowMapper, get_sql("get_multival_request"),
                                        user_name,
                                        dataset_id,
                                        value_time_start,
                                        value_time_end,
                                        resolution,
                                        value_columns
    );
}

MultivalRequest_ptr PgRequestDAO::get_latest_multival_request(const std::string &user_name, bigint dataset_id)
{
    MultivalRequestRowMapper rowMapper;
    return data_source.query_for_object(&rowMapper, get_sql("get_latest_multival_request"),
                                        user_name,
                                        dataset_id);
}

std::vector<MultivalRequest_ptr> PgRequestDAO::get_active_multival_requests(const std::string &user_name, bigint dataset_id, std::string const &inputQueueName)
{
    MultivalRequestRowMapper rowMapper;

    return data_source.query_for_array(rowMapper, get_sql("get_active_multival_requests"), user_name, dataset_id);
}

std::vector<MultivalResponse_ptr> PgRequestDAO::get_multival_results(
        const std::string &user_name, bigint dataset_id, const bpt::ptime &value_time_start, const bpt::ptime &value_time_end, const size_t resolution)
{
    MultivalResponseRowMapper rowMapper;
    return data_source.query_for_array(rowMapper, get_sql("get_multival_results"),
                                        //user_name,
                                        //dataset_id,
                                        value_time_start,
                                        value_time_end
                                        //resolution
    );
}

std::vector<MultivalResponse_ptr> PgRequestDAO::get_multival_results_column(
        const std::string &user_name, const std::string &column_name,
        bigint dataset_id, const bpt::ptime &value_time_start, const bpt::ptime &value_time_end, const size_t resolution)
{
    MultivalResponseRowMapper rowMapper;
    return data_source.query_for_array(rowMapper, get_sql("get_multival_results_column"),
                                       user_name,
                                       column_name,
                                       dataset_id,
                                       value_time_start,
                                       value_time_end,
                                       resolution
    );
}

int PgRequestDAO::save(const MultivalResponse_ptr &response) {

    int result = 0;
    if(response->get_id())
    {
        result = data_source.update(get_sql("value_update"),
                              response->request_id,
                              response->value_time,
                              response->value_column,
                              response->value,
                              response->get_id()
        );
    } else {
        response->set_id(get_next_result_id());
         result= data_source.update(get_sql("value_save"),
                                  response->get_id(),
                                  response->request_id,
                                  response->value_time,
                                  response->value_column,
                                  response->value
        );
    }
    return result;
}

int PgRequestDAO::force_finalize(const MultivalRequest_ptr &request)
{
    data_source.update(get_sql("force_finalize_request"), request->get_id());
    return 0;
}

void PgRequestDAO::prune_finalized_requests(bpt::ptime const & olderThan)
{
    data_source.update(get_sql("prune_finalized_requests"), olderThan);
}

} }
