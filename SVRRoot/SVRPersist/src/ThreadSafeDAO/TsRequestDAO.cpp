
#include "TsRequestDAO.hpp"

namespace svr {
namespace dao {

DEFINE_THREADSAFE_DAO_CONSTRUCTOR(TsRequestDAO, RequestDAO)
{}

bigint TsRequestDAO::get_next_id()
{
    return ts_call<bigint>(&RequestDAO::get_next_id);
}


bigint TsRequestDAO::get_next_result_id()
{
    return ts_call<bigint>(&RequestDAO::get_next_result_id);
}


int TsRequestDAO::save(const datamodel::MultivalRequest_ptr &request)
{
    const std::scoped_lock<std::recursive_mutex> scope_guard(mutex);
    return dao->save(request);
}

bool TsRequestDAO::exists(const std::string &user, const bigint dataset_id, const bpt::ptime &start_time, const bpt::ptime &end_time, const bpt::time_duration &resolution,
                         const std::string &value_columns)
{
    const std::scoped_lock<std::recursive_mutex> scope_guard(mutex);
    return dao->exists(user, dataset_id, start_time, end_time, resolution, value_columns);
}

bigint TsRequestDAO::make_request(
                const std::string &user, const std::string &dataset_id_str, const std::string &value_time_start_str, const std::string &value_time_end_str,
                const std::string &resolution_str, const std::string &value_columns)
{
    const std::scoped_lock<std::recursive_mutex> scope_guard(mutex);
    return dao->make_request(user, dataset_id_str, value_time_start_str, value_time_end_str, resolution_str, value_columns);
}

datamodel::MultivalRequest_ptr
TsRequestDAO::get_multival_request(const std::string &user_name, const bigint dataset_id, const bpt::ptime &value_time_start, const bpt::ptime &value_time_end)
{
    const std::scoped_lock<std::recursive_mutex> scope_guard(mutex);
    return dao->get_multival_request(user_name, dataset_id, value_time_start, value_time_end);
}


datamodel::MultivalRequest_ptr
TsRequestDAO::get_multival_request(const std::string &user_name, const bigint dataset_id, const bpt::ptime &value_time_start, const bpt::ptime &value_time_end,
                                   const bpt::time_duration &resolution, std::string const &value_columns)
{
    const std::scoped_lock<std::recursive_mutex> scope_guard(mutex);
    return dao->get_multival_request(user_name, dataset_id, value_time_start, value_time_end, resolution, value_columns);
}


datamodel::MultivalRequest_ptr TsRequestDAO::get_latest_multival_request(const std::string &user_name, const bigint dataset_id)
{
    return ts_call<datamodel::MultivalRequest_ptr>(&RequestDAO::get_latest_multival_request, user_name, dataset_id);
}


std::deque<datamodel::MultivalRequest_ptr> TsRequestDAO::get_active_multival_requests(const std::string &user_name, const bigint dataset_id)
{
    return ts_call<std::deque<datamodel::MultivalRequest_ptr>>(&RequestDAO::get_active_multival_requests, user_name, dataset_id);
}


std::deque<datamodel::MultivalResponse_ptr> TsRequestDAO::get_multival_results(
            const std::string &user_name, const bigint dataset_id, const bpt::ptime &value_time_start, const bpt::ptime &value_time_end, const bpt::time_duration &resolution)
{
    return ts_call<std::deque<datamodel::MultivalResponse_ptr>>(&RequestDAO::get_multival_results, user_name, dataset_id, value_time_start, value_time_end, resolution);
}

std::deque<datamodel::MultivalResponse_ptr> TsRequestDAO::get_multival_results_str(
        const std::string &user_name, const std::string &dataset_id, const std::string &value_time_start, const std::string &value_time_end, const std::string &resolution)
{
    return ts_call<std::deque<datamodel::MultivalResponse_ptr>>(&RequestDAO::get_multival_results_str, user_name, dataset_id, value_time_start, value_time_end, resolution);
}

std::deque<datamodel::MultivalResponse_ptr>
TsRequestDAO::get_multival_results_column(const std::string &user_name, const std::string &column_name, const bigint dataset_id, const bpt::ptime &value_time_start,
                                          const bpt::ptime &value_time_end, const bpt::time_duration &resolution)
{
    return ts_call<std::deque<datamodel::MultivalResponse_ptr>>(&RequestDAO::get_multival_results_column, user_name, column_name, dataset_id, value_time_start, value_time_end,
                                                                resolution);
}

int TsRequestDAO::save(const datamodel::MultivalResponse_ptr &response)
{
    const std::scoped_lock<std::recursive_mutex> scope_guard(mutex);
    return dao->save(response);
}


int TsRequestDAO::force_finalize(const datamodel::MultivalRequest_ptr &request)
{
    return ts_call<int>(&RequestDAO::force_finalize, request);
}


void TsRequestDAO::prune_finalized_requests(bpt::ptime const &before)
{
    ts_call<void>(&RequestDAO::prune_finalized_requests, before);
}


}
}
