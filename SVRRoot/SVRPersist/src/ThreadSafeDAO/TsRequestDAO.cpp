
#include "TsRequestDAO.hpp"

namespace svr{
namespace dao{

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


int TsRequestDAO::save(const MultivalRequest_ptr& request)
{
    std::scoped_lock<std::recursive_mutex> scope_guard(mutex);
    return dao->save(request);
}


MultivalRequest_ptr TsRequestDAO::get_multival_request(const std::string &user_name, bigint dataset_id, const bpt::ptime &value_time_start, const bpt::ptime &value_time_end)
{
    std::scoped_lock<std::recursive_mutex> scope_guard(mutex);
    return dao->get_multival_request(user_name, dataset_id, value_time_start, value_time_end);
}


MultivalRequest_ptr TsRequestDAO::get_multival_request(const std::string &user_name, bigint dataset_id
        , const bpt::ptime &value_time_start, const bpt::ptime &value_time_end, size_t resolution, std::string const & value_columns)
{
    std::scoped_lock<std::recursive_mutex> scope_guard(mutex);
    return dao->get_multival_request(user_name, dataset_id, value_time_start, value_time_end, resolution, value_columns);
}


MultivalRequest_ptr TsRequestDAO::get_latest_multival_request(const std::string &user_name, bigint dataset_id)
{
    return ts_call<MultivalRequest_ptr>(&RequestDAO::get_latest_multival_request, user_name, dataset_id);
}


std::vector<MultivalRequest_ptr> TsRequestDAO::get_active_multival_requests(const std::string &user_name, bigint dataset_id, std::string const & inputQueueName)
{
    return ts_call<std::vector<MultivalRequest_ptr>>(&RequestDAO::get_active_multival_requests, user_name, dataset_id, inputQueueName);
}


std::vector<MultivalResponse_ptr> TsRequestDAO::get_multival_results(const std::string &user_name, bigint dataset_id
        , const bpt::ptime &value_time_start, const bpt::ptime &value_time_end, size_t resolution)
{
    return ts_call<std::vector<MultivalResponse_ptr>>(&RequestDAO::get_multival_results, user_name, dataset_id, value_time_start, value_time_end, resolution);
}


std::vector<MultivalResponse_ptr> TsRequestDAO::get_multival_results_column(const std::string &user_name, const std::string &column_name, bigint dataset_id
        , const bpt::ptime &value_time_start, const bpt::ptime &value_time_end, size_t resolution)
{
    return ts_call<std::vector<MultivalResponse_ptr>>(&RequestDAO::get_multival_results_column, user_name, column_name, dataset_id, value_time_start, value_time_end, resolution);
}

int TsRequestDAO::save(const MultivalResponse_ptr &response)
{
        std::scoped_lock<std::recursive_mutex> scope_guard(mutex);
    return dao->save(response);
}


int TsRequestDAO::force_finalize(const MultivalRequest_ptr &request)
{
    return ts_call<int>(&RequestDAO::force_finalize, request);
}


void TsRequestDAO::prune_finalized_requests(bpt::ptime const & olderThan)
{
    ts_call<void>(&RequestDAO::prune_finalized_requests, olderThan);
}


}
}
