
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


int TsRequestDAO::save(const datamodel::MultivalRequest_ptr& request)
{
    std::scoped_lock<std::recursive_mutex> scope_guard(mutex);
    return dao->save(request);
}


datamodel::MultivalRequest_ptr TsRequestDAO::get_multival_request(const std::string &user_name, const bigint dataset_id, const bpt::ptime &value_time_start, const bpt::ptime &value_time_end)
{
    std::scoped_lock<std::recursive_mutex> scope_guard(mutex);
    return dao->get_multival_request(user_name, dataset_id, value_time_start, value_time_end);
}


datamodel::MultivalRequest_ptr TsRequestDAO::get_multival_request(const std::string &user_name, const bigint dataset_id
        , const bpt::ptime &value_time_start, const bpt::ptime &value_time_end, size_t resolution, std::string const & value_columns)
{
    std::scoped_lock<std::recursive_mutex> scope_guard(mutex);
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


std::deque<datamodel::MultivalResponse_ptr> TsRequestDAO::get_multival_results(const std::string &user_name, const bigint dataset_id
        , const bpt::ptime &value_time_start, const bpt::ptime &value_time_end, size_t resolution)
{
    return ts_call<std::deque<datamodel::MultivalResponse_ptr>>(&RequestDAO::get_multival_results, user_name, dataset_id, value_time_start, value_time_end, resolution);
}


std::deque<datamodel::MultivalResponse_ptr> TsRequestDAO::get_multival_results_column(const std::string &user_name, const std::string &column_name, const bigint dataset_id
        , const bpt::ptime &value_time_start, const bpt::ptime &value_time_end, size_t resolution)
{
    return ts_call<std::deque<datamodel::MultivalResponse_ptr>>(&RequestDAO::get_multival_results_column, user_name, column_name, dataset_id, value_time_start, value_time_end, resolution);
}

int TsRequestDAO::save(const datamodel::MultivalResponse_ptr &response)
{
        std::scoped_lock<std::recursive_mutex> scope_guard(mutex);
    return dao->save(response);
}


int TsRequestDAO::force_finalize(const datamodel::MultivalRequest_ptr &request)
{
    return ts_call<int>(&RequestDAO::force_finalize, request);
}


void TsRequestDAO::prune_finalized_requests(bpt::ptime const & olderThan)
{
    ts_call<void>(&RequestDAO::prune_finalized_requests, olderThan);
}


}
}
