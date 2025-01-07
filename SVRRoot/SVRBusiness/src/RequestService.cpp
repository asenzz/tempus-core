#include "RequestService.hpp"
#include "model/User.hpp"

#include <model/InputQueue.hpp>
#include <DAO/RequestDAO.hpp>
#include <util/validation_utils.hpp>
#include <model/Request.hpp>
#include <model/Dataset.hpp>

using namespace svr::common;

namespace svr {
namespace business {

RequestService::RequestService(svr::dao::RequestDAO &request_dao)
        : request_dao(request_dao)
{}

int RequestService::save(const datamodel::MultivalRequest_ptr &request)
{
    return request_dao.save(request);
}

datamodel::MultivalRequest_ptr
RequestService::get_multival_request(const std::string &user_name, const bigint dataset_id, const bpt::ptime &value_time_start, const bpt::ptime &value_time_end)
{
    return request_dao.get_multival_request(user_name, dataset_id, value_time_start, value_time_end);
}

datamodel::MultivalRequest_ptr RequestService::get_multival_request(
        const std::string &user_name, const bigint dataset_id, const bpt::ptime &value_time_start, const bpt::ptime &value_time_end,
                                                                    const bpt::time_duration &resolution, const std::string &value_columns)
{
    return request_dao.get_multival_request(user_name, dataset_id, value_time_start, value_time_end, resolution, value_columns);
}

datamodel::MultivalRequest_ptr RequestService::get_latest_multival_request(svr::datamodel::User const &user, svr::datamodel::Dataset const &dataset)
{
    return request_dao.get_latest_multival_request(user.get_user_name(), dataset.get_id());
}

std::deque<datamodel::MultivalRequest_ptr> RequestService::get_active_multival_requests(
        svr::datamodel::User const &user, svr::datamodel::Dataset const &dataset)
{
    return request_dao.get_active_multival_requests(user.get_user_name(), dataset.get_id());
}

std::deque<datamodel::MultivalResponse_ptr>
RequestService::get_multival_results(const std::string &user_name, const bigint dataset_id, const bpt::ptime &value_time_start, const bpt::ptime &value_time_end,
                                     const bpt::time_duration &resolution)
{
    return request_dao.get_multival_results(user_name, dataset_id, value_time_start, value_time_end, resolution);
}

std::deque<datamodel::MultivalResponse_ptr>
RequestService::get_multival_results(const std::string &user_name, const std::string &dataset_id, const std::string &value_time_start, const std::string &value_time_end,
                                     const std::string &resolution)
{
    return request_dao.get_multival_results_str(user_name, dataset_id, value_time_start, value_time_end, resolution);
}

std::deque<datamodel::MultivalResponse_ptr>
RequestService::get_multival_results(const std::string &user_name, const std::string &column_name, const bigint dataset_id, const bpt::ptime &value_time_start,
                                     const bpt::ptime &value_time_end, const bpt::time_duration &resolution)
{
    return request_dao.get_multival_results_column(user_name, column_name, dataset_id, value_time_start, value_time_end, resolution);
}

int RequestService::force_finalize(const datamodel::MultivalRequest_ptr &request)
{
    LOG4_DEBUG("Forced finalize of request with id " << request->get_id());

    return request_dao.force_finalize(request);
}

int RequestService::save(const datamodel::MultivalResponse_ptr &response)
{
    LOG4_TRACE("Saving " << *response);
    return request_dao.save(response);
}

void RequestService::prune_finalized_requests(bpt::ptime const &older_than)
{
    request_dao.prune_finalized_requests(older_than);
}

bool RequestService::exists(const std::string &user, const bigint dataset_id, const boost::posix_time::ptime &start_time, const boost::posix_time::ptime &end_time,
                            const bpt::time_duration &resolution, const std::string &value_columns)
{
    return request_dao.exists(user, dataset_id, start_time, end_time, resolution, value_columns);
}

bigint RequestService::make_request(
        const std::string &user, const std::string &dataset_id_str, const std::string &value_time_start_str, const std::string &value_time_end_str,
        const std::string &resolution_str, const std::string &value_columns)
{
    return request_dao.make_request(user, dataset_id_str, value_time_start_str, value_time_end_str, resolution_str, value_columns);
}

} /* namespace business */
}
