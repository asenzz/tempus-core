#include "RequestService.hpp"
#include "model/User.hpp"

#include <model/InputQueue.hpp>
#include <DAO/RequestDAO.hpp>
#include <util/ValidationUtils.hpp>
#include <model/Request.hpp>
#include <model/Dataset.hpp>

using namespace svr::common;
namespace svr {
namespace business {

RequestService::RequestService(svr::dao::RequestDAO &request_dao)
        : requestDao(request_dao)
{}

int RequestService::save(const datamodel::MultivalRequest_ptr &request)
{
    return requestDao.save(request);
}

datamodel::MultivalRequest_ptr
RequestService::get_multival_request(const std::string &user_name, const bigint dataset_id, const bpt::ptime &value_time_start, const bpt::ptime &value_time_end)
{
    return requestDao.get_multival_request(user_name, dataset_id, value_time_start, value_time_end);
}

datamodel::MultivalRequest_ptr RequestService::get_multival_request(const std::string &user_name,
                                                                    bigint dataset_id, const bpt::ptime &value_time_start, const bpt::ptime &value_time_end,
                                                                    const size_t resolution, std::string const &value_columns)
{
    return requestDao.get_multival_request(user_name, dataset_id, value_time_start, value_time_end, resolution, value_columns);
}

datamodel::MultivalRequest_ptr RequestService::get_latest_multival_request(svr::datamodel::User const &user, svr::datamodel::Dataset const &dataset)
{
    return requestDao.get_latest_multival_request(user.get_user_name(), dataset.get_id());
}

std::deque<datamodel::MultivalRequest_ptr> RequestService::get_active_multival_requests(
        svr::datamodel::User const &user, svr::datamodel::Dataset const &dataset)
{
    return requestDao.get_active_multival_requests(user.get_user_name(), dataset.get_id());
}

std::deque<datamodel::MultivalResponse_ptr>
RequestService::get_multival_results(const std::string &user_name, const bigint dataset_id, const bpt::ptime &value_time_start, const bpt::ptime &value_time_end,
                                     const size_t resolution)
{
    return requestDao.get_multival_results(user_name, dataset_id, value_time_start, value_time_end, resolution);
}

std::deque<datamodel::MultivalResponse_ptr>
RequestService::get_multival_results(const std::string &user_name, const std::string &column_name, const bigint dataset_id, const bpt::ptime &value_time_start,
                                     const bpt::ptime &value_time_end, const size_t resolution)
{
    return requestDao.get_multival_results_column(user_name, column_name, dataset_id, value_time_start, value_time_end, resolution);
}

int RequestService::force_finalize(const datamodel::MultivalRequest_ptr &request)
{
    LOG4_DEBUG("Forced finalize of request with id " << request->get_id());

    return requestDao.force_finalize(request);
}

int RequestService::save(const datamodel::MultivalResponse_ptr &response)
{
    return requestDao.save(response);
}

void RequestService::prune_finalized_requests(bpt::ptime const &older_than)
{
    requestDao.prune_finalized_requests(older_than);
}

}
}
