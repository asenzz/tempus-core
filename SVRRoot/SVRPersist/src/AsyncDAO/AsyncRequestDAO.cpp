#include "AsyncRequestDAO.hpp"
#include "model/Request.hpp"
#include "AsyncImplBase.hpp"
#include "../PgDAO/PgRequestDAO.hpp"

namespace svr {
namespace dao {

namespace {
static const auto cmp_primary_key = [](datamodel::MultivalResponse_ptr const &lhs, datamodel::MultivalResponse_ptr const &rhs) {
    return reinterpret_cast<uint64_t>(lhs.get()) && reinterpret_cast<uint64_t>(rhs.get())
           && lhs->request_id == rhs->request_id
           && lhs->value_column == rhs->value_column
           && lhs->value_time == rhs->value_time;
};
static const auto cmp_whole_value = [](datamodel::MultivalResponse_ptr const &lhs, datamodel::MultivalResponse_ptr const &rhs) {
    return reinterpret_cast<uint64_t>(lhs.get()) && reinterpret_cast<uint64_t>(rhs.get())
           && cmp_primary_key(lhs, rhs)
           && lhs->value == rhs->value;
};
}

struct AsyncRequestDAO::AsyncImpl
        : AsyncImplBase<datamodel::MultivalResponse_ptr, DTYPE(cmp_primary_key), DTYPE(cmp_whole_value), PgRequestDAO> {
    AsyncImpl(common::PropertiesReader &tempus_config, dao::DataSource &data_source)
            : AsyncImplBase(tempus_config, data_source, cmp_primary_key, cmp_whole_value, 10, 10)
    {}
};

AsyncRequestDAO::AsyncRequestDAO(common::PropertiesReader &tempus_config, dao::DataSource &data_source)
        : RequestDAO(tempus_config, data_source), pImpl(*new AsyncImpl(tempus_config, data_source))
{}

AsyncRequestDAO::~AsyncRequestDAO()
{
    delete &pImpl;
}

bigint AsyncRequestDAO::get_next_id()
{
    const std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.get_next_id();
}

bigint AsyncRequestDAO::get_next_result_id()
{
    const std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.get_next_result_id();
}

int AsyncRequestDAO::force_finalize(const datamodel::MultivalRequest_ptr &p_request)
{
    const std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.force_finalize(p_request);
}

int AsyncRequestDAO::save(const datamodel::MultivalRequest_ptr &request)
{
    const std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.save(request);
}

bool AsyncRequestDAO::exists(const std::string &user, const bigint dataset_id, const boost::posix_time::ptime &start_time, const boost::posix_time::ptime &end_time,
                             const bpt::time_duration &resolution, const std::string &value_columns)
{
    const std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.exists(user, dataset_id, start_time, end_time, resolution, value_columns);
}

bigint AsyncRequestDAO::make_request(
        const std::string &user, const std::string &dataset_id_str, const std::string &value_time_start_str, const std::string &value_time_end_str,
        const std::string &resolution_str, const std::string &value_columns)
{
    const std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.make_request(user, dataset_id_str, value_time_start_str, value_time_end_str, resolution_str, value_columns);
}

datamodel::MultivalRequest_ptr
AsyncRequestDAO::get_multival_request(const std::string &user_name, const bigint dataset_id, const bpt::ptime &value_time_start, const bpt::ptime &value_time_end)
{
    const std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.get_multival_request(user_name, dataset_id, value_time_start, value_time_end);
}

datamodel::MultivalRequest_ptr AsyncRequestDAO::get_multival_request(
        const std::string &user_name, const bigint dataset_id, const bpt::ptime &value_time_start, const bpt::ptime &value_time_end, const bpt::time_duration &resolution,
        const std::string &value_columns)
{
    const std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.get_multival_request(user_name, dataset_id, value_time_start, value_time_end, resolution, value_columns);
}

datamodel::MultivalRequest_ptr AsyncRequestDAO::get_latest_multival_request(const std::string &user_name, const bigint dataset_id)
{
    const std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.get_latest_multival_request(user_name, dataset_id);
}

std::deque<datamodel::MultivalRequest_ptr> AsyncRequestDAO::get_active_multival_requests(const std::string &user_name, const bigint dataset_id)
{
    pImpl.flush();
    const std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.get_active_multival_requests(user_name, dataset_id);
}

std::deque<datamodel::MultivalResponse_ptr> AsyncRequestDAO::get_multival_results(
        const std::string &user_name, const bigint dataset_id, const bpt::ptime &value_time_start, const bpt::ptime &value_time_end, const bpt::time_duration &resolution)
{
    pImpl.flush();
    const std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.get_multival_results(user_name, dataset_id, value_time_start, value_time_end, resolution);
}

std::deque<datamodel::MultivalResponse_ptr> AsyncRequestDAO::get_multival_results_str(
        const std::string &user_name, const std::string &dataset_id, const std::string &value_time_start, const std::string &value_time_end, const std::string &resolution)
{
    pImpl.flush();
    const std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.get_multival_results_str(user_name, dataset_id, value_time_start, value_time_end, resolution);
}

std::deque<datamodel::MultivalResponse_ptr>
AsyncRequestDAO::get_multival_results_column(
        const std::string &user_name, const std::string &column_name, const bigint dataset_id, const bpt::ptime &value_time_start, const bpt::ptime &value_time_end,
        const bpt::time_duration &resolution)
{
    pImpl.flush();
    const std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.get_multival_results_column(user_name, column_name, dataset_id, value_time_start, value_time_end, resolution);
}

int AsyncRequestDAO::save(const datamodel::MultivalResponse_ptr &response)
{
    pImpl.cache(response);
    return 1;
}

void AsyncRequestDAO::prune_finalized_requests(bpt::ptime const &before)
{
    pImpl.flush();
    const std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.prune_finalized_requests(before);
}

}
}
