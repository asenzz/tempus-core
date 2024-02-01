#include "AsyncRequestDAO.hpp"
#include <model/Request.hpp>
#include "AsyncImplBase.hpp"
#include "../PgDAO/PgRequestDAO.hpp"

namespace svr {
namespace dao {

namespace
{
    static bool cmp_primary_key(datamodel::MultivalResponse_ptr const & lhs, datamodel::MultivalResponse_ptr const & rhs)
    {
        return reinterpret_cast<unsigned long>(lhs.get()) && reinterpret_cast<unsigned long>(rhs.get())
                && lhs->request_id == rhs->request_id
                && lhs->value_column == rhs->value_column
                && lhs->value_time == rhs->value_time
                ;
    }
    static bool cmp_whole_value(datamodel::MultivalResponse_ptr const & lhs, datamodel::MultivalResponse_ptr const & rhs)
    {
        return reinterpret_cast<unsigned long>(lhs.get()) && reinterpret_cast<unsigned long>(rhs.get())
                && cmp_primary_key(lhs, rhs)
                && lhs->value == rhs->value
                ;
    }
}

struct AsyncRequestDAO::AsyncImpl
    : AsyncImplBase<datamodel::MultivalResponse_ptr, decltype(std::ptr_fun(cmp_primary_key)), decltype(std::ptr_fun(cmp_whole_value)), PgRequestDAO>
{
    AsyncImpl(svr::common::PropertiesFileReader& sqlProperties, svr::dao::DataSource& dataSource)
    :AsyncImplBase(sqlProperties, dataSource, std::ptr_fun(cmp_primary_key), std::ptr_fun(cmp_whole_value), 10, 10)
    {}
};

AsyncRequestDAO::AsyncRequestDAO(svr::common::PropertiesFileReader& sqlProperties, svr::dao::DataSource& dataSource)
: RequestDAO(sqlProperties, dataSource), pImpl(* new AsyncImpl(sqlProperties, dataSource))
{}

AsyncRequestDAO::~AsyncRequestDAO()
{
    delete &pImpl;
}

bigint AsyncRequestDAO::get_next_id()
{
    std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.get_next_id();
}

bigint AsyncRequestDAO::get_next_result_id()
{
    std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.get_next_result_id();
}

int AsyncRequestDAO::force_finalize(const datamodel::MultivalRequest_ptr &p_request)
{
    std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.force_finalize(p_request);
}

int AsyncRequestDAO::save(const datamodel::MultivalRequest_ptr &request)
{
    const std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.save(request);
}

datamodel::MultivalRequest_ptr AsyncRequestDAO::get_multival_request(const std::string &user_name, const bigint dataset_id, const bpt::ptime &value_time_start, const bpt::ptime &value_time_end)
{

    const std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.get_multival_request(user_name, dataset_id, value_time_start, value_time_end);
}

datamodel::MultivalRequest_ptr AsyncRequestDAO::get_multival_request(const std::string &user_name, const bigint dataset_id,
        const bpt::ptime &value_time_start, const bpt::ptime &value_time_end, size_t resolution, std::string const & value_columns)
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

std::deque<datamodel::MultivalResponse_ptr> AsyncRequestDAO::get_multival_results(const std::string &user_name, const bigint dataset_id, const bpt::ptime &value_time_start, const bpt::ptime &value_time_end, const size_t resolution)
{
    pImpl.flush();
    const std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.get_multival_results(user_name, dataset_id, value_time_start, value_time_end, resolution);
}

std::deque<datamodel::MultivalResponse_ptr> AsyncRequestDAO::get_multival_results_column(const std::string &user_name, const std::string &column_name, const bigint dataset_id, const bpt::ptime &value_time_start, const bpt::ptime &value_time_end, const size_t resolution)
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

void AsyncRequestDAO::prune_finalized_requests(bpt::ptime const & olderThan)
{
    pImpl.flush();
    const std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.prune_finalized_requests(olderThan);
}

} }
