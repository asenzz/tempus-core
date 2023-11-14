#include "AsyncDeconQueueDAO.hpp"
#include "AsyncImplBase.hpp"

#include <model/DeconQueue.hpp>
#include "util/ValidationUtils.hpp"

#include "../PgDAO/PgDeconQueueDAO.hpp"

using svr::datamodel::DeconQueue;
using svr::common::reject_empty;

namespace svr {    
namespace dao {

namespace {
bool cmp_primary_key(DeconQueue_ptr const & lhs, DeconQueue_ptr const & rhs)
{
    return reinterpret_cast<unsigned long>(lhs.get()) && reinterpret_cast<unsigned long>(rhs.get())
            && lhs->get_table_name() == rhs->get_table_name();
}
bool cmp_whole_value(DeconQueue_ptr const & lhs, DeconQueue_ptr const & rhs)
{
    return reinterpret_cast<unsigned long>(lhs.get()) && reinterpret_cast<unsigned long>(rhs.get())
            && lhs->get_table_name() == rhs->get_table_name();
}
}

struct AsyncDeconQueueDAO::AsyncImpl
    : public AsyncImplBase<DeconQueue_ptr, decltype(std::ptr_fun(cmp_primary_key)), decltype(std::ptr_fun(cmp_whole_value)), class PgDeconQueueDAO>
{

    AsyncImpl(svr::common::PropertiesFileReader& sqlProperties, svr::dao::DataSource& dataSource)
    :AsyncImplBase(sqlProperties, dataSource, std::ptr_fun(cmp_primary_key), std::ptr_fun(cmp_whole_value), 10, 100)
    {}

};

AsyncDeconQueueDAO::AsyncDeconQueueDAO(svr::common::PropertiesFileReader& sqlProperties, svr::dao::DataSource& dataSource)
: DeconQueueDAO(sqlProperties, dataSource), pImpl(* new AsyncDeconQueueDAO::AsyncImpl(sqlProperties, dataSource))
{}

AsyncDeconQueueDAO::~AsyncDeconQueueDAO()
{
    delete & pImpl;
}

DeconQueue_ptr AsyncDeconQueueDAO::get_decon_queue_by_table_name(const std::string &tableName)
{
    DeconQueue_ptr dq(new DeconQueue());
    dq->set_table_name(tableName);

    pImpl.seekAndCache(dq, &PgDeconQueueDAO::get_decon_queue_by_table_name, tableName);
    return dq;
}

std::deque<DataRow_ptr> AsyncDeconQueueDAO::get_data(const std::string& deconQueueTableName, bpt::ptime const &timeFrom, bpt::ptime const &timeTo, const size_t limit)
{
    pImpl.flush();
    std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.get_data(deconQueueTableName, timeFrom, timeTo, limit);
}

std::deque<DataRow_ptr> AsyncDeconQueueDAO::get_latest_data(const std::string& deconQueueTableName, bpt::ptime const &timeTo, const size_t limit)
{
    pImpl.flush();
    std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.get_latest_data(deconQueueTableName, timeTo, limit);
}

bool AsyncDeconQueueDAO::exists(const DeconQueue_ptr& deconQueue)
{
    return exists(deconQueue->get_table_name());
}

bool AsyncDeconQueueDAO::exists(const std::string &tableName)
{
    DeconQueue_ptr dq(new DeconQueue());
    dq->set_table_name(tableName);

    if(pImpl.cached(dq))
        return true;

    std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.exists(dq);
}

int AsyncDeconQueueDAO::remove(DeconQueue_ptr const &queue)
{
    return pImpl.remove(queue);
}

void AsyncDeconQueueDAO::save(DeconQueue_ptr const &queue, const boost::posix_time::ptime &start_time)
{
    DeconQueue_ptr nq = DeconQueue_ptr( new DeconQueue( *queue )  );
    for(auto const & r : queue->get_data())
        nq->get_data().push_back(r);

    DeconQueue_ptr existing = get_decon_queue_by_table_name(queue->get_table_name());

    if (existing)
    {
        std::scoped_lock lg(pImpl.pgMutex);
        if(pImpl.pgDao.decon_table_needs_recreation(queue, existing))
            pImpl.cache_remove(existing);
    }

    pImpl.cache(nq);
    std::scoped_lock lg(pImpl.pgMutex);
    pImpl.pgDao.save_metadata(nq); // It creates a Decon Queue table in the DB. That should be done immediately
}

std::deque<DataRow_ptr> AsyncDeconQueueDAO::get_data_having_update_time_greater_than(const std::string &deconQueueTableName, const bpt::ptime &updateTime, const size_t limit) {
    std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.get_data_having_update_time_greater_than(deconQueueTableName, updateTime, limit);
}

int AsyncDeconQueueDAO::clear(const DeconQueue_ptr &deconQueue)
{
    pImpl.flush();

    std::scoped_lock lg(pImpl.pgMutex);
    int res = pImpl.pgDao.clear(deconQueue);

    DeconQueue_ptr dq = deconQueue;
    pImpl.seekAndCache(dq, &PgDeconQueueDAO::get_decon_queue_by_table_name, dq->get_table_name());

    return res;
}

long AsyncDeconQueueDAO::count(const DeconQueue_ptr &deconQueue) {
    std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.count(deconQueue);
}

}
}
