#include "AsyncDeconQueueDAO.hpp"
#include "AsyncImplBase.hpp"
#include "model/DeconQueue.hpp"
#include "util/validation_utils.hpp"
#include "../PgDAO/PgDeconQueueDAO.hpp"


namespace svr {
namespace dao {

namespace {

bool cmp_primary_key(datamodel::DeconQueue_ptr const &lhs, datamodel::DeconQueue_ptr const &rhs)
{
    return reinterpret_cast<uint64_t>(lhs.get()) && reinterpret_cast<uint64_t>(rhs.get()) && lhs->get_table_name() == rhs->get_table_name();
}

bool cmp_whole_value(datamodel::DeconQueue_ptr const &lhs, datamodel::DeconQueue_ptr const &rhs)
{
    return reinterpret_cast<uint64_t>(lhs.get()) && reinterpret_cast<uint64_t>(rhs.get()) && lhs->get_table_name() == rhs->get_table_name();
}

}

struct AsyncDeconQueueDAO::AsyncImpl : public AsyncImplBase<datamodel::DeconQueue_ptr, DTYPE(cmp_primary_key), DTYPE(cmp_whole_value), class PgDeconQueueDAO> {

    AsyncImpl(common::PropertiesReader &tempus_config, dao::DataSource &data_source)
            : AsyncImplBase(tempus_config, data_source, cmp_primary_key, cmp_whole_value, 10, 100)
    {}
};

AsyncDeconQueueDAO::AsyncDeconQueueDAO(common::PropertiesReader &tempus_config, dao::DataSource &data_source)
        : DeconQueueDAO(tempus_config, data_source), pImpl(*new AsyncDeconQueueDAO::AsyncImpl(tempus_config, data_source))
{}

AsyncDeconQueueDAO::~AsyncDeconQueueDAO()
{
    delete &pImpl;
}

datamodel::DeconQueue_ptr AsyncDeconQueueDAO::get_decon_queue_by_table_name(const std::string &table_name)
{
    auto dq = otr<datamodel::DeconQueue>();
    dq->set_table_name(table_name);
    pImpl.seekAndCache(dq, &PgDeconQueueDAO::get_decon_queue_by_table_name, table_name);
    return dq;
}

std::deque<datamodel::DataRow_ptr> AsyncDeconQueueDAO::get_data(const std::string &deconQueueTableName, bpt::ptime const &timeFrom, bpt::ptime const &timeTo, const size_t limit)
{
    pImpl.flush();
    const std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.get_data(deconQueueTableName, timeFrom, timeTo, limit);
}

std::deque<datamodel::DataRow_ptr> AsyncDeconQueueDAO::get_latest_data(const std::string &deconQueueTableName, bpt::ptime const &timeTo, const size_t limit)
{
    pImpl.flush();
    const std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.get_latest_data(deconQueueTableName, timeTo, limit);
}

bool AsyncDeconQueueDAO::exists(const datamodel::DeconQueue_ptr &deconQueue)
{
    return exists(deconQueue->get_table_name());
}

bool AsyncDeconQueueDAO::exists(const std::string &table_name)
{
    auto dq = otr<datamodel::DeconQueue>();
    dq->set_table_name(table_name);
    if (pImpl.cached(dq)) return true;
    const std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.exists(dq);
}

int AsyncDeconQueueDAO::remove(datamodel::DeconQueue_ptr const &queue)
{
    return pImpl.remove(queue);
}

void AsyncDeconQueueDAO::save(datamodel::DeconQueue_ptr const &queue, const boost::posix_time::ptime &start_time)
{
    auto nq = otr(*queue);
    for (auto const &r: queue->get_data()) nq->get_data().emplace_back(r);
    datamodel::DeconQueue_ptr existing = get_decon_queue_by_table_name(queue->get_table_name());
    if (existing) {
        const std::scoped_lock lg(pImpl.pgMutex);
        if (pImpl.pgDao.decon_table_needs_recreation(queue, existing)) pImpl.cache_remove(existing);
    }

    pImpl.cache(nq);
    const std::scoped_lock lg(pImpl.pgMutex);
    pImpl.pgDao.save_metadata(nq); // It creates a Decon Queue table in the DB. That should be done immediately
}

std::deque<datamodel::DataRow_ptr>
AsyncDeconQueueDAO::get_data_having_update_time_greater_than(const std::string &deconQueueTableName, const bpt::ptime &updateTime, const size_t limit)
{
    const std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.get_data_having_update_time_greater_than(deconQueueTableName, updateTime, limit);
}

int AsyncDeconQueueDAO::clear(const datamodel::DeconQueue_ptr &deconQueue)
{
    pImpl.flush();
    const std::scoped_lock lg(pImpl.pgMutex);
    int res = pImpl.pgDao.clear(deconQueue);
    datamodel::DeconQueue_ptr dq = deconQueue;
    pImpl.seekAndCache(dq, &PgDeconQueueDAO::get_decon_queue_by_table_name, dq->get_table_name());
    return res;
}

long AsyncDeconQueueDAO::count(const datamodel::DeconQueue_ptr &deconQueue)
{
    const std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.count(deconQueue);
}

}
}
