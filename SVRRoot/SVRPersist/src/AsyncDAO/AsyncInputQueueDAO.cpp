
#include "AsyncInputQueueDAO.hpp"

#include "../PgDAO/PgInputQueueDAO.hpp"
#include <model/DBTable.hpp>
#include <model/InputQueue.hpp>
#include "AsyncImplBase.hpp"
#include "InputQueueService.hpp"

using svr::datamodel::InputQueue;

namespace svr {
namespace dao {

namespace {
static const auto cmp_primary_key = [](datamodel::InputQueue_ptr const &lhs, datamodel::InputQueue_ptr const &rhs) {
    return reinterpret_cast<uint64_t>(lhs.get()) && reinterpret_cast<uint64_t>(rhs.get())
           && lhs->get_table_name() == rhs->get_table_name();
};

static const auto cmp_whole_value = [](datamodel::InputQueue_ptr const &lhs, datamodel::InputQueue_ptr const &rhs) {
    return reinterpret_cast<uint64_t>(lhs.get()) && reinterpret_cast<uint64_t>(rhs.get())
           && lhs->get_table_name() == rhs->get_table_name();
};
}

struct AsyncInputQueueDAO::AsyncImpl
        : AsyncImplBase<datamodel::InputQueue_ptr, DTYPE(cmp_primary_key), DTYPE(cmp_whole_value), PgInputQueueDAO>
{
    AsyncImpl(svr::common::PropertiesFileReader &tempus_config, svr::dao::DataSource &data_source)
            : AsyncImplBase(tempus_config, data_source, cmp_primary_key, cmp_whole_value, 10, 100)
    {}

};

AsyncInputQueueDAO::AsyncInputQueueDAO(svr::common::PropertiesFileReader &properties, svr::dao::DataSource &source)
        : InputQueueDAO(properties, source), pImpl(*new AsyncImpl(properties, source))
{}

AsyncInputQueueDAO::~AsyncInputQueueDAO()
{
    delete &pImpl;
}

datamodel::InputQueue_ptr AsyncInputQueueDAO::get_queue_metadata(const std::string &user_name, const std::string &logical_name,
                                                                 const bpt::time_duration &resolution)
{
    return get_queue_metadata(business::InputQueueService::make_queue_table_name(user_name, logical_name, resolution));
}

datamodel::InputQueue_ptr AsyncInputQueueDAO::get_queue_metadata(const std::string &table_name)
{
    datamodel::InputQueue_ptr queue(new InputQueue());
    queue->set_table_name(table_name);

    if (pImpl.cached(queue))
        return datamodel::InputQueue_ptr(new InputQueue(queue->get_copy_metadata()));

    const std::scoped_lock lg(pImpl.pgMutex);
    queue = pImpl.pgDao.get_queue_metadata(table_name);
    pImpl.cache_no_store(queue);
    return queue;
}

std::deque<datamodel::DataRow_ptr> AsyncInputQueueDAO::get_queue_data_by_table_name(
        const std::string &table_name,
        const bpt::ptime &time_from,
        const bpt::ptime &time_to,
        const size_t limit)
{
    pImpl.flush();

    const std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.get_queue_data_by_table_name(table_name, time_from, time_to);
}

std::deque<datamodel::DataRow_ptr> AsyncInputQueueDAO::get_latest_queue_data_by_table_name(
        const std::string &table_name,
        const size_t limit,
        const bpt::ptime &last_time)
{
    pImpl.flush();

    const std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.get_latest_queue_data_by_table_name(table_name, limit, last_time);
}

datamodel::DataRow_ptr AsyncInputQueueDAO::get_nth_last_row(const std::string &table_name, const size_t position, const bpt::ptime target_time)
{
    pImpl.flush();

    const std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.get_nth_last_row(table_name, position, target_time);
}


size_t AsyncInputQueueDAO::get_count_from_start(
        const std::string &table_name,
        const bpt::ptime target_time)
{
    pImpl.flush();

    const std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.get_count_from_start(table_name, target_time);
}

size_t AsyncInputQueueDAO::save(const datamodel::InputQueue_ptr &p_input_queue, const boost::posix_time::ptime &start_time)
{
    long ret = 0;

    bool exist = exists(p_input_queue);

    if (p_input_queue->size() > 0) {
        ret = save_data(p_input_queue, start_time);
    } else {
        if (exist)
            ret = update_metadata(p_input_queue);
        else
            ret = save_metadata(p_input_queue);
    }

    return ret;
}

size_t AsyncInputQueueDAO::save_metadata(const datamodel::InputQueue_ptr &queue)
{
    datamodel::InputQueue_ptr md(new InputQueue(queue->get_copy_metadata()));
    pImpl.cache(md);
    return 1;
}

size_t AsyncInputQueueDAO::save_data(const datamodel::InputQueue_ptr &queue, const boost::posix_time::ptime &last_time)
{
    pImpl.cache(queue);
    return queue->size();
}

size_t AsyncInputQueueDAO::update_metadata(const datamodel::InputQueue_ptr &queue)
{
    return save_metadata(queue);
}

bool AsyncInputQueueDAO::exists(const std::string &table_name)
{
    datamodel::InputQueue_ptr queue(new InputQueue());
    queue->set_table_name(table_name);

    if (pImpl.cached(queue))
        return true;
    const std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.exists(table_name);
}

bool AsyncInputQueueDAO::exists(const std::string &user_name, const std::string &logical_name,
                                const bpt::time_duration &resolution)
{
    return exists(business::InputQueueService::make_queue_table_name(user_name, logical_name, resolution));
}

bool AsyncInputQueueDAO::exists(const datamodel::InputQueue_ptr &p_input_queue)
{
    p_input_queue->set_table_name(p_input_queue->get_table_name());
    return exists(p_input_queue->get_table_name());
}

void AsyncInputQueueDAO::upsert_row_str(CRPTR(char) table_name, CRPTR(char) value_time, CRPTR(char) update_time, CRPTR(char) volume, CRPTR(char *) values, const uint16_t n_values)
{
    pImpl.flush();
    const std::scoped_lock lg(pImpl.pgMutex);
    pImpl.pgDao.upsert_row_str(table_name, value_time, update_time, volume, values, n_values);
}

size_t AsyncInputQueueDAO::remove(const datamodel::InputQueue_ptr &queue)
{
    return pImpl.remove(queue);
}

size_t AsyncInputQueueDAO::clear(const datamodel::InputQueue_ptr &queue)
{
    return pImpl.pgDao.clear(queue);
}

datamodel::DataRow_ptr AsyncInputQueueDAO::find_oldest_record(const datamodel::InputQueue_ptr &queue)
{
    pImpl.flush();
    const std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.find_oldest_record(queue);
}

datamodel::DataRow_ptr AsyncInputQueueDAO::find_newest_record(const datamodel::InputQueue_ptr &queue)
{
    pImpl.flush();
    const std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.find_newest_record(queue);
}

std::deque<std::shared_ptr<std::string>> AsyncInputQueueDAO::get_db_table_column_names(const datamodel::InputQueue_ptr &queue)
{
    pImpl.flush();
    const std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.get_db_table_column_names(queue);
}

std::deque<datamodel::InputQueue_ptr> AsyncInputQueueDAO::get_all_user_queues(const std::string &user_name)
{
    const std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.get_all_user_queues(user_name);
}


std::deque<datamodel::InputQueue_ptr> AsyncInputQueueDAO::get_all_queues_with_sign(bool uses_fix_connection)
{
    const std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.get_all_queues_with_sign(uses_fix_connection);
}


/******************************************************************************/
/******************************************************************************/
/******************************************************************************/

OptionalTimeRange AsyncInputQueueDAO::get_missing_hours(datamodel::InputQueue_ptr const &queue, TimeRange const &from_range)
{
    const std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.get_missing_hours(queue, from_range);
}

void AsyncInputQueueDAO::purge_missing_hours(datamodel::InputQueue_ptr const &queue)
{
    const std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.purge_missing_hours(queue);
}

}
}
