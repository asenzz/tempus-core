#include "TsDeconQueueDAO.hpp"

namespace svr {
namespace dao {

DEFINE_THREADSAFE_DAO_CONSTRUCTOR (TsDeconQueueDAO, DeconQueueDAO)
{}

datamodel::DeconQueue_ptr TsDeconQueueDAO::get_decon_queue_by_table_name(const std::string &table_name)
{
    return ts_call<datamodel::DeconQueue_ptr>(&DeconQueueDAO::get_decon_queue_by_table_name, table_name);
}


std::deque<datamodel::DataRow_ptr> TsDeconQueueDAO::get_data(const std::string& deconQueueTableName, const bpt::ptime& timeFrom, const bpt::ptime& timeTo, const size_t limit)
{
    return ts_call<std::deque<datamodel::DataRow_ptr>>(&DeconQueueDAO::get_data, deconQueueTableName, timeFrom, timeTo, limit);
}


std::deque<datamodel::DataRow_ptr> TsDeconQueueDAO::get_latest_data(const std::string& deconQueueTableName, const bpt::ptime& timeTo, const size_t limit)
{
    return ts_call<std::deque<datamodel::DataRow_ptr>>(&DeconQueueDAO::get_latest_data, deconQueueTableName, timeTo, limit);
}


std::deque<datamodel::DataRow_ptr> TsDeconQueueDAO::get_data_having_update_time_greater_than(const std::string& deconQueueTableName, const bpt::ptime& updateTime, const size_t limit)
{
    return ts_call<std::deque<datamodel::DataRow_ptr>>(&DeconQueueDAO::get_data_having_update_time_greater_than, deconQueueTableName, updateTime, limit);
}


bool TsDeconQueueDAO::exists(const std::string& table_name)
{
    std::scoped_lock<std::recursive_mutex> scope_guard(mutex);
    return dao->exists(table_name);
}


bool TsDeconQueueDAO::exists(const datamodel::DeconQueue_ptr& deconQueue)
{
    std::scoped_lock<std::recursive_mutex> scope_guard(mutex);
    return dao->exists(deconQueue);
}


void TsDeconQueueDAO::save(const datamodel::DeconQueue_ptr& deconQueue, const boost::posix_time::ptime &start_time)
{
    return ts_call<void>(&DeconQueueDAO::save, deconQueue, start_time);
}


int TsDeconQueueDAO::remove(const datamodel::DeconQueue_ptr& deconQueue)
{
    return ts_call<int>(&DeconQueueDAO::remove, deconQueue);
}


int TsDeconQueueDAO::clear(const datamodel::DeconQueue_ptr& deconQueue)
{
    return ts_call<int>(&DeconQueueDAO::clear, deconQueue);
}


long TsDeconQueueDAO::count(const datamodel::DeconQueue_ptr& deconQueue)
{
    return ts_call<long>(&DeconQueueDAO::count, deconQueue);
}


} }
