#pragma once

#include "TsDaoBase.hpp"
#include <DAO/DeconQueueDAO.hpp>

namespace svr {
namespace dao {

THREADSAFE_DAO_CLASS_DECLARATION_HEADER (TsDeconQueueDAO, DeconQueueDAO)

    virtual datamodel::DeconQueue_ptr get_decon_queue_by_table_name(const std::string &tableName);

    virtual std::deque<datamodel::DataRow_ptr> get_data(const std::string& deconQueueTableName, const bpt::ptime& timeFrom = bpt::min_date_time, const bpt::ptime& timeTo = bpt::max_date_time, size_t limit = 0);
    virtual std::deque<datamodel::DataRow_ptr> get_latest_data(const std::string& deconQueueTableName, const bpt::ptime& timeTo = bpt::max_date_time, const size_t limit = 0);
    virtual std::deque<datamodel::DataRow_ptr> get_data_having_update_time_greater_than(const std::string& deconQueueTableName, const bpt::ptime& updateTime, const size_t limit = 0);

    virtual bool exists(const std::string& tableName);
    virtual bool exists(const datamodel::DeconQueue_ptr& deconQueue);

    virtual void save(const datamodel::DeconQueue_ptr& deconQueue, const boost::posix_time::ptime &start_time = boost::posix_time::min_date_time);

    virtual int remove(const datamodel::DeconQueue_ptr& deconQueue);
    virtual int clear(const datamodel::DeconQueue_ptr& deconQueue);

    virtual long count(const datamodel::DeconQueue_ptr& deconQueue);
};

} }

using DeconQueueDAO_ptr = std::shared_ptr <svr::dao::DeconQueueDAO>;
