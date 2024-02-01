#pragma once

#include <DAO/AbstractDAO.hpp>

namespace svr { namespace datamodel {
    class DeconQueue;
    class DataRow;
    using DeconQueue_ptr = std::shared_ptr<DeconQueue>;
    using DataRow_ptr = std::shared_ptr<DataRow>;
} }

namespace svr {
namespace dao {

class DeconQueueDAO : public AbstractDAO {
public:
    static DeconQueueDAO * build(svr::common::PropertiesFileReader& sql_properties, svr::dao::DataSource& data_source, svr::common::ConcreteDaoType daoType, bool use_threadsafe_dao);

    explicit DeconQueueDAO(svr::common::PropertiesFileReader& sql_properties, svr::dao::DataSource& data_source);

    virtual datamodel::DeconQueue_ptr get_decon_queue_by_table_name(const std::string &tableName) = 0;

    virtual std::deque<datamodel::DataRow_ptr> get_data(const std::string& deconQueueTableName, const bpt::ptime& timeFrom = bpt::min_date_time, const bpt::ptime& timeTo = bpt::max_date_time, size_t limit = 0) = 0;
    virtual std::deque<datamodel::DataRow_ptr> get_latest_data(const std::string& deconQueueTableName, const bpt::ptime& timeTo = bpt::max_date_time, const size_t limit = 0) = 0;
    virtual std::deque<datamodel::DataRow_ptr> get_data_having_update_time_greater_than(const std::string& deconQueueTableName, const bpt::ptime& updateTime, const size_t limit = 0) = 0;

    virtual bool exists(const std::string& tableName) = 0;
    virtual bool exists(const datamodel::DeconQueue_ptr& deconQueue) = 0;

    virtual void save(const datamodel::DeconQueue_ptr& deconQueue, const boost::posix_time::ptime &start_time = boost::posix_time::min_date_time) = 0;

    virtual int remove(const datamodel::DeconQueue_ptr& deconQueue) = 0;
    virtual int clear(const datamodel::DeconQueue_ptr& deconQueue) = 0;

    virtual long count(const datamodel::DeconQueue_ptr& deconQueue) = 0;
};

} }

using DeconQueueDAO_ptr = std::shared_ptr <svr::dao::DeconQueueDAO>;
