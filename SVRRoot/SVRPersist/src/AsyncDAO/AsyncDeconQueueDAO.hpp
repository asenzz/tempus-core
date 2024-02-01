#ifndef ASYNCDECONQUEUEDAO_HPP
#define ASYNCDECONQUEUEDAO_HPP

#include <DAO/DeconQueueDAO.hpp>

namespace svr {
namespace dao {

class AsyncDeconQueueDAO : public DeconQueueDAO
{
public:
    explicit AsyncDeconQueueDAO(svr::common::PropertiesFileReader& sql_properties, svr::dao::DataSource& data_source);
    ~AsyncDeconQueueDAO();

    datamodel::DeconQueue_ptr          get_decon_queue_by_table_name(const std::string &tableName);

    std::deque<datamodel::DataRow_ptr> get_data(const std::string& deconQueueTableName, const bpt::ptime& timeFrom = bpt::min_date_time, const bpt::ptime& timeTo = bpt::max_date_time, const size_t limit = 0);
    std::deque<datamodel::DataRow_ptr> get_latest_data(const std::string& deconQueueTableName, const bpt::ptime& timeTo = bpt::max_date_time, const size_t limit = 0); /* TODO Andrey test please :) */
    std::deque<datamodel::DataRow_ptr> get_data_having_update_time_greater_than(const std::string& deconQueueTableName, const bpt::ptime& updateTime, const size_t limit = 0);

    bool exists(const std::string& tableName);
    bool exists(const datamodel::DeconQueue_ptr& p_decon_queue);

    void save(const datamodel::DeconQueue_ptr& p_decon_queue, const boost::posix_time::ptime &start_time = boost::posix_time::min_date_time);

    int remove(const datamodel::DeconQueue_ptr& p_decon_queue);
    int clear(const datamodel::DeconQueue_ptr& p_decon_queue);

    long count(const datamodel::DeconQueue_ptr& p_decon_queue);
private:
    struct AsyncImpl;
    AsyncImpl & pImpl;
};

} }


#endif /* ASYNCDECONQUEUEDAO_HPP */

