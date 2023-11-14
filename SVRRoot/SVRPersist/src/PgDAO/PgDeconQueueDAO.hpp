#ifndef PGDECONQUEUEDAO_H
#define PGDECONQUEUEDAO_H

#include <DAO/DeconQueueDAO.hpp>

namespace svr {
namespace dao {

class PgDeconQueueDAO : public DeconQueueDAO
{
public:
    explicit PgDeconQueueDAO(svr::common::PropertiesFileReader& sql_properties, svr::dao::DataSource& data_source);

    DeconQueue_ptr          get_decon_queue_by_table_name(const std::string &tableName);

    std::deque<DataRow_ptr> get_data(const std::string& deconQueueTableName, const bpt::ptime& timeFrom = bpt::min_date_time, const bpt::ptime& timeTo = bpt::max_date_time, const size_t limit = 0);
    std::deque<DataRow_ptr> get_latest_data(const std::string& deconQueueTableName, bpt::ptime const &timeTo = bpt::max_date_time, const size_t limit = 0);
    std::deque<DataRow_ptr> get_data_having_update_time_greater_than(const std::string& deconQueueTableName, const bpt::ptime& updateTime, const size_t limit = 0);

    bool exists(const std::string& tableName);
    bool exists(const DeconQueue_ptr& p_decon_queue);

    void save(const DeconQueue_ptr& p_decon_queue, const boost::posix_time::ptime &start_time = boost::posix_time::min_date_time);
    int save_metadata(const DeconQueue_ptr& p_decon_queue);
    long save_data(const DeconQueue_ptr& p_decon_queue, const boost::posix_time::ptime &start_time = boost::posix_time::min_date_time);

    int remove(const DeconQueue_ptr& p_decon_queue);
    int clear(const DeconQueue_ptr& p_decon_queue);

    long count(const DeconQueue_ptr& p_decon_queue);

    bool decon_table_needs_recreation(DeconQueue_ptr const &existing_queue, DeconQueue_ptr const &new_queue);

private:
    void create_decon_table_no_trx(DeconQueue_ptr const & p_decon_queue); //no_trx means caller code takes care of transactions
    void remove_decon_table_no_trx(DeconQueue_ptr const & p_decon_queue);
    int  update_metadata_no_trx(DeconQueue_ptr const & p_decon_queue);

    size_t get_level_count_db(DeconQueue_ptr const & p_decon_queue);
    size_t get_level_count(DeconQueue_ptr const & p_decon_queue);
};

} }

#endif /* PGDECONQUEUEDAO_H */

