#ifndef PGINPUTQUEUEDAO_HPP
#define PGINPUTQUEUEDAO_HPP

#include <DAO/InputQueueDAO.hpp>

namespace svr {
namespace dao {

class PgInputQueueDAO : public InputQueueDAO
{
public:
    explicit PgInputQueueDAO(svr::common::PropertiesFileReader& sql_properties, svr::dao::DataSource& data_source);

    std::deque<datamodel::InputQueue_ptr> get_all_user_queues(const std::string& user_name);
    std::deque<datamodel::InputQueue_ptr> get_all_queues_with_sign(bool uses_fix_connection);

    datamodel::InputQueue_ptr get_queue_metadata(const std::string &userName, const std::string &logicalName, const bpt::time_duration &resolution);
    datamodel::InputQueue_ptr get_queue_metadata(const std::string &tableName);

    std::deque<datamodel::DataRow_ptr> get_queue_data_by_table_name(
            const std::string &tableName, const bpt::ptime &timeFrom = bpt::min_date_time, const bpt::ptime &timeTo = bpt::max_date_time, size_t limit = 0);
    std::deque<datamodel::DataRow_ptr> get_latest_queue_data_by_table_name(
            const std::string &tableName, const size_t limit = 0, const bpt::ptime &last_time = bpt::max_date_time);
    datamodel::DataRow_ptr get_nth_last_row(const std::string &table_name, const size_t position, const bpt::ptime target_time = bpt::max_date_time);

    size_t get_count_from_start(const std::string &table_name, const bpt::ptime target_time = bpt::max_date_time);

    size_t save(const datamodel::InputQueue_ptr& p_input_queue, const bpt::ptime &start_time = bpt::min_date_time);
    size_t save_data(const datamodel::InputQueue_ptr& p_input_queue, const bpt::ptime &start_time = bpt::min_date_time);
    size_t save_metadata(const datamodel::InputQueue_ptr& inputQueue);
    size_t update_metadata(const datamodel::InputQueue_ptr& inputQueue);
    size_t remove(const datamodel::InputQueue_ptr& inputQueue);
    size_t clear(const datamodel::InputQueue_ptr& inputQueue);

    bool exists(const std::string &table_name);
    bool exists(const datamodel::InputQueue_ptr &input_queue);
    bool exists(const std::string &user_name, const std::string &logical_name, const bpt::time_duration &resolution);

    datamodel::DataRow_ptr find_oldest_record(const datamodel::InputQueue_ptr& queue);
    datamodel::DataRow_ptr find_newest_record(const datamodel::InputQueue_ptr& queue);

    std::deque<std::shared_ptr<std::string>> get_db_table_column_names(const datamodel::InputQueue_ptr& queue);

    OptionalTimeRange get_missing_hours(datamodel::InputQueue_ptr const &, TimeRange const &);
    void purge_missing_hours(datamodel::InputQueue_ptr const & queue);
private:
    bool row_exists(const std::string& tableName, const bpt::ptime& valueTime);
};

}}

#endif /* PGINPUTQUEUEDAO_HPP */

