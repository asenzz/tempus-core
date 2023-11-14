#ifndef ASYNCINPUTQUEUEDAO_HPP
#define ASYNCINPUTQUEUEDAO_HPP

#include <DAO/InputQueueDAO.hpp>

namespace svr {
namespace dao {

class AsyncInputQueueDAO : public InputQueueDAO
{
public:
    explicit AsyncInputQueueDAO(svr::common::PropertiesFileReader &sql_properties, svr::dao::DataSource &data_source);

    ~AsyncInputQueueDAO();

    std::vector<InputQueue_ptr> get_all_user_queues(const std::string &user_name);

    std::vector<InputQueue_ptr> get_all_queues_with_sign(bool uses_fix_connection);

    InputQueue_ptr get_queue_metadata(const std::string &userName, const std::string &logicalName, const bpt::time_duration &resolution);
    InputQueue_ptr get_queue_metadata(const std::string &tableName);

    std::deque<DataRow_ptr> get_queue_data_by_table_name(
            const std::string &table_name,
            const bpt::ptime &time_from = bpt::min_date_time,
            const bpt::ptime &time_to = bpt::max_date_time,
            size_t limit = 0);

    std::deque<DataRow_ptr> get_latest_queue_data_by_table_name(
            const std::string &table_name,
            const size_t limit = 0,
            const bpt::ptime &last_time = bpt::max_date_time);

    DataRow_ptr get_nth_last_row(const std::string &table_name, const size_t position, const bpt::ptime target_time = bpt::max_date_time);

    size_t get_count_from_start(
            const std::string &table_name,
            const bpt::ptime target_time = bpt::max_date_time);

    size_t save(const InputQueue_ptr &queue, const boost::posix_time::ptime &start_time = boost::posix_time::min_date_time);

    size_t save_data(const InputQueue_ptr &queue, const boost::posix_time::ptime &start_time = boost::posix_time::min_date_time);

    size_t save_metadata(const InputQueue_ptr &queue);

    size_t update_metadata(const InputQueue_ptr &queue);

    size_t remove(const InputQueue_ptr &queue);

    size_t clear(const InputQueue_ptr &queue);

    bool exists(const std::string &table_name);

    bool exists(const InputQueue_ptr &input_queue);

    bool exists(const std::string &user_name, const std::string &logical_name, const bpt::time_duration &resolution);

    DataRow_ptr find_oldest_record(const InputQueue_ptr &queue);

    DataRow_ptr find_newest_record(const InputQueue_ptr &queue);

    std::vector<std::shared_ptr<std::string>> get_db_table_column_names(const InputQueue_ptr &queue);

    OptionalTimeRange get_missing_hours(InputQueue_ptr const &, TimeRange const &);
    void purge_missing_hours(InputQueue_ptr const & queue);
private:
    struct AsyncImpl;
    AsyncImpl &pImpl;

};

}
}
#endif /* ASYNCINPUTQUEUEDAO_HPP */

