#pragma once

#include <DAO/AbstractDAO.hpp>
#include <boost/optional.hpp>

namespace svr {
namespace datamodel {
class InputQueue;

class DataRow;
}
}
using InputQueue_ptr = std::shared_ptr<svr::datamodel::InputQueue>;
using DataRow_ptr = std::shared_ptr<svr::datamodel::DataRow>;

namespace svr {
namespace dao {

using TimeRange = std::pair<boost::posix_time::ptime, boost::posix_time::ptime>;
using OptionalTimeRange = boost::optional<TimeRange>;

class InputQueueDAO : public AbstractDAO
{
public:
    static InputQueueDAO *build(
            svr::common::PropertiesFileReader &sql_properties,
            svr::dao::DataSource &data_source,
            svr::common::ConcreteDaoType daoType,
            bool use_threadsafe_dao);

    explicit InputQueueDAO(svr::common::PropertiesFileReader &sql_properties, svr::dao::DataSource &data_source);

    virtual std::vector<InputQueue_ptr> get_all_user_queues(const std::string &user_name) = 0;

    virtual std::vector<InputQueue_ptr> get_all_queues_with_sign(bool uses_fix_connection) = 0;

    virtual InputQueue_ptr get_queue_metadata(
            const std::string &user_name,
            const std::string &logicalName,
            const bpt::time_duration &resolution) = 0;

    virtual InputQueue_ptr get_queue_metadata(const std::string &table_name) = 0;

    virtual std::deque<DataRow_ptr> get_queue_data_by_table_name(
            const std::string &table_name,
            const bpt::ptime &time_from = bpt::min_date_time,
            const bpt::ptime &time_to = bpt::max_date_time,
            size_t limit = 0) = 0;

    virtual std::deque<DataRow_ptr> get_latest_queue_data_by_table_name(
            const std::string &table_name,
            size_t limit = 0,
            const bpt::ptime &last_time = bpt::max_date_time) = 0;

    virtual DataRow_ptr get_nth_last_row(
            const std::string &table_name,
            const size_t position,
            const bpt::ptime target_time = bpt::max_date_time) = 0;

    virtual size_t get_count_from_start(
            const std::string &table_name,
            const bpt::ptime target_time = bpt::max_date_time) = 0;

    virtual size_t save(const InputQueue_ptr &input_queue, const bpt::ptime &start_time = bpt::min_date_time) = 0;

    virtual size_t save_data(const InputQueue_ptr &input_queue, const bpt::ptime &start_time = bpt::min_date_time) = 0;

    virtual size_t save_metadata(const InputQueue_ptr &input_queue) = 0;

    virtual size_t update_metadata(const InputQueue_ptr &input_queue) = 0;

    virtual size_t remove(const InputQueue_ptr &input_queue) = 0;

    virtual size_t clear(const InputQueue_ptr &input_queue) = 0;

    virtual bool exists(const std::string &table_name) = 0;

    virtual bool exists(const InputQueue_ptr &input_queue) = 0;

    virtual bool
    exists(const std::string &user_name, const std::string &logicalName, const bpt::time_duration &resolution) = 0;

    virtual DataRow_ptr find_oldest_record(const InputQueue_ptr &queue) = 0;

    virtual DataRow_ptr find_newest_record(const InputQueue_ptr &queue) = 0;

    virtual std::vector<std::shared_ptr<std::string>> get_db_table_column_names(const InputQueue_ptr &queue) = 0;

    virtual OptionalTimeRange get_missing_hours(InputQueue_ptr const &, TimeRange const &) = 0;

    virtual void purge_missing_hours(InputQueue_ptr const &queue) = 0;
};

} /* namespace dao */
} /* namespace svr */

using InputQueueDAO_ptr = std::shared_ptr<svr::dao::InputQueueDAO>;
