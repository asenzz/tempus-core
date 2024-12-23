#pragma once

#include <boost/optional.hpp>
#include "DAO/AbstractDAO.hpp"
#include "common/compatibility.hpp"

namespace svr {
namespace datamodel {
class InputQueue;

class DataRow;
using InputQueue_ptr = std::shared_ptr<InputQueue>;
using DataRow_ptr = std::shared_ptr<DataRow>;
}
}

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
            svr::common::ConcreteDaoType dao_type,
            bool use_threadsafe_dao);

    explicit InputQueueDAO(svr::common::PropertiesFileReader &sql_properties, svr::dao::DataSource &data_source);

    virtual std::deque<datamodel::InputQueue_ptr> get_all_user_queues(const std::string &user_name) = 0;

    virtual std::deque<datamodel::InputQueue_ptr> get_all_queues_with_sign(bool uses_fix_connection) = 0;

    virtual datamodel::InputQueue_ptr get_queue_metadata(
            const std::string &user_name,
            const std::string &p_input_queue,
            const bpt::time_duration &resolution) = 0;

    virtual datamodel::InputQueue_ptr get_queue_metadata(const std::string &table_name) = 0;

    virtual std::deque<datamodel::DataRow_ptr> get_queue_data_by_table_name(
            const std::string &table_name,
            const bpt::ptime &time_from = bpt::min_date_time,
            const bpt::ptime &time_to = bpt::max_date_time,
            size_t limit = 0) = 0;

    virtual std::deque<datamodel::DataRow_ptr> get_latest_queue_data_by_table_name(
            const std::string &table_name,
            size_t limit = 0,
            const bpt::ptime &last_time = bpt::max_date_time) = 0;

    virtual datamodel::DataRow_ptr get_nth_last_row(
            const std::string &table_name,
            const size_t position,
            const bpt::ptime target_time = bpt::max_date_time) = 0;

    virtual size_t get_count_from_start(
            const std::string &table_name,
            const bpt::ptime target_time = bpt::max_date_time) = 0;

    virtual size_t save(const datamodel::InputQueue_ptr &input_queue, const bpt::ptime &start_time = bpt::min_date_time) = 0;

    virtual size_t save_data(const datamodel::InputQueue_ptr &input_queue, const bpt::ptime &start_time = bpt::min_date_time) = 0;

    virtual size_t save_metadata(const datamodel::InputQueue_ptr &input_queue) = 0;

    virtual size_t update_metadata(const datamodel::InputQueue_ptr &input_queue) = 0;

    virtual size_t remove(const datamodel::InputQueue_ptr &input_queue) = 0;

    virtual size_t clear(const datamodel::InputQueue_ptr &input_queue) = 0;

    virtual bool exists(const std::string &table_name) = 0;

    virtual bool exists(const datamodel::InputQueue_ptr &input_queue) = 0;

    virtual bool exists(const std::string &user_name, const std::string &p_input_queue, const bpt::time_duration &resolution) = 0;

    virtual void upsert_row_str(CRPTR(char) table_name, CRPTR(char) value_time, CRPTR(char) update_time, CRPTR(char) volume, CRPTR(char *) values, const uint16_t n_values) = 0;

    virtual datamodel::DataRow_ptr find_oldest_record(const datamodel::InputQueue_ptr &queue) = 0;

    virtual datamodel::DataRow_ptr find_newest_record(const datamodel::InputQueue_ptr &queue) = 0;

    virtual std::deque<std::shared_ptr<std::string>> get_db_table_column_names(const datamodel::InputQueue_ptr &queue) = 0;

    virtual OptionalTimeRange get_missing_hours(datamodel::InputQueue_ptr const &, TimeRange const &) = 0;

    virtual void purge_missing_hours(datamodel::InputQueue_ptr const &queue) = 0;
};

} /* namespace dao */
} /* namespace svr */

using InputQueueDAO_ptr = std::shared_ptr<svr::dao::InputQueueDAO>;
