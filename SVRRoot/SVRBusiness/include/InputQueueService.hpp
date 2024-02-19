#pragma once

#include <vector>
#include <deque>
#include <memory>
#include <boost/date_time.hpp>
#include <boost/optional.hpp>
#include "model/DataRow.hpp"
#include "model/InputQueue.hpp"
#include "model/DeconQueue.hpp"
#include "model/Dataset.hpp"


namespace svr {
namespace dao {
class InputQueueDAO;
using TimeRange = std::pair<boost::posix_time::ptime, boost::posix_time::ptime>;
using OptionalTimeRange = boost::optional<TimeRange>;
}
namespace datamodel {
class Dataset;
using Dataset_ptr = std::shared_ptr<Dataset>;
}
}


namespace svr {
namespace business {

class InputQueueService
{
    svr::dao::InputQueueDAO &input_queue_dao;

public:
    /**
     * This method uses the inputQueue's resolution and legal_time_deviation values to calculate the value_time
     * which will lay on the time-grid.
     *
     * Returns the adjusted value_time or not_a_date_time if the value_time cannot be set on the time-grid)
     *
     * A time-grid is a space-in-time object represented by these values (resolution and legal_time_deviation)
     * which may only have values on certain time-slots. Each slot is away from the next/previous exactly
     * seconds specified by the resolution field, allowing deviation of legal_time_deviation seconds from the exact
     * time. All calculations are performed on the base of seconds so any fractional seconds are ignored.
     */
    static bpt::ptime adjust_time_on_grid(const datamodel::InputQueue_ptr &p_input_queue, const bpt::ptime &value_time);

    explicit InputQueueService(svr::dao::InputQueueDAO &input_queue_dao);

    virtual ~InputQueueService();

    std::deque<datamodel::InputQueue_ptr> get_all_user_queues(const std::string &user_name);

    std::deque<datamodel::InputQueue_ptr> get_all_queues_with_sign(const bool uses_fix_connector);

    datamodel::InputQueue_ptr get_queue_metadata(const std::string &user_name, const std::string &logical_name, const bpt::time_duration &resolution);

    datamodel::InputQueue_ptr get_queue_metadata(const std::string &input_queue_table_name);

    data_row_container
    load(const std::string &table_name, const bpt::ptime &time_from = bpt::min_date_time, const bpt::ptime &timeTo = bpt::max_date_time, size_t limit = 0);

    void load(const datamodel::InputQueue_ptr &p_input_queue);

    void
    load(const datamodel::InputQueue_ptr &p_input_queue, const bpt::time_period &range, const size_t limit = 0);

    void
    load_latest(datamodel::InputQueue_ptr p_input_queue, const size_t limit = 0, const bpt::ptime &last_time = bpt::max_date_time);

    datamodel::DataRow_ptr
    load_nth_last_row(const datamodel::InputQueue_ptr &input_queue, const size_t position, const bpt::ptime target_time = bpt::max_date_time);

    size_t save(const datamodel::InputQueue_ptr &p_input_queue);

    size_t save(const datamodel::InputQueue_ptr &p_input_queue, const bpt::ptime &start_time);

    bool exists(const std::string &user_name, const std::string &logical_name, const bpt::time_duration &resolution);

    int remove(const datamodel::InputQueue_ptr &p_input_queue);

    int clear(const datamodel::InputQueue_ptr &p_input_queue);

    bool add_row(datamodel::InputQueue_ptr &p_input_queue, datamodel::DataRow_ptr p_row, bool concatenate = false);

    datamodel::DataRow_ptr find_oldest_record(const datamodel::InputQueue_ptr &queue);

    datamodel::DataRow_ptr find_newest_record(const datamodel::InputQueue_ptr &queue);

    static size_t get_value_column_index(const datamodel::InputQueue_ptr &p_input_queue, const std::string &column_name);

    datamodel::InputQueue_ptr clone_with_data(const datamodel::InputQueue_ptr &p_input_queue, const bpt::time_period &time_range,
                                   const size_t minimum_rows_count = std::numeric_limits<size_t>::min());

    static svr::datamodel::DataRow::container
    get_column_data(const datamodel::InputQueue_ptr &p_input_queue, const std::string &column_name);

    static data_row_container
    get_column_data(const datamodel::InputQueue_ptr &queue, const size_t column_index);

    std::deque<std::string> get_db_table_column_names(const datamodel::InputQueue_ptr &queue);

    svr::dao::OptionalTimeRange get_missing_hours(datamodel::InputQueue_ptr const &, svr::dao::TimeRange const &) const;

    void purge_missing_hours(datamodel::InputQueue_ptr const &);

    static void prepare_queues(datamodel::Dataset_ptr &p_dataset, const bool trim = false);

    static boost::posix_time::ptime
    compare_to_decon_queue(
            const datamodel::InputQueue_ptr &p_input_queue,
            const datamodel::DeconQueue_ptr &p_decon_queue);

    size_t get_count_from_start(const datamodel::InputQueue_ptr &p_input_queue, const boost::posix_time::ptime &time);

    //data_row_container shift_times_forward(const data_row_container &data, const bpt::time_duration &resolution);

    static std::string make_queue_table_name(const std::string &user_name, const std::string &logical_name, const bpt::time_duration &resolution);

private:
    data_row_container
    load_latest_from_mmf(datamodel::InputQueue_ptr const &input_queue, const bpt::ptime &last_time);
    static void prepare_input_data(datamodel::Dataset_ptr &p_dataset);
};

} /* namespace business */
} /* namespace svr */

using InputQueueService_ptr = std::shared_ptr<svr::business::InputQueueService>;
