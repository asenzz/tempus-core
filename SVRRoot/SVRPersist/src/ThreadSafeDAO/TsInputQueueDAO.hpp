#pragma once

#include "TsDaoBase.hpp"
#include "DAO/InputQueueDAO.hpp"

namespace svr {
namespace dao {

THREADSAFE_DAO_CLASS_DECLARATION_HEADER (TsInputQueueDAO, InputQueueDAO)
    virtual std::deque<datamodel::InputQueue_ptr> get_all_user_queues(const std::string& user_name) override;
    virtual std::deque<datamodel::InputQueue_ptr> get_all_queues_with_sign(bool uses_fix_connection) override;
    virtual datamodel::InputQueue_ptr get_queue_metadata(const std::string &user_name, const std::string &logical_name,
                                              const bpt::time_duration &resolution) override;
    virtual datamodel::InputQueue_ptr get_queue_metadata(const std::string &table_name) override;

    virtual std::deque<datamodel::DataRow_ptr> get_queue_data_by_table_name(
            const std::string &table_name, const bpt::ptime &time_from = bpt::min_date_time, const bpt::ptime &time_to = bpt::max_date_time, const size_t limit = 0) override;
    virtual std::deque<datamodel::DataRow_ptr> get_latest_queue_data_by_table_name(const std::string &table_name,
            const size_t limit, const bpt::ptime &last_time = bpt::max_date_time) override;
    virtual datamodel::DataRow_ptr get_nth_last_row(
            const std::string &table_name, const size_t position, const bpt::ptime target_time = bpt::max_date_time) override;
    virtual size_t get_count_from_start(const std::string &table_name, const bpt::ptime target_time = bpt::max_date_time) override;

    virtual size_t save(const datamodel::InputQueue_ptr& input_queue, const bpt::ptime &start_time = bpt::min_date_time) override;
    virtual size_t save_data(const datamodel::InputQueue_ptr& input_queue, const bpt::ptime &start_time = bpt::min_date_time) override;
    virtual size_t save_metadata(const datamodel::InputQueue_ptr& input_queue) override;
    virtual size_t update_metadata(const datamodel::InputQueue_ptr& input_queue) override;
    virtual size_t remove(const datamodel::InputQueue_ptr& input_queue) override;
    virtual size_t clear(const datamodel::InputQueue_ptr& input_queue) override;

    virtual void upsert_row_str(CRPTR(char) table_name, CRPTR(char) value_time, CRPTR(char) update_time, CRPTR(char) volume, CRPTR(char *) values, const uint16_t n_values) override;

    virtual bool exists(const std::string &table_name) override;
    virtual bool exists(const datamodel::InputQueue_ptr& input_queue) override;
    virtual bool exists(const std::string &user_name, const std::string &logical_name, const bpt::time_duration &resolution) override;

    virtual datamodel::DataRow_ptr find_oldest_record(const datamodel::InputQueue_ptr& queue) override;
    virtual datamodel::DataRow_ptr find_newest_record(const datamodel::InputQueue_ptr& queue) override;

    virtual std::deque<std::shared_ptr<std::string>> get_db_table_column_names(const datamodel::InputQueue_ptr& queue) override;

    virtual OptionalTimeRange get_missing_hours(datamodel::InputQueue_ptr const &, TimeRange const &) override;
    virtual void purge_missing_hours(datamodel::InputQueue_ptr const & queue) override;
};

} /* namespace dao */
} /* namespace svr */

using InputQueueDAO_ptr = std::shared_ptr<svr::dao::InputQueueDAO>;
