#pragma once

#include <common/types.hpp>
#include <util/string_utils.hpp>
#include "DBTable.hpp"
#include "DataRow.hpp"
#include "StoreBufferPushMerge.hpp"


#define DEFAULT_INPUT_QUEUE_RESOLUTION (60) /* seconds */


namespace svr {
namespace datamodel { class InputQueue; }
namespace business { class InputQueueService; }
}


namespace svr {
namespace datamodel {

using InputQueue_ptr = std::shared_ptr<InputQueue>;

class InputQueue final : public Queue
{
    friend svr::business::InputQueueService;

    std::string logical_name_;
    std::string owner_user_name_;
    std::string description_;

    bpt::time_duration resolution_;
    bpt::time_duration legal_time_deviation_;

    std::string time_zone_;
    std::deque<std::string> value_columns_;

    void reinit_table_name();

    bool uses_fix_connection = false;

public:

    explicit InputQueue(
            const std::string &table_name = std::string(),
            const std::string &logical_name = std::string(),
            const std::string &owner_user_name = std::string(),
            const std::string &description = std::string(),
            const bpt::time_duration &resolution = bpt::seconds(DEFAULT_INPUT_QUEUE_RESOLUTION),
            const bpt::time_duration &legal_time_deviation = bpt::seconds(0),
            const std::string &time_zone = "UTC",
            const std::deque<std::string> &value_columns = std::deque<std::string>(),
            const bool uses_fix_connection = false,
            const data_row_container &rows = data_row_container());

    datamodel::InputQueue_ptr clone(const size_t start_ix, const size_t end_ix) const;

    const std::string &get_description() const;

    void set_description(const std::string &description);

    const bpt::time_duration &get_legal_time_deviation() const;

    void set_legal_time_deviation(const bpt::time_duration &legalTimeDeviation);

    const std::string &get_logical_name() const;

    void set_logical_name(const std::string &p_input_queue);

    const std::string &get_owner_user_name() const;

    void set_owner_user_name(const std::string &ownerUserName);

    const bpt::time_duration &get_resolution() const;

    void set_resolution(const bpt::time_duration &resolution);

    const std::string &get_time_zone() const;

    void set_time_zone(const std::string &time_zone);

    const std::string &get_value_column(const size_t i) const;

    const std::deque<std::string> &get_value_columns() const;

    void set_value_columns(const std::deque<std::string> &value_columns);

    bool is_tick_queue() const;

    bpt::time_duration const &get_missing_hours_retention() const;

    // end getters and setters
    InputQueue get_copy_metadata() const;

    datamodel::InputQueue_ptr clone_empty() const;

    size_t get_value_column_index(const std::string &column_name) const;

    std::deque<double> get_column_values(
            const std::string &column_name,
            const size_t start_pos = 0,
            const size_t count = std::numeric_limits<size_t>::max()) const;

    std::string metadata_to_string() const override;

    bool get_uses_fix_connection() const;

    void set_uses_fix_connection(const bool value);
};

template<typename T> std::basic_ostream<T> &
operator<<(std::basic_ostream<T> &s, const InputQueue &iq)
{
    return s << iq.to_string();
}

template<typename T> std::basic_ostream<T> &
operator<<(const InputQueue &iq, std::basic_ostream<T> &s)
{
    return s << iq.to_string();
}

} /* namespace model */

template<>
inline void store_buffer_push_merge<svr::datamodel::InputQueue_ptr>(svr::datamodel::InputQueue_ptr &dest, svr::datamodel::InputQueue_ptr const &src)
{
    dest->get_data().insert(dest->end(), src->begin(), src->end());
    dest->set_value_columns(src->get_value_columns());
    dest->set_description(src->get_description());
    dest->set_legal_time_deviation(src->get_legal_time_deviation());
    dest->set_logical_name(src->get_logical_name());
    dest->set_owner_user_name(src->get_owner_user_name());
    dest->set_resolution(src->get_resolution());
    dest->set_table_name(src->get_table_name());
    dest->set_time_zone(src->get_time_zone());
}

} /* namespace svr */