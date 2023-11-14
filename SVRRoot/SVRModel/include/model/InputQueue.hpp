#pragma once

#include <common/types.hpp>
#include <util/string_utils.hpp>
#include "DBTable.hpp"
#include "DataRow.hpp"
#include "StoreBufferPushMerge.hpp"


#define DEFAULT_INPUT_QUEUE_RESOLUTION (60) /* seconds */


namespace svr { namespace datamodel { class InputQueue; }}
namespace svr { namespace business { class InputQueueService; }}

using InputQueue_ptr = std::shared_ptr<svr::datamodel::InputQueue>;


namespace svr {
namespace datamodel {


class InputQueue : public Queue
{
    friend svr::business::InputQueueService;
private:
    std::string logical_name_;
    std::string owner_user_name_;
    std::string description_;

    bpt::time_duration resolution_;
    bpt::time_duration legal_time_deviation_;

    std::string time_zone_;
    std::vector<std::string> value_columns_;

    void reinit_table_name();

    bool uses_fix_connection;

public:

    InputQueue(
            const std::string& table_name = std::string(),
            const std::string& logical_name = std::string(),
            const std::string& owner_user_name = std::string(),
            const std::string& description = std::string(),
            const bpt::time_duration& resolution = bpt::seconds(DEFAULT_INPUT_QUEUE_RESOLUTION),
            const bpt::time_duration& legal_time_deviation = bpt::seconds(0),
            const std::string &time_zone = "UTC",
            const std::vector<std::string> value_columns = std::vector<std::string>(),
            bool uses_fix_connection = false,
            const data_row_container &rows = data_row_container());

    inline const std::string& get_description() const { return description_; }
    inline void set_description(const std::string& description) { this->description_ = description; }

    inline const bpt::time_duration& get_legal_time_deviation() const {	return legal_time_deviation_;}
    inline void set_legal_time_deviation(const bpt::time_duration& legalTimeDeviation) {legal_time_deviation_ = legalTimeDeviation;}

    inline const std::string& get_logical_name() const { return logical_name_; }
    void set_logical_name(const std::string& logicalName);

    inline const std::string& get_owner_user_name() const { return owner_user_name_; }
    void set_owner_user_name(const std::string& ownerUserName);

    inline const bpt::time_duration& get_resolution() const { return resolution_; }
    void set_resolution(const bpt::time_duration& resolution);

    inline const std::string& get_time_zone() const {return time_zone_;}
    inline void set_time_zone(const std::string &time_zone) {time_zone_ = time_zone;}

    inline const std::vector<std::string>& get_value_columns() const {return value_columns_;}
    inline void set_value_columns(const std::vector<std::string>& value_columns){ value_columns_ = value_columns; }

    inline const bool is_tick_queue() const { return resolution_ < bpt::seconds(1); }

    /* TODO Implement
    inline const std::vector<std::string> &get_feature_columns() const;
    inline void set_feature_columns(const std::vector<std::string> &feature_columns) { feature_columns_ = feature_columns; }
    */

    inline bpt::time_duration const & get_missing_hours_retention() const { static auto const weeks2 = bpt::hours(24 * 14); return weeks2; }

	// end getters and setters
    InputQueue get_copy_metadata() const;
    InputQueue_ptr clone_empty() const;

    size_t get_value_column_index(const std::string &column_name) const
    {
        const auto pos = find(value_columns_.begin(), value_columns_.end(), column_name);
        if (pos == value_columns_.end()) THROW_EX_FS(
                std::invalid_argument, "Column " << column_name << " is not part of input queue " << table_name_);
        return std::distance(value_columns_.begin(), pos);
    }

    std::vector<double> get_column_values(
            const std::string &column_name,
            const size_t start_pos = 0,
            const size_t count = std::numeric_limits<size_t>::max()) const
    {
        return Queue::get_column_values(get_value_column_index(column_name), start_pos, count);
    }

    std::string metadata_to_string() const override;

    static std::string make_queue_table_name(const std::string &user_name, const std::string &logical_name, const bpt::time_duration &resolution);

    bool get_uses_fix_connection() const ;
    void set_uses_fix_connection(bool value);
};

} /* namespace model */
} /* namespace svr */


template<>
inline void store_buffer_push_merge<InputQueue_ptr>(InputQueue_ptr & dest, InputQueue_ptr const & src)
{
    dest->get_data().insert(dest->get_data().end(), src->get_data().begin(), src->get_data().end());
    dest->set_value_columns(src->get_value_columns());
    dest->set_description(src->get_description());
    dest->set_legal_time_deviation(src->get_legal_time_deviation());
    dest->set_logical_name(src->get_logical_name());
    dest->set_owner_user_name(src->get_owner_user_name());
    dest->set_resolution(src->get_resolution());
    dest->set_table_name(src->get_table_name());
    dest->set_time_zone(src->get_time_zone());
}
