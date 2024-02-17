#pragma once

#include "common/types.hpp"
#include "DBTable.hpp"
#include "DataRow.hpp"
#include "StoreBufferPushMerge.hpp"
#include "DeconQueue.hpp"

namespace svr {

namespace business { class DeconQueueService; }

namespace datamodel {


class DeconQueue: public Queue
{
    friend svr::business::DeconQueueService;
public:
    DeconQueue();
    DeconQueue(
            std::string const &table_name,
            std::string const &input_queue_table_name_,
            std::string const &input_queue_column_name_,
            const bigint dataset_id_ = 0,
            const size_t decon_level_number_ = 1,
            const data_row_container &data = data_row_container());
    virtual ~DeconQueue(){};

    std::shared_ptr<DeconQueue> clone_empty() const;

    std::shared_ptr<DeconQueue> clone(const size_t start_ix = std::numeric_limits<size_t>::min(), const size_t end_ix = std::numeric_limits<size_t>::max()) const;

    virtual void update_data(const DataRow::container &new_data, const bool overwrite = true) override;

    inline std::string get_input_queue_table_name() const
    {
        return input_queue_table_name_;
    }

    void set_input_queue_table_name(const std::string &input_queue_table_name_);

    inline std::string get_input_queue_column_name() const
    {
        return input_queue_column_name_;
    }

    void set_input_queue_column_name(const std::string &input_queue_column_name_);

    inline bigint get_dataset_id() const
    {
        return dataset_id_;
    }

    void set_dataset_id(const bigint dataset_id_);

    size_t get_decon_level_number() const;

    virtual std::string metadata_to_string() const override;

    bool operator==(const DeconQueue &other) const;

    virtual std::string data_to_string() const override;

    std::string data_to_string(const size_t data_size = DISPLAY_ROWS_LIMIT) const;

    size_t get_column_count() const
    {
        return data_.empty() ? decon_level_number_ : data_.front()->get_values().size();
    }

    std::vector<double> get_actual_values(const data_row_container::const_iterator &target_iter) const;
    void erase_until(const data_row_container::iterator &target_iter);
    void erase_until(const boost::posix_time::ptime &target_time);

    static DeconQueue load(const std::string &file_path);

private:
    std::string input_queue_table_name_; // TODO Replace with pointer to Input Queue
    std::string input_queue_column_name_; // TODO Replace with pointer to Input Queue
    bigint dataset_id_ = 0; // TODO Replace with pointer to Dataset
    size_t decon_level_number_ = 0;

    void reinit_table_name();
};

template<typename T> std::basic_ostream<T> &
operator<<(std::basic_ostream<T> &os, const DeconQueue &e)
{
    os << e.to_string();
    return os;
}

using DeconQueue_ptr = std::shared_ptr<DeconQueue>;

}
}

template<>
inline void store_buffer_push_merge<svr::datamodel::DeconQueue_ptr>(svr::datamodel::DeconQueue_ptr &dest, const svr::datamodel::DeconQueue_ptr &src)
{
    for(auto const &row: src->get_data()) dest->get_data().emplace_back(row);
    dest->set_table_name(src->get_table_name());
    dest->set_input_queue_table_name(src->get_input_queue_table_name());
    dest->set_input_queue_column_name(src->get_input_queue_column_name());
    dest->set_dataset_id(src->get_dataset_id());
}
