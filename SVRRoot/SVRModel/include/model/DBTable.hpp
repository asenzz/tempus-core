#pragma once

#include <string>
#include <memory>
#include "DataRow.hpp"

#define DISPLAY_ROWS_LIMIT (5)


namespace svr {
namespace datamodel {

class Queue : public Entity
{
protected:
    std::string table_name_;
    data_row_container data_; // TODO Replace with pointer to data_row_container

public:
    Queue() = default;

    Queue(const Queue &rhs);

    explicit Queue(const std::string &table_name, const data_row_container &data = data_row_container());

    virtual void init_id() override;

    std::string get_table_name() const;

    void set_table_name(const std::string &tableName);

    virtual std::string to_string() const override;

    virtual std::string data_to_string() const;

    virtual std::string metadata_to_string() const = 0;

    virtual void update_data(const DataRow::container &new_data, const bool overwrite = true);

    virtual const DataRow::container &get_data() const;

    virtual DataRow::container &get_data();

    operator datarow_range();

    operator datarow_crange() const;

    operator data_row_container &();

    operator const data_row_container &() const;

    void set_data(const DataRow::container &data);

    size_t size() const;

    bool empty() const;

    datamodel::DataRow_ptr operator[](const size_t i) const;

    datamodel::DataRow_ptr &operator[](const size_t i);

    datamodel::DataRow_ptr at(const size_t i) const;

    datamodel::DataRow_ptr &at(const size_t i);

    datamodel::DataRow_ptr front() const;

    datamodel::DataRow_ptr &front();

    datamodel::DataRow_ptr back() const;

    datamodel::DataRow_ptr &back();

    DataRow::container::const_iterator cbegin() const;

    DataRow::container::iterator begin();

    DataRow::container::reverse_iterator rbegin();

    DataRow::container::const_iterator cend() const;

    DataRow::container::iterator end();

    DataRow::container::reverse_iterator rend();

    DataRow::container get_data(const size_t row_count, const bpt::ptime &time_to) const;

    DataRow::container get_data(const size_t tail_length, const bpt::time_period &range) const;

    virtual DataRow::container get_data(const bpt::time_period &range) const;

    std::deque<double>
    get_column_values(
            const size_t column_index,
            const boost::posix_time::ptime start_time = boost::posix_time::min_date_time,
            const boost::posix_time::ptime end_time = boost::posix_time::max_date_time) const;

    std::deque<double> get_column_values(
            const size_t column_index,
            const size_t start_pos = 0,
            const size_t count = std::numeric_limits<size_t>::max()) const;

    bool get_column_values(
            const size_t column_index,
            std::deque<double> &output_values,
            const bpt::time_period &range,
            const bpt::time_duration &resolution);

    void trim(const bpt::ptime &start_time, const bpt::ptime &end_time = bpt::max_date_time);

    void trim(const ssize_t lag, const bpt::ptime &start_time, const bpt::ptime &end_time = bpt::max_date_time);

    void trim(const bpt::time_period &time_range);

    std::deque<double> get_tick_volume() const;
};

} /* namespace model */
} /* namespace svr */
