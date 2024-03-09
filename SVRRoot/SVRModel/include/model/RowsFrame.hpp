#pragma once
#include "DataRow.hpp"
#include "util/string_utils.hpp"
#include "common/exceptions.hpp"

namespace svr{
namespace datamodel{

/**
* This class is supposed to contain subset of InputQueue data
* which will ease the use of swt algorithm because it requires
* constant number of elements.
*/
class RowsFrame
{
private:
    DataRow::container data;
    size_t frame_size;
    bpt::ptime first_time;
    bpt::time_duration resolution;
    std::vector<std::string> value_columns;

public:

    RowsFrame() { }

    RowsFrame(size_t frame_size, bpt::ptime const &first_time, bpt::time_duration const &resolution, std::vector<std::string> const &value_columns)
            : frame_size(frame_size), first_time(first_time), resolution(resolution), value_columns(value_columns) {
    }

    const DataRow::container &get_data() const { return data; }
    DataRow::container &get_data() { return data; }

    void set_data(const DataRow::container &data_) {
        this->data = data_;
    }

    size_t get_frame_size() const { return frame_size; }

    size_t size() const { return data.size(); }

    void set_frame_size(size_t frame_size_) {
        this->frame_size = frame_size_;
    }

    size_t get_value_columns_count() const {
        return value_columns.size();
    }

    void set_value_columns_count(size_t value_columns_count) {
        this->value_columns.reserve(value_columns_count);
    }

    bpt::ptime const &get_first_time() const {
        return first_time;
    }

    void set_first_time(bpt::ptime const &first_time_) {
        this->first_time = first_time_;
    }

    bpt::time_duration const &get_resolution() const {
        return resolution;
    }

    void set_resolution(bpt::time_duration const &resolution_) {
        this->resolution = resolution_;
    }


    std::vector<std::string> const &get_value_columns() const {
        return value_columns;
    }

    void set_value_columns(std::vector<std::string> const &value_columns_) {
        this->value_columns = value_columns_;
    }

    bool is_frame_full() const{
        return this->data.size() == frame_size;
    }

    std::string to_string() const {
        std::stringstream ss;

        ss  <<"Frame size: " << get_frame_size()
            << ", First time: " << bpt::to_simple_string(get_first_time())
            << ", resolution: " << bpt::to_simple_string(get_resolution())
            << ", value columns: " << svr::common::to_string(get_value_columns());

        return ss.str();
    }
};
}
}

using RowsFrame_ptr = std::shared_ptr<svr::datamodel::RowsFrame>;
