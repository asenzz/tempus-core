#include <model/DBTable.hpp>

#include <algorithm>

namespace svr {
namespace datamodel {

Queue::Queue(const Queue &rhs) : table_name_(rhs.table_name_)
{
    this->data_ = svr::common::clone_shared_ptr_elements(rhs.data_);
}


Queue::Queue(const std::string& table_name, const data_row_container &data):
    table_name_(table_name)
{
    this->data_ = svr::common::clone_shared_ptr_elements(data);
}


void Queue::set_table_name(const std::string& table_name)
{
    table_name_ = table_name;
}


std::string Queue::to_string() const
{
    return "Table name " + table_name_ + ", " + metadata_to_string() + ", " + data_to_string();
}


void Queue::update_data(const DataRow::container &new_data, const bool overwrite /* default true */)
{
    LOG4_BEGIN();

    if (!data_.empty()
        && !new_data.empty()
        && new_data.front()->get_value_time() < data_.back()->get_value_time())
        data_.erase(lower_bound_back(data_, new_data.front()->get_value_time()), data_.end());

    data_.insert(data_.end(), new_data.begin(), new_data.end());

    LOG4_END();
}


std::string Queue::data_to_string() const
{
    size_t rows_ct = 0;
    std::stringstream ss;
    for (const auto &row : this->data_) {
        if (rows_ct++ > DISPLAY_ROWS_LIMIT) {
            ss << ". . . " << (get_data().size() - DISPLAY_ROWS_LIMIT) << " more" << std::endl;
            break;
        }
        ss << row.get()->to_string() << "\n\t";
    }
    return ss.str();
}


}
}
