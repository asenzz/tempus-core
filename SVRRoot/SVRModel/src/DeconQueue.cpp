#include <csignal>
#include <appcontext.hpp>
#include "model/DeconQueue.hpp"
#include "DeconQueueService.hpp"

namespace svr {
namespace datamodel {

DeconQueue::DeconQueue() : Queue() {}

DeconQueue::DeconQueue(
        std::string const &table_name,
        std::string const &input_queue_table_name,
        std::string const &input_queue_column_name,
        const bigint dataset_id,
        const size_t decon_level_number,
        const data_row_container &data
)
        : Queue(table_name.empty() ? business::DeconQueueService::make_queue_table_name(input_queue_table_name, dataset_id, input_queue_column_name) : table_name, data),
          input_queue_table_name_(input_queue_table_name),
          input_queue_column_name_(input_queue_column_name),
          dataset_id_(dataset_id),
          decon_level_number_(decon_level_number)
{
}


datamodel::DeconQueue_ptr DeconQueue::clone_empty() const
{
    return ptr<DeconQueue>(
            "clone_" + table_name_, input_queue_table_name_, input_queue_column_name_, dataset_id_, decon_level_number_);
}

datamodel::DeconQueue_ptr DeconQueue::clone(const size_t start_ix, const size_t end_ix) const
{
    auto p_new_decon_queue = clone_empty();
    if (data_.empty()) return p_new_decon_queue;
    else if (data_.size() > start_ix) p_new_decon_queue->data_ = clone_datarows(
            data_.cbegin() + std::min(start_ix, data_.size() - 1),
            data_.cbegin() + std::min(end_ix, data_.size()));
    else
        LOG4_ERROR("Start index " << start_ix << " exceeds data size " << data_.size());
    return p_new_decon_queue;
}

void DeconQueue::erase_until(const data_row_container::iterator &target_iter)
{
//    reset_anchor(target_iter);
    data_.erase(data_.begin(), target_iter);
}

void DeconQueue::erase_until(const bpt::ptime &target_time)
{
    const auto target_iter = lower_bound(data_, target_time);
    data_.erase(data_.begin(), target_iter);
}

std::vector<double> DeconQueue::get_actual_values(const data_row_container::const_iterator &target_iter) const
{
    return business::DeconQueueService::get_actual_values(data_, target_iter);
}


void DeconQueue::update_data(const DataRow::container &new_data, const bool overwrite /* true */)
{
    if (new_data.empty()) return;
    if (data_.empty()) {
        data_ = new_data;
        return;
    }
    if (svr::datamodel::DataRow::fast_compare(data_, new_data)) {
        LOG4_DEBUG("New data seems identical to present data, skipping.");
        return;
    }

    LOG4_DEBUG("Updating " << input_queue_table_name_ << " " << input_queue_column_name_ << " new data from " << new_data.front()->get_value_time() << " until " << new_data.back()->get_value_time() <<
                            " existing data from " << data_.front()->get_value_time() << " until " << data_.back()->get_value_time());

    // Sanity checks against existing decon data
    if (!new_data.empty() && data_.size() > 1)
        if (new_data.front()->get_value_time() - data_.back()->get_value_time() > std::next(new_data.begin())->get()->get_value_time() - new_data.front()->get_value_time()) // TODO Max gap time
            THROW_EX_FS(std::runtime_error,
                    "Adding new data will introduce a gap in decon queue " << input_queue_column_name_ << " time series!");

    if (new_data.front()->get_value_time() <= data_.front()->get_value_time() && new_data.back()->get_value_time() >= data_.back()->get_value_time()) {
        data_ = new_data;
    } else if (new_data.front()->get_value_time() < data_.front()->get_value_time()) {
        const auto data_iter_end = upper_bound_back(data_, new_data.back()->get_value_time());
        if (data_iter_end != data_.end()) data_.erase(data_.begin(), data_iter_end);
        data_.insert(data_.begin(), new_data.begin(), new_data.end());
    } else if (new_data.front()->get_value_time() >= data_.front()->get_value_time()) {
        data_.erase(upper_bound(data_, new_data.front()->get_value_time()), data_.end());
        data_.insert(data_.end(), new_data.begin(), new_data.end());
    }

    LOG4_END();
}

std::string DeconQueue::get_input_queue_table_name() const
{
    return input_queue_table_name_;
}

std::string DeconQueue::get_input_queue_column_name() const
{
    return input_queue_column_name_;
}

bigint DeconQueue::get_dataset_id() const
{
    return dataset_id_;
}

size_t DeconQueue::get_column_count() const
{
    return data_.empty() ? decon_level_number_ : data_.front()->get_values().size();
}

void DeconQueue::set_input_queue_table_name(const std::string &input_queue_table_name)
{
    input_queue_table_name_ = input_queue_table_name;
    reinit_table_name();
}

void DeconQueue::set_input_queue_column_name(const std::string &input_queue_column_name)
{
    input_queue_column_name_ = input_queue_column_name;
    reinit_table_name();
}

void DeconQueue::set_dataset_id(const bigint dataset_id)
{
    dataset_id_ = dataset_id;
    reinit_table_name();
}

size_t DeconQueue::get_decon_level_number() const
{
    return decon_level_number_;
}


void DeconQueue::reinit_table_name()
{
    if (input_queue_table_name_.empty() || input_queue_column_name_.empty()) {
        LOG4_WARN("Aborting, input queue column name " << input_queue_column_name_ << ", input queue column name " << input_queue_column_name_ << " not initialized!");
        return;
    }
    set_table_name(business::DeconQueueService::make_queue_table_name(input_queue_table_name_, dataset_id_, input_queue_column_name_));
}


std::string DeconQueue::metadata_to_string() const
{
    std::stringstream ss;
    ss << "Table name " << this->table_name_
       << ", input queue table name " << this->input_queue_table_name_
       << ", input queue column name " << this->input_queue_column_name_
       << ", dataset id " << this->dataset_id_
       << ", rows count " << this->data_.size();

    return ss.str();
}


bool DeconQueue::operator==(const DeconQueue &other) const
{
    if (this == &other) return true;

    return table_name_ == other.table_name_
           && input_queue_table_name_ == other.input_queue_table_name_
           && input_queue_column_name_ == other.input_queue_column_name_
           && dataset_id_ == other.dataset_id_;
}

std::string DeconQueue::data_to_string() const
{
    return data_to_string(DISPLAY_ROWS_LIMIT);
}

std::string DeconQueue::data_to_string(const size_t data_size) const
{
    size_t row_id = 0;
    std::stringstream ss;
    for (const auto &row : get_data()) {
        if (row_id++ > data_size) {
            ss << ". . . " << (size() - data_size) << " more" << std::endl;
            break;
        }
        ss << row->to_string() << '\n';
    }
    return ss.str();
}

DeconQueue DeconQueue::load(const std::string &file_path)
{
    DeconQueue res;
    auto &data = res.get_data();
    std::fstream fin;
    fin.open(file_path, std::ios::in);
    std::vector<std::string> row;
    std::string line, word, temp;

    while (fin >> temp) {
        row.clear();

        // read an entire row and
        // store it in a string variable 'line'
        getline(fin, line);

        // used for breaking words
        std::stringstream s(line);

        // read every column data of a row and
        // store it in a string variable, 'word'
        while (std::getline(s, word, ',')) {
            // add all the column data
            // of a row to a vector
            const auto colon_position = word.find(':');
            if (colon_position != std::string::npos) {
                word = word.substr(colon_position+2);
            }
            row.push_back(word);
        }
        std::vector<double> values;
        for (size_t i = 3; i < row.size() - (row.size() - 3) / 5; ++i) {
            values.push_back(std::atof(row[i].c_str()));
        }
        const auto data_row_ptr = ptr<svr::datamodel::DataRow>(
                boost::posix_time::time_from_string(row[0]),
                boost::posix_time::time_from_string(row[1]),
                std::atoll(row[2].c_str()),
                values);
        data.push_back(data_row_ptr);
    }
    return res;
}

}
}
