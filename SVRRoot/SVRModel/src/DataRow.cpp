#include <armadillo>
#include "common/constants.hpp"
#include "util/time_utils.hpp"
#include "util/math_utils.hpp"
#include "model/DataRow.hpp"
#include "appcontext.hpp"

namespace svr {
namespace datamodel {


DataRow::DataRow(const bpt::ptime &value_time) :
        update_time_(bpt::second_clock::local_time()),
        tick_volume_(common::C_default_value_tick_volume),
        values_(DEFAULT_SVRPARAM_DECON_LEVEL + 1)
{}

DataRow::DataRow(
        const bpt::ptime &value_time,
        const bpt::ptime &update_time,
        const double tick_volume,
        const size_t levels) :
        value_time_(value_time),
        update_time_(update_time),
        tick_volume_(tick_volume),
        values_(levels)
{}

DataRow::DataRow(
        const bpt::ptime &value_time,
        const bpt::ptime &update_time,
        const double tick_volume,
        const size_t levels,
        const double value) :
        value_time_(value_time),
        update_time_(update_time),
        tick_volume_(tick_volume),
        values_(levels, value)
{}

DataRow::DataRow(
        const bpt::ptime &value_time,
        const bpt::ptime &update_time,
        const double tick_volume,
        const std::vector<double> &values) :
        value_time_(value_time),
        update_time_(update_time),
        tick_volume_(tick_volume),
        values_(values)
{}

DataRow::DataRow(
        const bpt::ptime &value_time,
        const bpt::ptime &update_time,
        const double tick_volume,
        const double *values_ptr,
        const size_t values_size) :
        value_time_(value_time),
        update_time_(update_time),
        tick_volume_(tick_volume),
        values_(values_size)
{
    memcpy(values_.data(), values_ptr, values_size * sizeof(double));
}

std::vector<double> &DataRow::get_values()
{
    return values_;
}

void DataRow::set_values(const std::vector<double> &values)
{
    values_ = values;
}

double DataRow::get_value(const size_t column_index) const
{
    return values_[column_index];
}

double &DataRow::get_value(const size_t column_index)
{
    return values_[column_index];
}

double DataRow::operator()(const size_t column_index) const
{
    return values_[column_index];
}

double DataRow::at(const size_t column_index) const
{
    if (column_index >= values_.size()) LOG4_THROW("Index " << column_index << " larger or equal to " << values_.size());
    return values_[column_index];
}

double DataRow::operator[](const size_t column_index) const
{
    return values_[column_index];
}

double &DataRow::operator[](const size_t column_index)
{
    return values_[column_index];
}

double &DataRow::operator()(const size_t column_index)
{
    return values_[column_index];
}

double &DataRow::at(const size_t column_index)
{
    return values_[column_index];
}

size_t DataRow::size() const
{
    return values_.size();
}

const bpt::ptime &DataRow::get_update_time() const
{
    return update_time_;
}

void DataRow::set_update_time(const bpt::ptime &update_time)
{
    update_time_ = update_time;
}

const bpt::ptime &DataRow::get_value_time() const
{
    return value_time_;
}

void DataRow::set_value_time(const bpt::ptime &value_time)
{
    value_time_ = value_time;
}

double DataRow::get_tick_volume() const
{
    return tick_volume_;
}

void DataRow::set_tick_volume(const double weight)
{
    tick_volume_ = weight;
}

DataRow::DataRow(const std::string &csv)
{
    const auto p_row = load(csv);
    if (p_row) *this = *p_row;
    else
        LOG4_FATAL("Failed constructing data row from string " << csv);
}

DataRow::container
DataRow::construct(const std::deque<datamodel::MultivalResponse_ptr> &responses)
{
    container result;
    for (auto iter_res = responses.begin(); iter_res != responses.end(); ++iter_res) {
        const auto &response = **iter_res;
        (void) result.emplace_back(ptr<DataRow>(
                response.value_time,
                bpt::second_clock::local_time(),
                common::C_default_value_tick_volume,
                std::vector{response.value}));
    }
    return result;
}

void DataRow::set_value(const size_t column_index, const double value)
{
    if (values_.size() <= column_index) LOG4_THROW("Invalid column index " << column_index << " of " << values_.size() << " columns.");
    values_[column_index] = value;
}

std::string DataRow::to_string() const
{
    std::stringstream st;
    st.precision(std::numeric_limits<long double>::max_digits10);
    st << "Value time " << value_time_ << ", update time " << update_time_ <<
       ", volume " << tick_volume_ << ", values ";
    for (size_t i = 0; i < values_.size() - 1; ++i) st << values_[i] << ", ";
    st << values_.back();
    return st.str();
}

std::vector<std::string> DataRow::to_tuple() const
{
    std::vector<std::string> result;

    result.push_back(bpt::to_simple_string(value_time_));
    result.push_back(bpt::to_simple_string(update_time_));
    result.push_back(common::to_string_with_precision(tick_volume_));

    for (const double v: values_) result.push_back(common::to_string_with_precision(v));

    return result;
}

bool DataRow::operator==(const DataRow &other) const
{
    return this->value_time_ == other.value_time_
           && this->tick_volume_ == other.tick_volume_
           && this->values_.size() == other.values_.size()
           && std::equal(values_.begin(), values_.end(), other.values_.begin());
}

bool DataRow::fast_compare(const DataRow::container &lhs, const DataRow::container &rhs)
{
    return lhs.size() == rhs.size()
           && lhs.begin()->get()->get_values().size() == rhs.begin()->get()->get_values().size()
           && lhs.begin()->get()->get_value_time() == rhs.begin()->get()->get_value_time()
           && lhs.rbegin()->get()->get_value_time() == rhs.rbegin()->get()->get_value_time();
}

std::shared_ptr<DataRow> DataRow::load(const std::string &s, const char delim)
{
    std::istringstream is(s);
    char fld[common::C_max_csv_token_size];
    size_t tix = 0;
    auto r = ptr<DataRow>();
    while (is.getline(fld, common::C_max_csv_token_size, delim)) {
        switch (tix) {
            case 0:
                r->value_time_ = bpt::time_from_string(fld);
                break;
            case 1:
                r->update_time_ = bpt::time_from_string(fld);
                break;
            case 2: {
                const auto v = std::strtod(fld, nullptr);
                if (!common::isnormalz(v))
                    LOG4_ERROR("Reading tick volume " << v << " is not znormal.");
                r->tick_volume_ = v;
                break;
            }
            default: {
                const auto v = std::strtod(fld, nullptr);
                if (!common::isnormalz(v))
                    LOG4_ERROR("Reading price column " << tix - 3 << " " << v << " is not znormal.");
                r->values_.emplace_back(v);
                break;
            }
        }
        ++tix;
    }
    return r;
}

}

data_row_container
clone_datarows(data_row_container::const_iterator it, const data_row_container::const_iterator &end)
{
    const size_t res_size = std::distance(it, end);
    data_row_container res(res_size);
#pragma omp parallel for schedule(static, 1 + std::thread::hardware_concurrency() / res_size) num_threads(adj_threads(res_size))
    for (size_t i = 0; i < res_size; ++i) res[i] = ptr<datamodel::DataRow>(**(it + i));
    return res;
}

static const auto comp_lb = [](const datamodel::DataRow_ptr &el, const bpt::ptime &tt) -> bool {
    return el->get_value_time() < tt;
};


datamodel::DataRow::container::const_iterator
lower_bound(const datamodel::DataRow::container &c, const bpt::ptime &t)
{
    return std::lower_bound(c.cbegin(), c.cend(), t, comp_lb);
}


datamodel::DataRow::container::iterator
lower_bound(datamodel::DataRow::container &c, const bpt::ptime &t)
{
    return std::lower_bound(c.begin(), c.end(), t, comp_lb);
}


static const auto comp_ub = [](const bpt::ptime &tt, const datamodel::DataRow_ptr &el) {
    return tt < el->get_value_time();
};

datamodel::DataRow::container::iterator find(datamodel::DataRow::container &data, const bpt::ptime &value_time)
{
    auto res = lower_bound(data, value_time);
    if (res == data.end()) return res;
    else if (res->get()->get_value_time() == value_time) return res;
    else return data.end();
}


datamodel::DataRow::container::const_iterator find(const datamodel::DataRow::container &data, const bpt::ptime &value_time)
{
    auto res = lower_bound(data, value_time);
    if (res == data.end()) return res;
    else if (res->get()->get_value_time() == value_time) return res;
    else return data.end();
}


datamodel::DataRow::container::iterator
find_nearest(
        datamodel::DataRow::container &data,
        const boost::posix_time::ptime &time)
{
    if (data.empty()) {
        LOG4_ERROR("Data is empty");
        return data.end();
    }
    auto iter = lower_bound(data, time);
    if (iter == data.end()) {
        LOG4_ERROR("Row for time " << time << " not found.");
        return --iter;
    }
    if (iter != data.begin() && time - (**(iter - 1)).get_value_time() < (**iter).get_value_time() - time) --iter;
    return iter;
}


datamodel::DataRow::container::const_iterator
find_nearest(
        const datamodel::DataRow::container &data,
        const boost::posix_time::ptime &time)
{
    if (data.empty()) {
        LOG4_ERROR("Data is empty");
        return data.cend();
    }
    auto iter = lower_bound(data, time);
    if (iter == data.cend()) {
        LOG4_ERROR("Row for time " << time << " not found.");
        return --iter;
    }
    if (iter != data.cbegin() && time - (**(iter - 1)).get_value_time() < (**iter).get_value_time() - time) --iter;
    return iter;
}

datamodel::DataRow::container::const_iterator
find_nearest_back(
        const datamodel::DataRow::container &data,
        const datamodel::DataRow::container::const_iterator &hint,
        const boost::posix_time::ptime &time)
{
    if (data.empty()) {
        LOG4_ERROR("Data is empty");
        return data.cend();
    }
    auto iter = hint;
    if (iter == data.cend()) --iter;
    while (iter != data.cbegin() && (**iter).get_value_time() > time) --iter;
    if (iter != data.cbegin() && time - (**(iter - 1)).get_value_time() < (**iter).get_value_time() - time) --iter;
    return iter;
}


datamodel::DataRow::container::iterator
find_nearest(
        datamodel::DataRow::container &data,
        const boost::posix_time::ptime &time,
        const boost::posix_time::time_duration &max_gap,
        const size_t lag_count)
{
    auto iter = find_nearest(data, time);
    if (ABSDIF(iter->get()->get_value_time(), time) > max_gap)
        THROW_EX_FS(common::insufficient_data, "Difference between " << time << " and " << iter->get()->get_value_time()
                                                                     << " is greater than max gap time "
                                                                     << max_gap <<
                                                                     ", data available is from "
                                                                     << data.begin()->get()->get_value_time()
                                                                     << " until "
                                                                     << data.rbegin()->get()->get_value_time());
    if (lag_count == std::numeric_limits<size_t>::max()) return iter;
    auto dist = std::distance(data.begin(), iter);
    if (dist < decltype(dist)(lag_count))
        THROW_EX_FS(common::insufficient_data,
                    "Distance from beginning " << dist << " is less than needed lag count " << lag_count <<
                                               ", data available is from " << data.begin()->get()->get_value_time() << " until "
                                               << data.rbegin()->get()->get_value_time());
    return iter;
}


datamodel::DataRow::container::const_iterator
find_nearest(
        const datamodel::DataRow::container &data,
        const boost::posix_time::ptime &time,
        const boost::posix_time::time_duration &max_gap,
        const size_t lag_count)
{
    auto iter = find_nearest(data, time);
    if (iter->get()->get_value_time() - time > max_gap)
        THROW_EX_FS(common::insufficient_data,
                    "Difference between " << time << " and " << iter->get()->get_value_time()
                                          << " is greater than max gap time " << max_gap <<
                                          ", data available is from " << data.front()->get_value_time() << " until "
                                          << data.back()->get_value_time());
    if (lag_count == std::numeric_limits<size_t>::max()) return iter;
    auto dist = std::distance(data.begin(), iter);
    if (dist < decltype(dist)(lag_count))
        THROW_EX_FS(common::insufficient_data,
                    "Distance from beginning " << dist << " is less than needed lag count " << lag_count <<
                                               ", data available is from " << data.front()->get_value_time() << " until "
                                               << data.back()->get_value_time());
    return iter;
}


datamodel::DataRow::container::const_iterator
find_nearest_after(
        const datamodel::DataRow::container &data,
        const boost::posix_time::ptime &time,
        const boost::posix_time::time_duration &max_gap,
        const size_t lag_count)
{
    auto iter = lower_bound(data, time);
    if (iter->get()->get_value_time() - time > max_gap)
        THROW_EX_FS(common::insufficient_data,
                    "Difference between " << time << " and " << iter->get()->get_value_time()
                                          << " is greater than max gap time " << max_gap <<
                                          ", data available is from " << data.begin()->get()->get_value_time()
                                          << " until " << data.rbegin()->get()->get_value_time());
    if (lag_count == std::numeric_limits<size_t>::max()) return iter;
    auto dist = std::distance(data.begin(), iter);
    if (dist < decltype(dist)(lag_count))
        THROW_EX_FS(common::insufficient_data,
                    "Distance from beginning " << dist << " is less than needed lag count " << lag_count <<
                                               ", data available is from " << data.begin()->get()->get_value_time() << " until "
                                               << data.rbegin()->get()->get_value_time());
    return iter;
}


datamodel::DataRow::container::const_iterator
find_nearest_before(
        const datamodel::DataRow::container &data,
        const boost::posix_time::ptime &time,
        const boost::posix_time::time_duration &max_gap,
        const size_t lag_count)
{
    auto iter = lower_bound(data, time);
    if (iter == data.begin())
        THROW_EX_FS(common::insufficient_data,
                    "No value before or at " << time << ", data available is from " << data.begin()->get()->get_value_time() << " until "
                                             << data.rbegin()->get()->get_value_time());

    --iter;
    if (ABSDIF((**iter).get_value_time(), time) > max_gap)
        THROW_EX_FS(common::insufficient_data,
                    "Difference between " << time << " and "
                                          << iter->get()->get_value_time()
                                          << " is greater than max gap time "
                                          << max_gap <<
                                          ", data available is from "
                                          << data.begin()->get()->get_value_time()
                                          << " until "
                                          << data.rbegin()->get()->get_value_time());

    if (!lag_count) return iter;

    off_t dist = std::distance(data.begin(), iter);
    if (dist < off_t(lag_count))
        THROW_EX_FS(common::insufficient_data,
                    "Distance from beginning " << dist << " is less than needed lag count " << lag_count <<
                                               ", data available is from " << data.begin()->get()->get_value_time() << " until "
                                               << data.rbegin()->get()->get_value_time());

    return iter;
}


datamodel::DataRow::container::iterator
find_nearest_before(
        datamodel::DataRow::container &data,
        const boost::posix_time::ptime &time,
        const size_t lag_count)
{
    auto iter = lower_bound(data, time);
    if (iter == data.begin())
        THROW_EX_FS(common::insufficient_data,
                    "No value before or at " << time << ", data available is from " << data.begin()->get()->get_value_time() << " until "
                                             << data.rbegin()->get()->get_value_time());
    --iter;

    if (!lag_count) return iter;

    off_t dist = std::distance(data.begin(), iter);
    if (dist < off_t(lag_count))
        THROW_EX_FS(common::insufficient_data,
                    "Distance from beginning " << dist << " is less than needed lag count " << lag_count << ", data available is from "
                                               << data.begin()->get()->get_value_time() << " until " << data.rbegin()->get()->get_value_time());

    return iter;
}


data_row_container::const_iterator
lower_bound(const data_row_container &data, const data_row_container::const_iterator &hint_start, const bpt::ptime &key)
{
    return std::lower_bound(hint_start, data.end(), key, comp_lb);
}

// TODO Unify the functions below using templates and macros
data_row_container::const_iterator
lower_bound_back(const data_row_container &data, const data_row_container::const_iterator &hint_end, const bpt::ptime &time_key)
{
    if (data.empty()) {
        LOG4_ERROR("Data is empty!");
        return data.end();
    }
    data_row_container::const_iterator row_iter = hint_end == data.cend() ? std::prev(hint_end) : hint_end;
    while (row_iter != data.cbegin() && (**row_iter).get_value_time() > time_key) --row_iter;
    if ((**row_iter).get_value_time() < time_key) {
        do { ++row_iter; }
        while (row_iter != data.cend() && (**row_iter).get_value_time() < time_key);
    }
    return row_iter;
}

data_row_container::iterator
lower_bound_back(data_row_container &data, const data_row_container::iterator &hint_end, const bpt::ptime &time_key)
{
    if (data.empty()) {
        LOG4_ERROR("Data is empty!");
        return data.end();
    }
    data_row_container::iterator row_iter = hint_end == data.end() ? std::prev(hint_end) : hint_end;
    while (row_iter != data.begin() && row_iter->get()->get_value_time() > time_key) --row_iter;
    if (row_iter->get()->get_value_time() < time_key) {
        do { ++row_iter; }
        while (row_iter != data.end() && row_iter->get()->get_value_time() < time_key);
    }
    return row_iter;
}

data_row_container::const_iterator
lower_bound_back(const data_row_container &data, const bpt::ptime &time_key)
{
    return lower_bound_back(data, data.end(), time_key);
}

data_row_container::iterator
lower_bound_back(data_row_container &data, const bpt::ptime &time_key)
{
    return lower_bound_back(data, data.end(), time_key);
}

// Find lower bound or before starting from back, using hint
data_row_container::const_iterator
lower_bound_back_before(const data_row_container &data, const data_row_container::const_iterator &hint_end, const bpt::ptime &time_key)
{
    if (data.empty()) {
        LOG4_ERROR("Data is empty!");
        return data.end();
    }
    data_row_container::const_iterator row_iter = hint_end == data.end() ? std::prev(hint_end) : hint_end;
    while (row_iter != data.begin() && (**row_iter).get_value_time() >= time_key) --row_iter;
    if ((**row_iter).get_value_time() > time_key || row_iter == data.end())
        LOG4_ERROR("Couldn't find equal or before to " << time_key << ", but found nearest match " << (**row_iter).get_value_time());
    return row_iter;
}

data_row_container::const_iterator
lower_bound_or_before_back(const data_row_container &data, const data_row_container::const_iterator &hint_end, const bpt::ptime &time_key)
{
    if (data.empty()) {
        LOG4_ERROR("Data is empty!");
        return data.end();
    }
    data_row_container::const_iterator row_iter = hint_end == data.end() ? std::prev(hint_end) : hint_end;
    while (row_iter != data.begin() && (**row_iter).get_value_time() > time_key) --row_iter;
    if ((**row_iter).get_value_time() > time_key)
        LOG4_ERROR("Couldn't find equal or before to " << time_key << ", but found nearest match " << (**row_iter).get_value_time());
    return row_iter;
}


data_row_container::const_iterator
lower_bound_back_before(const data_row_container &data, const bpt::ptime &time_key)
{
    return lower_bound_back_before(data, data.end(), time_key);
}

// Find equal or before
data_row_container::const_iterator
lower_bound_before(const data_row_container &data, const bpt::ptime &time_key)
{
    auto found = lower_bound(data, time_key);
    if (found->get()->get_value_time() == time_key) return found;
    if (found == data.begin()) {
        LOG4_ERROR("Couldn't find equal or before " << time_key << ", found " << found->get()->get_value_time());
        return found;
    }
    --found;
    if (found->get()->get_value_time() > time_key)
        LOG4_ERROR(
                "Couldn't find " << time_key << " lower bound or before, nearest found is " << found->get()->get_value_time());
    return found;
}


data_row_container::iterator
upper_bound_back(data_row_container &data, const data_row_container::iterator &hint_end, const bpt::ptime &time_key)
{
    if (data.empty()) {
        LOG4_ERROR("Data is empty!");
        return data.end();
    }
    data_row_container::iterator row_iter = hint_end == data.end() ? std::prev(hint_end) : hint_end;
    while (row_iter != data.begin() and row_iter->get()->get_value_time() > time_key) --row_iter;
    if (row_iter->get()->get_value_time() <= time_key) {
        if (row_iter == data.end()) {
            LOG4_ERROR("Couldn't find later then " << time_key << ", but found nearest match " << row_iter->get()->get_value_time());
        } else {
            do { ++row_iter; }
            while (row_iter != data.end() && row_iter->get()->get_value_time() <= time_key);
        }
    }
    return row_iter;
}

data_row_container::const_iterator
upper_bound_back(const data_row_container &data, const data_row_container::const_iterator &hint_end, const bpt::ptime &time_key)
{
    if (data.empty()) {
        LOG4_ERROR("Data is empty!");
        return data.end();
    }
    data_row_container::const_iterator row_iter = hint_end == data.end() ? std::prev(hint_end) : hint_end;
    while (row_iter != data.begin() and row_iter->get()->get_value_time() > time_key) --row_iter;
    if (row_iter->get()->get_value_time() <= time_key) {
        if (row_iter == data.end()) {
            LOG4_ERROR("Couldn't find later then " << time_key << ", but found nearest match " << row_iter->get()->get_value_time());
        } else {
            do { ++row_iter; }
            while (row_iter != data.end() && row_iter->get()->get_value_time() <= time_key);
        }
    }
    return row_iter;
}

data_row_container::iterator
upper_bound_back(data_row_container &data, const bpt::ptime &time_key)
{
    return upper_bound_back(data, data.end(), time_key);
}

data_row_container::const_iterator
upper_bound_back(const data_row_container &data, const bpt::ptime &time_key)
{
    return upper_bound_back(data, data.end(), time_key);
}


datamodel::DataRow::container::const_iterator
upper_bound(const datamodel::DataRow::container &c, const bpt::ptime &t)
{
    return std::upper_bound(c.begin(), c.end(), t, comp_ub);
}


datamodel::DataRow::container::iterator
upper_bound(datamodel::DataRow::container &c, const bpt::ptime &t)
{
    return std::upper_bound(c.begin(), c.end(), t, comp_ub);
}

// [start_it, it_end)
// [start_time, end_time)
bool
generate_twap(
        const datamodel::DataRow::container::const_iterator &start_it, // At start time or before
        const datamodel::DataRow::container::const_iterator &it_end,
        const bpt::ptime &start_time,
        const boost::posix_time::ptime &end_time,
        const bpt::time_duration &hf_resolution,
        const size_t colix,
        arma::rowvec &row)
{
    row.zeros();
    const auto outlen = row.size();
    auto valit = start_it;
    const ssize_t inlen = (end_time - start_time) / hf_resolution;
    arma::rowvec ct_prices(outlen);
    auto time_iter = start_time;
    ssize_t inctr = 0;
    const auto inout_ratio = double(outlen) / double(inlen);
    auto last_price = (**start_it)[colix];
    while (time_iter < end_time) {
        while (valit != it_end && (**valit).get_value_time() <= time_iter) {
            last_price = (**valit)[colix];
            ++valit;
        }
        const auto outctr = inctr * inout_ratio;
        row[outctr] += last_price;
        ct_prices[outctr] += 1;
        time_iter += hf_resolution;
        ++inctr;
    }
    const auto dist_it = std::distance(start_it, valit);
    if (inctr != inlen || dist_it < 1) {
        LOG4_ERROR("Could not calculate TWAP for " << start_time << ", column " << colix << ", HF resolution " << hf_resolution);
        return false;
    }
    if (dist_it != inlen) LOG4_TRACE("HF price rows " << dist_it << " different than expected " << inlen);
    row /= ct_prices;
    return true;
}


double
calc_twap(
        const datamodel::DataRow::container::const_iterator &start_iter, // At start time or before
        const datamodel::DataRow::container::const_iterator &it_end,
        const bpt::ptime &start_time,
        const boost::posix_time::ptime &end_time,
        const bpt::time_duration &hf_resolution,
        const size_t col_ix)
{
    arma::rowvec labels_row(1);
    generate_twap(start_iter, it_end, start_time, end_time, hf_resolution, col_ix, labels_row);
    return labels_row[0];
}


bpt::ptime // Returns last inserted row value time // TODO Add phases
datamodel::DataRow::insert_rows(
        data_row_container &rows_container,
        const arma::mat &data,
        const bpt::ptime &start_time,
        const bpt::time_duration &resolution,
        const size_t level,
        const size_t level_ct,
        const bool merge)
{
    std::set<bpt::ptime> times;
    for (size_t i = 0; i < data.n_rows; ++i)
        times.emplace(start_time + double(i) * resolution);
    insert_rows(rows_container, data, times, level, level_ct, merge);
    return *times.rbegin();
}

data_row_container
datamodel::DataRow::insert_rows(
        const arma::mat &data,
        const std::set<bpt::ptime> &times,
        const size_t level,
        const size_t level_ct,
        const bool merge)
{
    data_row_container rows_container;
    insert_rows(rows_container, data, times, level, level_ct, merge);
    return rows_container;
}

void
datamodel::DataRow::insert_rows(
        data_row_container &rows_container,
        const arma::mat &data,
        const std::set<bpt::ptime> &times,
        const size_t level,
        const size_t level_ct,
        const bool merge)
{
    if (times.size() != data.n_rows)
        LOG4_THROW("Times " << times.size() << " and data rows " << data.n_rows << " do not match.");

    LOG4_DEBUG("Inserting " << arma::size(data) << " rows, starting at " << *times.begin());

    if (!rows_container.empty() && data.n_cols != rows_container.begin()->get()->get_values().size())
        LOG4_WARN("Data columns " << data.n_cols << " does not equal existing data columns " << rows_container.begin()->get()->get_values().size());
    if (!merge) {
        auto ins_it = lower_bound_back(rows_container, *times.begin());
        if (ins_it != rows_container.end()) rows_container.erase(ins_it, rows_container.end());
    }

    const auto time_now = bpt::second_clock::local_time();
    const auto prev_size = rows_container.size();
    rows_container.resize(prev_size + data.n_rows);
#pragma omp parallel for num_threads(adj_threads(data.n_rows))
    for (size_t row_ix = 0; row_ix < data.n_rows; ++row_ix) {
        const auto row_time = times ^ row_ix;
        data_row_container::iterator it_row;
#pragma omp critical (DataRow_insert_rows)
        {
            if ((it_row = lower_bound_back(rows_container, row_time)) == rows_container.end())
                it_row = rows_container.insert(it_row, ptr<datamodel::DataRow>(row_time, time_now, common::C_default_value_tick_volume, level_ct, 0.));
        }
        (**it_row).set_value(level, arma::mean(arma::vectorise(data.row(row_ix))));
    }
    LOG4_END();
}

[[maybe_unused]] datamodel::DataRow::container::const_iterator static
find(
        const datamodel::DataRow::container &data,
        const boost::posix_time::ptime &row_time,
        const boost::posix_time::time_duration &deviation)
{
    auto it_row = lower_bound(data, row_time);
    if (it_row->get()->get_value_time() == row_time) return it_row;
    if (std::prev(it_row)->get()->get_value_time() - row_time > it_row->get()->get_value_time() - row_time) std::advance(it_row, -1);
    if (it_row->get()->get_value_time() - row_time > deviation) return data.end();
    return it_row;
}

}