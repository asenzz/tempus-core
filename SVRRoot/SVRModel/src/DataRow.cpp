#include "model/DataRow.hpp"
#include "util/TimeUtils.hpp"
#include "util/math_utils.hpp"
#include "appcontext.hpp"
#include <armadillo>


#define DURMS(T1, T2) size_t(std::abs(bpt::time_duration(T1 - T2).total_milliseconds()))


namespace svr {
namespace datamodel {

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
        const auto response = **iter_res;
        result.push_back(std::make_shared<DataRow>(
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
    return lhs.size() == rhs.size() and
           lhs.begin()->get()->get_values().size() == rhs.begin()->get()->get_values().size() and
           lhs.begin()->get()->get_value_time() == rhs.begin()->get()->get_value_time() and
           lhs.rbegin()->get()->get_value_time() == rhs.rbegin()->get()->get_value_time();
}

std::shared_ptr<DataRow> DataRow::load(const std::string &s)
{
    std::istringstream is(s);
    char coef_str[MAX_CSV_TOKEN_SIZE];
    size_t tix = 0;
    auto r = std::make_shared<DataRow>();
    while (is.getline(coef_str, MAX_CSV_TOKEN_SIZE, ',')) {
        switch (tix) {
            case 0:
                r->value_time_ = bpt::time_from_string(coef_str);
                break;
            case 1:
                r->update_time_ = bpt::time_from_string(coef_str);
                break;
            case 2: {
                const auto v = std::strtod(coef_str, nullptr);
                if (!common::isnormalz(v))
                    LOG4_ERROR("Reading tick volume " << v << " is not znormal.");
                r->tick_volume_ = v;
                break;
            }
            default: {
                const auto v = std::strtod(coef_str, nullptr);
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
    for (size_t i = 0; i < res_size; ++i) res[i] = std::make_shared<datamodel::DataRow>(**(it + i));
    return res;
}

static const auto comp_lb = [](const datamodel::DataRow_ptr &el, const bpt::ptime &tt) -> bool {
    return el->get_value_time() < tt;
};


datamodel::DataRow::container::const_iterator
lower_bound(const datamodel::DataRow::container &c, const bpt::ptime &t)
{
    return std::lower_bound(c.begin(), c.end(), t, comp_lb);
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
    auto iter = lower_bound(data, time);
    if (iter == data.end()) LOG4_THROW("Row for time " << time << " not found.");
    if (iter != data.begin() and DURMS(time, std::prev(iter)->get()->get_value_time()) < DURMS(time, iter->get()->get_value_time())) --iter;
    return iter;
}


datamodel::DataRow::container::const_iterator
find_nearest(
        const datamodel::DataRow::container &data,
        const boost::posix_time::ptime &time)
{
    auto iter = lower_bound(data, time);
    if (iter == data.end()) LOG4_THROW("Row for time " << time << " not found.");
    if (iter != data.begin() and DURMS(time, std::prev(iter)->get()->get_value_time()) < DURMS(iter->get()->get_value_time(), time)) --iter;
    return iter;
}

datamodel::DataRow::container::const_iterator
find_nearest(
        const datamodel::DataRow::container &data,
        const boost::posix_time::ptime &time,
        bool &check)
{
    auto iter = lower_bound(data, time);
    if (iter == data.end()) {
        LOG4_ERROR("Row for time " << time << " not found.");
        check &= false;
        return iter;
    }
    if (iter != data.begin() and DURMS(time, std::prev(iter)->get()->get_value_time()) < DURMS(iter->get()->get_value_time(), time)) --iter;
    check &= true;
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
    if (DURMS(iter->get()->get_value_time(), time) > size_t(max_gap.total_milliseconds()))
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
    if (DURMS(iter->get()->get_value_time(), time) > size_t(max_gap.total_milliseconds()))
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
    data_row_container::const_iterator row_iter = hint_end == data.end() ? std::prev(hint_end) : hint_end;
    while (row_iter != data.begin() && row_iter->get()->get_value_time() > time_key) --row_iter;
    if (row_iter->get()->get_value_time() < time_key) {
        if (row_iter == data.end()) {
            LOG4_ERROR("Couldn't find equal or before to " << time_key << ", but found nearest match " << row_iter->get()->get_value_time());
        } else {
            do { ++row_iter; }
            while (row_iter != data.end() && row_iter->get()->get_value_time() < time_key);
        }
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
    while (row_iter != data.begin() and row_iter->get()->get_value_time() > time_key) --row_iter;
    if (row_iter->get()->get_value_time() < time_key) {
        if (row_iter == data.end()) {
            LOG4_ERROR("Couldn't find equal or before to " << time_key << ", but found nearest match " << row_iter->get()->get_value_time());
        } else {
            do { ++row_iter; }
            while (row_iter != data.end() && row_iter->get()->get_value_time() < time_key);
        }
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
    while (row_iter != data.begin() and row_iter->get()->get_value_time() >= time_key) --row_iter;
    if (row_iter->get()->get_value_time() > time_key)
        LOG4_ERROR(
                "Couldn't find equal or before to " << time_key << ", but found nearest match " << row_iter->get()->get_value_time());
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


bool
generate_labels(
        const datamodel::DataRow::container::const_iterator &start_iter, // At start time or before
        const datamodel::DataRow::container::const_iterator &it_end,
        const bpt::ptime &start_time,
        const boost::posix_time::ptime &end_time,
        const bpt::time_duration &hf_resolution,
        const size_t col_ix,
        arma::rowvec &labels_row)
{
    const size_t outlen = labels_row.size();
    labels_row.fill(0);
    datamodel::DataRow::container::const_iterator valit = start_iter;
    const size_t inlen = (end_time - start_time).total_seconds() / hf_resolution.total_seconds();
    arma::rowvec ct_prices(outlen, arma::fill::zeros);
    bpt::ptime time_iter = start_time;
    size_t inctr = 0;
    const double inout_ratio = double(outlen) / (double((end_time - start_time).total_seconds()) / double(hf_resolution.total_seconds()));
    // if (inout_ratio > 1) LOG4_THROW("Output len " << outlen << " is larger than input period " << inlen << " samples.");
    double last_price = start_iter->get()->get_value(col_ix);
    while (time_iter < end_time) {
        bpt::ptime valtime;
        while (valit != it_end && (valtime = valit->get()->get_value_time()) <= time_iter) {
            last_price = valit->get()->get_value(col_ix);
            ++valit;
        }
        const size_t outctr = inctr * inout_ratio;
        labels_row[outctr] += last_price;
        ct_prices[outctr] += 1;
        time_iter += hf_resolution;
        ++inctr;
    }
    const size_t dist_it = std::distance(start_iter, valit);
    if (inctr != inlen || dist_it < 1) {
        LOG4_ERROR("Could not calculate TWAP for " << start_time << ", column " << col_ix << ", HF resolution " << hf_resolution);
        labels_row.fill(std::numeric_limits<double>::quiet_NaN());
        return false;
    }
    if (dist_it != inlen) LOG4_TRACE("HF price rows " << dist_it << " different than expected " << inlen);
    labels_row /= ct_prices;
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
    generate_labels(start_iter, it_end, start_time, end_time, hf_resolution, col_ix, labels_row);
    return labels_row[0];
}


bpt::ptime // Returns last inserted row value time // TODO Add phases
datamodel::DataRow::insert_rows(
        data_row_container &rows_container,
        const arma::mat &data,
        const bpt::ptime &start_time,
        const bpt::time_duration &resolution)
{
    std::set<bpt::ptime> times;
    for (size_t i = 0; i < data.n_rows; ++i)
        times.emplace(start_time + double(i) * resolution);
    insert_rows(rows_container, data, times);
    return *times.rbegin();
}

data_row_container
datamodel::DataRow::insert_rows(
        const arma::mat &data,
        const std::set<bpt::ptime> &times)
{
    data_row_container rows_container;
    insert_rows(rows_container, data, times);
    return rows_container;
}

void
datamodel::DataRow::insert_rows(
        data_row_container &rows_container,
        const arma::mat &data,
        const std::set<bpt::ptime> &times)
{
    if (times.size() != data.n_rows)
        LOG4_THROW("Times " << times.size() << " and data rows " << data.n_rows << " do not match.");

    LOG4_DEBUG("Inserting " << arma::size(data) << " rows, starting at " << *times.begin());

    if (!rows_container.empty() && data.n_cols != rows_container.begin()->get()->get_values().size())
        LOG4_WARN("Data columns " << data.n_cols << " does not equal existing data columns " << rows_container.begin()->get()->get_values().size());
    auto ins_it = lower_bound_back(rows_container, *times.begin());
    if (ins_it != rows_container.end()) rows_container.erase(ins_it, rows_container.end());

    const auto time_now = bpt::second_clock::local_time();
    const auto prev_size = rows_container.size();
    rows_container.resize(prev_size + data.n_rows);
#pragma omp parallel for num_threads(adj_threads(data.n_rows))
    for (size_t row_ix = 0; row_ix < data.n_rows; ++row_ix)
        rows_container[prev_size + row_ix] = std::make_shared<DataRow>(
                times ^ row_ix, time_now, common::C_default_value_tick_volume, arma::conv_to<std::vector<double>>::from(data.row(row_ix)));

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