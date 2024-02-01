#include "model/DataRow.hpp"
#include "util/TimeUtils.hpp"
#include "util/math_utils.hpp"
#include "appcontext.hpp"
#include <armadillo>


#define DURMS(T1, T2) size_t(std::abs(bpt::time_duration(T1 - T2).total_milliseconds()))


namespace svr {

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

    if (lag_count == std::numeric_limits<size_t>::max()) return iter;

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

    if (lag_count == std::numeric_limits<size_t>::max()) return iter;

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
    LOG4_DEBUG("Inserting data " << arma::size(data) << " starting time " << start_time);
    if (!rows_container.empty() and data.n_cols != rows_container.begin()->get()->get_values().size())
        LOG4_WARN(
                "data.n_cols " << data.n_cols << " != rows_container.begin()->second->get_values().size() " << rows_container.begin()->get()->get_values().size());
    auto ins_it = lower_bound(rows_container, start_time);
    if (ins_it != rows_container.end()) rows_container.erase(ins_it, rows_container.end());

    for (size_t row_ix = 0; row_ix < data.n_rows; ++row_ix) {
        const auto row_time = start_time + resolution * row_ix;
#ifdef EMO_DIFF
        const auto feature_end_iter = lower_bound_back(rows_container, row_time - resolution * OFFSET_PRED_MUL);
        if (feature_end_iter == rows_container.end()) {
            LOG4_ERROR("Couldn't find anchor row for " << row_time << " in container.");
            continue;
        } else {
            LOG4_DEBUG("Anchor row for " << row_time << " found at " << std::prev(feature_end_iter)->get()->get_value_time());
        }
        const std::vector<double> &feature_end_vals = std::prev(feature_end_iter)->get()->get_values();
#endif
        std::vector<double> val_row(data.n_cols);
        for (size_t col_ix = 0; col_ix < data.n_cols; ++col_ix) {
#ifdef EMO_DIFF
            val_row[col_ix] = data(row_ix, col_ix) + feature_end_vals[col_ix];
            LOG4_TRACE("col_ix " << col_ix << ", val_row[col_ix] " << val_row[col_ix] << ", data " << data(row_ix, col_ix) << ", feature_end_vals " << feature_end_vals[col_ix]);
#else
            val_row[col_ix] = data(row_ix, col_ix);
#endif
        }
        const auto p_new_row = std::make_shared<DataRow>(row_time, bpt::second_clock::local_time(), common::C_default_value_tick_volume, val_row);
        LOG4_DEBUG("Inserting row " << p_new_row->to_string());
        rows_container.push_back(p_new_row);
    }
    LOG4_END();
    return start_time + resolution * (data.n_rows - 1);
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