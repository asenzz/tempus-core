//
// Created by zarko on 27/07/2025.
//

#include <iterator>
#include "DataRowService.hpp"
#include "util/math_utils.hpp"
#include "common/exceptions.hpp"

namespace svr {
namespace business {
datamodel::data_row_container
clone_datarows(datamodel::data_row_container::const_iterator it, const datamodel::data_row_container::const_iterator &end)
{
    const auto res_size = std::distance(it, end);
    datamodel::data_row_container res(res_size);
    OMP_FOR_i(res_size) res[i] = otr<datamodel::DataRow>(**(it + i));
    return res;
}

namespace {
constexpr auto comp_lb = [](const datamodel::DataRow_ptr &el, const bpt::ptime &tt) { return el->get_value_time() < tt; };
constexpr auto comp_ub = [](const bpt::ptime &tt, const datamodel::DataRow_ptr &el) { return tt < el->get_value_time(); };
}

datamodel::DataRow::container::iterator
lower_bound(const datamodel::DataRow::container::iterator &begin, const datamodel::DataRow::container::iterator &end, const bpt::ptime &t)
{
    return std::lower_bound(begin, end, t, comp_lb);
}

datamodel::DataRow::container::const_iterator
lower_bound(const datamodel::DataRow::container::const_iterator &cbegin, const datamodel::DataRow::container::const_iterator &cend, const bpt::ptime &t)
{
    return std::lower_bound(cbegin, cend, t, comp_lb);
}

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


datamodel::DataRow::container::const_iterator find(
    const datamodel::DataRow::container &data, const boost::posix_time::ptime &vtime, const boost::posix_time::time_duration &deviation)
{
    auto iter = lower_bound(data, vtime);
    if ((**iter).get_value_time() == vtime) return iter;
    if (vtime - (**std::prev(iter)).get_value_time() < (**iter).get_value_time() - vtime) --iter;
    if ((**iter).get_value_time() - vtime > deviation) return data.cend();
    return iter;
}


datamodel::DataRow::container::iterator find(datamodel::DataRow::container &data, const bpt::ptime &value_time)
{
    auto res = lower_bound(data, value_time);
    if (res == data.cend()) return res;
    else if ((**res).get_value_time() == value_time) return res;
    else return data.end();
}


datamodel::DataRow::container::const_iterator find(const datamodel::DataRow::container &data, const bpt::ptime &value_time)
{
    auto res = lower_bound(data, value_time);
    if (res == data.end()) return res;
    else if ((**res).get_value_time() == value_time) return res;
    else return data.end();
}


datamodel::DataRow::container::iterator find_nearest(datamodel::DataRow::container &data, const boost::posix_time::ptime &time)
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
    if (iter != data.cbegin() && time - (**(iter - 1)).get_value_time() < (**iter).get_value_time() - time) --iter;
    return iter;
}

datamodel::DataRow::container::const_iterator find_nearest(
    const datamodel::DataRow::container::const_iterator &cbegin, const datamodel::DataRow::container::const_iterator &cend, const boost::posix_time::ptime &time) noexcept
{
    if (cbegin == cend) {
        LOG4_ERROR("Data is empty");
        return cend;
    }
    auto iter = lower_bound(cbegin, cend, time);
    if (iter == cend) {
        LOG4_ERROR("Row for time " << time << " not found.");
        return --iter;
    }
    if (iter != cbegin && time - (**(iter - 1)).get_value_time() < (**iter).get_value_time() - time) --iter;
    return iter;
}

datamodel::DataRow::container::const_iterator find_nearest(const datamodel::DataRow::container &data, const boost::posix_time::ptime &time) noexcept
{
    return find_nearest(data.cbegin(), data.cend(), time);
}


datamodel::DataRow::container::const_iterator find_nearest_back(
    const datamodel::DataRow::container &data, const datamodel::DataRow::container::const_iterator &hint, const boost::posix_time::ptime &time)
{
    if (data.empty()) {
        LOG4_ERROR("Data is empty");
        return data.cend();
    }
    auto iter = lower_bound(data, hint, time);
    if (iter != data.cbegin() && time - (**(iter - 1)).get_value_time() < (**iter).get_value_time() - time) --iter;
    return iter;
}


datamodel::DataRow::container::iterator find_nearest(
    datamodel::DataRow::container &data,
    const boost::posix_time::ptime &time,
    const boost::posix_time::time_duration &max_gap,
    const size_t lag_count)
{
    auto iter = find_nearest(data, time);
    if (_ABSDIF((**iter).get_value_time(), time) > max_gap)
        THROW_EX_FS(common::insufficient_data, "Difference between " << time << " and " << (**iter).get_value_time() << " is greater than max gap time " << max_gap <<
                ", data available is from " << data.front()->get_value_time() << " until " << data.back()->get_value_time());
    if (lag_count == std::numeric_limits<size_t>::max()) return iter;
    auto dist = std::distance(data.begin(), iter);
    if (dist < DTYPE(dist)(lag_count))
        THROW_EX_FS(common::insufficient_data,
                "Distance from beginning " << dist << " is less than needed lag count " << lag_count << ", data available is from " << data.front()->get_value_time() << " until " << data.
                back()->get_value_time());
    return iter;
}


datamodel::DataRow::container::const_iterator find_nearest(
    const datamodel::DataRow::container &data,
    const boost::posix_time::ptime &time,
    const boost::posix_time::time_duration &max_gap,
    const size_t lag_count)
{
    auto iter = find_nearest(data, time);
    if ((**iter).get_value_time() - time > max_gap)
        THROW_EX_FS(common::insufficient_data,
                "Difference between " << time << " and " << (**iter).get_value_time()
                << " is greater than max gap time " << max_gap <<
                ", data available is from " << data.front()->get_value_time() << " until "
                << data.back()->get_value_time());
    if (lag_count == std::numeric_limits<size_t>::max()) return iter;
    auto dist = std::distance(data.cbegin(), iter);
    if (dist < DTYPE(dist)(lag_count))
        THROW_EX_FS(common::insufficient_data,
                "Distance from beginning " << dist << " is less than needed lag count " << lag_count <<
                ", data available is from " << data.front()->get_value_time() << " until "
                << data.back()->get_value_time());
    return iter;
}


datamodel::DataRow::container::const_iterator find_nearest_after(
    const datamodel::DataRow::container &data,
    const boost::posix_time::ptime &time,
    const boost::posix_time::time_duration &max_gap,
    const size_t lag_count)
{
    auto iter = lower_bound(data, time);
    if ((**iter).get_value_time() - time > max_gap)
        THROW_EX_FS(common::insufficient_data,
                "Difference between " << time << " and " << (**iter).get_value_time()
                << " is greater than max gap time " << max_gap <<
                ", data available is from " << data.front()->get_value_time()
                << " until " << data.back()->get_value_time());
    if (lag_count == std::numeric_limits<size_t>::max()) return iter;
    auto dist = std::distance(data.cbegin(), iter);
    if (dist < DTYPE(dist)(lag_count))
        THROW_EX_FS(common::insufficient_data,
                "Distance from beginning " << dist << " is less than needed lag count " << lag_count <<
                ", data available is from " << data.front()->get_value_time() << " until "
                << data.back()->get_value_time());
    return iter;
}


datamodel::DataRow::container::const_iterator find_nearest_before(
    const datamodel::DataRow::container &data,
    const boost::posix_time::ptime &time,
    const boost::posix_time::time_duration &max_gap,
    const size_t lag_count)
{
    auto iter = lower_bound(data, time);
    if (iter == data.cbegin())
        THROW_EX_FS(common::insufficient_data,
                "No value before or at " << time << ", data available is from " << data.front()->get_value_time() << " until "
                << data.back()->get_value_time());

    --iter;
    if (_ABSDIF((**iter).get_value_time(), time) > max_gap)
        THROW_EX_FS(common::insufficient_data,
                "Difference between " << time << " and " << (**iter).get_value_time()
                << " is greater than max gap time " << max_gap <<
                ", data available is from " << data.front()->get_value_time()
                << " until " << data.back()->get_value_time());

    if (!lag_count) return iter;

    const auto dist = std::distance(data.begin(), iter);
    if (dist < DTYPE(dist)(lag_count))
        THROW_EX_FS(common::insufficient_data,
                "Distance from beginning " << dist << " is less than needed lag count " << lag_count <<
                ", data available is from " << data.front()->get_value_time() << " until "
                << data.back()->get_value_time());

    return iter;
}


datamodel::DataRow::container::iterator find_nearest_before(
    datamodel::DataRow::container &data,
    const boost::posix_time::ptime &time,
    const size_t lag_count)
{
    auto iter = lower_bound(data, time);
    if (iter == data.cbegin())
        THROW_EX_FS(common::insufficient_data,
                "No value before or at " << time << ", data available is from " << data.front()->get_value_time() << " until "
                << data.back()->get_value_time());
    --iter;

    if (!lag_count) return iter;

    const auto dist = std::distance(data.begin(), iter);
    if (dist < DTYPE(dist)(lag_count))
        THROW_EX_FS(common::insufficient_data,
                "Distance from beginning " << dist << " is less than needed lag count " << lag_count << ", data available is from "
                << data.front()->get_value_time() << " until " << data.back()->get_value_time());

    return iter;
}


datamodel::data_row_container::const_iterator
lower_bound(const datamodel::data_row_container &data, const datamodel::data_row_container::const_iterator &hint_start, const bpt::ptime &key)
{
    if (data.empty()) return data.end();
    return std::lower_bound(hint_start, data.cend(), key, comp_lb);
}

// TODO Unify the functions below using templates and macros
datamodel::data_row_container::const_iterator
lower_bound_back(const datamodel::data_row_container &data, const datamodel::data_row_container::const_iterator &hint_end, const bpt::ptime &time_key)
{
    if (data.empty()) return data.end();
    return std::lower_bound(data.cbegin(), hint_end, time_key, comp_lb);
}

datamodel::data_row_container::iterator
lower_bound_back(const datamodel::data_row_container::iterator &begin, const datamodel::data_row_container::iterator &end, const bpt::ptime &time_key)
{
    if (std::distance(begin, end) < 1) return end;
    return std::lower_bound(begin, end, time_key, comp_lb);
}

datamodel::data_row_container::iterator
lower_bound_back(datamodel::data_row_container &data, const datamodel::data_row_container::iterator &hint_end, const bpt::ptime &time_key)
{
    if (data.empty()) return data.end();
    return std::lower_bound(data.begin(), hint_end, time_key, comp_lb);
}

datamodel::data_row_container::const_iterator
lower_bound_or_before(const datamodel::data_row_container &data, const datamodel::data_row_container::const_iterator &hint_end, const bpt::ptime &time_key)
{
    if (data.empty()) {
        LOG4_ERROR("Data is empty!");
        return data.cend();
    }
    auto row_iter = lower_bound_back(data, hint_end, time_key);
    if (row_iter == data.cend()) {
        if (row_iter == data.cbegin()) return row_iter;
        --row_iter;
    }
    while (row_iter != data.cbegin() && (**row_iter).get_value_time() > time_key) --row_iter;
    if ((**row_iter).get_value_time() > time_key)
        LOG4_ERROR(
        "Couldn't find equal or before to " << time_key << ", but found nearest match " << (**row_iter).get_value_time());
    return row_iter;
}

datamodel::data_row_container::const_iterator
lower_bound_or_before_back(const datamodel::data_row_container &data, const bpt::ptime &time_key)
{
    if (data.empty()) {
        LOG4_ERROR("Data is empty!");
        return data.cend();
    }
    auto row_iter = lower_bound_back(data, data.cend(), time_key);
    if (row_iter == data.cend()) {
        if (row_iter == data.cbegin()) return row_iter;
        --row_iter;
    }
    while (row_iter != data.cbegin() && (**row_iter).get_value_time() > time_key) --row_iter;
    if ((**row_iter).get_value_time() > time_key)
        LOG4_ERROR(
        "Couldn't find equal or before to " << time_key << ", but found nearest match " << (**row_iter).get_value_time());
    return row_iter;
}

// Find lower bound or before starting from back, using hint
datamodel::data_row_container::const_iterator
lower_bound_back_before(const datamodel::data_row_container &data, const datamodel::data_row_container::const_iterator &hint_end, const bpt::ptime &time_key)
{
    if (data.empty()) {
        LOG4_ERROR("Data is empty!");
        return data.cend();
    }
    auto row_iter = lower_bound_back(data, hint_end, time_key);
    if (row_iter == data.cend()) {
        if (row_iter == data.cbegin()) return row_iter;
        --row_iter;
    }
    while (row_iter != data.cbegin() && (**row_iter).get_value_time() >= time_key) --row_iter;
    if ((**row_iter).get_value_time() >= time_key)
        LOG4_ERROR("Couldn't find before to " << time_key << ", but found nearest match " << (**row_iter).get_value_time());
    return row_iter;
}

datamodel::data_row_container::iterator
lower_bound_back_before(datamodel::data_row_container &data, const datamodel::data_row_container::iterator &hint_end, const bpt::ptime &time_key)
{
    if (data.empty()) {
        LOG4_ERROR("Data is empty!");
        return data.end();
    }
    auto row_iter = lower_bound_back(data, hint_end, time_key);
    if (row_iter == data.end()) {
        if (row_iter == data.begin()) return row_iter;
        --row_iter;
    }
    while (row_iter != data.cbegin() && (**row_iter).get_value_time() >= time_key) --row_iter;
    if ((**row_iter).get_value_time() > time_key)
        LOG4_ERROR("Couldn't find equal or before to " << time_key << ", but found nearest match " << (**row_iter).get_value_time());
    return row_iter;
}

datamodel::data_row_container::const_iterator
lower_bound_before(const datamodel::data_row_container::const_iterator &cbegin, const datamodel::data_row_container::const_iterator &cend, const bpt::ptime &time_key)
{
    if (cbegin == cend) {
        LOG4_ERROR("Data is empty!");
        return cend;
    }
    auto row_iter = lower_bound(cbegin, cend, time_key);
    if (row_iter == cend) {
        if (row_iter == cbegin) return row_iter;
        --row_iter;
    }
    while (row_iter != cbegin && (**row_iter).get_value_time() >= time_key) --row_iter;
    if ((**row_iter).get_value_time() > time_key)
        LOG4_ERROR("Couldn't find before or equal to " << time_key << ", but found nearest match " << (**row_iter).get_value_time());

    return row_iter;
}

datamodel::data_row_container::const_iterator
lower_bound_before(const datamodel::data_row_container &data, const bpt::ptime &time_key)
{
    return lower_bound_back_before(data, data.cend(), time_key);
}

datamodel::data_row_container::iterator
lower_bound_back_before(datamodel::data_row_container &data, const bpt::ptime &time_key)
{
    return lower_bound_back_before(data, data.end(), time_key);
}

// Find equal or before
datamodel::data_row_container::const_iterator
lower_bound_or_before(const datamodel::data_row_container &data, const bpt::ptime &time_key)
{
    return lower_bound_or_before(data.cbegin(), data.cend(), time_key);
}

datamodel::data_row_container::const_iterator
lower_bound_or_before(const datamodel::data_row_container::const_iterator &cbegin, const datamodel::data_row_container::const_iterator &cend, const bpt::ptime &time_key)
{
    auto found = lower_bound(cbegin, cend, time_key);
    if (found == cend || (**found).get_value_time() == time_key) return found;
    if (found == cbegin) {
        LOG4_ERROR("Couldn't find equal or before " << time_key << ", found " << (**found).get_value_time());
        return found;
    }
    --found;
    if ((**found).get_value_time() > time_key)
        LOG4_ERROR("Couldn't find " << time_key << " lower bound or before, nearest found is " << (**found).get_value_time());
    return found;
}

datamodel::data_row_container::iterator
upper_bound(datamodel::data_row_container &data, const datamodel::data_row_container::iterator &hint_end, const bpt::ptime &time_key)
{
    return std::upper_bound(data.begin(), hint_end, time_key, comp_ub);
}

datamodel::data_row_container::const_iterator
upper_bound(const datamodel::data_row_container &data, const datamodel::data_row_container::const_iterator &hint_end, const bpt::ptime &time_key)
{
    return std::upper_bound(data.cbegin(), hint_end, time_key, comp_ub);
}

datamodel::DataRow::container::const_iterator
upper_bound(const datamodel::DataRow::container &c, const bpt::ptime &t)
{
    return std::upper_bound(c.cbegin(), c.cend(), t, comp_ub);
}


datamodel::DataRow::container::iterator
upper_bound(datamodel::DataRow::container &c, const bpt::ptime &t)
{
    return std::upper_bound(c.begin(), c.end(), t, comp_ub);
}

// [start_it, it_end)
// [start_time, end_time)
bool
generate_twav( // Time-weighted average volume
    const datamodel::DataRow::container::const_iterator &start_it, // At start time or before
    const datamodel::DataRow::container::const_iterator &it_end,
    const bpt::ptime &start_time,
    const boost::posix_time::ptime &end_time,
    const bpt::time_duration &hf_resolution,
    const size_t colix,
    arma::subview<double> out) // Needs to be zeroed out before submitting to this function
{
    assert(it_end >= start_it);
    assert(end_time >= start_time);
    auto volit = start_it;
    const unsigned inlen = (end_time - start_time) / hf_resolution;
    unsigned inctr = 0;
    const auto inout_ratio = double(out.n_elem) / double(inlen);
    auto last_volume = (**start_it).get_tick_volume();
    UNROLL()
    for (auto time_iter = start_time; time_iter < end_time; time_iter += hf_resolution) {
        UNROLL()
        for (; volit != it_end && (**volit).get_value_time() <= time_iter; ++volit) last_volume = (**volit).get_tick_volume();
        out[inctr * inout_ratio] += last_volume;
        ++inctr;
    }
#ifndef NDEBUG
    const unsigned dist_it = std::distance(start_it, volit);
    if (inctr != inlen || dist_it < 1)
        LOG4_THROW("Could not calculate TWAP for " << start_time << ", column " << colix << ", HF resolution " << hf_resolution);
    if (dist_it != inlen) LOG4_TRACE("HF price rows " << dist_it << " different than expected " << inlen);
    if (out.has_nonfinite()) LOG4_THROW(
            "Out " << out << ", hf_resolution " << hf_resolution << ", inlen " << inlen << ", inout_ratio " << inout_ratio << ", inctr " << inctr);
#endif
    return true;
}

arma::mat to_arma_mat(const datamodel::data_row_container &c)
{
    arma::mat v(c.size(), c.front()->size());
    OMP_FOR_(v.n_elem, simd collapse(2))
    for (unsigned i = 0; i < v.n_rows; ++i)
        for (unsigned j = 0; j < v.n_cols; ++j)
            v(i, j) = c[i]->at(j);
    return v;
}

datamodel::DataRow::container::const_iterator DataRowService::get_start(
    const datamodel::DataRow::container &cont,
    const uint32_t decremental_offset,
    const boost::posix_time::ptime &model_last_time,
    const boost::posix_time::time_duration &resolution)
{
    return get_start(cont.cbegin(), cont.cend(), decremental_offset, model_last_time, resolution);
}

datamodel::DataRow::container::const_iterator DataRowService::get_start(
    const datamodel::DataRow::container::const_iterator &cbegin,
    const datamodel::DataRow::container::const_iterator &cend,
    const uint32_t count,
    const boost::posix_time::ptime &last_time,
    const boost::posix_time::time_duration &resolution)
{
    if (count < 1) {
        LOG4_ERROR("Decremental offset " << count << " returning end.");
        return cend;
    }
    const auto len = std::distance(cbegin, cend);
    // Returns an iterator with the earliest value time needed to train a model with the most current data.
    LOG4_DEBUG("Size is " << len << " decrement " << count);
    if (len <= count) {
        LOG4_WARN("Container size " << len << " is less or equal to needed size " << count);
        return cbegin;
    }

    if (last_time == boost::posix_time::min_date_time) return std::next(cbegin, len - count);

    return find_nearest(cbegin, cend, last_time + resolution);
}

}
}
