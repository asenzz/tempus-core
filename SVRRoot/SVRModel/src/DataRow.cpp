#include <armadillo>
#include "common/constants.hpp"
#include "util/time_utils.hpp"
#include "util/math_utils.hpp"
#include "model/DataRow.hpp"
#include "appcontext.hpp"
#include "common/parallelism.hpp"

namespace svr {
namespace datamodel {


DataRow::DataRow(const bpt::ptime &value_time) :
        update_time_(bpt::second_clock::local_time()),
        tick_volume_(common::C_default_value_tick_volume),
        values_(C_default_svrparam_decon_level + 1)
{}

DataRow::DataRow(
        const bpt::ptime &value_time,
        const bpt::ptime &update_time,
        const double tick_volume,
        const unsigned levels) :
        value_time_(value_time),
        update_time_(update_time),
        tick_volume_(tick_volume),
        values_(levels)
{}

DataRow::DataRow(
        const bpt::ptime &value_time,
        const bpt::ptime &update_time,
        const double tick_volume,
        const unsigned levels,
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
        CPTRd values_ptr,
        const unsigned values_size) :
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

double DataRow::get_value(const unsigned column_index) const
{
    return values_[column_index];
}

double &DataRow::get_value(const unsigned column_index)
{
    return values_[column_index];
}

double &DataRow::operator*()
{
    return values_.front();
}

const double &DataRow::operator*() const
{
    return values_.front();
}

double DataRow::operator()(const unsigned column_index) const
{
    return values_[column_index];
}

double DataRow::at(const unsigned column_index) const
{
    if (column_index >= values_.size()) LOG4_THROW("Index " << column_index << " larger or equal to " << values_.size());
    return values_[column_index];
}

double DataRow::operator[](const unsigned column_index) const
{
    return values_[column_index];
}

double &DataRow::operator[](const unsigned column_index)
{
    return values_[column_index];
}

double &DataRow::operator()(const unsigned column_index)
{
    return values_[column_index];
}

double &DataRow::at(const unsigned column_index)
{
    return values_[column_index];
}

double *DataRow::p(const unsigned column_index)
{
    return values_.data() + column_index;
}

unsigned DataRow::size() const
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
    else LOG4_FATAL("Failed constructing data row from string " << csv);
}

DataRow::container
DataRow::construct(const std::deque<MultivalResponse_ptr> &responses)
{
    container result;
    for (auto iter_res = responses.begin(); iter_res != responses.end(); ++iter_res) {
        const auto p_response = *iter_res;
        result.emplace_back(otr<DataRow>(p_response->value_time, bpt::second_clock::local_time(), common::C_default_value_tick_volume, std::vector{p_response->value}));
    }
    return result;
}

void DataRow::set_value(const unsigned column_index, const double value)
{
    if (values_.size() <= column_index) LOG4_THROW("Invalid column index " << column_index << " of " << values_.size() << " columns.");
    values_[column_index] = value;
}

std::string DataRow::to_string() const
{
    std::stringstream s;
    s.precision(std::numeric_limits<double>::max_digits10);
    s << "Value time " << value_time_ << ", update time " << update_time_ << ", volume " << tick_volume_ << ", values ";
    for (unsigned i = 0; i < values_.size() - 1; ++i) s << values_[i] << ", ";
    s << values_.back();
    return s.str();
}

std::vector<std::string> DataRow::to_tuple() const
{
    std::vector<std::string> result;

    result.emplace_back(bpt::to_simple_string(value_time_));
    result.emplace_back(bpt::to_simple_string(update_time_));
    result.emplace_back(common::to_string_with_precision(tick_volume_));
    for (const double v: values_) result.emplace_back(common::to_string_with_precision(v));

    return result;
}

bool DataRow::operator==(const DataRow &other) const
{
    return value_time_ == other.value_time_
           && tick_volume_ == other.tick_volume_
           && values_.size() == other.values_.size()
           && std::equal(C_default_exec_policy, values_.cbegin(), values_.cend(), other.values_.cbegin());
}

bool DataRow::fast_compare(const DataRow::container &lhs, const DataRow::container &rhs)
{
    return lhs.size() == rhs.size()
           && lhs.front()->size() == rhs.front()->size()
           && lhs.front()->get_value_time() == rhs.front()->get_value_time()
           && lhs.back()->get_value_time() == rhs.back()->get_value_time();
}

std::shared_ptr<DataRow> DataRow::load(const std::string &s, const char delim)
{
    std::istringstream is(s);
    char field[common::C_max_csv_token_size];
    unsigned tix = 0;
    auto r = ptr<DataRow>();
    while (is.getline(field, common::C_max_csv_token_size, delim)) {
        switch (tix) {
            case 0:
                r->value_time_ = bpt::time_from_string(field);
                break;
            case 1:
                r->update_time_ = bpt::time_from_string(field);
                break;
            case 2: {
                const auto v = std::strtod(field, nullptr);
                if (!common::isnormalz(v)) LOG4_ERROR("Reading tick volume " << v << " from field " << field << " is not znormal.");
                r->tick_volume_ = v;
                break;
            }
            default: {
                const auto v = std::strtod(field, nullptr);
                if (!common::isnormalz(v))
                    LOG4_ERROR("Reading price column " << tix - 3 << " " << v << " from field " << field << " is not znormal.");
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
    const auto res_size = std::distance(it, end);
    data_row_container res(res_size);
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


datamodel::DataRow::container::const_iterator
find(
        const datamodel::DataRow::container &data,
        const boost::posix_time::ptime &vtime,
        const boost::posix_time::time_duration &deviation)
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
    if (iter != data.cbegin() && time - (**(iter - 1)).get_value_time() < (**iter).get_value_time() - time) --iter;
    return iter;
}


datamodel::DataRow::container::const_iterator
find_nearest(
        const datamodel::DataRow::container &data,
        const boost::posix_time::ptime &time) noexcept
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
    auto iter = lower_bound(data, hint, time);
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
    if (_ABSDIF((**iter).get_value_time(), time) > max_gap)
        THROW_EX_FS(common::insufficient_data, "Difference between " << time << " and " << (**iter).get_value_time()
                                                                     << " is greater than max gap time " << max_gap <<
                                                                     ", data available is from "
                                                                     << data.front()->get_value_time()
                                                                     << " until "
                                                                     << data.back()->get_value_time());
    if (lag_count == std::numeric_limits<size_t>::max()) return iter;
    auto dist = std::distance(data.begin(), iter);
    if (dist < DTYPE(dist)(lag_count))
        THROW_EX_FS(common::insufficient_data,
                    "Distance from beginning " << dist << " is less than needed lag count " << lag_count <<
                                               ", data available is from " << data.front()->get_value_time() << " until "
                                               << data.back()->get_value_time());
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


datamodel::DataRow::container::const_iterator
find_nearest_after(
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


datamodel::DataRow::container::const_iterator
find_nearest_before(
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


datamodel::DataRow::container::iterator
find_nearest_before(
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


data_row_container::const_iterator
lower_bound(const data_row_container &data, const data_row_container::const_iterator &hint_start, const bpt::ptime &key)
{
    return std::lower_bound(hint_start, data.cend(), key, comp_lb);
}

// TODO Unify the functions below using templates and macros
data_row_container::const_iterator
lower_bound_back(const data_row_container &data, const data_row_container::const_iterator &hint_end, const bpt::ptime &time_key)
{
    return std::lower_bound(data.cbegin(), hint_end, time_key, comp_lb);
}

data_row_container::iterator
lower_bound_back(const data_row_container::iterator &begin, const data_row_container::iterator &end, const bpt::ptime &time_key)
{
    return std::lower_bound(begin, end, time_key, comp_lb);
}

data_row_container::iterator
lower_bound_back(data_row_container &data, const data_row_container::iterator &hint_end, const bpt::ptime &time_key)
{
    return std::lower_bound(data.begin(), hint_end, time_key, comp_lb);
}

data_row_container::const_iterator
lower_bound_back(const data_row_container &data, const bpt::ptime &time_key)
{
    return lower_bound_back(data, data.cend(), time_key);
}

data_row_container::iterator
lower_bound_back(data_row_container &data, const bpt::ptime &time_key)
{
    return lower_bound_back(data, data.end(), time_key);
}

data_row_container::const_iterator
lower_bound_or_before_back(const data_row_container &data, const data_row_container::const_iterator &hint_end, const bpt::ptime &time_key)
{
    if (data.empty()) {
        LOG4_ERROR("Data is empty!");
        return data.cend();
    }
    auto row_iter = lower_bound_back(data, hint_end, time_key);
    while (row_iter != data.cbegin() && (**row_iter).get_value_time() > time_key) --row_iter;
    if ((**row_iter).get_value_time() > time_key) LOG4_ERROR(
            "Couldn't find equal or before to " << time_key << ", but found nearest match " << (**row_iter).get_value_time());
    return row_iter;
}

data_row_container::const_iterator
lower_bound_or_before_back(const data_row_container &data, const bpt::ptime &time_key)
{
    if (data.empty()) {
        LOG4_ERROR("Data is empty!");
        return data.cend();
    }
    auto row_iter = lower_bound_back(data, data.cend(), time_key);
    while (row_iter != data.cbegin() && (**row_iter).get_value_time() > time_key) --row_iter;
    if ((**row_iter).get_value_time() > time_key) LOG4_ERROR(
            "Couldn't find equal or before to " << time_key << ", but found nearest match " << (**row_iter).get_value_time());
    return row_iter;
}

// Find lower bound or before starting from back, using hint
data_row_container::const_iterator
lower_bound_back_before(const data_row_container &data, const data_row_container::const_iterator &hint_end, const bpt::ptime &time_key)
{
    if (data.empty()) {
        LOG4_ERROR("Data is empty!");
        return data.cend();
    }
    auto row_iter = lower_bound_back(data, hint_end, time_key);
    while (row_iter != data.cbegin() && (**row_iter).get_value_time() >= time_key) --row_iter;
    if ((**row_iter).get_value_time() >= time_key) LOG4_ERROR(
            "Couldn't find equal or before to " << time_key << ", but found nearest match " << (**row_iter).get_value_time());
    return row_iter;
}

data_row_container::iterator
lower_bound_back_before(data_row_container &data, const data_row_container::iterator &hint_end, const bpt::ptime &time_key)
{
    if (data.empty()) {
        LOG4_ERROR("Data is empty!");
        return data.end();
    }
    auto row_iter = lower_bound_back(data, hint_end, time_key);
    while (row_iter != data.cbegin() && (**row_iter).get_value_time() >= time_key) --row_iter;
    if ((**row_iter).get_value_time() > time_key)
        LOG4_ERROR("Couldn't find equal or before to " << time_key << ", but found nearest match " << (**row_iter).get_value_time());
    return row_iter;
}

data_row_container::const_iterator
lower_bound_back_before(const data_row_container &data, const bpt::ptime &time_key)
{
    return lower_bound_back_before(data, data.cend(), time_key);
}

data_row_container::iterator
lower_bound_back_before(data_row_container &data, const bpt::ptime &time_key)
{
    return lower_bound_back_before(data, data.end(), time_key);
}

// Find equal or before
data_row_container::const_iterator
lower_bound_before(const data_row_container &data, const bpt::ptime &time_key)
{
    return lower_bound_before(data.cbegin(), data.cend(), time_key);
}

data_row_container::const_iterator
lower_bound_before(const data_row_container::const_iterator &cbegin, const data_row_container::const_iterator &cend, const bpt::ptime &time_key)
{
    auto found = lower_bound(cbegin, cend, time_key);
    if ((**found).get_value_time() == time_key) return found;
    if (found == cbegin) {
        LOG4_ERROR("Couldn't find equal or before " << time_key << ", found " << (**found).get_value_time());
        return found;
    }
    --found;
    if ((**found).get_value_time() > time_key)
        LOG4_ERROR("Couldn't find " << time_key << " lower bound or before, nearest found is " << (**found).get_value_time());
    return found;
}

data_row_container::iterator
upper_bound_back(data_row_container &data, const data_row_container::iterator &hint_end, const bpt::ptime &time_key)
{
    return std::upper_bound(data.begin(), hint_end, time_key, comp_ub);
}

data_row_container::const_iterator
upper_bound_back(const data_row_container &data, const data_row_container::const_iterator &hint_end, const bpt::ptime &time_key)
{
    return std::upper_bound(data.cbegin(), hint_end, time_key, comp_ub);
}

data_row_container::iterator
upper_bound_back(data_row_container &data, const bpt::ptime &time_key)
{
    return upper_bound_back(data, data.end(), time_key);
}

data_row_container::const_iterator
upper_bound_back(const data_row_container &data, const bpt::ptime &time_key)
{
    return upper_bound_back(data, data.cend(), time_key);
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
        for (;volit != it_end && (**volit).get_value_time() <= time_iter; ++volit) last_volume = (**volit).get_tick_volume();
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


bpt::ptime // Returns last inserted row value time // TODO Add phases
datamodel::DataRow::insert_rows(
        data_row_container &rows_container,
        const arma::mat &data,
        const bpt::ptime &start_time,
        const bpt::time_duration &resolution,
        const unsigned level,
        const unsigned level_ct,
        const bool merge)
{
    std::deque<bpt::ptime> times;
    for (unsigned i = 0; i < data.n_rows; ++i)
        times.emplace_back(start_time + double(i) * resolution);
    insert_rows(rows_container, data, times, level, level_ct, merge);
    return *times.rbegin();
}

data_row_container
datamodel::DataRow::insert_rows(
        const arma::mat &data,
        const std::deque<bpt::ptime> &times,
        const unsigned level,
        const unsigned level_ct,
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
        const std::deque<bpt::ptime> &times,
        const unsigned level,
        const unsigned level_ct,
        const bool merge)
{
    if (times.size() != data.n_rows) LOG4_THROW("Times " << times.size() << " and data rows " << data.n_rows << " do not match.");

    if (rows_container.size() && (data.n_cols != (**rows_container.cbegin()).size() || (**rows_container.cbegin()).size() != level_ct))
        LOG4_WARN("Data columns " << data.n_cols << " does not equal existing data columns " << (**rows_container.cbegin()).size() << " or level count " << level_ct);

    if (!merge) rows_container.erase(lower_bound_back(rows_container, *times.cbegin()), rows_container.end());

    LOG4_DEBUG("Inserting " << arma::size(data) << " rows, starting at " << *times.cbegin());

    const auto time_now = bpt::second_clock::local_time();
    const auto prev_size = rows_container.size();
    rows_container.resize(prev_size + data.n_rows);
    OMP_FOR_i(data.n_rows) {
        auto row_iter = rows_container.begin() + prev_size + i;
        if (!*row_iter) *row_iter = ptr<datamodel::DataRow>(times[i], time_now, common::C_default_value_tick_volume, level_ct, 0.);
        (**row_iter)[level] += arma::mean(data.row(i));
    }
    LOG4_END();
}


void datamodel::DataRow::insert_rows(
        datamodel::DataRow::container &rows_container,
        const arma::mat &data,
        const datamodel::DataRow::container &times,
        const unsigned level,
        const unsigned level_ct,
        const bool merge)
{
    if (times.size() != data.n_rows) LOG4_THROW("Times " << times.size() << " and data rows " << data.n_rows << " do not match.");

    if (rows_container.size() && (data.n_cols != (**rows_container.cbegin()).size() || (**rows_container.cbegin()).size() != level_ct))
        LOG4_WARN("Data columns " << data.n_cols << " does not equal existing data columns " << (**rows_container.cbegin()).size() << " or level count " << level_ct);

    if (!merge) rows_container.erase(lower_bound_back(rows_container, times.front()->get_value_time()), rows_container.end());

    LOG4_DEBUG("Inserting " << arma::size(data) << " rows, starting at " << *times.cbegin());

    const auto time_now = bpt::second_clock::local_time();
    const auto prev_size = rows_container.size();
    rows_container.resize(prev_size + data.n_rows);
    OMP_FOR_i(data.n_rows) {
        auto row_iter = rows_container.begin() + prev_size + i;
        if (!*row_iter) *row_iter = ptr<datamodel::DataRow>(times[i]->get_value_time(), time_now, common::C_default_value_tick_volume, level_ct, 0.);
        (**row_iter)[level] += arma::mean(data.row(i));
    }
    LOG4_END();
}

arma::mat to_arma_mat(const data_row_container &c)
{
    arma::mat v(c.size(), c.front()->size());
    OMP_FOR_(v.n_elem, simd collapse(2))
    for (unsigned i = 0; i < v.n_rows; ++i)
        for (unsigned j = 0; j < v.n_cols; ++j)
            v(i, j) = c[i]->at(j);
    return v;
}

}
