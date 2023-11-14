#pragma once

#include <vector>
#include <tsl/ordered_map.h>

#include "common/compatibility.hpp"
#include "common/Logging.hpp"
#include "common/constants.hpp"
#include "common/exceptions.hpp"
#include "model/Request.hpp"

namespace svr {
namespace datamodel{


class DataRow
{
private:
    bpt::ptime value_time_;
    bpt::ptime update_time_;
    double tick_volume_;
    std::vector<double> values_;
    bool anchor = false;

public:
    using container = std::deque<std::shared_ptr<DataRow>>;

    /* TODO Finish implementing
     *
     *  class container {
     *      arma::mat values;
     *
     *      std::vector<boost::posix_time::ptime> row_times;
     *      std::pair<bpt::ptime &, arma::sub_view<double> &> operator[] (const bpt::ptime &ptime) {}
     *      std::pair<bpt::ptime &, arma::sub_view<double> &> operator[] (const size_t ix) {}
     *  }
     *
     */
#if 0
    static std::vector<double>
    diff_return(const std::vector<double> &inp)
    {
        LOG4_DEBUG("inp size " << inp.size());
        if (inp.size() < 2) THROW_EX_FS(std::invalid_argument, "Illegal input size " << inp.size());
        std::vector<double> res(inp.size() - 1);
        __omp_tpfor(size_t, t, 1, inp.size(), res[t - 1] = inp[t] - inp[t - 1])
        return res;
    }

    static void
    inv_diff_return(const double anchor, arma::rowvec &inp)
    {
        inp[0] = anchor + inp[0];
        for (size_t t = 1; t < inp.size(); ++t)
            inp[t] = inp[t - 1] + inp[t];
    }
#endif

    static void
    inv_diff_return( // Convert container contents from diffs to actual prices
            const container &orig_data,
            container &inp,
            const size_t orig_col)
    {
        for (auto iter = inp.begin(); iter != inp.end(); ++iter) {
            double anchor_val = 0;
            auto orig_iter = find(orig_data, iter->get()->get_value_time(), bpt::seconds(0));
            if (orig_iter != orig_data.end())
                anchor_val = std::prev(orig_iter)->get()->get_value(orig_col);
            else if (std::prev(iter)->get()->is_anchor())
                anchor_val = std::prev(iter)->get()->get_value(0);
            else
                LOG4_THROW("Could not find anchor value for " << iter->get()->get_value_time());
            iter->get()->set_value(0, anchor_val + iter->get()->get_value(0));
            iter->get()->set_anchor(true);
        }
    }

    static bpt::ptime // Returns last inserted row value time
    insert_rows(
            container &rows_container,
            const arma::mat &data,
            const bpt::ptime &start_time,
            const bpt::time_duration &resolution);


    static container
    construct(const std::vector<MultivalResponse_ptr> &responses)
    {
        container result;
        for (auto iter_res = responses.begin(); iter_res != responses.end(); ++iter_res) {
            const auto response = **iter_res;
            const std::vector<double> v{response.value};
            result.push_back(std::make_shared<DataRow>(
                    response.value_time,
                    bpt::second_clock::local_time(),
                    svr::common::C_default_value_tick_volume,
                    v));
        }
        return result;
    }

    DataRow()
    {}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
    DataRow(
            const bpt::ptime& value_time,
            const bpt::ptime& update_time = bpt::second_clock::local_time(),
            double tick_volume = svr::common::C_default_value_tick_volume,
            const std::vector<double> &values = {}) :
        value_time_(value_time),
        update_time_(update_time),
        tick_volume_(tick_volume),
        values_(values)
    {}
#pragma GCC diagnostic pop

    void set_anchor(const bool new_anchor_val) { anchor = new_anchor_val; }
    bool is_anchor() { return anchor; }

    std::vector<double>& get_values() { return values_; }
    void set_values(const std::vector<double>& values) { values_ = values; }
    double get_value(const size_t column_index) { return values_[column_index]; } const
    double operator() (const size_t column_index) { return values_[column_index]; }
    void set_value(const size_t column_index, const double value)
    {
        if (values_.size() <= column_index) LOG4_THROW("Invalid column index " << column_index << " of " << values_.size() << " columns.");
        values_[column_index] = value;
    }

    const bpt::ptime& get_update_time() const {return update_time_;}
    void set_update_time(const bpt::ptime& update_time) { update_time_ = update_time;}

    const bpt::ptime& get_value_time() const {return value_time_;}
    void set_value_time(const bpt::ptime& value_time) { value_time_ = value_time;}

    double get_tick_volume() const { return tick_volume_; }
    void set_tick_volume(const double weight) { tick_volume_ = weight; }

    std::string to_string() const
    {
        std::stringstream st;
        st.precision(std::numeric_limits<long double>::max_digits10);
        st << "Value time " << value_time_ << ", update time " << update_time_ <<
           ", volume " << tick_volume_ << ", values ";
        for (size_t i = 0; i < values_.size() - 1; ++i) st << values_[i] << ", ";
        st << values_.back();
        return st.str();
    }

    std::vector <std::string> to_tuple() const
    {
        std::vector<std::string> result;

        result.push_back(bpt::to_simple_string(value_time_));
        result.push_back(bpt::to_simple_string(update_time_));
        result.push_back(common::to_string_with_precision(tick_volume_));

        for (const double v: values_) result.push_back(common::to_string_with_precision(v));

        return result;
    }

    bool operator == (const DataRow& other) const
    {
        return this->value_time_ == other.value_time_
                && this->tick_volume_ == other.tick_volume_
                && this->values_.size() == other.values_.size()
                && std::equal(values_.begin(), values_.end(), other.values_.begin());
    }

    static bool fast_compare(const DataRow::container &lhs, const DataRow::container &rhs)
    {
        return lhs.size() == rhs.size() and
                lhs.begin()->get()->get_values().size() == rhs.begin()->get()->get_values().size() and
                lhs.begin()->get()->get_value_time() == rhs.begin()->get()->get_value_time() and
                lhs.rbegin()->get()->get_value_time() == rhs.rbegin()->get()->get_value_time();
    }

    DataRow::container::const_iterator static find(
            const DataRow::container &data,
            const boost::posix_time::ptime &row_time,
            const boost::posix_time::time_duration &deviation);
};


class datarow_range
{
public:
    datarow_range(
            DataRow::container &container
    )
        : begin_(container.begin())
        , end_(container.end())
        , container_(container)
    {
        reinit();
    }

    datarow_range(
            const DataRow::container::iterator &start,
            const DataRow::container::iterator &end,
            DataRow::container &container
    )
        : begin_(start)
        , end_(end)
        , container_(container)
    {
        reinit();
    }


    size_t distance() const { return static_cast<size_t>(std::abs(std::distance(begin_, end_))); } // TODO Rewrite this container or remove it

    DataRow::container::iterator begin() const { return begin_; }
    DataRow::container::iterator end() const { return end_; }
    DataRow::container::iterator begin() { return begin_; }
    DataRow::container::iterator end() { return end_; }

    DataRow::container::reverse_iterator rbegin() const { return DataRow::container::reverse_iterator(end_); }
    DataRow::container::reverse_iterator rend() const { return DataRow::container::reverse_iterator(begin_); }
    DataRow::container::reverse_iterator rbegin() { return DataRow::container::reverse_iterator(end_); }
    DataRow::container::reverse_iterator rend() { return DataRow::container::reverse_iterator(begin_);}

    const DataRow::container &get_container() const { return container_; }
    DataRow::container &get_container() { return container_; }

    void set_begin(const DataRow::container::iterator &begin)
    {
        begin_ = begin;
        reinit();
    }

    void set_end(const DataRow::container::iterator &end)
    {
        end_ = end;
        reinit();
    }

    datarow_range(const datarow_range &rhs) :
            begin_(rhs.begin_),
            end_(rhs.end_),
            container_(rhs.container_)
    {
        reinit();
    }

    datarow_range &operator = (const datarow_range &rhs)
    {
        begin_ = rhs.begin_;
        end_ = rhs.end_;
        container_ = rhs.container_;
        reinit();
        return *this;
    }

    datarow_range() = delete;

private:
    size_t distance_;
    DataRow::container::iterator begin_, end_;
    DataRow::container &container_;

    void reinit()
    {
        distance_ = std::abs<ssize_t>(std::distance(begin_, end_));
    }
};

}
}

svr::datamodel::DataRow::container::const_iterator
lower_bound(const svr::datamodel::DataRow::container &c, const bpt::ptime &t);

svr::datamodel::DataRow::container::iterator
lower_bound(svr::datamodel::DataRow::container &c, const bpt::ptime &t);

svr::datamodel::DataRow::container::const_iterator
upper_bound(const svr::datamodel::DataRow::container &c, const bpt::ptime &t);

svr::datamodel::DataRow::container::iterator
upper_bound(svr::datamodel::DataRow::container &c, const bpt::ptime &t);

svr::datamodel::DataRow::container::iterator find(svr::datamodel::DataRow::container &data, const bpt::ptime &value_time);

svr::datamodel::DataRow::container::const_iterator find(const svr::datamodel::DataRow::container &data, const bpt::ptime &value_time);


svr::datamodel::DataRow::container::const_iterator
find_nearest_before(
        const svr::datamodel::DataRow::container &data,
        const boost::posix_time::ptime &time,
        const boost::posix_time::time_duration &max_gap,
        const size_t lag_count = std::numeric_limits<size_t>::max());

svr::datamodel::DataRow::container::iterator
find_nearest_before(
        svr::datamodel::DataRow::container &data,
        const boost::posix_time::ptime &time,
        const size_t lag_count = std::numeric_limits<size_t>::max());

svr::datamodel::DataRow::container::iterator
find_nearest(
        svr::datamodel::DataRow::container &data,
        const boost::posix_time::ptime &time);

svr::datamodel::DataRow::container::const_iterator
find_nearest(
        const svr::datamodel::DataRow::container &data,
        const boost::posix_time::ptime &time);

svr::datamodel::DataRow::container::const_iterator
find_nearest(
        const svr::datamodel::DataRow::container &data,
        const boost::posix_time::ptime &time,
        bool &check);

svr::datamodel::DataRow::container::iterator
find_nearest(
        svr::datamodel::DataRow::container &data,
        const boost::posix_time::ptime &time,
        const boost::posix_time::time_duration &max_gap,
        const size_t lag_count = std::numeric_limits<size_t>::max());

svr::datamodel::DataRow::container::const_iterator
find_nearest(
        const svr::datamodel::DataRow::container &data,
        const boost::posix_time::ptime &time,
        const boost::posix_time::time_duration &max_gap,
        const size_t lag_count = std::numeric_limits<size_t>::max());

using DataRow_ptr = std::shared_ptr<svr::datamodel::DataRow>;
using data_row_container = svr::datamodel::DataRow::container;
using data_row_container_ptr = std::shared_ptr<svr::datamodel::DataRow::container>;

/*
template<typename T> T
lower_bound(const data_row_container &data, const T &hint_start, const bpt::ptime &key)
{
    return std::lower_bound(hint_start, data.end(), key, [](const DataRow_ptr &el, const bpt::ptime &tt){
        return el->get_value_time() < tt;
    });
}
*/

svr::datamodel::DataRow::container::const_iterator
find_nearest_after(
        const svr::datamodel::DataRow::container &data,
        const boost::posix_time::ptime &time,
        const boost::posix_time::time_duration &max_gap,
        const size_t lag_count);

data_row_container::const_iterator
lower_bound(const data_row_container &data, const data_row_container::const_iterator &hint_start, const bpt::ptime &key);

data_row_container::const_iterator
lower_bound_back(const data_row_container &data, const data_row_container::const_iterator &hint, const bpt::ptime &time_key);

data_row_container::const_iterator
lower_bound_back(const data_row_container &data, const bpt::ptime &time_key);

svr::datamodel::DataRow::container::const_iterator
lower_bound_back_before(
        const svr::datamodel::DataRow::container &data,
        const svr::datamodel::DataRow::container::const_iterator &hint,
        const bpt::ptime &time_key);

svr::datamodel::DataRow::container::const_iterator
lower_bound_back_before(
        const svr::datamodel::DataRow::container &data,
        const bpt::ptime &time_key);

svr::datamodel::DataRow::container::const_iterator
lower_bound_before(const data_row_container &data, const bpt::ptime &time_key);

bool
generate_labels(
        const svr::datamodel::DataRow::container::const_iterator &start_iter, // At start time or before
        const svr::datamodel::DataRow::container::const_iterator &it_end,
        const bpt::ptime &start_time,
        const boost::posix_time::ptime &end_time,
        const bpt::time_duration &hf_resolution,
        const size_t col_ix,
        arma::rowvec &labels_row);

double calc_twap(
        const svr::datamodel::DataRow::container::const_iterator &start_iter, // At start time or before
        const svr::datamodel::DataRow::container::const_iterator &it_end,
        const boost::posix_time::ptime &start_time,
        const boost::posix_time::ptime &end_time,
        const bpt::time_duration &hf_resolution,
        const size_t col_ix);
