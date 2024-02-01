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

class DataRow;
using DataRow_ptr = std::shared_ptr<DataRow>;

class DataRow
{
private:
    bpt::ptime value_time_;
    bpt::ptime update_time_;
    double tick_volume_;
    std::vector<double> values_;

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

    static bpt::ptime // Returns last inserted row value time
    insert_rows(
            container &rows_container,
            const arma::mat &data,
            const bpt::ptime &start_time,
            const bpt::time_duration &resolution);


    static container
    construct(const std::deque<datamodel::MultivalResponse_ptr> &responses)
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

    DataRow()
    {}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
    DataRow(
            const bpt::ptime& value_time,
            const bpt::ptime& update_time = bpt::second_clock::local_time(),
            double tick_volume = common::C_default_value_tick_volume,
            const std::vector<double> &values = {}) :
        value_time_(value_time),
        update_time_(update_time),
        tick_volume_(tick_volume),
        values_(values)
    {}

    DataRow(
            const bpt::ptime& value_time,
            const bpt::ptime& update_time,
            double tick_volume = common::C_default_value_tick_volume,
            double *values_ptr = nullptr,
            const size_t values_size = 0) :
            value_time_(value_time),
            update_time_(update_time),
            tick_volume_(tick_volume),
            values_(common::wrap_vector<double>(values_ptr, values_size))
    {}

#pragma GCC diagnostic pop

    std::vector<double>& get_values() { return values_; }
    void set_values(const std::vector<double>& values) { values_ = values; }
    double get_value(const size_t column_index) const { return values_[column_index]; }
    double &get_value(const size_t column_index) { return values_[column_index]; }
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
};


template<typename C = DataRow::container, typename C_range_iter = typename C::iterator, typename T = typename C::value_type>
class container_range
{
    using C_iter = typename C::iterator;
    using C_citer = typename C::const_iterator;
    using C_riter = typename C::reverse_iterator;

    ssize_t distance_;
    C_range_iter begin_, end_;
    C &container_;

public:
    explicit container_range(C &container)
        : begin_(container.begin())
        , end_(container.end())
        , container_(container)
    {
        reinit();
    }

    container_range(
            const C_range_iter &start,
            const C_range_iter &end,
            C &container
    )
            : begin_(start)
            , end_(end)
            , container_(container)
    {
        reinit();
    }

    container_range(
            C_range_iter start,
            C &container
    )
            : begin_(start)
            , end_(container.end())
            , container_(container)
    {
        reinit();
    }

    container_range(
            C &container,
            C_range_iter end
    )
            : begin_(container.start())
            , end_(end)
            , container_(container)
    {
        reinit();
    }

    container_range(const container_range &rhs) :
            begin_(rhs.begin_),
            end_(rhs.end_),
            container_(rhs.container_)
    {
        reinit();
    }

    container_range(container_range &rhs) :
            begin_(rhs.begin_),
            end_(rhs.end_),
            container_(rhs.container_)
    {
        reinit();
    }

    container_range &operator = (const container_range &rhs)
    {
        if (this == &rhs) return *this;
        begin_ = rhs.begin_;
        end_ = rhs.end_;
        container_ = rhs.container_;
        reinit();
        return *this;
    }

    T &operator[](const size_t index)
    {
        return *(begin_ + index);
    }


    T operator[](const size_t index) const
    {
        return *(begin_ + index);
    }

    container_range() = delete;

    ssize_t distance() const { return distance_; } // TODO Rewrite this container or remove it

    C_range_iter begin() const { return begin_; }
    C_range_iter end() const { return end_; }
    C_riter rbegin() const { return C_riter(end_); }
    C_riter rend() const { return C_riter(begin_); }

    C &get_container() const { return container_; }
    // C &get_container() { return container_; }

    void set_range(const C_range_iter &start, const C_range_iter &end)
    {
        begin_ = start;
        end_ = end;
        reinit();
    }

    void set_begin(C_range_iter &begin)
    {
        begin_ = begin;
        reinit();
    }

    void set_end(C_range_iter &end)
    {
        end_ = end;
        reinit();
    }

    void reinit() // Call whenever changes to the container are made
    {
        distance_ = std::distance(begin_, end_);
    }
};

typedef container_range<const DataRow::container, DataRow::container::const_iterator> datarow_crange;
typedef container_range<DataRow::container, DataRow::container::reverse_iterator> datarow_rrange;
typedef container_range<DataRow::container, DataRow::container::iterator> datarow_range;

}

using data_row_container = datamodel::DataRow::container;
using data_row_container_ptr = std::shared_ptr<data_row_container>;

data_row_container::const_iterator
lower_bound(const data_row_container &c, const bpt::ptime &t);

data_row_container::iterator
lower_bound(data_row_container &c, const bpt::ptime &t);

data_row_container::const_iterator
upper_bound(const data_row_container &c, const bpt::ptime &t);

data_row_container::iterator
upper_bound(data_row_container &c, const bpt::ptime &t);

data_row_container::const_iterator
upper_bound_back(const data_row_container &data, const data_row_container::const_iterator &hint_end, const bpt::ptime &time_key);

data_row_container::iterator
upper_bound_back(data_row_container &data, const data_row_container::iterator &hint_end, const bpt::ptime &time_key);

data_row_container::iterator
upper_bound_back(data_row_container &data, const bpt::ptime &time_key);

data_row_container::const_iterator
upper_bound_back(const data_row_container &data, const bpt::ptime &time_key);

data_row_container::const_iterator
lower_bound(const data_row_container &data, const data_row_container::const_iterator &hint_start, const bpt::ptime &key);

data_row_container::const_iterator
lower_bound_back(const data_row_container &data, const data_row_container::const_iterator &hint, const bpt::ptime &time_key);

data_row_container::const_iterator
lower_bound_back(const data_row_container &data, const bpt::ptime &time_key);

data_row_container::const_iterator
lower_bound_back_before(
        const data_row_container &data,
        const data_row_container::const_iterator &hint,
        const bpt::ptime &time_key);

data_row_container::const_iterator
lower_bound_back_before(
        const data_row_container &data,
        const bpt::ptime &time_key);

data_row_container::const_iterator lower_bound_before(const data_row_container &data, const bpt::ptime &time_key);
data_row_container::iterator lower_bound_back(data_row_container &data, const bpt::ptime &time_key);
data_row_container::iterator lower_bound_back(data_row_container &data, const data_row_container::iterator &hint_end, const bpt::ptime &time_key);

data_row_container::iterator find(data_row_container &data, const bpt::ptime &value_time);

data_row_container::const_iterator find(const data_row_container &data, const bpt::ptime &value_time);


data_row_container::const_iterator
find_nearest_before(
        const data_row_container &data,
        const boost::posix_time::ptime &time,
        const boost::posix_time::time_duration &max_gap,
        const size_t lag_count = std::numeric_limits<size_t>::max());

data_row_container::iterator
find_nearest_before(
        data_row_container &data,
        const boost::posix_time::ptime &time,
        const size_t lag_count = std::numeric_limits<size_t>::max());

data_row_container::iterator
find_nearest(
        data_row_container &data,
        const boost::posix_time::ptime &time);

data_row_container::const_iterator
find_nearest(
        const data_row_container &data,
        const boost::posix_time::ptime &time);

data_row_container::const_iterator
find_nearest(
        const data_row_container &data,
        const boost::posix_time::ptime &time,
        bool &check);

data_row_container::iterator
find_nearest(
        data_row_container &data,
        const boost::posix_time::ptime &time,
        const boost::posix_time::time_duration &max_gap,
        const size_t lag_count = std::numeric_limits<size_t>::max());

data_row_container::const_iterator
find_nearest(
        const data_row_container &data,
        const boost::posix_time::ptime &time,
        const boost::posix_time::time_duration &max_gap,
        const size_t lag_count = std::numeric_limits<size_t>::max());

data_row_container::const_iterator
find_nearest_after(
        const data_row_container &data,
        const boost::posix_time::ptime &time,
        const boost::posix_time::time_duration &max_gap,
        const size_t lag_count);


bool
generate_labels(
        const data_row_container::const_iterator &start_iter, // At start time or before
        const data_row_container::const_iterator &it_end,
        const bpt::ptime &start_time,
        const boost::posix_time::ptime &end_time,
        const bpt::time_duration &hf_resolution,
        const size_t col_ix,
        arma::rowvec &labels_row);

double calc_twap(
        const data_row_container::const_iterator &start_iter, // At start time or before
        const data_row_container::const_iterator &it_end,
        const boost::posix_time::ptime &start_time,
        const boost::posix_time::ptime &end_time,
        const bpt::time_duration &hf_resolution,
        const size_t col_ix);

}
