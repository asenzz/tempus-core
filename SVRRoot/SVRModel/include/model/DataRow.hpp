#pragma once

#include <armadillo>
#include <boost/date_time/posix_time/posix_time_config.hpp>
#include <vector>

#include "common/compatibility.hpp"
#include "common/logging.hpp"
#include "common/constants.hpp"
#include "common/exceptions.hpp"
#include "model/Request.hpp"

namespace svr {
namespace datamodel {

typedef std::function<double(const double)> t_iqscaler;

class DataRow;

using DataRow_ptr = std::shared_ptr<DataRow>;

class DataRow
{
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
     *      std::vector<boost::posix_time::ptime> row_times;
     *      std::pair<bpt::ptime &, arma::sub_view<double> &> operator[] (const bpt::ptime &ptime) {}
     *      std::pair<bpt::ptime &, arma::sub_view<double> &> operator[] (const size_t ix) {}
     *      ...
     *  }
     *
     */

    static bpt::ptime // Returns last inserted row value time
    insert_rows(
            container &rows_container,
            const arma::mat &data,
            const bpt::ptime &start_time,
            const bpt::time_duration &resolution,
            const unsigned level,
            const unsigned level_ct,
            const bool merge);

    static void
    insert_rows(
            container &rows_container,
            const arma::mat &data,
            const std::deque<bpt::ptime> &times,
            const unsigned level,
            const unsigned level_ct,
            const bool merge);

    static container
    insert_rows(
            const arma::mat &data,
            const std::deque<bpt::ptime> &times,
            const unsigned level,
            const unsigned level_ct,
            const bool merge);

    static void insert_rows(
            container &rows_container,
            const arma::mat &data,
            const container &times,
            const unsigned level,
            const unsigned level_ct,
            const bool merge);

    static container
    construct(const std::deque<datamodel::MultivalResponse_ptr> &responses);

    static void sort(container &rows_container);

    DataRow() = default;

    explicit DataRow(const std::string &csv);

    DataRow(const bpt::ptime &value_time);

    DataRow(
            const bpt::ptime &value_time,
            const bpt::ptime &update_time,
            const double tick_volume,
            const unsigned levels);

    DataRow(
            const bpt::ptime &value_time,
            const bpt::ptime &update_time,
            const double tick_volume,
            const unsigned levels,
            const double value);

    DataRow(
            const bpt::ptime &value_time,
            const bpt::ptime &update_time,
            const double tick_volume,
            const std::vector<double> &values);

    explicit DataRow(
            const bpt::ptime &value_time,
            const bpt::ptime &update_time,
            const double tick_volume,
            CPTRd values_ptr,
            const unsigned values_size);

    std::vector<double> &get_values();

    void set_values(const std::vector<double> &values);

    double get_value(const unsigned column_index) const;

    double &get_value(const unsigned column_index);

    const double &operator*() const;

    double &operator*();

    double operator()(const unsigned column_index) const;

    double &operator[](const unsigned column_index);

    double operator[](const unsigned column_index) const;

    double &operator()(const unsigned column_index);

    double at(const unsigned column_index) const;

    double &at(const unsigned column_index);

    double *p(const unsigned column_index = 0);

    void set_value(const unsigned column_index, const double value);

    unsigned size() const;

    const bpt::ptime &get_update_time() const;

    void set_update_time(const bpt::ptime &update_time);

    const bpt::ptime &get_value_time() const;

    void set_value_time(const bpt::ptime &value_time);

    double get_tick_volume() const;

    void set_tick_volume(const double weight);

    std::string to_string() const;

    std::vector<std::string> to_tuple() const;

    bool operator==(const DataRow &other) const;

    static bool fast_compare(const DataRow::container &lhs, const DataRow::container &rhs);

    static std::shared_ptr<DataRow> load(const std::string &s, const char delim = ',');
};

template<typename T> std::basic_ostream<T> &operator <<(std::basic_ostream<T> &o, const datamodel::DataRow &r)
{
    return o << r.to_string();
}

template<typename C = DataRow::container, typename C_range_iter = typename C::iterator, typename T = typename C::value_type>
class container_range
{
    using C_iter = C_range_iter;
    using C_citer = typename C::const_iterator;
    using C_riter = typename C::reverse_iterator;
    using C_criter = typename C::const_reverse_iterator;

    ssize_t distance_;
    C_range_iter begin_, end_;
    C &container_;

public:
    explicit container_range(C &container);

    container_range(const C_range_iter &start, const C_range_iter &end, C &container);

    container_range(C_range_iter start, C &container);

    container_range(C &container, C_range_iter end);

    container_range(const container_range &rhs);

    container_range(container_range &rhs);

    container_range &operator=(const container_range &rhs);

    T &operator[](const size_t index);

    T operator[](const size_t index) const;

    C_range_iter operator()(const ssize_t index) const;

    C_range_iter it(const ssize_t index) const;

    container_range() = delete;

    ssize_t distance() const;

    C_range_iter contbegin() const;

    C_range_iter contend() const;

    C_citer contcbegin() const;

    C_citer contcend() const;

    unsigned contsize() const;

    C_range_iter begin() const;

    C_range_iter end() const;

    C_citer cbegin() const;

    C_citer cend() const;

    T &front();

    T front() const;

    T &back();

    T back() const;

    C_riter rbegin() const;

    C_criter crbegin() const;

    C_riter rend() const;

    C_criter crend() const;

    C &get_container() const;


    void set_range(const C_range_iter &start, const C_range_iter &end);

    void set_begin(C_range_iter &begin);

    void set_end(C_range_iter &end);

    void reinit();

    size_t levels() const;
};

typedef container_range<const DataRow::container, DataRow::container::const_iterator> datarow_crange;
typedef container_range<DataRow::container, DataRow::container::reverse_iterator> datarow_rrange;
typedef container_range<DataRow::container, DataRow::container::iterator> datarow_range;

}

using data_row_container = datamodel::DataRow::container;
using data_row_container_ptr = std::shared_ptr<data_row_container>;

datamodel::DataRow::container
clone_datarows(datamodel::DataRow::container::const_iterator it, const datamodel::DataRow::container::const_iterator &end);

datamodel::DataRow::container::iterator
lower_bound(const datamodel::DataRow::container::iterator &begin, const datamodel::DataRow::container::iterator &end, const bpt::ptime &t);

datamodel::DataRow::container::const_iterator
lower_bound(const datamodel::DataRow::container::const_iterator &cbegin, const datamodel::DataRow::container::const_iterator &cend, const bpt::ptime &t);

data_row_container::const_iterator
lower_bound(const data_row_container &c, const bpt::ptime &t);

data_row_container::iterator
lower_bound(data_row_container &c, const bpt::ptime &t);

data_row_container::const_iterator
upper_bound(const data_row_container &c, const bpt::ptime &t);

data_row_container::iterator
upper_bound(data_row_container &c, const bpt::ptime &t);

data_row_container::const_iterator
upper_bound(const data_row_container &data, const data_row_container::const_iterator &hint_end, const bpt::ptime &time_key);

data_row_container::iterator
upper_bound(data_row_container &data, const data_row_container::iterator &hint_end, const bpt::ptime &time_key);

data_row_container::iterator
upper_bound(data_row_container &data, const bpt::ptime &time_key);

data_row_container::const_iterator
upper_bound(const data_row_container &data, const bpt::ptime &time_key);

data_row_container::const_iterator
lower_bound(const data_row_container &data, const data_row_container::const_iterator &hint_start, const bpt::ptime &key);

data_row_container::const_iterator
lower_bound_back(const data_row_container &data, const data_row_container::const_iterator &hint, const bpt::ptime &time_key);

data_row_container::iterator lower_bound_back(data_row_container &data, const data_row_container::iterator &hint_end, const bpt::ptime &time_key);

data_row_container::const_iterator
lower_bound(const data_row_container &data, const bpt::ptime &time_key);

data_row_container::const_iterator
lower_bound_back_before(
        const data_row_container &data,
        const data_row_container::const_iterator &hint,
        const bpt::ptime &time_key);

data_row_container::const_iterator
lower_bound_before(
        const data_row_container &data,
        const bpt::ptime &time_key);

data_row_container::iterator
lower_bound_back_before(data_row_container &data, const bpt::ptime &time_key);

data_row_container::const_iterator
lower_bound_or_before(
        const data_row_container &data,
        const data_row_container::const_iterator &hint_end,
        const bpt::ptime &time_key);

data_row_container::const_iterator lower_bound_or_before_back(const data_row_container &data, const bpt::ptime &time_key);

data_row_container::const_iterator lower_bound_or_before(const data_row_container &data, const bpt::ptime &time_key);

data_row_container::const_iterator
lower_bound_before(const data_row_container::const_iterator &cbegin, const data_row_container::const_iterator &cend, const bpt::ptime &time_key);


data_row_container::const_iterator
lower_bound_or_before(const data_row_container::const_iterator &cbegin, const data_row_container::const_iterator &cend, const bpt::ptime &time_key);

data_row_container::iterator lower_bound(data_row_container &data, const bpt::ptime &time_key);

datamodel::DataRow::container::const_iterator
find(
        const datamodel::DataRow::container &data,
        const boost::posix_time::ptime &vtime,
        const boost::posix_time::time_duration &deviation);

data_row_container::iterator find(data_row_container &data, const bpt::ptime &value_time);

data_row_container::const_iterator find(const data_row_container &data, const bpt::ptime &value_time);

data_row_container::const_iterator
find_nearest_before(
        const data_row_container &data,
        const boost::posix_time::ptime &time,
        const boost::posix_time::time_duration &max_gap,
        const size_t lag_count = 0);

data_row_container::iterator
find_nearest_before(
        data_row_container &data,
        const boost::posix_time::ptime &time,
        const size_t lag_count = 0);

data_row_container::iterator
find_nearest(
        data_row_container &data,
        const boost::posix_time::ptime &time);

data_row_container::const_iterator
find_nearest(
        const datamodel::DataRow::container &data,
        const boost::posix_time::ptime &time) noexcept;

datamodel::DataRow::container::const_iterator
find_nearest_back(
        const datamodel::DataRow::container &data,
        const datamodel::DataRow::container::const_iterator &hint,
        const boost::posix_time::ptime &time);

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

inline std::vector<unsigned>
generate_twap_indexes(
        const std::deque<bpt::ptime>::const_iterator &cbegin, // Begin of container
        const std::deque<bpt::ptime>::const_iterator &start_it, // At start time or before
        const std::deque<bpt::ptime>::const_iterator &it_end, // At end time or after
        const bpt::ptime &start_time, // Exact start time
        const boost::posix_time::ptime &end_time, // Exact end time
        const bpt::time_duration &resolution, // Aux input queue resolution
        const unsigned n_out); // Input column index

template<typename I> inline uint32_t /* index of extrema */ generate_twap_bias(
        uint32_t *const out, // Output array
        const bool maxmin, // Min or max
        const I &cbegin, // Begin of container
        const I &start_it, // At start time or before
        const I &it_end, // At end time or after
        const bpt::ptime &start_time, // Exact start time
        const bpt::ptime &end_time, // Exact end time
        const bpt::time_duration &resolution, // Aux input queue resolution
        const uint32_t n_out, // Count of positions to output
        const uint16_t level // Level
        );

bool
generate_twav( // Time-weighted average volume
        const datamodel::DataRow::container::const_iterator &start_it, // At start time or before
        const datamodel::DataRow::container::const_iterator &it_end,
        const bpt::ptime &start_time,
        const boost::posix_time::ptime &end_time,
        const bpt::time_duration &hf_resolution,
        const size_t colix,
        arma::subview<double> out); // Needs to be zeroed out before submitting to this function

arma::mat to_arma_mat(const data_row_container &c);

}

#include "DataRow.tpp"
