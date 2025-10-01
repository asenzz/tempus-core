#ifndef DATAROW_HPP
#define DATAROW_HPP

#include <armadillo>
#include <boost/date_time/posix_time/posix_time_config.hpp>
#include <vector>
#include "common/compatibility.hpp"
#include "model/Request.hpp"

namespace svr {
namespace datamodel {
typedef std::function<double(double)> t_iqscaler;

class DataRow;

using DataRow_ptr = std::shared_ptr<DataRow>;

class DataRow
{
    bpt::ptime value_time_;
    bpt::ptime update_time_;
    double tick_volume_;
    std::vector<double> values_;

public:
    using container = std::deque<DataRow_ptr>;
    using manifold_container = std::deque<std::pair<DataRow_ptr, DataRow_ptr> >;

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

    static bpt::ptime insert_rows(container &rows_container, const arma::mat &data, const bpt::ptime &start_time, const bpt::time_duration &resolution, const unsigned level,
                                  const unsigned level_ct, const bool merge);

    static void insert_rows(container &rows_container, const arma::mat &data, const std::deque<bpt::ptime> &times, const unsigned level, const unsigned level_ct, const bool merge);

    static container insert_rows(const arma::mat &data, const std::deque<bpt::ptime> &times, const unsigned level, const unsigned level_ct, const bool merge);

    static void insert_rows(container &rows_container, const arma::mat &data, const container &times, const unsigned level, const unsigned level_ct, const bool merge);

    static container construct(const std::deque<datamodel::MultivalResponse_ptr> &responses);

    static void sort(container &rows_container);

    DataRow() = default;

    explicit DataRow(const std::string &csv);

    DataRow(const bpt::ptime &value_time);

    DataRow(const bpt::ptime &value_time, const bpt::ptime &update_time, const double tick_volume, const unsigned levels);

    DataRow(const bpt::ptime &value_time, const bpt::ptime &update_time, const double tick_volume, const unsigned levels, const double value);

    DataRow(const bpt::ptime &value_time, const bpt::ptime &update_time, const double tick_volume, const std::vector<double> &values);

    explicit DataRow(const bpt::ptime &value_time, const bpt::ptime &update_time, const double tick_volume, CPTRd values_ptr, const unsigned values_size);

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

using data_row_container = DataRow::container; // Is a queue of DataRow objects, sorted by value_time, ascending
using data_row_container_ptr = std::shared_ptr<data_row_container>;

}
}

#include "DataRow.tpp"

#endif
