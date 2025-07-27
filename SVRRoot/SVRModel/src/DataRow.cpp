#include <armadillo>
#include "common/parallelism.hpp"
#include "common/constants.hpp"
#include "util/time_utils.hpp"
#include "util/math_utils.hpp"
#include "model/DataRow.hpp"

#include "DataRowService.hpp"
#include "model/SVRParameters.hpp"

namespace svr {
namespace datamodel {
DataRow::DataRow(const bpt::ptime &value_time) : update_time_(bpt::second_clock::local_time()),
                                                 tick_volume_(common::C_default_value_tick_volume),
                                                 values_(C_default_svrparam_decon_level + 1)
{
}

DataRow::DataRow(
    const bpt::ptime &value_time,
    const bpt::ptime &update_time,
    const double tick_volume,
    const unsigned levels) : value_time_(value_time),
                             update_time_(update_time),
                             tick_volume_(tick_volume),
                             values_(levels)
{
}

DataRow::DataRow(
    const bpt::ptime &value_time,
    const bpt::ptime &update_time,
    const double tick_volume,
    const unsigned levels,
    const double value) : value_time_(value_time),
                          update_time_(update_time),
                          tick_volume_(tick_volume),
                          values_(levels, value)
{
}

DataRow::DataRow(
    const bpt::ptime &value_time,
    const bpt::ptime &update_time,
    const double tick_volume,
    const std::vector<double> &values) : value_time_(value_time),
                                         update_time_(update_time),
                                         tick_volume_(tick_volume),
                                         values_(values)
{
}

DataRow::DataRow(
    const bpt::ptime &value_time,
    const bpt::ptime &update_time,
    const double tick_volume,
    CPTRd values_ptr,
    const unsigned values_size) : value_time_(value_time),
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
    if (column_index >= values_.size())
        LOG4_THROW("Index " << column_index << " larger or equal to " << values_.size());
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
    else
        LOG4_FATAL("Failed constructing data row from string " << csv);
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

void DataRow::sort(container &rows_container)
{
    std::sort(C_default_exec_policy, rows_container.begin(), rows_container.end(), [](const DataRow_ptr &lhs, const DataRow_ptr &rhs) {
        return lhs->get_value_time() < rhs->get_value_time();
    });
}

void DataRow::set_value(const unsigned column_index, const double value)
{
    if (values_.size() <= column_index)
        LOG4_THROW("Invalid column index " << column_index << " of " << values_.size() << " columns.");
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
                if (!common::isnormalz(v))
                    LOG4_ERROR("Reading tick volume " << v << " from field " << field << " is not znormal.");
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

datamodel::data_row_container
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
    if (times.size() != data.n_rows)
        LOG4_THROW("Times " << times.size() << " and data rows " << data.n_rows << " do not match.");

    if (rows_container.size() && (data.n_cols != (**rows_container.cbegin()).size() || (**rows_container.cbegin()).size() != level_ct))
        LOG4_WARN("Data columns " << data.n_cols << " does not equal existing data columns " << (**rows_container.cbegin()).size() << " or level count " << level_ct);

    if (!merge) rows_container.erase(business::lower_bound(rows_container, *times.cbegin()), rows_container.end());

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
    if (times.size() != data.n_rows)
        LOG4_THROW("Times " << times.size() << " and data rows " << data.n_rows << " do not match.");

    if (rows_container.size() && (data.n_cols != (**rows_container.cbegin()).size() || (**rows_container.cbegin()).size() != level_ct))
        LOG4_WARN("Data columns " << data.n_cols << " does not equal existing data columns " << (**rows_container.cbegin()).size() << " or level count " << level_ct);

    if (!merge) rows_container.erase(business::lower_bound(rows_container, times.front()->get_value_time()), rows_container.end());

    LOG4_DEBUG("Inserting " << arma::size(data) << " rows, starting at " << (**times.cbegin()).get_value_time());

    const auto time_now = bpt::second_clock::local_time();
    const auto prev_size = rows_container.size();
    rows_container.resize(prev_size + data.n_rows);
    const auto it_begin = rows_container.begin() + prev_size;
    OMP_FOR_i(data.n_rows) {
        auto row_iter = it_begin + i;
        if (!*row_iter) *row_iter = ptr<datamodel::DataRow>(times[i]->get_value_time(), time_now, common::C_default_value_tick_volume, level_ct, 0.);
        (**row_iter)[level] += arma::mean(data.row(i));
        LOG4_TRACE("Row " << i << ", level " << level << ", value " << (**row_iter)[level] <<
            ", input row " << common::present(data.row(i)) << ", mean " << arma::mean(data.row(i))); // TODO Remove after testing
    }
    LOG4_END();
}
}
}
