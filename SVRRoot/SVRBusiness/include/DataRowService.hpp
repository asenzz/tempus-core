//
// Created by zarko on 27/07/2025.
//

#ifndef DATAROWSERVICE_HPP
#define DATAROWSERVICE_HPP

#include "model/DataRow.hpp"

namespace svr {
namespace business {

class DataRowService
{
public:
    static datamodel::DataRow::container::const_iterator get_start(
        const datamodel::DataRow::container &cont, uint32_t decremental_offset, const boost::posix_time::ptime &model_last_time, const boost::posix_time::time_duration &resolution);

    static datamodel::DataRow::container::const_iterator get_start(
        const datamodel::DataRow::container::const_iterator &cbegin, const datamodel::DataRow::container::const_iterator &cend,
        uint32_t count, const boost::posix_time::ptime &last_time, const boost::posix_time::time_duration &resolution);
};

datamodel::DataRow::container clone_datarows(datamodel::DataRow::container::const_iterator it, const datamodel::DataRow::container::const_iterator &end);

datamodel::DataRow::container::iterator lower_bound(const datamodel::DataRow::container::iterator &begin, const datamodel::DataRow::container::iterator &end, const bpt::ptime &t);

datamodel::DataRow::container::const_iterator
lower_bound(const datamodel::DataRow::container::const_iterator &cbegin, const datamodel::DataRow::container::const_iterator &cend, const bpt::ptime &t);

datamodel::data_row_container::const_iterator lower_bound(const datamodel::data_row_container &c, const bpt::ptime &t);

datamodel::data_row_container::iterator lower_bound(datamodel::data_row_container &c, const bpt::ptime &t);

datamodel::data_row_container::const_iterator upper_bound(const datamodel::data_row_container &c, const bpt::ptime &t);

datamodel::data_row_container::iterator upper_bound(datamodel::data_row_container &c, const bpt::ptime &t);

datamodel::data_row_container::const_iterator upper_bound(const datamodel::data_row_container &data, const datamodel::data_row_container::const_iterator &hint_end, const bpt::ptime &time_key);

datamodel::data_row_container::iterator upper_bound(datamodel::data_row_container &data, const datamodel::data_row_container::iterator &hint_end, const bpt::ptime &time_key);

datamodel::data_row_container::iterator upper_bound(datamodel::data_row_container &data, const bpt::ptime &time_key);

datamodel::data_row_container::const_iterator upper_bound(const datamodel::data_row_container &data, const bpt::ptime &time_key);

datamodel::data_row_container::const_iterator lower_bound(const datamodel::data_row_container &data, const datamodel::data_row_container::const_iterator &hint_start, const bpt::ptime &key);

datamodel::data_row_container::const_iterator lower_bound_back(const datamodel::data_row_container &data, const datamodel::data_row_container::const_iterator &hint, const bpt::ptime &time_key);

datamodel::data_row_container::iterator lower_bound_back(datamodel::data_row_container &data, const datamodel::data_row_container::iterator &hint_end, const bpt::ptime &time_key);

datamodel::data_row_container::const_iterator lower_bound(const datamodel::data_row_container &data, const bpt::ptime &time_key);

datamodel::data_row_container::const_iterator lower_bound_back_before(const datamodel::data_row_container &data, const datamodel::data_row_container::const_iterator &hint, const bpt::ptime &time_key);

datamodel::data_row_container::const_iterator lower_bound_before(const datamodel::data_row_container &data, const bpt::ptime &time_key);

datamodel::data_row_container::iterator lower_bound_back_before(datamodel::data_row_container &data, const bpt::ptime &time_key);

datamodel::data_row_container::const_iterator lower_bound_or_before(const datamodel::data_row_container &data, const datamodel::data_row_container::const_iterator &hint_end, const bpt::ptime &time_key);

datamodel::data_row_container::const_iterator lower_bound_or_before_back(const datamodel::data_row_container &data, const bpt::ptime &time_key);

datamodel::data_row_container::const_iterator lower_bound_or_before(const datamodel::data_row_container &data, const bpt::ptime &time_key);

datamodel::data_row_container::const_iterator lower_bound_before(const datamodel::data_row_container::const_iterator &cbegin, const datamodel::data_row_container::const_iterator &cend, const bpt::ptime &time_key);


datamodel::data_row_container::const_iterator lower_bound_or_before(const datamodel::data_row_container::const_iterator &cbegin, const datamodel::data_row_container::const_iterator &cend, const bpt::ptime &time_key);

datamodel::data_row_container::iterator lower_bound(datamodel::data_row_container &data, const bpt::ptime &time_key);

datamodel::DataRow::container::const_iterator find(const datamodel::DataRow::container &data, const boost::posix_time::ptime &vtime, const boost::posix_time::time_duration &deviation);

datamodel::data_row_container::iterator find(datamodel::data_row_container &data, const bpt::ptime &value_time);

datamodel::data_row_container::const_iterator find(const datamodel::data_row_container &data, const bpt::ptime &value_time);

datamodel::data_row_container::const_iterator find_nearest_before(
    const datamodel::data_row_container &data, const boost::posix_time::ptime &time, const boost::posix_time::time_duration &max_gap, size_t lag_count = 0);

datamodel::data_row_container::iterator find_nearest_before(datamodel::data_row_container &data, const boost::posix_time::ptime &time, size_t lag_count = 0);

datamodel::data_row_container::iterator find_nearest(datamodel::data_row_container &data, const boost::posix_time::ptime &time);

datamodel::DataRow::container::const_iterator
find_nearest(
        const datamodel::DataRow::container::const_iterator &cbegin,
        const datamodel::DataRow::container::const_iterator &cend,
        const boost::posix_time::ptime &time) noexcept;

datamodel::data_row_container::const_iterator find_nearest(const datamodel::DataRow::container &data, const boost::posix_time::ptime &time) noexcept;

datamodel::DataRow::container::const_iterator find_nearest_back(const datamodel::DataRow::container &data, const datamodel::DataRow::container::const_iterator &hint,
                                                                const boost::posix_time::ptime &time);

datamodel::data_row_container::iterator find_nearest(
    datamodel::data_row_container &data, const boost::posix_time::ptime &time, const boost::posix_time::time_duration &max_gap, size_t lag_count = std::numeric_limits<size_t>::max());

datamodel::data_row_container::const_iterator find_nearest(
    const datamodel::data_row_container &data, const boost::posix_time::ptime &time, const boost::posix_time::time_duration &max_gap, size_t lag_count = std::numeric_limits<size_t>::max());

datamodel::data_row_container::const_iterator find_nearest_after(
    const datamodel::data_row_container &data, const boost::posix_time::ptime &time, const boost::posix_time::time_duration &max_gap, size_t lag_count);

template<typename I> inline void generate_twap_indexes(
    const I &cbegin, // Begin of container
    const I &start_it, // At start time or before
    const I &it_end, // At end time or after
    const bpt::ptime &start_time, // Exact start time
    const bpt::time_duration &duration, // Exact end time
    uint32_t n_out, // Count of positions to output
    RPTR(uint32_t) out);
    
template<typename I> inline uint32_t /* index of extrema */ generate_twap_bias(
    RPTR(uint32_t) out, // Output array
    bool maxmin, // Min or max
    const I &cbegin, // Begin of container
    const I &start_it, // At start time or before
    const I &it_end, // At end time or after
    const bpt::ptime &start_time, // Exact start time
    const bpt::ptime &end_time, // Exact end time
    const bpt::time_duration &resolution, // Aux input queue resolution
    uint32_t n_out, // Count of positions to output
    uint16_t level // Level
);

bool
generate_twav( // Time-weighted average volume
    const datamodel::DataRow::container::const_iterator &start_it, // At start time or before
    const datamodel::DataRow::container::const_iterator &it_end,
    const bpt::ptime &start_time,
    const boost::posix_time::ptime &end_time,
    const bpt::time_duration &hf_resolution,
    size_t colix,
    arma::subview<double> out); // Needs to be zeroed out before submitting to this function

arma::mat to_arma_mat(const datamodel::data_row_container &c);

}
}

#include "DataRowService.tpp"

#endif //DATAROWSERVICE_HPP
