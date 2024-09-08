#include <fftw3.h>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <appcontext.hpp>

#include "online_emd.hpp"
#include "util/string_utils.hpp"
#include "util/math_utils.hpp"
#include "oemd_coefficient_search.hpp"


namespace svr {
namespace oemd {

t_coefs_cache online_emd::oemd_coefs_cache;

void online_emd::transform(
        datamodel::DeconQueue &decon_queue,
        const unsigned decon_start_ix,
        const unsigned test_offset,
        const unsigned custom_residuals_ct)
{
    const unsigned residuals_ct = custom_residuals_ct == std::numeric_limits<unsigned>::max() ? get_residuals_length(decon_queue.get_table_name()) : custom_residuals_ct;
    const unsigned in_colix = business::SVRParametersService::get_trans_levix(decon_queue.front()->size());
    auto start_decon_iter = decon_queue.begin() + decon_start_ix;
    const auto dist = std::distance(decon_queue.begin(), start_decon_iter);
    size_t tail_len;
    if (dist >= dtype(dist)(residuals_ct)) {
        start_decon_iter -= residuals_ct;
        tail_len = 0;
    } else {
        tail_len = residuals_ct - dist;
        start_decon_iter = decon_queue.begin();
        if (decon_start_ix) LOG4_WARN("Not enough data " << dist <<  " to produce time invariant output " << residuals_ct << " for " << decon_queue.get_table_name() <<
            ", generating tail of length " << tail_len);
    }

    std::vector<double> tail;
    datamodel::datarow_crange in_crange(start_decon_iter, decon_queue.cend(), decon_queue.get_data());
    if (tail_len) business::DeconQueueService::mirror_tail(in_crange, in_crange.distance() + tail_len, tail, in_colix);
#ifdef INTEGRATION_TEST
    datamodel::datarow_crange in_range_test(start_decon_iter, decon_queue.end() - test_offset, decon_queue.get_data());
    if (in_range_test.distance() < ssize_t(residuals_ct))
        business::DeconQueueService::mirror_tail(in_range_test, residuals_ct, tail, in_colix);
    const auto p_coefs = get_masks(in_range_test, tail, decon_queue.get_table_name(), in_colix, std::identity());
#else
    const auto p_coefs = get_masks(in_range, tail, decon_queue.get_table_name());
#endif
    datamodel::datarow_range in_range{start_decon_iter, decon_queue.end(), decon_queue.get_data()};
    PROFILE_EXEC_TIME(transform(
            in_crange,
            in_range,
            tail,
            p_coefs->siftings,
            p_coefs->masks,
            stretch_coef,
            levels,
            in_colix,
            std::identity()), "OEMD inplace transform of " << in_crange.distance() + tail.size() << " values.");
}

void online_emd::transform(
        const datamodel::InputQueue &input_queue,
        datamodel::DeconQueue &decon_queue,
        const unsigned in_colix,
        const unsigned test_offset,
        const datamodel::t_iqscaler &scaler,
        const unsigned custom_residuals_ct)
{
    const unsigned residuals_ct = custom_residuals_ct == std::numeric_limits<unsigned>::max() ? get_residuals_length(decon_queue.get_table_name()) : custom_residuals_ct;
    data_row_container::const_iterator start_input_iter;
    if (decon_queue.empty()) start_input_iter = input_queue.cbegin();
    else start_input_iter = upper_bound(input_queue.get_data(), decon_queue.back()->get_value_time());
    unsigned tail_len;
    if (start_input_iter - input_queue.cbegin() < residuals_ct) {
        tail_len = residuals_ct - (start_input_iter - input_queue.cbegin());
        start_input_iter = input_queue.cbegin();
    } else {
        start_input_iter -= residuals_ct;
        tail_len = 0;
    }

    std::vector<double> tail;
    datamodel::datarow_crange in_range(start_input_iter, input_queue.cend(), input_queue.get_data());
    if (tail_len) business::DeconQueueService::mirror_tail(in_range, in_range.distance() + tail_len, tail, in_colix);

#ifdef INTEGRATION_TEST
    datamodel::datarow_crange in_range_test(start_input_iter, input_queue.cend() - test_offset, input_queue.get_data());
    if (in_range_test.distance() < ssize_t(residuals_ct))
        business::DeconQueueService::mirror_tail(in_range_test, residuals_ct, tail, in_colix);
    const auto p_coefs = get_masks(in_range_test, tail, decon_queue.get_table_name(), in_colix, scaler);
#else
    const auto p_coefs = get_masks(in_range, tail, decon_queue.get_table_name(), in_colix);
#endif

    const auto prev_size = decon_queue.size();
    const auto time_now = bpt::second_clock::local_time();
#ifdef EMD_ONLY
    const auto out_levels = levels;
#else
    const auto out_levels = levels * 4;
#endif
    for (auto &in_iter: in_range) decon_queue.get_data().emplace_back(
            std::make_shared<datamodel::DataRow>(in_iter->get_value_time(), time_now, in_iter->get_tick_volume() / out_levels, out_levels));
    datamodel::datarow_range out_range(decon_queue.begin() + prev_size, decon_queue.end(), decon_queue.get_data());
    PROFILE_EXEC_TIME(transform(
            in_range,
            out_range,
            tail,
            p_coefs->siftings,
            p_coefs->masks,
            stretch_coef,
            levels,
            in_colix,
            scaler), "OEMD transform of " << in_range.distance() + tail.size() << " values.");
}

t_oemd_coefficients_ptr online_emd::get_masks(
        const datamodel::datarow_crange &input, const std::vector<double> &tail, const std::string &queue_name, const unsigned in_colix, const datamodel::t_iqscaler &scaler) const
{
    const oemd::oemd_coefficients_search oemd_search(levels);
    const auto in_tail_size = input.distance() + tail.size();
    const auto coefs_key = std::pair{levels, queue_name};
    const auto it_coefs = oemd_coefs_cache.find(coefs_key);
    if (it_coefs != oemd_coefs_cache.end()) return it_coefs->second;

    const size_t fir_search_input_window_start = in_tail_size > oemd_coefficients_search::C_fir_validation_window ?
            in_tail_size - oemd_coefficients_search::C_fir_validation_window : 0;

    auto coefs = oemd_coefficients::load(levels, queue_name);
    if (coefs && !coefs->siftings.empty() && *std::min_element(coefs->siftings.begin(), coefs->siftings.end()) && !coefs->masks.empty() && !common::empty(coefs->masks))
        goto __bail;

    if (in_tail_size < oemd_coefficients_search::C_fir_validation_window)
        LOG4_WARN("Input size " << input.distance() << " smaller than recommended " << oemd_coefficients_search::C_fir_validation_window);

    if (!coefs) coefs = ptr<oemd_coefficients>();
    oemd::oemd_coefficients_search::prepare_masks(coefs->masks, coefs->siftings, levels);

    PROFILE_EXEC_TIME(
            oemd_search.optimize_levels(input, tail, coefs->masks, coefs->siftings, fir_search_input_window_start, in_tail_size, queue_name, in_colix, scaler),
            "OEMD FIR coefficients search");

    __bail:
    if (!coefs) LOG4_THROW("Could not prepare coefficients for " << levels << ", " << queue_name);

    const auto [it_loaded_coefs, rc] = oemd_coefs_cache.emplace(coefs_key, coefs);
    if (rc) return it_loaded_coefs->second;
    else {
        LOG4_ERROR("Error storing coefficients for " << levels << ", " << queue_name);
        return coefs;
    }
}


online_emd::online_emd(const unsigned _levels, const double _stretch_coef)
        : spectral_transform(std::string("oemd"), _levels), levels(_levels), stretch_coef(_stretch_coef)
{
}


#ifndef OEMD_CUDA
static void
oemd_mean_compute(
        const std::vector<double> &x,
        const std::vector<double> &masks,
        std::vector<double> &rx)
{
    printf("pre %lu:%f:%f\n", 100ul, x[100], rx[100]);
    printf("pre %lu:%f:%f\n", 1000ul, x[1000], rx[1000]);
    omp_tpfor__(unsigned, t, masks.size() - 1, x.size(),
        rx[t] = 0;
        for (unsigned m = 0; m < masks.size(); ++m) {
            rx[t] += masks[m] * x[t - masks.size() + m + 1];
        }
    )

    const unsigned lim = masks.size() - 1 < x.size() ? masks.size() - 1 : x.size();
    omp_tpfor__(unsigned, 0, lim,
        rx[t] = 0;
        double sum = 0;
        for (unsigned j = 0; j <= t; j++) {
            rx[t] += masks[masks.size() - 1 - t + j] * x[j];
            sum += masks[masks.size() - 1 - t + j];
        }
        rx[t] = rx[t] / sum;
    )
    printf("post %lu:%f:%f\n", 100ul, x[100], rx[100]);
    printf("post %lu:%f:%f\n", 1000ul, x[1000], rx[1000]);
    fflush(stdout);
    fflush(stderr);
}
#endif


void online_emd::inverse_transform(
        const std::vector<double> &decon,
        std::vector<double> &recon,
        const size_t padding) const
{
    const unsigned input_size = decon.size() / levels;
    recon = std::vector<double>(input_size, 0.);

    //by rows  --> decon_column_values[row_index + level_ix * frame_size]
#pragma omp parallel for num_threads(adj_threads(input_size))
    for (unsigned i = 0; i < input_size; ++i) {
        double recon_i = 0;
        for (unsigned j = 0; j < levels; ++j)
            recon_i += decon[i + j * input_size];
        recon[i] = recon_i;
    }
}

unsigned mult_residuals(const unsigned mask_len, const double stretch, const unsigned siftings)
{
    return mask_len * stretch * siftings * 8;
}

unsigned online_emd::get_residuals_length(const oemd_coefficients &coefs, const double stretch_coef)
{
    //return coefs.masks.back().size(); // Minimum
    //return coefs.masks.back().size() * p_oemd_coef->siftings.back(); // Tradeoff
    //return 6 * svr::common::next_power_of_two(coefs.masks.back().size() * coefs.siftings.back()); // Assured time invariance first
    return mult_residuals(common::max_size(coefs.masks), stretch_coef, common::max(coefs.siftings)); /* time invariance 1: * oemd_level_coefs->second.siftings.back() * 7 */ //
}


unsigned online_emd::get_residuals_length(const double _stretch_coef, const unsigned siftings)
{
    return mult_residuals(oemd_coefficients_search::C_fir_mask_end_len, _stretch_coef, siftings);
}


unsigned online_emd::get_residuals_length(const std::string &queue_name)
{
    LOG4_WARN("Getting default masks residuals length.");
    const auto oemd_coefs = oemd_coefs_cache.find({levels, queue_name});
    if (levels < 1)
        LOG4_THROW("Unsupported number of levels " << levels << ", " << queue_name << " requested for OEMD.");
    if (levels == 1) return 0;
    if (oemd_coefs == oemd_coefs_cache.end()) return get_residuals_length(stretch_coef);
    return get_residuals_length(*oemd_coefs->second, stretch_coef);
}


}
}
