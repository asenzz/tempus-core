#include <fftw3.h>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <appcontext.hpp>

#include "online_emd.hpp"
#include "util/string_utils.hpp"
#include "util/math_utils.hpp"
#include "common/gpu_handler.hpp"
#include "oemd_coefficients_search.hpp"


namespace svr {

t_coefs_cache online_emd::oemd_coefs_cache;

void online_emd::transform(
        datamodel::DeconQueue_ptr &p_decon_queue,
        const size_t decon_start_ix,
        const size_t test_offset,
        const size_t custom_residuals_ct)
{
    const size_t residuals_ct = custom_residuals_ct == std::numeric_limits<size_t>::max() ? get_residuals_length(p_decon_queue->get_table_name()) : custom_residuals_ct;

    auto start_decon_iter = p_decon_queue->begin() + decon_start_ix;
    if (std::distance(p_decon_queue->begin(), start_decon_iter) >= ssize_t(residuals_ct))
        start_decon_iter -= residuals_ct;
    else {
        LOG4_ERROR("Not enough data to produce time invariant output " << residuals_ct);
        start_decon_iter = p_decon_queue->begin();
    }

    std::vector<double> tail;
    datamodel::datarow_range in_range(start_decon_iter, p_decon_queue->end(), p_decon_queue->get_data());
    if (in_range.distance() < ssize_t(residuals_ct)) mirror_tail(in_range, residuals_ct, tail);

#ifdef MANIFOLD_TEST
    datamodel::datarow_range in_range_test(start_decon_iter, p_decon_queue->end() - test_offset, p_decon_queue->get_data());
    if (in_range_test.distance() < ssize_t(residuals_ct))
        mirror_tail(in_range_test, residuals_ct, tail);
    const auto p_coefs = get_masks(in_range_test, tail, p_decon_queue->get_table_name());
#else
    const auto p_coefs = get_masks(in_range, tail, p_decon_queue->get_table_name());
#endif

    PROFILE_EXEC_TIME(transform(
            in_range,
            tail,
            p_coefs->siftings,
            p_coefs->masks,
            stretch_coef), "OEMD transform of " << in_range.distance() + tail.size() << " values.");
}


t_oemd_coefficients_ptr online_emd::get_masks(
        const datamodel::datarow_range &input,
        const std::vector<double> &tail,
        const std::string &queue_name) const
{
    const auto in_tail_size = input.distance() + tail.size();
    const auto coefs_key = std::pair{levels, queue_name};
    const auto it_coefs = oemd_coefs_cache.find(coefs_key);
    if (it_coefs != oemd_coefs_cache.end()) return it_coefs->second;

    const auto fir_search_input_window_start = std::max<size_t>(0, in_tail_size - oemd_search::fir_search_window_end);

    auto coefs = oemd_coefficients::load(levels, queue_name);
    if (coefs && !coefs->siftings.empty()
        && *std::min_element(coefs->siftings.begin(), coefs->siftings.end())
        && !coefs->masks.empty()
        && !common::empty(coefs->masks))
        goto __bail;

    if (in_tail_size < oemd_search::fir_search_window_end)
        LOG4_WARN("Input size " << input.distance() << " smaller than recommended " << oemd_search::fir_search_window_end);

    if (!coefs) coefs = std::make_shared<oemd_coefficients>();
    oemd_search::oemd_coefficients_search::prepare_masks(coefs->masks, coefs->siftings, levels);

    PROFILE_EXEC_TIME(
            oemd_search::oemd_coefficients_search::get().optimize_levels(input, tail, coefs->masks, coefs->siftings, fir_search_input_window_start, in_tail_size, queue_name),
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


online_emd::online_emd(const size_t _levels, const double _stretch_coef)
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
    __omp_tpfor(size_t, t, masks.size() - 1, x.size(),
        rx[t] = 0;
        for (size_t m = 0; m < masks.size(); ++m) {
            rx[t] += masks[m] * x[t - masks.size() + m + 1];
        }
    )

    const size_t lim = masks.size() - 1 < x.size() ? masks.size() - 1 : x.size();
    __omp_tpfor(size_t, 0, lim,
        rx[t] = 0;
        double sum = 0;
        for (size_t j = 0; j <= t; j++) {
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
    const size_t input_size = decon.size() / levels;
    recon = std::vector<double>(input_size, 0.);

    //by rows  --> decon_column_values[row_index + level_ix * frame_size]
#pragma omp parallel for num_threads(adj_threads(input_size))
    for (size_t i = 0; i < input_size; ++i) {
        double recon_i = 0;
        for (size_t j = 0; j < levels; ++j)
            recon_i += decon[i + j * input_size];
        recon[i] = recon_i;
    }
}


size_t online_emd::get_residuals_length(const oemd_coefficients &coefs, const double stretch_coef)
{
    //return coefs.masks.back().size(); // Minimum
    //return coefs.masks.back().size() * p_oemd_coef->siftings.back(); // Tradeoff
    //return 6 * svr::common::next_power_of_two(coefs.masks.back().size() * coefs.siftings.back()); // Assured time invariance first
    return common::max_size(coefs.masks) * stretch_coef * common::max(coefs.siftings) * 7; /* time invariance 1: * oemd_level_coefs->second.siftings.back() * 7 */ //
}


size_t online_emd::get_residuals_length(const double _stretch_coef, const size_t siftings)
{
    return oemd_search::fir_search_end_size * _stretch_coef * siftings * 7;
}


size_t online_emd::get_residuals_length(const std::string &queue_name)
{
    LOG4_WARN("Getting default masks residuals length!");
    const auto oemd_coefs = oemd_coefs_cache.find({levels, queue_name});
    if (levels < 1)
        LOG4_THROW("Unsupported number of levels " << levels << ", " << queue_name << " requested for OEMD.");
    if (levels == 1) return 0;
    if (oemd_coefs == oemd_coefs_cache.end())
        return get_residuals_length(stretch_coef);
    return get_residuals_length(*oemd_coefs->second, stretch_coef);
}

}
