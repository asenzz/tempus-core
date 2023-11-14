#include "online_emd.hpp"
#include "util/string_utils.hpp"
#include "util/math_utils.hpp"
#include "common/gpu_handler.hpp"
#include "oemd_coefficients_search.hpp"
#include "online_emd_impl.cuh"

#include <fftw3.h>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <cmath>
#include <appcontext.hpp>


namespace svr {

static const size_t fir_search_start_size = 500;
static const size_t fir_search_end_size = 20000; // 18k is the maximum for 16GB video card that won't make OSQP crash
static const size_t fir_search_window_start = 0; // 10000000; // Tail size, set to 0 if tail is presented trimmed
static const size_t fir_search_window_end = fir_search_window_start + fir_search_end_size * 20;


void online_emd::find_fir_coefficients(const std::vector<double> &input)
{
    LOG4_BEGIN();
    if (input.size() < fir_search_window_end)
        LOG4_ERROR("Input size " << input.size() << " smaller than needed window end " << fir_search_window_end);

    const size_t fir_search_input_window_start = std::max<size_t>(0, input.size() - fir_search_window_end);
    PROFILE_EXEC_TIME(
            oemd_search::oemd_coefs_search(
                    input, fir_search_start_size, fir_search_end_size, fir_search_input_window_start, input.size(), levels, p_oemd_coef->masks, p_oemd_coef->siftings),
                      "OEMD FIR coefficients search");
    fir_coefs_initialized = true;
    LOG4_END();
}


online_emd::online_emd(const size_t _levels, const double _stretch_coef, const bool force_find_fir_coefs)
        : spectral_transform(std::string("oemd"), _levels), levels(_levels), stretch_coef(_stretch_coef)
{
    if (p_oemd_coef) {
        LOG4_TRACE("Filter coefficients already initialized.");
        return;
    }
    auto it_coefs = oemd_available_levels.find(_levels);
    if (it_coefs == oemd_available_levels.end()) {
        LOG4_DEBUG("Coefs for " << levels << " not found.");
        p_oemd_coef = std::make_shared<oemd_coefficients>(levels, std::vector<size_t>(levels - 1, DEFAULT_SIFTINGS), std::vector<std::vector<double>>(levels - 1));
        fir_coefs_initialized = false;
    } else {
        LOG4_DEBUG("Initializing OEMD coefficients from loaded.");
        p_oemd_coef = std::make_shared<oemd_coefficients>(it_coefs->second);
        if (force_find_fir_coefs) {
            fir_coefs_initialized = false;
        } else {
            fir_coefs_initialized = !p_oemd_coef->siftings.empty() && *std::min_element(p_oemd_coef->siftings.begin(), p_oemd_coef->siftings.end()) && !p_oemd_coef->masks.empty();
            for (const auto &m: p_oemd_coef->masks) fir_coefs_initialized &= !m.empty();
        }
    }
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

#ifdef OEMD_TAIL_COMPUTE
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
#endif
}

#endif


#if 0
static int
oemd_fast_mean_compute(const std::vector<double> &x, const double *masks, int mask_size, std::vector<double> &rx)
{
    int len = x.size();
    fftw_plan forward_plan;
    fftw_plan backward_plan;
    fftw_complex *output = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * (len / 2 + 1));
    fftw_complex *mult = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * (len / 2 + 1));
    forward_plan = fftw_plan_dft_r2c_1d(len, &rx[0], output, FFTW_ESTIMATE);
    backward_plan = fftw_plan_dft_c2r_1d(len, output, &rx[0], FFTW_ESTIMATE);
    for (int i = 0; i < len; i++) {
        rx[i] = i < mask_size ? masks[mask_size - 1 - i] : 0.;
    }
    fftw_execute(forward_plan);
    for (int i = 0; i < len / 2 + 1; i++) {
        mult[i][0] = output[i][0];
        mult[i][1] = output[i][1];
    }
    for (int i = 0; i < len; i++) {
        rx[i] = x[i];
    }
    fftw_execute(forward_plan);
    for (int i = 0; i < len / 2 + 1; i++) {
        double o_real = output[i][0] * mult[i][0] - output[i][1] * mult[i][1];
        double o_imag = output[i][0] * mult[i][1] + output[i][1] * mult[i][0];
        output[i][0] = o_real;
        output[i][1] = o_imag;
    }
    fftw_execute(backward_plan);
    for (int i = 0; i < mask_size - 1; i++) {
        rx[i] = 0.;
        double sum = 0.;
        for (int j = 0; j <= i; j++) {
            rx[i] += masks[mask_size - 1 - i + j] * x[j];
            sum += masks[mask_size - 1 - i + j];
        }
        rx[i] = rx[i] / sum;
    }
    for (int i = mask_size - 1; i < len; i++) {
        rx[i] = rx[i] / len;
    }
    fftw_free(output);
    fftw_free(mult);
    fftw_destroy_plan(forward_plan);
    fftw_destroy_plan(backward_plan);
    return 0;
}

oemd_fptr online_emd::select_method(size_t level, size_t input_length) const
{
    return &oemd_mean_compute;
    /*
    if (level < 2) {
        return &oemd_mean_compute;
    } else {
        if (input_length < p_oemd_coef->masks[level].size()){
            return &oemd_mean_compute;
        }else{
            return &oemd_fast_mean_compute;
        }
    }
	*/
}
#endif


void
online_emd::transform(
        const std::vector<double> &input,
        std::vector<std::vector<double>> &decon,
        const size_t padding = 0)
{
    if (!fir_coefs_initialized) LOG4_THROW("FIR coefs not initialized!");
    if (input.size() < get_residuals_length())
        LOG4_ERROR("Input size " << input.size() << " too short, needed " << get_residuals_length());
    else
        LOG4_DEBUG("Input size " << input.size());
#ifdef OEMD_CUDA
#ifdef CUDA_OEMD_MULTIGPU
    std::vector<std::shared_ptr<common::gpu_context>> gpu_ctxs;
    while (common::gpu_handler::get_instance().get_free_gpus())
        gpu_ctxs.emplace_back(std::make_shared<common::gpu_context>());
#else
    common::gpu_context ctx;
#endif

    svr::cuoemd::transform(
            input, decon,
#ifdef CUDA_OEMD_MULTIGPU
            gpu_ctxs,
#else
            ctx.phy_id(),
#endif
            p_oemd_coef->siftings,
            p_oemd_coef->masks,
            stretch_coef,
            levels);
#else // Use CPU, slow!
    init_oemd();

    size_t input_length = input.size();
    LOG4_TRACE("Size of input is " << input_length);
    std::vector<double> remainder = input;
    std::vector<double> rx = remainder;
    std::vector<double> rx2(input_length, 0.);
    if (decon.size() != input_length) decon.resize(input_length);
    for (size_t l = 0; l < levels - 1; l++) {
        //oemd_fptr mean_compute = select_method(l, input_length);
        for (size_t j = 0; j < p_oemd_coef->siftings[l]; ++j) {
            oemd_mean_compute(rx, p_oemd_coef->masks[l], rx2);
            for (size_t t = 0; t < input_length; ++t) {
                rx[t] -= rx2[t];
            }
        }
        __omp_tpfor(size_t, t, 0, input_length,
            if (decon[t].size() != levels) decon[t].resize(levels, 0.);
            //result[k][l+1] = rx[k];
            decon[t][levels - l - 1] = rx[t];
            remainder[t] -= rx[t];
            rx[t] = remainder[t];
        )
    }
    for (size_t t = 0; t < input_length; ++t) {
        decon[t][0] = rx[t];
    }

#if 0 // Debugging
    {
        std::stringstream ss;
        for (size_t i = 0; i < decon.size(); ++i) {
            for (size_t j = 0; j < decon[i].size(); ++j)
                ss << ", " << decon[i][j];
            ss << std::endl;
        }
        LOG4_FILE("oemd_decon.txt", ss.str().c_str());
        exit(0);
    }
#endif
#endif
}


void online_emd::inverse_transform(
        const std::vector<double> &decon,
        std::vector<double> &recon,
        const size_t padding = 0) const
{
    const size_t input_size = decon.size() / levels;
    recon = std::vector<double>(input_size, 0.);

    //by rows  --> decon_column_values[row_index + level_ix * frame_size]
    __omp_pfor_i(0, input_size,
        double recon_i = 0;
        for (size_t j = 0; j < levels; ++j)
            recon_i += decon[i + j * input_size];
        recon[i] = recon_i;
    )
}


size_t online_emd::get_residuals_length(const oemd_coefficients &coefs, const double stretch_coef)
{
    //return coefs.masks.back().size(); // Minimum
    //return coefs.masks.back().size() * p_oemd_coef->siftings.back(); // Tradeoff
    //return 6 * svr::common::next_power_of_two(coefs.masks.back().size() * coefs.siftings.back()); // Assured time invariance first
    return common::max_size(coefs.masks) * stretch_coef * common::max(coefs.siftings) * 7; /* time invariance 1: * oemd_level_coefs->second.siftings.back() * 7 */ //
}


size_t online_emd::get_residuals_length(const size_t levels)
{
    LOG4_WARN("Getting default masks residuals length!");
    const auto oemd_level_coefs = oemd_available_levels.find(levels);
    if (levels < 2 or oemd_level_coefs == oemd_available_levels.end())
        LOG4_THROW("Unsupported number of levels " << levels << " requested for OEMD.");
    return get_residuals_length(oemd_level_coefs->second, OEMD_STRETCH_COEF);
}


size_t online_emd::get_residuals_length()
{
#ifdef OEMD_CUDA
    return get_residuals_length(*p_oemd_coef, stretch_coef);
#else
    if (!p_oemd_coef) init_oemd();
    return p_oemd_coef->masks.back().size();
#endif
}


}
