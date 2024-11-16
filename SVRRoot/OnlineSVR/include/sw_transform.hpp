/*
 * Wavelet.hpp
 *
 *  Wrapper of wavelib. https://github.com/rafat/wavelib
 *
 *  Provides convenient way to make forward and inverse wavelet transformations
 *  hiding wavelib's internal objects.
 *
 *  Created on: Jun 17, 2016
 *      Author: sgeorgiev
 */

#pragma once

#include <vector>
#include <string>
#include "common/compatibility.hpp"
#include "spectral_transform.hpp"

namespace svr {

class sw_transform : public spectral_transform
{
public:
    enum class padding_strategy
    {
        average = 1,
        constant_value_at_end = 2,
        first_order_simple_approximation = 3,
        antisymmetric = 4,
        symmetric = 5
    };

    // TODO: delete
    std::string unnecessary_output_suffix;

    sw_transform(
            const std::string &wavelet_name,
            const size_t levels,
            sw_transform::padding_strategy swt_padding_strategy = padding_strategy::constant_value_at_end);

    virtual ~sw_transform();

    void
    transform(
            const std::vector<double> &input,
            std::vector<std::vector<double> > &decon,
            const size_t padding = 0) override;

    void
    inverse_transform(
            const std::vector<double> &decon,
            std::vector<double> &recon,
            const size_t padding = 0) const override;

    virtual size_t get_minimal_input_length(const size_t decremental_offset, const size_t lag_count) const;


protected:
    const spectral_transform::wavelet wavelet_;
    const padding_strategy swt_padding_strategy_;

    void
    fill_swt_data_structure(
            std::vector<double> input, std::vector<double> &padded_data_container,
            const size_t padding,
            size_t &out_coercion_padding) const;

    void
    fill_iswt_data_structure(
            const std::vector<double> &frame,
            std::vector<std::pair<std::vector<double>, std::vector<double>>> &deconstructed,
            const size_t padding,
            const size_t coercion_padding) const;

    /* SWT decomposition at given level */
    void
    double_swt_a_cls(
            CPTRd  input,
            const size_t input_len,
            const spectral_transform::wavelet &wavelet,
            double *const output,
            const size_t output_len,
            const size_t level) const;

    void
    double_swt_d_cls(
            CPTRd  input,
            const size_t input_len,
            const spectral_transform::wavelet &wavelet,
            double *const output,
            const size_t output_len,
            const size_t level) const;

    void
    double_swt_cls(
            CPTRd  input,
            const size_t input_len,
            CPTRd  filter,
            const size_t filter_len,
            double *const output,
            const size_t output_len,
            const size_t level) const;

    void
    double_downsampling_convolution_periodization_cls(
            CPTRd  input,
            const size_t N,
            CPTRd  filter,
            const size_t F,
            double *const output,
            const size_t step,
            const size_t fstep) const;

    void
    double_idwt_cls(
            CPTRd  coeffs_a,
            const size_t coeffs_a_len,
            CPTRd  coeffs_d,
            const size_t coeffs_d_len,
            double *const output,
            const size_t output_len,
            const spectral_transform::wavelet &wavelet) const;

    void
    double_upsampling_convolution_valid_sf_cls(
            CPTRd  input,
            const size_t N,
            CPTRd  filter,
            const size_t F,
            double *const output, const size_t O) const;

    void
    double_upsampling_convolution_valid_sf_periodization_cls(
            CPTRd  input,
            const size_t N,
            CPTRd  filter,
            const size_t F,
            double *const output,
            const size_t O) const;

    std::vector<double> idwt_single(
            std::vector<double> cA,
            std::vector<double> cD,
            const spectral_transform::wavelet &wavelet) const;

    static unsigned char swt_max_level(const size_t input_len);
};

}
