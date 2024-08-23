/*
 * spectral_transform.hpp
 *
 *  Abstracts a transformation of an N-sized vector, to an [N-sized x nLevels-sized] matrix, which has high stationarity and reconstruction accuracy.
 *
 *  Created on: Apr 6, 2017
 *      Author: Boyko Perfanov
 */

#pragma once

#include <deque>
#include <vector>
#include <string>
#include <memory>
#include "wavelet_coefs.hpp"

namespace svr {

class spectral_transform {
protected:
    std::string transformation_name_;
    size_t wavelet_order_;
    size_t levels_;
    size_t filter_order_;

public:
    static size_t modwt_levels_to_frame_length(const size_t modwt_levels, const size_t wavelet_order);

    static size_t modwt_residuals_length(const size_t modwt_levels);

    static size_t modwt_filter_order_from(const std::string& filter_order);

    static std::unique_ptr<spectral_transform> create(const std::string& transformation_name, const size_t levels, const double stretch_coef = 1, const bool force_find_oemd_coefs = false);

    spectral_transform(const std::string& transformation_name, const size_t& levels);
    virtual ~spectral_transform();

    // Returns a row-major (rows * levels) matrix.
    // TODO Make it return a vmatrix instead of vector of a vector. Same for inverse.
    virtual void transform(
            const std::vector<double> &input,
            std::vector<std::vector<double> > &decon,
            const size_t padding = 0) = 0;

    virtual void inverse_transform(
            const std::vector<double> &decon,
            std::vector<double> &recon,
            const size_t padding = 0) const = 0;

    static size_t get_min_frame_length(const size_t, const size_t, const size_t, const std::string &);

    void summary() const;

    // TODO: extract wavelet-specific symbols to the corresponding class.
    class wavelet
    {
    protected:
        std::string name_;
        size_t order_;
        double* dec_hi_double_;  /* highpass decomposition */
        double* dec_lo_double_;  /* lowpass decomposition */
        double* rec_hi_double_;  /* highpass reconstruction */
        double* rec_lo_double_;  /* lowpass reconstruction */
        size_t dec_len_;   /* length of decomposition filter */
        size_t rec_len_;   /* length of reconstruction filter */

        int vanishing_moments_psi;
        int vanishing_moments_phi;

        static wavelet blank_discrete_wavelet(int filters_length);
        wavelet() { };

    public:
        explicit wavelet(std::string name);
        ~wavelet();

        inline size_t rec_len() const { return rec_len_; };
        inline size_t dec_len() const { return dec_len_; };
        inline double* dec_hi_double() const { return dec_hi_double_; };
        inline double* dec_lo_double() const { return dec_lo_double_; };
        inline double* rec_hi_double() const { return rec_hi_double_; };
        inline double* rec_lo_double() const { return rec_lo_double_; };
        std::string name() const { return name_; };
        size_t order() const { return order_; };
        void print() const;
    };
};

}