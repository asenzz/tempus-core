//
// Created by zarko on 10/3/22.
//

#ifndef SVR_OEMD_COEFFICIENTS_SEARCH_HPP
#define SVR_OEMD_COEFFICIENTS_SEARCH_HPP

#include <boost/interprocess/sync/named_semaphore.hpp>
#include <vector>
#include <cufft.h>
#include <oneapi/tbb/mutex.h>
#include "oemd_coefficients.hpp"
#include "common/gpu_handler.hpp"
#include "model/DataRow.hpp"
#include "align_features.cuh"

namespace svr {
namespace oemd {

class oemd_coefficients_search {

    constexpr static uint8_t C_column_interleave = 8;
    constexpr static uint8_t C_quantisation_skipdiv = 3;
    constexpr static uint16_t particles = 10;
    constexpr static uint16_t iterations = 40;
    constexpr static double lambda1 = .30;
    constexpr static double lambda2 = .33;
    constexpr static double allowed_eps_up = .1;
    constexpr static double norm_thresh = 1. + allowed_eps_up;
    constexpr static uint8_t multistep = 1;

    const bpt::time_duration resolution;
    const double sample_rate;
    const uint8_t levels;

    static std::tuple<double, double, double, double> sift_the_mask(
            const uint32_t mask_size, const uint16_t siftings, CPTRd d_mask, const cufftHandle plan_sift_forward, const cufftHandle plan_sift_backward,
            CPTRd d_expanded_mask, const cufftDoubleComplex *d_expanded_mask_fft, CPTRd d_global_sift_matrix_ptr, const uint16_t gpu_id);

    void smoothen_mask(std::vector<double> &mask, common::t_drand48_data_ptr buffer);

    std::vector<double>
    fill_auto_matrix(const uint32_t M, const uint16_t siftings, const uint32_t N, CPTRd x);

    void transform(double *d_values, CPTRd d_mask, const uint32_t input_len, const uint32_t mask_size, const uint16_t siftings, double *d_temp, const cudaStream_t custream) const;

    // 18k mask len is the maximum for 16GB video card that won't make OSQP crash
    //int do_osqp(const uint32_t mask_size, std::vector<double> &good_mask, const uint32_t input_size, CPTRd x, const int gpu_id);

    void
    create_random_mask(
            const uint32_t position, double step, const uint32_t mask_size, std::vector<double> &mask,
            CPTRd start_mask,
            common::t_drand48_data_ptr buffer, cufftHandle plan_mask_forward, cufftHandle plan_mask_backward,
            const uint16_t gpu_id);

    // static void do_mult(const uint32_t m, const uint32_t n, const std::vector<double> &A_x, std::vector<double> &H);

    static void do_diff_mult(const uint32_t M, const uint32_t N, const std::vector<double> &diff,
                             std::vector<double> &Bmatrix);

    static void prep_x_matrix(const uint32_t M, const uint32_t N, CPTRd x, std::vector<double> &Bmatrix,
                              std::vector<double> &Fmatrix);

    static void save_mask(const std::vector<double> &mask, const std::string &queue_name, const uint16_t level, const uint16_t levels);

public:
    constexpr static double C_smooth_factor = 1000;
    constexpr static uint32_t C_fir_max_len = 10'000;
    constexpr static uint32_t C_sifted_fir_max_len = C_fir_max_len * oemd_coefficients::C_default_siftings;
    const double stretch_coef = oemd_coefficients::C_oemd_stretch_coef;
    const uint32_t max_row_len;
    const uint32_t label_len;
    const uint32_t fir_validation_window;

    explicit oemd_coefficients_search(const uint16_t levels, const bpt::time_duration &resolution, const uint32_t label_len);

    double evaluate_mask(const double att, const double fp, const double fs, const std::span<double> &workspace, const uint8_t siftings, const uint32_t prev_masks_len,
                         const double meanabs_input, const std::vector<uint32_t> &times, const std::vector<t_label_ix> &label_ixs,
                         const std::vector<t_feat_params> &feat_params) const;

    static double do_quality(const std::vector<cufftDoubleComplex> &h_mask_fft, const uint16_t siftings);

    static double cu_quality(const cufftDoubleComplex *mask_fft, const uint32_t mask_size, const uint16_t siftings,
                             const cudaStream_t custream);

    static int do_filter(const std::vector<cufftDoubleComplex> &h_mask_fft);

    static void gauss_smoothen_mask(const uint32_t mask_size, std::vector<double> &mask, common::t_drand48_data_ptr buffer,
                        cufftHandle plan_mask_forward, cufftHandle plan_mask_backward, const uint16_t gpu_id);

    static void prepare_masks(std::deque<std::vector<double> > &masks, std::deque<uint16_t> &siftings, const uint16_t levels);

    void run(
            const datamodel::datarow_crange &input,
            const std::vector<double> &tail,
            std::deque<std::vector<double>> &masks,
            std::deque<uint16_t> &siftings,
            const uint32_t window_start,
            const uint32_t window_end,
            const std::string &queue_name,
            const uint16_t in_colix,
            const datamodel::t_iqscaler &scaler) const;

    void sift(const uint16_t siftings, const uint32_t full_input_len, const uint32_t mask_len, cudaStream_t const custream, CPTRd d_mask, double *const d_rx,
              double *const d_rx2) const noexcept;

    static double compute_spectral_entropy_cufft(double *d_signal, uint32_t N, const cudaStream_t custream);

    double dominant_frequency(const std::span<double> &input, const double percentile_greatest_peak, const cudaStream_t custream) const;
};

__global__ void G_quantise_features_quick(
        CRPTRd d_imf,
        RPTR(double) d_features,
        const uint32_t rows_q,
        const uint32_t cols,
        const uint32_t quantisation,
        const uint32_t cols_q,
        const uint32_t cols_rows_q,
        const uint32_t interleave,
        const uint32_t start_col_q,
        const uint32_t start_row,
        CRPTR(t_label_ix) d_label_ixs);

}
}

#endif //SVR_OEMD_COEFFICIENTS_SEARCH_HPP
