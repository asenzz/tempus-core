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

namespace svr {
namespace oemd {

class oemd_coefficients_search {
    constexpr static unsigned particles = 100;
    constexpr static unsigned iterations = 100;
    constexpr static double lambda1 = .30;
    constexpr static double lambda2 = .33;
    constexpr static double b_factor = 1000;
    constexpr static unsigned mask_expander = 2;
    constexpr static unsigned parallelism = 24; // TODO use gpu handler instead
    constexpr static double allowed_eps_up = .1;
    constexpr static double norm_thresh = 1. + allowed_eps_up;

    const double levels;
    double best_quality = 1;
    double best_result = common::C_bad_validation;
    unsigned max_gpus;
    std::deque<unsigned> gpuids;
    const double stretch_coef = oemd_coefficients::C_oemd_stretch_coef;

    static std::tuple<double, double, double, double> sift_the_mask(
            const unsigned mask_size, const unsigned siftings, const double *d_mask, const cufftHandle plan_sift_forward, const cufftHandle plan_sift_backward,
            const double *d_expanded_mask, const cufftDoubleComplex *d_expanded_mask_fft, const double *d_global_sift_matrix_ptr, const unsigned gpu_id);

    void smoothen_mask(std::vector<double> &mask, common::t_drand48_data_ptr buffer);

    std::vector<double>
    fill_auto_matrix(const unsigned M, const unsigned siftings, const unsigned N, const double *x);

    void
    transform(double *d_values, CPTR(double) d_mask, const unsigned input_len, const unsigned mask_size,
              const unsigned siftings, double *d_temp, const cudaStream_t custream) const;

    //int do_osqp(const unsigned mask_size, std::vector<double> &good_mask, const unsigned input_size, const double *x, const int gpu_id);

    static void fix_mask(const arma::vec &in_mask, std::vector<double> &out_mask);

    void
    create_random_mask(
            const unsigned position, double step, const unsigned mask_size, std::vector<double> &mask,
            const double *start_mask,
            common::t_drand48_data_ptr buffer, cufftHandle plan_mask_forward, cufftHandle plan_mask_backward,
            const unsigned gpu_id);

    // static void do_mult(const unsigned m, const unsigned n, const std::vector<double> &A_x, std::vector<double> &H);

    static void do_diff_mult(const unsigned M, const unsigned N, const std::vector<double> &diff,
                             std::vector<double> &Bmatrix);

    static void prep_x_matrix(const unsigned M, const unsigned N, const double *x, std::vector<double> &Bmatrix,
                              std::vector<double> &Fmatrix);

    static void save_mask(
            const std::vector<double> &mask,
            const std::string &queue_name,
            const unsigned level,
            const unsigned levels);

    double find_good_mask(const unsigned siftings, const unsigned valid_start_ix, const std::vector<double> &workspace, std::vector<double> &mask, const unsigned level) const;

public:

    constexpr static double C_smooth_factor = 1000;
    constexpr static unsigned C_fir_mask_start_len = 500;
    constexpr static unsigned C_fir_mask_end_len = 20'000; // 18k is the maximum for 16GB video card that won't make OSQP crash
    constexpr static unsigned C_fir_validation_window = C_test_len * 3600;

    double evaluate_mask(
            const std::vector<double> &h_mask, const std::vector<double> &h_workspace, const size_t validate_start_ix, const size_t validation_len, const unsigned siftings,
            const unsigned current_level, const std::deque<unsigned> &head_tail_sizes, const double meanabs_input, const unsigned gpu_id) const;

    static double do_quality(const std::vector<cufftDoubleComplex> &h_mask_fft, const unsigned siftings);

    static double cu_quality(const cufftDoubleComplex *mask_fft, const unsigned mask_size, const unsigned siftings,
                             const cudaStream_t custream);

    static int do_filter(const std::vector<cufftDoubleComplex> &h_mask_fft);

    static void
    gauss_smoothen_mask(const unsigned mask_size, std::vector<double> &mask, common::t_drand48_data_ptr buffer,
                        cufftHandle plan_mask_forward, cufftHandle plan_mask_backward, const unsigned gpu_id);

    static void prepare_masks(std::deque<std::vector<double> > &masks, std::deque<unsigned> &siftings, const unsigned levels);

    void optimize_levels(
            const datamodel::datarow_range &input, const std::vector<double> &tail, std::deque<std::vector<double> > &masks, std::deque<unsigned> &siftings,
            const unsigned window_start, const unsigned window_end, const std::string &queue_name) const;

    explicit oemd_coefficients_search(const unsigned levels);

    void sift(const unsigned siftings, const unsigned full_input_len, const unsigned mask_len, cudaStream_t const custream, CPTR(double) d_mask,
              double *const d_rx, double *const d_rx2) const noexcept;

    static double compute_spectral_entropy_cufft(const double *d_signal, const unsigned N, const cudaStream_t custream);
};
}
}

#endif //SVR_OEMD_COEFFICIENTS_SEARCH_HPP
