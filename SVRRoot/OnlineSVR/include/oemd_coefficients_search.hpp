//
// Created by zarko on 10/3/22.
//

#ifndef SVR_OEMD_COEFFICIENTS_SEARCH_HPP
#define SVR_OEMD_COEFFICIENTS_SEARCH_HPP

#include "common/gpu_handler.hpp"

#include <vector>
#include <cufft.h>
#include "common/gpu_handler.hpp"
#include "oemd_coefficients.hpp"
#include "model/DataRow.hpp"

namespace svr {
namespace oemd_search {

constexpr unsigned FIREFLY_PARTICLES = 400;
constexpr unsigned FIREFLY_ITERATIONS = 70;

constexpr double C_lambda1 = .30;
constexpr double C_lambda2 = .33;
constexpr double C_smooth_factor = 1000.;
constexpr double C_b_factor = 1000;
constexpr size_t C_mask_expander = 2;
constexpr size_t C_parallelism = 64;
constexpr size_t C_NUM_TRIES = 100000;
constexpr size_t C_filter_count = 100;
constexpr size_t fir_search_start_size = 500;
constexpr size_t fir_search_end_size = 20000; // 18k is the maximum for 16GB video card that won't make OSQP crash
constexpr size_t fir_search_window_end = fir_search_end_size * 20;
constexpr double allowed_eps_up = .1;
constexpr double norm_thresh = 1. + allowed_eps_up;

class oemd_coefficients_search
{
    sem_t sem_fft;
    std::mutex current_mask_mutex;
    std::atomic<double> best_quality = 1;
    std::atomic<double> best_result = std::numeric_limits<double>::max();
    std::mutex best_mtx;
    size_t max_gpus;
    std::deque<size_t> gpuids;

    oemd_coefficients_search();
    void fft_acquire();
    void fft_release();

    std::tuple<double, double, double, double>
    sift_the_mask(
            const size_t mask_size,
            const size_t siftings,
            const double *d_mask,
            const cufftHandle plan_sift_forward,
            const cufftHandle plan_sift_backward,
            const double *d_expanded_mask,
            const cufftDoubleComplex *d_expanded_mask_fft,
            const double *d_global_sift_matrix_ptr,
            const size_t gpu_id);

    static double do_quality(const std::vector<cufftDoubleComplex> &h_mask_fft, const size_t siftings);

    static int do_filter(const std::vector<cufftDoubleComplex> &h_mask_fft);

    void smoothen_mask(std::vector<double> &mask, common::t_drand48_data_ptr buffer);

    std::vector<double>
    fill_auto_matrix(const size_t M, const size_t siftings, const size_t N, const double *x);

    void
    expand_the_mask(const size_t mask_size, const size_t input_size, const double *dev_mask, double *dev_expanded_mask);

    void
    transform(double *d_values, double *h_mask, const size_t input_size, const size_t mask_size, const size_t siftings, double *d_temp, const size_t gpu_id);

    //int do_osqp(const size_t mask_size, std::vector<double> &good_mask, const size_t input_size, const double *x, const int gpu_id);
    static void fix_mask(std::vector<double> &mask);

    void
    create_random_mask(
            const size_t position, double step, const size_t mask_size, std::vector<double> &mask, const double *start_mask,
            common::t_drand48_data_ptr buffer, cufftHandle plan_mask_forward, cufftHandle plan_mask_backward, const size_t gpu_id);

    static void do_mult(const size_t m, const size_t n, const std::vector<double> &A_x, std::vector<double> &H);

    static void do_diff_mult(const size_t M, const size_t N, const std::vector<double> &diff, std::vector<double> &Bmatrix);

    static void prep_x_matrix(const size_t M, const size_t N, const double *x, std::vector<double> &Bmatrix, std::vector<double> &Fmatrix);

    void
    save_mask(
            const std::vector<double> &mask,
            const std::string &queue_name,
            const size_t level,
            const size_t levels);

    double
    evaluate_mask(
            const size_t siftings,
            const std::vector<double> &h_mask, const std::vector<double> &h_workspace,
            const std::vector<cufftDoubleComplex> &values_fft, const size_t val_start,
            const cufftHandle plan_expanded_forward, const cufftHandle plan_expanded_backward,
            const cufftHandle plan_mask_forward, const cufftHandle plan_sift_forward, const cufftHandle plan_sift_backward,
            const std::vector<double> &global_sift_matrix, const size_t current_level, const size_t gpu_id);

    double
    find_good_mask_ffly(
            const size_t siftings, const size_t valid_start_index,
            const std::vector<double> &h_workspace, const std::vector<cufftDoubleComplex> &h_workspace_fft, std::vector<double> &h_mask,
            std::deque<cufftHandle> &plan_full_forward, std::deque<cufftHandle> &plan_full_backward, std::deque<cufftHandle> &plan_mask_forward,
            std::deque<cufftHandle> &plan_sift_forward, std::deque<cufftHandle> &plan_sift_backward,
            const std::vector<double> &global_sift_matrix, const size_t current_level);

public:
    static void
    gauss_smoothen_mask(
            const size_t mask_size, std::vector<double> &mask, common::t_drand48_data_ptr buffer,
            cufftHandle plan_mask_forward, cufftHandle plan_mask_backward, const size_t gpu_id);

    static void
    prepare_masks(
            std::deque<std::vector<double>> &masks,
            std::deque<size_t> &siftings,
            const size_t levels);


    void
    optimize_levels(
            const datamodel::datarow_range &input,
            const std::vector<double> &tail,
            std::deque<std::vector<double>> &masks,
            std::deque<size_t> &siftings,
            const size_t window_start,
            const size_t window_end,
            const std::string &queue_name);

    static oemd_coefficients_search &get() {
        static oemd_coefficients_search instance;
        return instance;
    }
};

}
}

#endif //SVR_OEMD_COEFFICIENTS_SEARCH_HPP
