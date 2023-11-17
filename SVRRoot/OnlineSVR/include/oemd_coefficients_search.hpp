//
// Created by zarko on 10/3/22.
//

#ifndef SVR_OEMD_COEFFICIENTS_SEARCH_HPP
#define SVR_OEMD_COEFFICIENTS_SEARCH_HPP

#define GENERATE_SIFTINGS 1
#define USE_FIREFLY_SEARCH
#ifdef USE_FIREFLY_SEARCH
#define FIREFLY_PARTICLES 400
#define FIREFLY_ITERATIONS 50
#endif

#include <vector>
#include <cufft.h>
#include "oemd_coefficients.hpp"

namespace svr::oemd_search {


typedef struct drand48_data t_drand48_data;
typedef t_drand48_data *t_drand48_data_ptr;


extern const double C_lambda1;
extern const double C_lambda2;
extern const double C_smooth_factor;
extern const double C_b_factor;
extern const size_t C_mask_expander;
extern const size_t C_parallelism;
extern std::atomic<double> best_quality;


double do_quality(const cufftDoubleComplex *const h_mask_fft, const size_t fft_size, const size_t siftings);
//int do_osqp(const size_t mask_size, std::vector<double> &good_mask, const size_t input_size, const double *x, const int gpu_id);
void fix_mask(const size_t mask_size, double *mask);

void
create_random_mask(
        const size_t position, double step, const size_t mask_size, std::vector<double> &mask, const double *start_mask,
        t_drand48_data_ptr buffer, cufftHandle plan_mask_forward, cufftHandle plan_mask_backward, const size_t gpu_id);

void
first_matrix(
        const size_t mask_size, const size_t fft_size, const double lambda, const double gamma,
        const double smooth_factor, std::vector<double> &A_x, std::vector<double> &F_x);


void
oemd_coefs_search(
        const std::vector<double> &val,
        const size_t start_size,
        const size_t end_size,
        const size_t window_start,
        const size_t window_end,
        const size_t levels,
        std::vector<std::vector<double>> &mask,
        std::vector<size_t> &siftings);

#if 1
void
optimize_levels(
        const std::vector<double> &val,
        std::vector<std::vector<double>> &masks,
        std::vector<size_t> &siftings,
        const size_t window_start,
        const size_t window_end);
#else
void
optimize_levels(
        const std::vector<double> &val,
        std::vector<std::vector<double>> &masks,
        const std::vector<size_t> &siftings,
        const size_t window_start,
        const size_t window_end);
#endif
}

#endif //SVR_OEMD_COEFFICIENTS_SEARCH_HPP
