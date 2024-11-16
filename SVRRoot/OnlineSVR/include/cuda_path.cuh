//
// Created by zarko on 12/22/22.
//

#ifndef SVR_CUDA_PATH_CUH
#define SVR_CUDA_PATH_CUH

#include "common/compatibility.hpp"
#include "cuqrsolve.cuh"

#if 0
#define DIST(x) pow(abs(x), tau)
#else
constexpr int8_t C_dist_pow = 4;
#define DIST(x) pow((x), C_dist_pow)
#endif

namespace svr::kernel::path {

constexpr double C_kernel_path_tau = .25;

__global__  void
G_kernel_xy(const uint32_t X_cols, const uint32_t Y_cols, const uint32_t rows, const uint32_t lag, const uint32_t dim, const uint32_t lag_TILE_WIDTH, const double lambda,
            const double tau, const double *const X, const double *const Y, double *Z);

__global__  void
G_kernel_xy(const uint32_t X_cols, const uint32_t Y_cols, const uint32_t rows, const uint32_t lag, const uint32_t dim, const uint32_t lag_TILE_WIDTH, const double lambda,
            const double tau, const double gamma, const double *const X, const double *const Y, double *Z);

void cu_distances_xx(const uint32_t cols, const uint32_t rows, const uint32_t lag, const double lambda, const double tau, CRPTR(double) X, RPTR(double) Z);

void
cu_distances_xy(const uint32_t X_cols, const uint32_t Xy_cols, const uint32_t rows, const uint32_t lag, const double lambda, const double tau, CRPTRd X, CRPTRd Xy, RPTR(double) Z);

void cu_kernel_xx(const uint32_t cols, const uint32_t rows, const uint32_t lag, const double lambda, const double tau, const double gamma, CRPTRd X, RPTR(double) Z);

void cu_kernel_xy(const uint32_t X_cols, const uint32_t Xy_cols, const uint32_t rows, const uint32_t lag, const double lambda, const double gamma,
                  CPTRd X, CPTRd Xy, RPTR(double) Z);

void cu_threshold(RPTR(double) v, const uint32_t n, const cudaStream_t custream);

inline __host__ __device__ double calc_g(const double n, const double Z_mm, const double L_mm)
{
    // return n * Z_mm / (n - L_mm);
    return n * Z_mm / L_mm;
}

/* In order to get min gamma, provide min Z and max L, and vice versa */
inline __host__ __device__ double calc_qgamma(const double Z_mean, const double Z_maxmin, const double L_mean, const double L_minmax, const double train_len)
{
    constexpr double q_1 = solvers::C_gamma_variance + 1;
    const auto Z_mm = (solvers::C_gamma_variance * Z_mean + Z_maxmin) / q_1;
    const auto L_mm = (solvers::C_gamma_variance * L_mean + L_minmax) / q_1;
    const auto g = kernel::path::calc_g(train_len, Z_mm, L_mm);
    // printf("calc_qgamma: Zmm %f, Lmm %f, n %f, gamma %f, quantile %f\n", Z_mm, L_mm, train_len, g, solvers::C_gamma_variance);
    return g;
}


}

#endif
