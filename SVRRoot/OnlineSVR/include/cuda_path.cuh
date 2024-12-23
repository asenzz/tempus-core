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
constexpr uint8_t C_dist_pow = 4;
#define DIST(x) pow((x), C_dist_pow)
#endif

namespace svr::kernel::path {

constexpr double C_kernel_path_tau = .25;

std::pair<double, double> cu_kernel_xy(const uint32_t X_cols, const uint32_t Y_cols, const uint32_t rows, const uint16_t lag, const uint16_t dim, const uint16_t lag_TILE_WIDTH, const double lambda,
            const double tau, const double gamma, CRPTRd X, CRPTRd Y, RPTR(double) Z, const cudaStream_t custream);

std::pair<double, double> cu_distances_xy(const uint32_t X_cols, const uint32_t Y_cols, const uint32_t rows, const uint16_t lag, const uint16_t dim, const uint16_t lag_TILE_WIDTH, const double lambda,
                                          const double tau, CRPTRd X, CRPTRd Y, RPTR(double) Z, const cudaStream_t custream);

std::pair<double, double> distances_xx(const uint32_t cols, const uint32_t rows, const uint16_t lag, const double lambda, const double tau, CRPTR(double) X, RPTR(double) Z);

void distances_xy(const uint32_t X_cols, const uint32_t Xy_cols, const uint32_t rows, const uint16_t lag, const double lambda, const double tau, const double min_Z, const double max_Z, CRPTRd X, CRPTRd Xy, RPTR(double) Z);

std::pair<double, double> kernel_xx(const uint32_t cols, const uint32_t rows, const uint16_t lag, const double gamma, const double lambda, const double tau, const double *const X, double *Z);

void kernel_xy(
    const uint32_t X_cols, const uint32_t Xy_cols, const uint32_t rows, const uint16_t lag, const double tau, const double lambda, const double gamma, const double min_Z, const double max_Z,
      const double *const X, const double *const Xy, double *Z);

// Distances to kernel value
__forceinline__ __host__ __device__ double z2k(const double z, const double gamma)
{
    // return Z / gamma;
    return (1. - z) / gamma;
}

__forceinline__ __host__ __device__ double calc_g(const double n, const double Z_mm, const double L_mm)
{
    // return n * Z_mm / L_mm;
    return (n - n * Z_mm) / L_mm;
}

/* In order to get min gamma, provide min Z and max L, and vice versa */
__forceinline__ __host__ __device__ double calc_qgamma(const double Z_mean, const double Z_maxmin, const double L_mean, const double L_minmax, const double train_len)
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
