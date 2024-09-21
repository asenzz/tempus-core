//
// Created by zarko on 12/22/22.
//

#ifndef SVR_CUDA_PATH_HPP
#define SVR_CUDA_PATH_HPP

#include "common/compatibility.hpp"

#define DISTPOW 4 // Should be even number
#define DIST(x) pow((x), DISTPOW)

namespace svr::kernel::path {

__global__  void
G_kernel_xy(const uint32_t X_cols, const uint32_t Y_cols, const uint32_t rows, const uint32_t lag, const uint32_t dim, const uint32_t lag_TILE_WIDTH, const double lambda,
            CRPTR(double) X, CRPTR(double) Y, RPTR(double) Z);

__global__  void
G_kernel_xy(const uint32_t X_cols, const uint32_t Y_cols, const uint32_t rows, const uint32_t lag, const uint32_t dim, const uint32_t lag_TILE_WIDTH,
            const double lambda, const double gamma, CPTR(double) X, CPTR(double) Y, double *Z);

__global__  void
G_kernel_xy(const uint32_t X_cols, const uint32_t Y_cols, const uint32_t lag, const unsigned lag_TILE_WIDTH, const double lambda,
            CRPTR(double) X, CRPTR(double) Y, RPTR(double) Z);

void do_gpu_kernel_compute_mat_xy(
        const uint32_t sizeX, const uint32_t sizeY, const uint32_t startX, const uint32_t startY, const uint32_t numX, const uint32_t numY,
        const uint32_t total_len_features, const uint32_t dim, CPTR(double) X, CPTR(double) Y, double *Z, const double param2, const double param3, const double param4);

void cu_distances_xx(const uint32_t cols, const uint32_t lag, CPTR(double) X, const double lambda, double *Z);

void cu_distances_xx(const uint32_t cols, const uint32_t rows, const uint32_t lag, const double lambda, CPTR(double) X, double *Z);

void cu_kernel_xx(const uint32_t cols, const uint32_t rows, const uint32_t lag, const double lambda, const double gamma, CPTR(double) X, double *Z);

void cu_distances_xy(const uint32_t X_cols, const uint32_t Xy_cols, const uint32_t lag, const double lambda, CPTR(double) X, CPTR(double) Xy, double *Z);

void cu_distances_xy(const uint32_t X_cols, const uint32_t Xy_cols, const uint32_t rows, const uint32_t lag, const double lambda, CPTR(double) X, CPTR(double) Xy, double *Z);

void cu_kernel_xy(const uint32_t X_cols, const uint32_t Xy_cols, const uint32_t rows, const uint32_t lag, const double lambda, const double gamma,
                  CPTR(double) X, CPTR(double) Xy, double *Z);

inline double calc_g(const double n, const double Z_mm, const double L_mm)
{
    return std::pow(- (L_mm - n) / (double(n) * Z_mm), -1. / DISTPOW);
}


}

#endif
