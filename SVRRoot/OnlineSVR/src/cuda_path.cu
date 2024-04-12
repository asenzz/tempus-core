#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include "common/defines.h"
#include "cuda_path.hpp"
#include "common/cuda_util.cuh"
#include "common/gpu_handler.hpp"
#include "common/constants.hpp"

namespace svr::kernel::path {

#define blockX(i, j) (X[(i) * lag + (j)])
#define blockYY(i, j) (Y[(i) * lag + (j)])

__global__  void
gpu_kernel_xx_compute(const size_t cols, const size_t end_col, const size_t end_row, const size_t lag, const double *X, double *Z, const double lambda)
{
    if (blockIdx.x * blockDim.x >= end_col
        || blockIdx.y * blockDim.y >= end_row
        || blockIdx.x * blockDim.x > blockIdx.y * blockDim.y + blockDim.y - 1)
        return;

    __shared__ double power_mult[TILE_WIDTH];
    __shared__ double ta[TILE_WIDTH][TILE_WIDTH];
    __shared__ double tam1[TILE_WIDTH][TILE_WIDTH]; // for index-1
    __shared__ double tb[TILE_WIDTH][TILE_WIDTH];
    __shared__ double tbm1[TILE_WIDTH][TILE_WIDTH]; // for index-1
    const auto kk = threadIdx.x + blockIdx.x * blockDim.x;
    const auto mm = threadIdx.y + blockIdx.y * blockDim.y;
    const auto tx = threadIdx.x;
    const auto ty = threadIdx.y;
    const auto len_TILE_WIDTH = lag / TILE_WIDTH + (lag % TILE_WIDTH == 0 ? 0 : 1);
    const auto do_matrix_product_sum = mm < end_row && kk < end_col && kk <= mm;
    double matrix_prod_sum = 0;
    for (size_t kk_internal_big = 0; kk_internal_big < len_TILE_WIDTH; ++kk_internal_big) {
        const auto kk_internal_big_TILE_WIDTH = kk_internal_big * TILE_WIDTH;
        const auto ty_kk_internal_big_TILE_WIDTH = ty + kk_internal_big_TILE_WIDTH;
        if (ty_kk_internal_big_TILE_WIDTH < lag) {
            if (!tx) power_mult[ty] = pow(1. / double(lag - ty_kk_internal_big_TILE_WIDTH), lambda);
            if (kk < end_col) {
                ta[tx][ty] = blockX(kk, ty_kk_internal_big_TILE_WIDTH);
                if (ty_kk_internal_big_TILE_WIDTH)
                    tam1[tx][ty] = ta[tx][ty] - blockX(kk, ty_kk_internal_big_TILE_WIDTH - 1);
            }
        }
        const auto tx_kk_internal_big_TILE_WIDTH = tx + kk_internal_big_TILE_WIDTH;
        if (mm < end_row && tx_kk_internal_big_TILE_WIDTH < lag) {
            tb[ty][tx] = blockX(mm, tx_kk_internal_big_TILE_WIDTH);
            if (tx_kk_internal_big_TILE_WIDTH)
                tbm1[ty][tx] = tb[ty][tx] - blockX(mm, tx_kk_internal_big_TILE_WIDTH - 1);
        }
        __syncthreads();

        if (do_matrix_product_sum)
            for (size_t kk_internal_small = 0; kk_internal_small < TILE_WIDTH; ++kk_internal_small) {
                const auto kk_internal = kk_internal_small + kk_internal_big_TILE_WIDTH;
                if (kk_internal >= lag) continue;
                matrix_prod_sum += (pow(ta[tx][kk_internal_small] - tb[ty][kk_internal_small], 2)
                                    + (kk_internal ? common::C_kernel_path_tau * pow(tam1[tx][kk_internal_small] - tbm1[ty][kk_internal_small], 2) : 0.))
                                   * power_mult[kk_internal_small];
            }
        __syncthreads();
    }

    if (do_matrix_product_sum) {
        Z[kk * cols + mm] = matrix_prod_sum;
        if (kk < mm) Z[mm * cols + kk] = matrix_prod_sum;
    }
}


__global__  void
gpu_kernel_xy_compute(const size_t Xy_cols, const size_t end_col, const size_t end_row, const size_t lag, const double *X, const double *Y, double *Z,
                      const double lambda)
{
    if (blockIdx.x * blockDim.x >= end_col || blockIdx.y * blockDim.y >= end_row) return;

    __shared__ double power_mult[TILE_WIDTH];
    __shared__ double ta[TILE_WIDTH][TILE_WIDTH];
    __shared__ double tam1[TILE_WIDTH][TILE_WIDTH]; // for index-1
    __shared__ double tb[TILE_WIDTH][TILE_WIDTH];
    __shared__ double tbm1[TILE_WIDTH][TILE_WIDTH]; // for index-1

    const auto kk = threadIdx.x + blockIdx.x * blockDim.x;
    const auto mm = threadIdx.y + blockIdx.y * blockDim.y;
    const auto tx = threadIdx.x;
    const auto ty = threadIdx.y;
    const auto len_TILE_WIDTH = lag / TILE_WIDTH + (lag % TILE_WIDTH == 0 ? 0 : 1);
    const auto do_matrix_product_sum = mm < end_row && kk < end_col;
    double matrix_prod_sum = 0;
    for (size_t kk_internal_big = 0; kk_internal_big < len_TILE_WIDTH; ++kk_internal_big) {
        const auto kk_internal_big_TILE_WIDTH = kk_internal_big * TILE_WIDTH;
        const auto ty_kk_internal_big_TILE_WIDTH = ty + kk_internal_big_TILE_WIDTH;
        if (ty_kk_internal_big_TILE_WIDTH < lag) {
            if (!tx) power_mult[ty] = pow(1. / double(lag - ty_kk_internal_big_TILE_WIDTH), lambda);
            if (kk < end_col) {
                ta[tx][ty] = blockX(kk, ty_kk_internal_big_TILE_WIDTH);
                if (ty_kk_internal_big_TILE_WIDTH) tam1[tx][ty] = ta[tx][ty] - blockX(kk, ty_kk_internal_big_TILE_WIDTH - 1);
            }
        }

        const auto tx_kk_internal_big_TILE_WIDTH = tx + kk_internal_big_TILE_WIDTH;
        if (mm < end_row && tx_kk_internal_big_TILE_WIDTH < lag) {
            tb[ty][tx] = blockYY(mm, tx_kk_internal_big_TILE_WIDTH);
            if (tx_kk_internal_big_TILE_WIDTH)
                tbm1[ty][tx] = tb[ty][tx] - blockYY(mm, tx_kk_internal_big_TILE_WIDTH - 1);
        }
        __syncthreads();

        if (do_matrix_product_sum)
            for (size_t kk_internal_small = 0; kk_internal_small < TILE_WIDTH; ++kk_internal_small) {
                const auto kk_internal = kk_internal_small + kk_internal_big_TILE_WIDTH;
                if (kk_internal >= lag) continue;
                matrix_prod_sum += (pow(ta[tx][kk_internal_small] - tb[ty][kk_internal_small], 2)
                                    + (kk_internal ? common::C_kernel_path_tau * pow(tam1[tx][kk_internal_small] - tbm1[ty][kk_internal_small], 2) : 0.))
                                   * power_mult[kk_internal_small];
            }
        __syncthreads();
    }

    if (do_matrix_product_sum) Z[kk * Xy_cols + mm] = matrix_prod_sum;
}


void
cu_distances_xx(const size_t lag, const size_t cols, const size_t end_col, const size_t end_row, const double *X, const double lambda, double *Z)
{
    const size_t X_size = cols * lag * sizeof(double);
    const size_t Z_size = cols * cols * sizeof(double);
    double *d_Z, *d_X;
    const common::gpu_context ctx;
    cu_errchk(cudaSetDevice(ctx.phy_id()));

    cu_errchk(cudaMalloc(&d_X, X_size))
    cu_errchk(cudaMalloc(&d_Z, Z_size));
    cu_errchk(cudaMemcpy(d_X, X, X_size, cudaMemcpyHostToDevice));

    const auto block_width = cols / TILE_WIDTH + (cols % TILE_WIDTH == 0 ? 0 : 1);
    gpu_kernel_xx_compute<<<dim3(block_width, block_width), dim3(TILE_WIDTH, TILE_WIDTH)>>>(cols, end_col, end_row, lag, d_X, d_Z, lambda);

    cu_errchk(cudaMemcpy(Z, d_Z, Z_size, cudaMemcpyDeviceToHost));
    cu_errchk(cudaFree(d_Z));
    cu_errchk(cudaFree(d_X));
    cu_errchk(cudaDeviceSynchronize());
}


void
cu_distances_xy(const size_t lag, const size_t X_cols, const size_t Xy_cols, const size_t end_col, const size_t end_row, const double *X, const double *Xy,
                const double lambda, double *Z)
{
    const auto X_size = X_cols * lag * sizeof(double);
    const auto Xy_size = Xy_cols * lag * sizeof(double);
    const auto Z_size = X_cols * Xy_cols * sizeof(double);
    const common::gpu_context ctx;
    cu_errchk(cudaSetDevice(ctx.phy_id()));
    double *d_X, *d_Xy, *d_Z;
    cu_errchk(cudaMalloc(&d_X, X_size));
    cu_errchk(cudaMalloc(&d_Xy, Xy_size));
    cu_errchk(cudaMalloc(&d_Z, Z_size));
    cu_errchk(cudaMemcpy(d_X, X, X_size, cudaMemcpyHostToDevice));
    cu_errchk(cudaMemcpy(d_Xy, Xy, Xy_size, cudaMemcpyHostToDevice));

    const auto block_width = X_cols / TILE_WIDTH + (X_cols % TILE_WIDTH == 0 ? 0 : 1);
    gpu_kernel_xy_compute<<<dim3(block_width, block_width), dim3(TILE_WIDTH, TILE_WIDTH)>>>(Xy_cols, end_col, end_row, lag, d_X, d_Xy, d_Z, lambda);
    cu_errchk(cudaMemcpy(Z, d_Z, Z_size, cudaMemcpyDeviceToHost));
    cu_errchk(cudaFree(d_X));
    cu_errchk(cudaFree(d_Xy));
    cu_errchk(cudaFree(d_Z));
    cu_errchk(cudaDeviceSynchronize());
}

}