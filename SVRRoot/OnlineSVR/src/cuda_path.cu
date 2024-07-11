#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include "common/defines.h"
#include "cuda_path.hpp"
#include "common/cuda_util.cuh"
#include "common/gpu_handler.tpp"
#include "common/constants.hpp"
#include "onlinesvr.hpp"
#include "model/SVRParameters.hpp"
#include "cuqrsolve.cuh"
#include "common/constants.hpp"

// #define HIFI_PATH // Actually lower precision when tuning parameters, so keep off for now

#define blockX(i, j) (X[(i) * lag + (j)])
#define blockYY(i, j) (Y[(i) * lag + (j)])
#define blockXX(i, j) (X[(i) * rows + (j)])
#define blockYYY(i, j) (Y[(i) * rows + (j)])

#define tx threadIdx.x
#define ty threadIdx.y

namespace svr {
namespace kernel::path {

// TODO Remove xx implementations
__global__  void
G_kernel_xx(const uint32_t cols, const uint32_t rows, const uint32_t lag, const uint32_t dim, const uint32_t len_TILE_WIDTH, const double lambda,
            const double *__restrict__ X, double *__restrict__ Z)
{
    if (blockIdx.x * blockDim.x >= cols || blockIdx.y * blockDim.y >= cols || blockIdx.x * blockDim.x >= blockIdx.y * blockDim.y + blockDim.y) return;

    __shared__ double power_mult[common::C_cu_tile_width];
    __shared__ double ta[common::C_cu_tile_width][common::C_cu_tile_width];
    __shared__ double tam1[common::C_cu_tile_width][common::C_cu_tile_width];
    __shared__ double tb[common::C_cu_tile_width][common::C_cu_tile_width];
    __shared__ double tbm1[common::C_cu_tile_width][common::C_cu_tile_width];
    const auto kk = threadIdx.x + blockIdx.x * blockDim.x;
    const auto mm = threadIdx.y + blockIdx.y * blockDim.y;
    const bool kk_numX = kk < cols;
    const bool mm_numY = mm < cols;
    const bool do_matrix_product_sum = kk_numX && mm_numY && kk <= mm;
    double matrix_prod_sum = 0;
#ifdef PRODUCTION_BUILD
#pragma unroll
#endif
    for (uint32_t jA = 0; jA < dim; ++jA) {
        const auto jA_lag = jA * lag;
#ifdef PRODUCTION_BUILD
#pragma unroll
#endif
        for (uint32_t kk_internal_big = 0; kk_internal_big < len_TILE_WIDTH; ++kk_internal_big) {
            const auto kk_internal_big_TILE_WIDTH = kk_internal_big * common::C_cu_tile_width;
            const auto ty_kk_internal_big_TILE_WIDTH = ty + kk_internal_big_TILE_WIDTH;
            if (!tx && ty_kk_internal_big_TILE_WIDTH < lag) power_mult[ty] = pow(1. / double(lag - ty_kk_internal_big_TILE_WIDTH), lambda);

            if (kk_numX && ty_kk_internal_big_TILE_WIDTH < lag) {
                ta[tx][ty] = blockXX(kk, ty_kk_internal_big_TILE_WIDTH + jA_lag);
                if (ty_kk_internal_big_TILE_WIDTH) tam1[tx][ty] = ta[tx][ty] - blockXX(kk, ty_kk_internal_big_TILE_WIDTH - 1 + jA_lag);
            }
            const auto tx_kk_internal_big_TILE_WIDTH = kk_internal_big_TILE_WIDTH + tx;
            if (mm_numY && tx_kk_internal_big_TILE_WIDTH < lag) {
                tb[ty][tx] = blockXX(mm, tx_kk_internal_big_TILE_WIDTH + jA_lag);
                if (tx_kk_internal_big_TILE_WIDTH) tbm1[ty][tx] = tb[ty][tx] - blockXX(mm, tx_kk_internal_big_TILE_WIDTH - 1 + jA_lag);
            }

            __syncthreads();

            if (do_matrix_product_sum)
#pragma unroll common::C_cu_tile_width
                for (uint32_t kk_internal_small = 0; kk_internal_small < common::C_cu_tile_width; ++kk_internal_small) {
                    const auto kk_internal = kk_internal_small + kk_internal_big_TILE_WIDTH;
                    if (kk_internal >= lag) continue;
#ifdef HIFI_PATH
                    matrix_prod_sum += (DIST(ta[tx][kk_internal_small] - tb[ty][kk_internal_small]) +
                                            (kk_internal ? (common::C_kernel_path_tau * DIST(tam1[tx][kk_internal_small] - tbm1[ty][kk_internal_small])) : 0.)) *
                                                    power_mult[kk_internal_small] / double(dim);
#else
                    matrix_prod_sum += (DIST(ta[tx][kk_internal_small] - tb[ty][kk_internal_small]) +
                                            (kk_internal ? (common::C_kernel_path_tau * DIST(tam1[tx][kk_internal_small] - tbm1[ty][kk_internal_small])) : 0.)) *
                                                    power_mult[kk_internal_small];
#endif
                }

            __syncthreads();
        }
    }
    if (do_matrix_product_sum) {
#ifdef HIFI_PATH
        Z[kk * cols + mm] = matrix_prod_sum;
#else
        Z[kk * cols + mm] = matrix_prod_sum / double(dim);
#endif
        if (kk < mm) Z[mm * cols + kk] = Z[kk * cols + mm];
    }
}

__global__  void
G_kernel_xx(const uint32_t cols, const uint32_t rows, const uint32_t lag, const uint32_t dim, const uint32_t len_TILE_WIDTH,
            const double lambda, const double gamma, const double *__restrict__ X, double *__restrict__ Z)
{
    if (blockIdx.x * blockDim.x >= cols || blockIdx.y * blockDim.y >= cols || blockIdx.x * blockDim.x >= blockIdx.y * blockDim.y + blockDim.y) return;

    __shared__ double power_mult[common::C_cu_tile_width];
    __shared__ double ta[common::C_cu_tile_width][common::C_cu_tile_width];
    __shared__ double tam1[common::C_cu_tile_width][common::C_cu_tile_width];//for index-1
    __shared__ double tb[common::C_cu_tile_width][common::C_cu_tile_width];
    __shared__ double tbm1[common::C_cu_tile_width][common::C_cu_tile_width];//for index-1
    const auto kk = threadIdx.x + blockIdx.x * blockDim.x;
    const auto mm = threadIdx.y + blockIdx.y * blockDim.y;
    const bool kk_numX = kk < cols;
    const bool mm_numY = mm < cols;
    const bool do_matrix_product_sum = kk_numX && mm_numY && kk <= mm;
    double matrix_prod_sum = 0;
#ifdef PRODUCTION_BUILD
#pragma unroll
#endif
    for (uint32_t jA = 0; jA < dim; ++jA) {
        const auto jA_lag = jA * lag;
#ifdef PRODUCTION_BUILD
#pragma unroll
#endif
        for (uint32_t kk_internal_big = 0; kk_internal_big < len_TILE_WIDTH; ++kk_internal_big) {
            const auto kk_internal_big_TILE_WIDTH = kk_internal_big * common::C_cu_tile_width;
            const auto ty_kk_internal_big_TILE_WIDTH = ty + kk_internal_big_TILE_WIDTH;
            if (!tx && ty_kk_internal_big_TILE_WIDTH < lag) power_mult[ty] = pow(1. / double(lag - ty_kk_internal_big_TILE_WIDTH), lambda);

            if (kk_numX && ty_kk_internal_big_TILE_WIDTH < lag) {
                ta[tx][ty] = blockXX(kk, ty_kk_internal_big_TILE_WIDTH + jA_lag);
                if (ty_kk_internal_big_TILE_WIDTH) tam1[tx][ty] = ta[tx][ty] - blockXX(kk, ty_kk_internal_big_TILE_WIDTH - 1 + jA_lag);
            }
            const auto tx_kk_internal_big_TILE_WIDTH = kk_internal_big_TILE_WIDTH + tx;
            if (mm_numY && tx_kk_internal_big_TILE_WIDTH < lag) {
                tb[ty][tx] = blockXX(mm, tx_kk_internal_big_TILE_WIDTH + jA_lag);
                if (tx_kk_internal_big_TILE_WIDTH) tbm1[ty][tx] = tb[ty][tx] - blockXX(mm, tx_kk_internal_big_TILE_WIDTH - 1 + jA_lag);
            }

            __syncthreads();

            if (do_matrix_product_sum)
#pragma unroll common::C_cu_tile_width
                for (uint32_t kk_internal_small = 0; kk_internal_small < common::C_cu_tile_width; ++kk_internal_small) {
                    const auto kk_internal = kk_internal_small + kk_internal_big_TILE_WIDTH;
                    if (kk_internal >= lag) continue;
#ifdef HIFI_PATH
                    matrix_prod_sum += (DIST(ta[tx][kk_internal_small] - tb[ty][kk_internal_small]) +
                                    (kk_internal ? (common::C_kernel_path_tau * DIST(tam1[tx][kk_internal_small] - tbm1[ty][kk_internal_small])) : 0.)) *
                                            power_mult[kk_internal_small] / double(dim) / gamma;
#else
                    matrix_prod_sum += (DIST(ta[tx][kk_internal_small] - tb[ty][kk_internal_small]) +
                                    (kk_internal ? (common::C_kernel_path_tau * DIST(tam1[tx][kk_internal_small] - tbm1[ty][kk_internal_small])) : 0.)) *
                                            power_mult[kk_internal_small];
#endif
                }

            __syncthreads();
        }
    }
    if (do_matrix_product_sum) {
#ifdef HIFI_PATH
        Z[kk * cols + mm] = 1. - matrix_prod_sum;
#else
        Z[kk * cols + mm] = 1. - matrix_prod_sum / double(dim) / gamma;
#endif
        if (kk < mm) Z[mm * cols + kk] = Z[kk * cols + mm];
    }
}

__global__  void
G_kernel_xx(const uint32_t cols, const uint32_t lag, const uint32_t len_TILE_WIDTH, const double lambda, const double *__restrict__ X, double *__restrict__ Z)
{
    if (blockIdx.x * blockDim.x >= cols || blockIdx.y * blockDim.y >= cols || blockIdx.x * blockDim.x >= blockIdx.y * blockDim.y + blockDim.y) return;

    __shared__ double power_mult[common::C_cu_tile_width];
    __shared__ double ta[common::C_cu_tile_width][common::C_cu_tile_width];
    __shared__ double tam1[common::C_cu_tile_width][common::C_cu_tile_width]; // for index-1
    __shared__ double tb[common::C_cu_tile_width][common::C_cu_tile_width];
    __shared__ double tbm1[common::C_cu_tile_width][common::C_cu_tile_width]; // for index-1
    const auto kk = threadIdx.x + blockIdx.x * blockDim.x;
    const auto mm = threadIdx.y + blockIdx.y * blockDim.y;
    const bool mm_cols = mm < cols;
    const bool kk_cols = kk < cols;
    const auto do_matrix_product_sum = mm_cols && kk_cols;
    double matrix_prod_sum = 0;
#ifdef PRODUCTION_BUILD
#pragma unroll
#endif
    for (uint32_t kk_internal_big = 0; kk_internal_big < len_TILE_WIDTH; ++kk_internal_big) {
        const auto kk_internal_big_TILE_WIDTH = kk_internal_big * common::C_cu_tile_width;
        const auto ty_kk_internal_big_TILE_WIDTH = ty + kk_internal_big_TILE_WIDTH;
        if (ty_kk_internal_big_TILE_WIDTH < lag) {
            if (!tx) power_mult[ty] = pow(1. / double(lag - ty_kk_internal_big_TILE_WIDTH), lambda);
            if (kk_cols) {
                ta[tx][ty] = blockX(kk, ty_kk_internal_big_TILE_WIDTH);
                if (ty_kk_internal_big_TILE_WIDTH) tam1[tx][ty] = ta[tx][ty] - blockX(kk, ty_kk_internal_big_TILE_WIDTH - 1);
            }
        }
        const auto tx_kk_internal_big_TILE_WIDTH = tx + kk_internal_big_TILE_WIDTH;
        if (mm_cols && tx_kk_internal_big_TILE_WIDTH < lag) {
            tb[ty][tx] = blockX(mm, tx_kk_internal_big_TILE_WIDTH);
            if (tx_kk_internal_big_TILE_WIDTH) tbm1[ty][tx] = tb[ty][tx] - blockX(mm, tx_kk_internal_big_TILE_WIDTH - 1);
        }
        __syncthreads();

        if (do_matrix_product_sum)
#pragma unroll common::C_cu_tile_width
            for (uint32_t kk_internal_small = 0; kk_internal_small < common::C_cu_tile_width; ++kk_internal_small) {
                const auto kk_internal = kk_internal_small + kk_internal_big_TILE_WIDTH;
                if (kk_internal >= lag) continue;
                matrix_prod_sum += (DIST(ta[tx][kk_internal_small] - tb[ty][kk_internal_small]) +
                        (kk_internal ? (common::C_kernel_path_tau * DIST(tam1[tx][kk_internal_small] - tbm1[ty][kk_internal_small])) : 0.)) *
                            power_mult[kk_internal_small];
            }
        __syncthreads();
    }

    if (do_matrix_product_sum) {
        Z[kk * cols + mm] = matrix_prod_sum;
        if (kk < mm) Z[mm * cols + kk] = matrix_prod_sum;
    }
}

__global__  void
G_kernel_xy(const uint32_t X_cols, const uint32_t Y_cols, const uint32_t rows, const uint32_t lag, const uint32_t dim, const uint32_t len_TILE_WIDTH, const double lambda,
            const double *__restrict__ X, const double *__restrict__ Y, double *__restrict__ Z)
{
    if (blockIdx.x * blockDim.x >= X_cols || blockIdx.y * blockDim.y >= Y_cols) return;

    __shared__ double power_mult[common::C_cu_tile_width];
    __shared__ double ta[common::C_cu_tile_width][common::C_cu_tile_width];
    __shared__ double tam1[common::C_cu_tile_width][common::C_cu_tile_width]; // for index-1
    __shared__ double tb[common::C_cu_tile_width][common::C_cu_tile_width];
    __shared__ double tbm1[common::C_cu_tile_width][common::C_cu_tile_width]; // for index-1

    const auto kk = threadIdx.x + blockIdx.x * blockDim.x;
    const auto mm = threadIdx.y + blockIdx.y * blockDim.y;
    const bool kk_X = kk < X_cols;
    const bool mm_Y = mm < Y_cols;
    const auto do_matrix_product_sum = mm_Y && kk_X;
    double matrix_prod_sum = 0;
#ifdef PRODUCTION_BUILD
#pragma unroll
#endif
    for (uint32_t jA = 0; jA < dim; ++jA) {
        const auto jA_lag = jA * lag;
#ifdef PRODUCTION_BUILD
#pragma unroll
#endif
        for (uint32_t kk_internal_big = 0; kk_internal_big < len_TILE_WIDTH; ++kk_internal_big) {
            const auto kk_internal_big_TILE_WIDTH = kk_internal_big * common::C_cu_tile_width;
            const auto ty_kk_internal_big_TILE_WIDTH = ty + kk_internal_big_TILE_WIDTH;
            if (ty_kk_internal_big_TILE_WIDTH < lag) {
                if (!tx) power_mult[ty] = pow(1. / double(lag - ty_kk_internal_big_TILE_WIDTH), lambda);
                if (kk_X) {
                    ta[tx][ty] = blockXX(kk, ty_kk_internal_big_TILE_WIDTH + jA_lag);
                    if (ty_kk_internal_big_TILE_WIDTH) tam1[tx][ty] = ta[tx][ty] - blockXX(kk, ty_kk_internal_big_TILE_WIDTH + jA_lag - 1);
                }
            }

            const auto tx_kk_internal_big_TILE_WIDTH = tx + kk_internal_big_TILE_WIDTH;
            if (mm_Y && tx_kk_internal_big_TILE_WIDTH < lag) {
                tb[ty][tx] = blockYYY(mm, tx_kk_internal_big_TILE_WIDTH + jA_lag);
                if (tx_kk_internal_big_TILE_WIDTH) tbm1[ty][tx] = tb[ty][tx] - blockYYY(mm, tx_kk_internal_big_TILE_WIDTH + jA_lag - 1);
            }
            __syncthreads();

            if (do_matrix_product_sum)
#pragma unroll common::C_cu_tile_width
                for (uint32_t kk_internal_small = 0; kk_internal_small < common::C_cu_tile_width; ++kk_internal_small) {
                    const auto kk_internal = kk_internal_small + kk_internal_big_TILE_WIDTH;
                    if (kk_internal >= lag) continue;
#ifdef HIFI_PATH
                    matrix_prod_sum += (DIST(ta[tx][kk_internal_small] - tb[ty][kk_internal_small]) +
                            (kk_internal ? (common::C_kernel_path_tau * DIST(tam1[tx][kk_internal_small] - tbm1[ty][kk_internal_small])) : 0.)) *
                                    power_mult[kk_internal_small] / double(dim);
#else
                    matrix_prod_sum += (DIST(ta[tx][kk_internal_small] - tb[ty][kk_internal_small]) +
                            (kk_internal ? (common::C_kernel_path_tau * DIST(tam1[tx][kk_internal_small] - tbm1[ty][kk_internal_small])) : 0.)) *
                                    power_mult[kk_internal_small];
#endif
                }
            __syncthreads();
        }
    }
    if (do_matrix_product_sum)
#ifdef HIFI_PATH
        Z[kk * Y_cols + mm] = matrix_prod_sum;
#else
        Z[kk * Y_cols + mm] = matrix_prod_sum / double(dim);
#endif
}

__global__  void
G_kernel_xy(const uint32_t X_cols, const uint32_t Y_cols, const uint32_t rows, const uint32_t lag, const uint32_t dim, const uint32_t len_TILE_WIDTH,
            const double lambda, const double gamma, const double *X, const double *Y, double *Z)
{
    if (blockIdx.x * blockDim.x >= X_cols || blockIdx.y * blockDim.y >= Y_cols) return;

    __shared__ double power_mult[common::C_cu_tile_width];
    __shared__ double ta[common::C_cu_tile_width][common::C_cu_tile_width];
    __shared__ double tam1[common::C_cu_tile_width][common::C_cu_tile_width];
    __shared__ double tb[common::C_cu_tile_width][common::C_cu_tile_width];
    __shared__ double tbm1[common::C_cu_tile_width][common::C_cu_tile_width];

    const auto kk = threadIdx.x + blockIdx.x * blockDim.x;
    const auto mm = threadIdx.y + blockIdx.y * blockDim.y;
    const bool mm_Y = mm < Y_cols;
    const bool kk_X = kk < X_cols;
    const auto do_matrix_product_sum = mm_Y && kk_X;
    double matrix_prod_sum = 0;
#ifdef PRODUCTION_BUILD
#pragma unroll
#endif
    for (uint32_t jA = 0; jA < dim; ++jA) {
        const auto jA_lag = jA * lag;
#ifdef PRODUCTION_BUILD
#pragma unroll
#endif
        for (uint32_t kk_internal_big = 0; kk_internal_big < len_TILE_WIDTH; ++kk_internal_big) {
            const auto kk_internal_big_TILE_WIDTH = kk_internal_big * common::C_cu_tile_width;
            const auto ty_kk_internal_big_TILE_WIDTH = ty + kk_internal_big_TILE_WIDTH;
            if (ty_kk_internal_big_TILE_WIDTH < lag) {
                if (!tx) power_mult[ty] = pow(1. / double(lag - ty_kk_internal_big_TILE_WIDTH), lambda);
                if (kk_X) {
                    ta[tx][ty] = blockXX(kk, ty_kk_internal_big_TILE_WIDTH + jA_lag);
                    if (ty_kk_internal_big_TILE_WIDTH) tam1[tx][ty] = ta[tx][ty] - blockXX(kk, ty_kk_internal_big_TILE_WIDTH + jA_lag - 1);
                }
            }

            const auto tx_kk_internal_big_TILE_WIDTH = tx + kk_internal_big_TILE_WIDTH;
            if (mm_Y && tx_kk_internal_big_TILE_WIDTH < lag) {
                tb[ty][tx] = blockYYY(mm, tx_kk_internal_big_TILE_WIDTH + jA_lag);
                if (tx_kk_internal_big_TILE_WIDTH) tbm1[ty][tx] = tb[ty][tx] - blockYYY(mm, tx_kk_internal_big_TILE_WIDTH + jA_lag - 1);
            }
            __syncthreads();

            if (do_matrix_product_sum)
#pragma unroll common::C_cu_tile_width
                for (uint32_t kk_internal_small = 0; kk_internal_small < common::C_cu_tile_width; ++kk_internal_small) {
                    const auto kk_internal = kk_internal_small + kk_internal_big_TILE_WIDTH;
                    if (kk_internal >= lag) continue;
#ifdef HIFI_PATH
                    matrix_prod_sum += (DIST(ta[tx][kk_internal_small] - tb[ty][kk_internal_small]) +
                            (kk_internal ? (common::C_kernel_path_tau * DIST(tam1[tx][kk_internal_small] - tbm1[ty][kk_internal_small])) : 0.)) *
                                    power_mult[kk_internal_small] / double(dim) / gamma;
#else
                    matrix_prod_sum += (DIST(ta[tx][kk_internal_small] - tb[ty][kk_internal_small]) +
                            (kk_internal ? (common::C_kernel_path_tau * DIST(tam1[tx][kk_internal_small] - tbm1[ty][kk_internal_small])) : 0.)) *
                                    power_mult[kk_internal_small];
#endif
                }
            __syncthreads();
        }
    }
    if (do_matrix_product_sum)
#ifdef HIFI_PATH
        Z[kk * Y_cols + mm] = 1. - matrix_prod_sum;
#else
        Z[kk * Y_cols + mm] = 1. - matrix_prod_sum / double(dim) / gamma;
#endif
}

__global__  void
G_kernel_xy(const uint32_t X_cols, const uint32_t Y_cols, const uint32_t lag, const unsigned len_TILE_WIDTH, const double lambda,
            const double *__restrict__ X, const double *__restrict__ Y, double *__restrict__ Z)
{
    if (blockIdx.x * blockDim.x >= X_cols || blockIdx.y * blockDim.y >= Y_cols) return;

    __shared__ double power_mult[common::C_cu_tile_width];
    __shared__ double ta[common::C_cu_tile_width][common::C_cu_tile_width];
    __shared__ double tam1[common::C_cu_tile_width][common::C_cu_tile_width]; // for index-1
    __shared__ double tb[common::C_cu_tile_width][common::C_cu_tile_width];
    __shared__ double tbm1[common::C_cu_tile_width][common::C_cu_tile_width]; // for index-1

    const auto kk = threadIdx.x + blockIdx.x * blockDim.x;
    const auto mm = threadIdx.y + blockIdx.y * blockDim.y;
    const bool kk_X = kk < X_cols;
    const bool mm_Y = mm < Y_cols;
    const auto do_matrix_product_sum = mm_Y && kk_X;
    double matrix_prod_sum = 0;
#ifndef __GNUC__
#pragma unroll
#endif
    for (uint32_t kk_internal_big = 0; kk_internal_big < len_TILE_WIDTH; ++kk_internal_big) {
        const auto kk_internal_big_TILE_WIDTH = kk_internal_big * common::C_cu_tile_width;
        const auto ty_kk_internal_big_TILE_WIDTH = ty + kk_internal_big_TILE_WIDTH;
        if (ty_kk_internal_big_TILE_WIDTH < lag) {
            if (!tx) power_mult[ty] = pow(1. / double(lag - ty_kk_internal_big_TILE_WIDTH), lambda);
            if (kk_X) {
                ta[tx][ty] = blockX(kk, ty_kk_internal_big_TILE_WIDTH);
                if (ty_kk_internal_big_TILE_WIDTH) tam1[tx][ty] = ta[tx][ty] - blockX(kk, ty_kk_internal_big_TILE_WIDTH - 1);
            }
        }

        const auto tx_kk_internal_big_TILE_WIDTH = tx + kk_internal_big_TILE_WIDTH;
        if (mm_Y && tx_kk_internal_big_TILE_WIDTH < lag) {
            tb[ty][tx] = blockYY(mm, tx_kk_internal_big_TILE_WIDTH);
            if (tx_kk_internal_big_TILE_WIDTH) tbm1[ty][tx] = tb[ty][tx] - blockYY(mm, tx_kk_internal_big_TILE_WIDTH - 1);
        }
        __syncthreads();

        if (do_matrix_product_sum)
#ifndef __GNUC__
#pragma unroll
#endif
            for (uint32_t kk_internal_small = 0; kk_internal_small < common::C_cu_tile_width; ++kk_internal_small) {
                const auto kk_internal = kk_internal_small + kk_internal_big_TILE_WIDTH;
                if (kk_internal >= lag) continue;
                matrix_prod_sum += (DIST(ta[tx][kk_internal_small] - tb[ty][kk_internal_small]) +
                        (kk_internal ? (common::C_kernel_path_tau * DIST(tam1[tx][kk_internal_small] - tbm1[ty][kk_internal_small])) : 0.)) *
                                power_mult[kk_internal_small];
            }
        __syncthreads();
    }

    if (do_matrix_product_sum) Z[kk * Y_cols + mm] = matrix_prod_sum;
}

void cu_distances_xx(const uint32_t cols, const uint32_t lag, const double *X, const double lambda, double *Z) // rows == lag
{
    const uint32_t X_size = cols * lag * sizeof(double);
    const uint32_t Z_size = cols * cols * sizeof(double);
    double *d_Z, *d_X;
    const common::gpu_context ctx;
    cu_errchk(cudaSetDevice(ctx.phy_id()));
    cudaStream_t cu_stream;
    cu_errchk(cudaStreamCreateWithFlags(&cu_stream, cudaStreamNonBlocking));
    cu_errchk(cudaMallocAsync(&d_X, X_size, cu_stream));
    cu_errchk(cudaMallocAsync(&d_Z, Z_size, cu_stream));
    cu_errchk(cudaMemcpyAsync(d_X, X, X_size, cudaMemcpyHostToDevice, cu_stream));
    G_kernel_xx<<<CUDA_THREADS_BLOCKS_2D(cols), 0, cu_stream>>>(cols, lag, _CEILDIV(lag, common::C_cu_tile_width), lambda, d_X, d_Z);
    cu_errchk(cudaMemcpyAsync(Z, d_Z, Z_size, cudaMemcpyDeviceToHost, cu_stream));
    cu_errchk(cudaFreeAsync(d_Z, cu_stream));
    cu_errchk(cudaFreeAsync(d_X, cu_stream));
    cu_errchk(cudaStreamSynchronize(cu_stream));
    cu_errchk(cudaStreamDestroy(cu_stream));
}

void cu_distances_xx(const uint32_t cols, const uint32_t rows, const uint32_t lag, const double lambda, const double *X, double *Z)
{
    const uint32_t X_size = cols * rows * sizeof(double);
    const uint32_t Z_size = cols * cols * sizeof(double);
    double *d_Z, *d_X;
    const common::gpu_context ctx;
    cu_errchk(cudaSetDevice(ctx.phy_id()));
    cudaStream_t cu_stream;
    cu_errchk(cudaStreamCreateWithFlags(&cu_stream, cudaStreamNonBlocking));
    cu_errchk(cudaMallocAsync(&d_X, X_size, cu_stream));
    cu_errchk(cudaMemcpyAsync(d_X, X, X_size, cudaMemcpyHostToDevice, cu_stream));
    cu_errchk(cudaMallocAsync(&d_Z, Z_size, cu_stream));
    G_kernel_xx<<<CUDA_THREADS_BLOCKS_2D(cols), 0, cu_stream>>>(cols, rows, lag, rows / lag, _CEILDIV(lag, common::C_cu_tile_width), lambda, d_X, d_Z);
    cu_errchk(cudaFreeAsync(d_X, cu_stream));
    cu_errchk(cudaMemcpyAsync(Z, d_Z, Z_size, cudaMemcpyDeviceToHost, cu_stream));
    cu_errchk(cudaFreeAsync(d_Z, cu_stream));
    cu_errchk(cudaStreamSynchronize(cu_stream));
    cu_errchk(cudaStreamDestroy(cu_stream));
}

void cu_kernel_xx(const uint32_t cols, const uint32_t rows, const uint32_t lag, const double lambda, const double gamma, const double *X, double *Z)
{
    const uint32_t X_size = cols * rows * sizeof(double);
    const uint32_t Z_size = cols * cols * sizeof(double);
    double *d_Z, *d_X;
    const common::gpu_context ctx;
    cu_errchk(cudaSetDevice(ctx.phy_id()));
    cudaStream_t cu_stream;
    cu_errchk(cudaStreamCreateWithFlags(&cu_stream, cudaStreamNonBlocking));
    cu_errchk(cudaMallocAsync(&d_X, X_size, cu_stream));
    cu_errchk(cudaMemcpyAsync(d_X, X, X_size, cudaMemcpyHostToDevice, cu_stream));
    cu_errchk(cudaMallocAsync(&d_Z, Z_size, cu_stream));
    G_kernel_xx<<<CUDA_THREADS_BLOCKS_2D(cols), 0, cu_stream>>>(cols, rows, lag, rows / lag, _CEILDIV(lag, common::C_cu_tile_width), lambda, DIST(gamma), d_X, d_Z);
    cu_errchk(cudaFreeAsync(d_X, cu_stream));
    cu_errchk(cudaMemcpyAsync(Z, d_Z, Z_size, cudaMemcpyDeviceToHost, cu_stream));
    cu_errchk(cudaFreeAsync(d_Z, cu_stream));
    cu_errchk(cudaStreamSynchronize(cu_stream));
    cu_errchk(cudaStreamDestroy(cu_stream));
}


void cu_distances_xy(const uint32_t X_cols, const uint32_t Xy_cols, const uint32_t lag, const double lambda, const double *X, const double *Xy, double *Z)
{
    const auto X_size = X_cols * lag * sizeof(double);
    const auto Xy_size = Xy_cols * lag * sizeof(double);
    const auto Z_size = X_cols * Xy_cols * sizeof(double);
    const common::gpu_context ctx;
    cu_errchk(cudaSetDevice(ctx.phy_id()));
    cudaStream_t cu_stream;
    cu_errchk(cudaStreamCreateWithFlags(&cu_stream, cudaStreamNonBlocking));
    double *d_X, *d_Xy, *d_Z;
    cu_errchk(cudaMallocAsync(&d_X, X_size, cu_stream));
    cu_errchk(cudaMemcpyAsync(d_X, X, X_size, cudaMemcpyHostToDevice, cu_stream));
    cu_errchk(cudaMallocAsync(&d_Xy, Xy_size, cu_stream));
    cu_errchk(cudaMemcpyAsync(d_Xy, Xy, Xy_size, cudaMemcpyHostToDevice, cu_stream));
    cu_errchk(cudaMallocAsync(&d_Z, Z_size, cu_stream));
    G_kernel_xy<<<CUDA_THREADS_BLOCKS_2D(X_cols), 0, cu_stream>>>(X_cols, Xy_cols, lag, _CEILDIV(lag, common::C_cu_tile_width), lambda, d_X, d_Xy, d_Z);
    cu_errchk(cudaFreeAsync(d_X, cu_stream));
    cu_errchk(cudaFreeAsync(d_Xy, cu_stream));
    cu_errchk(cudaMemcpyAsync(Z, d_Z, Z_size, cudaMemcpyDeviceToHost, cu_stream));
    cu_errchk(cudaFreeAsync(d_Z, cu_stream));
    cu_errchk(cudaStreamSynchronize(cu_stream));
    cu_errchk(cudaStreamDestroy(cu_stream));
}


void
cu_distances_xy(const uint32_t X_cols, const uint32_t Xy_cols, const uint32_t rows, const uint32_t lag, const double lambda, const double *X, const double *Xy, double *Z)
{
    const uint32_t X_size = X_cols * lag * sizeof(double);
    const uint32_t Xy_size = Xy_cols * lag * sizeof(double);
    const uint32_t Z_size = X_cols * Xy_cols * sizeof(double);
    double *d_X, *d_Xy, *d_Z;
    const common::gpu_context ctx;
    cu_errchk(cudaSetDevice(ctx.phy_id()));
    cudaStream_t cu_stream;
    cu_errchk(cudaStreamCreateWithFlags(&cu_stream, cudaStreamNonBlocking));
    cu_errchk(cudaMallocAsync(&d_X, X_size, cu_stream));
    cu_errchk(cudaMemcpyAsync(d_X, X, X_size, cudaMemcpyHostToDevice, cu_stream));
    cu_errchk(cudaMallocAsync(&d_Xy, Xy_size, cu_stream));
    cu_errchk(cudaMemcpyAsync(d_Xy, Xy, Xy_size, cudaMemcpyHostToDevice, cu_stream));
    cu_errchk(cudaMallocAsync(&d_Z, Z_size, cu_stream));
    G_kernel_xy<<<CUDA_THREADS_BLOCKS_2D(X_cols), 0, cu_stream>>>(X_cols, Xy_cols, lag, rows, rows / lag, _CEILDIV(lag, common::C_cu_tile_width), lambda, d_X, d_Xy, d_Z);
    cu_errchk(cudaFreeAsync(d_X, cu_stream));
    cu_errchk(cudaFreeAsync(d_Xy, cu_stream));
    cu_errchk(cudaMemcpyAsync(Z, d_Z, Z_size, cudaMemcpyDeviceToHost, cu_stream));
    cu_errchk(cudaFreeAsync(d_Z, cu_stream));
    cu_errchk(cudaStreamSynchronize(cu_stream));
    cu_errchk(cudaStreamDestroy(cu_stream));
}


void cu_kernel_xy(const uint32_t X_cols, const uint32_t Xy_cols, const uint32_t rows, const uint32_t lag, const double lambda, const double gamma,
                  const double *X, const double *Xy, double *Z)
{
    const auto X_size = X_cols * rows * sizeof(double);
    const auto Xy_size = Xy_cols * rows * sizeof(double);
    const auto Z_size = X_cols * Xy_cols * sizeof(double);
    double *d_X, *d_Xy, *d_Z;
    const common::gpu_context ctx;
    cu_errchk(cudaSetDevice(ctx.phy_id()));
    cudaStream_t cu_stream;
    cu_errchk(cudaStreamCreateWithFlags(&cu_stream, cudaStreamNonBlocking));
    cu_errchk(cudaMallocAsync(&d_X, X_size, cu_stream));
    cu_errchk(cudaMemcpyAsync(d_X, X, X_size, cudaMemcpyHostToDevice, cu_stream));
    cu_errchk(cudaMallocAsync(&d_Xy, Xy_size, cu_stream));
    cu_errchk(cudaMemcpyAsync(d_Xy, Xy, Xy_size, cudaMemcpyHostToDevice, cu_stream));
    cu_errchk(cudaMallocAsync(&d_Z, Z_size, cu_stream));
    G_kernel_xy<<<CUDA_THREADS_BLOCKS_2D(X_cols), 0, cu_stream>>>(X_cols, Xy_cols, rows, lag, rows / lag, _CEILDIV(lag, common::C_cu_tile_width), lambda, DIST(gamma), d_X, d_Xy, d_Z);
    cu_errchk(cudaFreeAsync(d_X, cu_stream));
    cu_errchk(cudaFreeAsync(d_Xy, cu_stream));
    cu_errchk(cudaMemcpyAsync(Z, d_Z, Z_size, cudaMemcpyDeviceToHost, cu_stream));
    cu_errchk(cudaFreeAsync(d_Z, cu_stream));
    cu_errchk(cudaStreamSynchronize(cu_stream));
    cu_errchk(cudaStreamDestroy(cu_stream));
}

}
}
