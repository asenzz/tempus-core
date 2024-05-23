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

namespace svr {
namespace kernel::path {

#define _old_blockX(i, j) (X[(startX + (i)) * total_len_features+(j)])
#define _old_blockYY(i, j) (Y[(startY + (i)) * total_len_features+(j)])


// TODO Replace with below kernel functions
__global__  void
_old_gpu_kernel_xy_compute(
        const uint32_t sizeX, const uint32_t sizeY, const uint32_t startX, const uint32_t startY, const uint32_t numX, const uint32_t numY, const uint32_t len, const uint32_t dim,
        const double *X, const double *Y, double *Z, const uint32_t full_sizeZ, const double param2, const double param3, const double param4)
{
    const double lambda = param2;
    const double tau = param3;
    const double w_sum_sym = param4;
    const auto total_len_features = len * dim;

    __shared__ double power_mult[CUDA_TILE_WIDTH];
    __shared__ double ta[CUDA_TILE_WIDTH][CUDA_TILE_WIDTH];
    __shared__ double tam1[CUDA_TILE_WIDTH][CUDA_TILE_WIDTH];//for index-1
    __shared__ double tb[CUDA_TILE_WIDTH][CUDA_TILE_WIDTH];
    __shared__ double tbm1[CUDA_TILE_WIDTH][CUDA_TILE_WIDTH];//for index-1

    const auto kk = threadIdx.x + blockIdx.x * blockDim.x;
    const auto mm = threadIdx.y + blockIdx.y * blockDim.y;

    const auto tx = threadIdx.x;
    const auto ty = threadIdx.y;
    //__syncthreads();

    if ((blockIdx.x * blockDim.x < numX) && (blockIdx.y * blockDim.y < numY)) {
        uint32_t kk_internal = 0;
        double matrix_prod_sum = 0;
        for (uint32_t jA = 0; jA < dim; ++jA) {
            //double s_mm = 0;
            for (uint32_t kk_internal_big = 0; kk_internal_big < len / CUDA_TILE_WIDTH + (len % CUDA_TILE_WIDTH == 0 ? 0 : 1); ++kk_internal_big) {
                if (tx == 0) {
                    if (ty + kk_internal_big * CUDA_TILE_WIDTH < len) {
                        power_mult[ty] = pow(1. / ((double) (len - (ty + kk_internal_big * CUDA_TILE_WIDTH))), /* 2. * */ lambda) * w_sum_sym;
                    }
                }
                if ((kk < numX) * (CUDA_TILE_WIDTH * kk_internal_big + ty < len)) {
                    ta[tx][ty] = _old_blockX(kk, CUDA_TILE_WIDTH * kk_internal_big + ty + jA * len);
                    if (CUDA_TILE_WIDTH * kk_internal_big + ty > 0) {
                        tam1[tx][ty] = ta[tx][ty] - _old_blockX(kk, CUDA_TILE_WIDTH * kk_internal_big + ty - 1 + jA * len);
                    }
                }
                if ((mm < numY) * (CUDA_TILE_WIDTH * kk_internal_big + tx < len)) {
                    tb[ty][tx] = _old_blockYY(mm, CUDA_TILE_WIDTH * kk_internal_big + tx + jA * len);
                    if (CUDA_TILE_WIDTH * kk_internal_big + tx > 0) {
                        tbm1[ty][tx] = tb[ty][tx] - _old_blockYY(mm, CUDA_TILE_WIDTH * kk_internal_big + tx - 1 + jA * len);
                    }
                }
                __syncthreads();
                if ((kk < numX) && (mm < numY)) {
                    for (int kk_internal_small = 0; kk_internal_small < CUDA_TILE_WIDTH; ++kk_internal_small) {
                        kk_internal = kk_internal_small + kk_internal_big * CUDA_TILE_WIDTH;
                        if (kk_internal < len) {
                            //mm_internal = kk_internal;
                            const double x_y = ta[tx][kk_internal_small] - tb[ty][kk_internal_small];
                            const double t_left = x_y * x_y;
                            double t_right = 0;
                            if (kk_internal > 0) {
                                double diff_x_y = 0;
                                diff_x_y = tam1[tx][kk_internal_small] - tbm1[ty][kk_internal_small];
                                t_right = diff_x_y * diff_x_y;
                            }
                            matrix_prod_sum += (t_left + tau * t_right) * power_mult[kk_internal_small];
                        }//end if kk_internal
                    }//end for kk_internal_small
                }//end if
                __syncthreads();//DO NOT REMOVE!
            }//end for kk_internal_big - tiles
            //matrix_prod_sum += s_mm;
        }//end for jA
        if ((kk < numX) && (mm < numY)) {
            //Z[(startX + kk) * sizeY + (startY + mm)] = 1. - matrix_prod_sum / (2. * sigma * sigma)+pow(matrix_prod_sum/(2*sigma*sigma),2)/2.;
            Z[(startX + kk) * sizeY + (startY + mm)] = matrix_prod_sum;
        }
    }//end if check get_global 0 and 1
}

/*
kernel::path::cu_distances_xy(features_t.n_cols, predict_features_t.n_cols, features_t.n_cols, predict_features_t.n_cols,
p_cums_X->at(i).n_rows, p_cums_X->at(i).mem, p_cums_Xy->at(i).mem, params.get_svr_kernel_param2(), z.memptr());
*/

void
do_gpu_kernel_compute_mat_xy(const uint32_t sizeX, const uint32_t sizeY, const uint32_t startX, const uint32_t startY, const uint32_t numX, const uint32_t numY, const uint32_t total_len_features, const uint32_t dim,
                             const double *X, const double *Y, double *Z, const double param2, const double param3, const double param4)
{
    common::gpu_context ctx;
    cudaSetDevice(ctx.phy_id());
    const uint32_t len = total_len_features / dim;

    const auto full_sizeX = sizeX * total_len_features;
    const auto full_sizeY = sizeY * total_len_features;
    const auto full_sizeZ = sizeX * sizeY;
    thrust::device_vector<double> d_X(full_sizeX);
    thrust::device_vector<double> d_Y(full_sizeY);
    cu_errchk(cudaMemcpy(thrust::raw_pointer_cast(d_X.data()), &X[0], sizeof(double) * full_sizeX, cudaMemcpyHostToDevice));
    cu_errchk(cudaMemcpy(thrust::raw_pointer_cast(d_Y.data()), &Y[0], sizeof(double) * full_sizeY, cudaMemcpyHostToDevice));

    double *d_Xptr = thrust::raw_pointer_cast(d_X.data());
    double *d_Yptr = thrust::raw_pointer_cast(d_Y.data());
    double *d_Zptr;
    cu_errchk(cudaMalloc(&d_Zptr, full_sizeZ * sizeof(double)));

    const uint32_t tile_x = CUDA_TILE_WIDTH;
    const uint32_t tile_y = CUDA_TILE_WIDTH;
    const dim3 thread_dim(tile_x, tile_y);
    const uint32_t block_x = (sizeX / tile_x) + (sizeX % tile_x == 0 ? 0 : 1);
    const uint32_t block_y = (sizeX / tile_y) + (sizeX % tile_y == 0 ? 0 : 1);
    const dim3 block_dim(block_x, block_y);

    _old_gpu_kernel_xy_compute<<<block_dim, thread_dim>>>(sizeX, sizeY, startX, startY, numX, numY, len, dim, d_Xptr, d_Yptr, d_Zptr, full_sizeZ, param2, param3, param4);
    cu_errchk(cudaMemcpy(&Z[0], d_Zptr, full_sizeZ * sizeof(double), cudaMemcpyDeviceToHost));
    cu_errchk(cudaFree(d_Zptr));
}


#define blockX(i, j) (X[(i) * lag + (j)])
#define blockYY(i, j) (Y[(i) * lag + (j)])
#define blockXX(i, j) (X[(i) * rows + (j)])
#define blockYYY(i, j) (Y[(i) * rows + (j)])
#define tx threadIdx.x
#define ty threadIdx.y


__global__  void
G_kernel_xx(const uint32_t cols, const uint32_t rows, const uint32_t lag, const uint32_t dim, const uint32_t len_TILE_WIDTH, const double lambda,
            const double *__restrict__ X, double *__restrict__ Z)
{
    if (blockIdx.x * blockDim.x >= cols || blockIdx.y * blockDim.y >= cols || blockIdx.x * blockDim.x >= blockIdx.y * blockDim.y + blockDim.y) return;

    __shared__ double power_mult[CUDA_TILE_WIDTH];
    __shared__ double ta[CUDA_TILE_WIDTH][CUDA_TILE_WIDTH];
    __shared__ double tam1[CUDA_TILE_WIDTH][CUDA_TILE_WIDTH];
    __shared__ double tb[CUDA_TILE_WIDTH][CUDA_TILE_WIDTH];
    __shared__ double tbm1[CUDA_TILE_WIDTH][CUDA_TILE_WIDTH];
    const auto kk = threadIdx.x + blockIdx.x * blockDim.x;
    const auto mm = threadIdx.y + blockIdx.y * blockDim.y;
    const bool kk_numX = kk < cols;
    const bool mm_numY = mm < cols;
    const bool do_matrix_product_sum = kk_numX && mm_numY && kk <= mm;
    double matrix_prod_sum = 0;
#ifndef __GNUC__
#pragma unroll
#endif
    for (uint32_t jA = 0; jA < dim; ++jA) {
        const auto jA_len = jA * lag;
#ifndef __GNUC__
#pragma unroll
#endif
        for (uint32_t kk_internal_big = 0; kk_internal_big < len_TILE_WIDTH; ++kk_internal_big) {
            const auto kk_internal_big_TILE_WIDTH = kk_internal_big * CUDA_TILE_WIDTH;
            const auto ty_kk_internal_big_TILE_WIDTH = ty + kk_internal_big_TILE_WIDTH;
            if (!tx) {
                if (ty_kk_internal_big_TILE_WIDTH < lag)
                    power_mult[ty] = pow(1. / double(lag - ty_kk_internal_big_TILE_WIDTH), lambda);
            }
            if (kk_numX && ty_kk_internal_big_TILE_WIDTH < lag) {
                ta[tx][ty] = blockXX(kk, ty_kk_internal_big_TILE_WIDTH + jA_len);
                if (ty_kk_internal_big_TILE_WIDTH)
                    tam1[tx][ty] = ta[tx][ty] - blockXX(kk, ty_kk_internal_big_TILE_WIDTH - 1 + jA_len);
            }
            const auto tx_kk_internal_big_TILE_WIDTH = kk_internal_big_TILE_WIDTH + tx;
            if (mm_numY && tx_kk_internal_big_TILE_WIDTH < lag) {
                tb[ty][tx] = blockXX(mm, tx_kk_internal_big_TILE_WIDTH + jA_len);
                if (tx_kk_internal_big_TILE_WIDTH)
                    tbm1[ty][tx] = tb[ty][tx] - blockXX(mm, tx_kk_internal_big_TILE_WIDTH - 1 + jA_len);
            }

            __syncthreads();

            if (do_matrix_product_sum) {
#ifndef __GNUC__
#pragma unroll
#endif
                for (uint32_t kk_internal_small = 0; kk_internal_small < CUDA_TILE_WIDTH; ++kk_internal_small) {
                    const auto kk_internal = kk_internal_small + kk_internal_big_TILE_WIDTH;
                    if (kk_internal >= lag) continue;
                    matrix_prod_sum += (pow(ta[tx][kk_internal_small] - tb[ty][kk_internal_small], 2) + common::C_kernel_path_tau *
                                                                                                        (kk_internal ? pow(
                                                                                                                tam1[tx][kk_internal_small] - tbm1[ty][kk_internal_small],
                                                                                                                2) : 0)) * power_mult[kk_internal_small];
                }
            }
            __syncthreads();
        }
    }
    if (kk_numX && mm_numY && kk <= mm) {
        Z[kk * cols + mm] = matrix_prod_sum / double(dim);
        if (kk < mm) Z[mm * cols + kk] = Z[kk * cols + mm];
    }
}

__global__  void
G_kernel_xx(const uint32_t cols, const uint32_t rows, const uint32_t lag, const uint32_t dim, const uint32_t len_TILE_WIDTH,
            const double lambda, const double gamma_2_pow_2, const double *__restrict__ X, double *__restrict__ Z)
{
    if (blockIdx.x * blockDim.x >= cols || blockIdx.y * blockDim.y >= cols || blockIdx.x * blockDim.x >= blockIdx.y * blockDim.y + blockDim.y) return;

    __shared__ double power_mult[CUDA_TILE_WIDTH];
    __shared__ double ta[CUDA_TILE_WIDTH][CUDA_TILE_WIDTH];
    __shared__ double tam1[CUDA_TILE_WIDTH][CUDA_TILE_WIDTH];//for index-1
    __shared__ double tb[CUDA_TILE_WIDTH][CUDA_TILE_WIDTH];
    __shared__ double tbm1[CUDA_TILE_WIDTH][CUDA_TILE_WIDTH];//for index-1
    const auto kk = threadIdx.x + blockIdx.x * blockDim.x;
    const auto mm = threadIdx.y + blockIdx.y * blockDim.y;
    const bool kk_numX = kk < cols;
    const bool mm_numY = mm < cols;
    const bool do_matrix_product_sum = kk_numX && mm_numY && kk <= mm;
    double matrix_prod_sum = 0;
#ifndef __GNUC__
#pragma unroll
#endif
    for (uint32_t jA = 0; jA < dim; ++jA) {
        const auto jA_len = jA * lag;
#ifndef __GNUC__
#pragma unroll
#endif
        for (uint32_t kk_internal_big = 0; kk_internal_big < len_TILE_WIDTH; ++kk_internal_big) {
            const auto kk_internal_big_TILE_WIDTH = kk_internal_big * CUDA_TILE_WIDTH;
            const auto ty_kk_internal_big_TILE_WIDTH = ty + kk_internal_big_TILE_WIDTH;
            if (!tx) {
                if (ty_kk_internal_big_TILE_WIDTH < lag)
                    power_mult[ty] = pow(1. / double(lag - ty_kk_internal_big_TILE_WIDTH), lambda);
            }
            if (kk_numX && ty_kk_internal_big_TILE_WIDTH < lag) {
                ta[tx][ty] = blockXX(kk, ty_kk_internal_big_TILE_WIDTH + jA_len);
                if (ty_kk_internal_big_TILE_WIDTH)
                    tam1[tx][ty] = ta[tx][ty] - blockXX(kk, ty_kk_internal_big_TILE_WIDTH - 1 + jA_len);
            }
            const auto tx_kk_internal_big_TILE_WIDTH = kk_internal_big_TILE_WIDTH + tx;
            if (mm_numY && tx_kk_internal_big_TILE_WIDTH < lag) {
                tb[ty][tx] = blockXX(mm, tx_kk_internal_big_TILE_WIDTH + jA_len);
                if (tx_kk_internal_big_TILE_WIDTH)
                    tbm1[ty][tx] = tb[ty][tx] - blockXX(mm, tx_kk_internal_big_TILE_WIDTH - 1 + jA_len);
            }

            __syncthreads();

            if (do_matrix_product_sum) {
#ifndef __GNUC__
#pragma unroll
#endif
                for (uint32_t kk_internal_small = 0; kk_internal_small < CUDA_TILE_WIDTH; ++kk_internal_small) {
                    const auto kk_internal = kk_internal_small + kk_internal_big_TILE_WIDTH;
                    if (kk_internal >= lag) continue;
                    matrix_prod_sum += (pow(ta[tx][kk_internal_small] - tb[ty][kk_internal_small], 2) + common::C_kernel_path_tau *
                                                                                                        (kk_internal ? pow(
                                                                                                                tam1[tx][kk_internal_small] - tbm1[ty][kk_internal_small],
                                                                                                                2) : 0)) * power_mult[kk_internal_small];
                }
            }
            __syncthreads();
        }
    }
    if (kk_numX && mm_numY && kk <= mm) {
        Z[kk * cols + mm] = 1. - matrix_prod_sum / double(dim) / gamma_2_pow_2;
        if (kk < mm) Z[mm * cols + kk] = Z[kk * cols + mm];
    }
}

__global__  void
G_kernel_xx(const uint32_t cols, const uint32_t lag, const uint32_t len_TILE_WIDTH, const double lambda, const double *__restrict__ X, double *__restrict__ Z)
{
    if (blockIdx.x * blockDim.x >= cols || blockIdx.y * blockDim.y >= cols || blockIdx.x * blockDim.x >= blockIdx.y * blockDim.y + blockDim.y) return;

    __shared__ double power_mult[CUDA_TILE_WIDTH];
    __shared__ double ta[CUDA_TILE_WIDTH][CUDA_TILE_WIDTH];
    __shared__ double tam1[CUDA_TILE_WIDTH][CUDA_TILE_WIDTH]; // for index-1
    __shared__ double tb[CUDA_TILE_WIDTH][CUDA_TILE_WIDTH];
    __shared__ double tbm1[CUDA_TILE_WIDTH][CUDA_TILE_WIDTH]; // for index-1
    const auto kk = threadIdx.x + blockIdx.x * blockDim.x;
    const auto mm = threadIdx.y + blockIdx.y * blockDim.y;
    const bool mm_cols = mm < cols;
    const bool kk_cols = kk < cols;
    const auto do_matrix_product_sum = mm_cols && kk_cols;
    double matrix_prod_sum = 0;
#ifndef __GNUC__
#pragma unroll
#endif
    for (uint32_t kk_internal_big = 0; kk_internal_big < len_TILE_WIDTH; ++kk_internal_big) {
        const auto kk_internal_big_TILE_WIDTH = kk_internal_big * CUDA_TILE_WIDTH;
        const auto ty_kk_internal_big_TILE_WIDTH = ty + kk_internal_big_TILE_WIDTH;
        if (ty_kk_internal_big_TILE_WIDTH < lag) {
            if (!tx) power_mult[ty] = pow(1. / double(lag - ty_kk_internal_big_TILE_WIDTH), lambda);
            if (kk_cols) {
                ta[tx][ty] = blockX(kk, ty_kk_internal_big_TILE_WIDTH);
                if (ty_kk_internal_big_TILE_WIDTH)
                    tam1[tx][ty] = ta[tx][ty] - blockX(kk, ty_kk_internal_big_TILE_WIDTH - 1);
            }
        }
        const auto tx_kk_internal_big_TILE_WIDTH = tx + kk_internal_big_TILE_WIDTH;
        if (mm_cols && tx_kk_internal_big_TILE_WIDTH < lag) {
            tb[ty][tx] = blockX(mm, tx_kk_internal_big_TILE_WIDTH);
            if (tx_kk_internal_big_TILE_WIDTH)
                tbm1[ty][tx] = tb[ty][tx] - blockX(mm, tx_kk_internal_big_TILE_WIDTH - 1);
        }
        __syncthreads();

        if (do_matrix_product_sum)
#ifndef __GNUC__
#pragma unroll
#endif
            for (uint32_t kk_internal_small = 0; kk_internal_small < CUDA_TILE_WIDTH; ++kk_internal_small) {
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
G_kernel_xy(const uint32_t X_cols, const uint32_t Y_cols, const uint32_t rows, const uint32_t lag, const uint32_t dim, const uint32_t len_TILE_WIDTH, const double lambda,
            const double *__restrict__ X, const double *__restrict__ Y, double *__restrict__ Z)
{
    if (blockIdx.x * blockDim.x >= X_cols || blockIdx.y * blockDim.y >= Y_cols) return;

    __shared__ double power_mult[CUDA_TILE_WIDTH];
    __shared__ double ta[CUDA_TILE_WIDTH][CUDA_TILE_WIDTH];
    __shared__ double tam1[CUDA_TILE_WIDTH][CUDA_TILE_WIDTH]; // for index-1
    __shared__ double tb[CUDA_TILE_WIDTH][CUDA_TILE_WIDTH];
    __shared__ double tbm1[CUDA_TILE_WIDTH][CUDA_TILE_WIDTH]; // for index-1

    const auto kk = threadIdx.x + blockIdx.x * blockDim.x;
    const auto mm = threadIdx.y + blockIdx.y * blockDim.y;
    const bool kk_X = kk < X_cols;
    const bool mm_Y = mm < Y_cols;
    const auto do_matrix_product_sum = mm_Y && kk_X;
    double matrix_prod_sum = 0;
#ifndef __GNUC__
#pragma unroll
#endif
    for (uint32_t jA = 0; jA < dim; ++jA) {
        const auto jA_len = jA * lag;
#ifndef __GNUC__
#pragma unroll
#endif
        for (uint32_t kk_internal_big = 0; kk_internal_big < len_TILE_WIDTH; ++kk_internal_big) {
            const auto kk_internal_big_TILE_WIDTH = kk_internal_big * CUDA_TILE_WIDTH;
            const auto ty_kk_internal_big_TILE_WIDTH = ty + kk_internal_big_TILE_WIDTH;
            if (ty_kk_internal_big_TILE_WIDTH < lag) {
                if (!tx) power_mult[ty] = pow(1. / double(lag - ty_kk_internal_big_TILE_WIDTH), lambda);
                if (kk_X) {
                    ta[tx][ty] = blockXX(kk, ty_kk_internal_big_TILE_WIDTH + jA_len);
                    if (ty_kk_internal_big_TILE_WIDTH) tam1[tx][ty] = ta[tx][ty] - blockXX(kk, ty_kk_internal_big_TILE_WIDTH + jA_len - 1);
                }
            }

            const auto tx_kk_internal_big_TILE_WIDTH = tx + kk_internal_big_TILE_WIDTH;
            if (mm_Y && tx_kk_internal_big_TILE_WIDTH < lag) {
                tb[ty][tx] = blockYYY(mm, tx_kk_internal_big_TILE_WIDTH + jA_len);
                if (tx_kk_internal_big_TILE_WIDTH)
                    tbm1[ty][tx] = tb[ty][tx] - blockYYY(mm, tx_kk_internal_big_TILE_WIDTH + jA_len - 1);
            }
            __syncthreads();

            if (do_matrix_product_sum)
#ifndef __GNUC__
#pragma unroll
#endif
                for (uint32_t kk_internal_small = 0; kk_internal_small < CUDA_TILE_WIDTH; ++kk_internal_small) {
                    const auto kk_internal = kk_internal_small + kk_internal_big_TILE_WIDTH;
                    if (kk_internal >= lag) continue;
                    matrix_prod_sum += (pow(ta[tx][kk_internal_small] - tb[ty][kk_internal_small], 2)
                                        + (kk_internal ? common::C_kernel_path_tau * pow(tam1[tx][kk_internal_small] - tbm1[ty][kk_internal_small], 2) : 0.))
                                       * power_mult[kk_internal_small];
                }
            __syncthreads();
        }
    }
    if (do_matrix_product_sum) Z[kk * Y_cols + mm] = matrix_prod_sum / double(dim);
}

__global__  void
G_kernel_xy(const uint32_t X_cols, const uint32_t Y_cols, const uint32_t rows, const uint32_t lag, const uint32_t dim, const uint32_t len_TILE_WIDTH,
            const double lambda, const double gamma_2_2, const double *X, const double *Y, double *Z)
{
    if (blockIdx.x * blockDim.x >= X_cols || blockIdx.y * blockDim.y >= Y_cols) return;

    __shared__ double power_mult[CUDA_TILE_WIDTH];
    __shared__ double ta[CUDA_TILE_WIDTH][CUDA_TILE_WIDTH];
    __shared__ double tam1[CUDA_TILE_WIDTH][CUDA_TILE_WIDTH]; // for index-1
    __shared__ double tb[CUDA_TILE_WIDTH][CUDA_TILE_WIDTH];
    __shared__ double tbm1[CUDA_TILE_WIDTH][CUDA_TILE_WIDTH]; // for index-1

    const auto kk = threadIdx.x + blockIdx.x * blockDim.x;
    const auto mm = threadIdx.y + blockIdx.y * blockDim.y;
    const bool mm_Y = mm < Y_cols;
    const bool kk_X = kk < X_cols;
    const auto do_matrix_product_sum = mm_Y && kk_X;
    double matrix_prod_sum = 0;
#ifndef __GNUC__
#pragma unroll
#endif
    for (uint32_t jA = 0; jA < dim; ++jA) {
        const auto jA_len = jA * lag;
#ifndef __GNUC__
#pragma unroll
#endif
        for (uint32_t kk_internal_big = 0; kk_internal_big < len_TILE_WIDTH; ++kk_internal_big) {
            const auto kk_internal_big_TILE_WIDTH = kk_internal_big * CUDA_TILE_WIDTH;
            const auto ty_kk_internal_big_TILE_WIDTH = ty + kk_internal_big_TILE_WIDTH;
            if (ty_kk_internal_big_TILE_WIDTH < lag) {
                if (!tx) power_mult[ty] = pow(1. / double(lag - ty_kk_internal_big_TILE_WIDTH), lambda);
                if (kk_X) {
                    ta[tx][ty] = blockXX(kk, ty_kk_internal_big_TILE_WIDTH + jA_len);
                    if (ty_kk_internal_big_TILE_WIDTH) tam1[tx][ty] = ta[tx][ty] - blockXX(kk, ty_kk_internal_big_TILE_WIDTH + jA_len - 1);
                }
            }

            const auto tx_kk_internal_big_TILE_WIDTH = tx + kk_internal_big_TILE_WIDTH;
            if (mm_Y && tx_kk_internal_big_TILE_WIDTH < lag) {
                tb[ty][tx] = blockYYY(mm, tx_kk_internal_big_TILE_WIDTH + jA_len);
                if (tx_kk_internal_big_TILE_WIDTH)
                    tbm1[ty][tx] = tb[ty][tx] - blockYYY(mm, tx_kk_internal_big_TILE_WIDTH + jA_len - 1);
            }
            __syncthreads();

            if (do_matrix_product_sum)
#ifndef __GNUC__
#pragma unroll
#endif
                for (uint32_t kk_internal_small = 0; kk_internal_small < CUDA_TILE_WIDTH; ++kk_internal_small) {
                    const auto kk_internal = kk_internal_small + kk_internal_big_TILE_WIDTH;
                    if (kk_internal >= lag) continue;
                    matrix_prod_sum += (pow(ta[tx][kk_internal_small] - tb[ty][kk_internal_small], 2)
                                        + (kk_internal ? common::C_kernel_path_tau * pow(tam1[tx][kk_internal_small] - tbm1[ty][kk_internal_small], 2) : 0.))
                                       * power_mult[kk_internal_small];
                }
            __syncthreads();
        }
    }
    if (do_matrix_product_sum) Z[kk * Y_cols + mm] = 1. - matrix_prod_sum / double(dim) / gamma_2_2;
}

__global__  void
G_kernel_xy(const uint32_t X_cols, const uint32_t Y_cols, const uint32_t lag, const unsigned len_TILE_WIDTH, const double lambda,
            const double *__restrict__ X, const double *__restrict__ Y, double *__restrict__ Z)
{
    if (blockIdx.x * blockDim.x >= X_cols || blockIdx.y * blockDim.y >= Y_cols) return;

    __shared__ double power_mult[CUDA_TILE_WIDTH];
    __shared__ double ta[CUDA_TILE_WIDTH][CUDA_TILE_WIDTH];
    __shared__ double tam1[CUDA_TILE_WIDTH][CUDA_TILE_WIDTH]; // for index-1
    __shared__ double tb[CUDA_TILE_WIDTH][CUDA_TILE_WIDTH];
    __shared__ double tbm1[CUDA_TILE_WIDTH][CUDA_TILE_WIDTH]; // for index-1

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
        const auto kk_internal_big_TILE_WIDTH = kk_internal_big * CUDA_TILE_WIDTH;
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
            if (tx_kk_internal_big_TILE_WIDTH)
                tbm1[ty][tx] = tb[ty][tx] - blockYY(mm, tx_kk_internal_big_TILE_WIDTH - 1);
        }
        __syncthreads();

        if (do_matrix_product_sum)
#ifndef __GNUC__
#pragma unroll
#endif
            for (uint32_t kk_internal_small = 0; kk_internal_small < CUDA_TILE_WIDTH; ++kk_internal_small) {
                const auto kk_internal = kk_internal_small + kk_internal_big_TILE_WIDTH;
                if (kk_internal >= lag) continue;
                matrix_prod_sum += (pow(ta[tx][kk_internal_small] - tb[ty][kk_internal_small], 2)
                                    + (kk_internal ? common::C_kernel_path_tau * pow(tam1[tx][kk_internal_small] - tbm1[ty][kk_internal_small], 2) : 0.))
                                   * power_mult[kk_internal_small];
            }
        __syncthreads();
    }

    if (do_matrix_product_sum) Z[kk * Y_cols + mm] = matrix_prod_sum;
}

void cu_distances_xx(const uint32_t cols, const uint32_t lag, const double *X, const double lambda, double *Z) // rows == lag
{
    const uint32_t X_size = cols * lag * sizeof(double);
    const uint32_t Z_size = cols * cols * sizeof(double);
    const uint32_t len_TILE_WIDTH = std::ceil(double(lag) / double(CUDA_TILE_WIDTH));
    double *d_Z, *d_X;
    const common::gpu_context ctx;
    cu_errchk(cudaSetDevice(ctx.phy_id()));
    cudaStream_t cu_stream;
    cu_errchk(cudaStreamCreate(&cu_stream));
    cu_errchk(cudaMallocAsync(&d_X, X_size, cu_stream));
    cu_errchk(cudaMallocAsync(&d_Z, Z_size, cu_stream));
    cu_errchk(cudaMemcpyAsync(d_X, X, X_size, cudaMemcpyHostToDevice, cu_stream));
    G_kernel_xx<<<CUDA_THREADS_BLOCKS_2D(cols), 0, cu_stream>>>(cols, lag, len_TILE_WIDTH, lambda, d_X, d_Z);
    cu_errchk(cudaMemcpyAsync(Z, d_Z, Z_size, cudaMemcpyDeviceToHost, cu_stream));
    cu_errchk(cudaFreeAsync(d_Z, cu_stream));
    cu_errchk(cudaFreeAsync(d_X, cu_stream));
    cu_errchk(cudaDeviceSynchronize());
    cu_errchk(cudaStreamDestroy(cu_stream));
}

void cu_distances_xx(const uint32_t cols, const uint32_t rows, const uint32_t lag, const double lambda, const double *X, double *Z)
{
    const uint32_t lag_TILE_WIDTH = std::ceil(double(lag) / double(CUDA_TILE_WIDTH));
    const uint32_t X_size = cols * rows * sizeof(double);
    const uint32_t Z_size = cols * cols * sizeof(double);
    double *d_Z, *d_X;
    const common::gpu_context ctx;
    cu_errchk(cudaSetDevice(ctx.phy_id()));
    cudaStream_t cu_stream;
    cu_errchk(cudaStreamCreate(&cu_stream));
    cu_errchk(cudaMallocAsync(&d_X, X_size, cu_stream));
    cu_errchk(cudaMemcpyAsync(d_X, X, X_size, cudaMemcpyHostToDevice, cu_stream));
    cu_errchk(cudaMallocAsync(&d_Z, Z_size, cu_stream));
    G_kernel_xx<<<CUDA_THREADS_BLOCKS_2D(cols), 0, cu_stream>>>(cols, rows, lag, rows / lag, lag_TILE_WIDTH, lambda, d_X, d_Z);
    cu_errchk(cudaFreeAsync(d_X, cu_stream));
    cu_errchk(cudaMemcpyAsync(Z, d_Z, Z_size, cudaMemcpyDeviceToHost, cu_stream));
    cu_errchk(cudaFreeAsync(d_Z, cu_stream));
    cu_errchk(cudaDeviceSynchronize());
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
    cu_errchk(cudaStreamCreate(&cu_stream));
    cu_errchk(cudaMallocAsync(&d_X, X_size, cu_stream));
    cu_errchk(cudaMemcpyAsync(d_X, X, X_size, cudaMemcpyHostToDevice, cu_stream));
    cu_errchk(cudaMallocAsync(&d_Z, Z_size, cu_stream));
    G_kernel_xx<<<CUDA_THREADS_BLOCKS_2D(cols), 0, cu_stream>>>(cols, rows, lag, rows / lag, uint32_t(std::ceil(double(lag) / double(CUDA_TILE_WIDTH))), lambda, 2 * gamma * gamma, d_X, d_Z);
    cu_errchk(cudaFreeAsync(d_X, cu_stream));
    cu_errchk(cudaMemcpyAsync(Z, d_Z, Z_size, cudaMemcpyDeviceToHost, cu_stream));
    cu_errchk(cudaFreeAsync(d_Z, cu_stream));
    cu_errchk(cudaDeviceSynchronize());
    cu_errchk(cudaStreamDestroy(cu_stream));
}


void cu_distances_xy(const uint32_t X_cols, const uint32_t Xy_cols, const uint32_t lag, const double lambda, const double *X, const double *Xy, double *Z)
{
    const uint32_t lag_TILE_WIDTH = std::ceil(double(lag) / double(CUDA_TILE_WIDTH));
    const auto X_size = X_cols * lag * sizeof(double);
    const auto Xy_size = Xy_cols * lag * sizeof(double);
    const auto Z_size = X_cols * Xy_cols * sizeof(double);
    const auto block_width = X_cols / CUDA_TILE_WIDTH + (X_cols % CUDA_TILE_WIDTH == 0 ? 0 : 1);
    const common::gpu_context ctx;
    cu_errchk(cudaSetDevice(ctx.phy_id()));
    cudaStream_t cu_stream;
    cu_errchk(cudaStreamCreate(&cu_stream));
    double *d_X, *d_Xy, *d_Z;
    cu_errchk(cudaMallocAsync(&d_X, X_size, cu_stream));
    cu_errchk(cudaMemcpyAsync(d_X, X, X_size, cudaMemcpyHostToDevice, cu_stream));
    cu_errchk(cudaMallocAsync(&d_Xy, Xy_size, cu_stream));
    cu_errchk(cudaMemcpyAsync(d_Xy, Xy, Xy_size, cudaMemcpyHostToDevice, cu_stream));
    cu_errchk(cudaMallocAsync(&d_Z, Z_size, cu_stream));
    G_kernel_xy<<<dim3(block_width, block_width), dim3(CUDA_TILE_WIDTH, CUDA_TILE_WIDTH), 0, cu_stream>>>(X_cols, Xy_cols, lag, lag_TILE_WIDTH, lambda, d_X, d_Xy, d_Z);
    cu_errchk(cudaFreeAsync(d_X, cu_stream));
    cu_errchk(cudaFreeAsync(d_Xy, cu_stream));
    cu_errchk(cudaMemcpyAsync(Z, d_Z, Z_size, cudaMemcpyDeviceToHost, cu_stream));
    cu_errchk(cudaFreeAsync(d_Z, cu_stream));
    cu_errchk(cudaDeviceSynchronize());
    cu_errchk(cudaStreamDestroy(cu_stream));
}


void cu_distances_xy(const uint32_t X_cols, const uint32_t Xy_cols, const uint32_t rows, const uint32_t lag, const double lambda, const double *X, const double *Xy, double *Z)
{
    const uint32_t X_size = X_cols * lag * sizeof(double);
    const uint32_t Xy_size = Xy_cols * lag * sizeof(double);
    const uint32_t Z_size = X_cols * Xy_cols * sizeof(double);
    double *d_X, *d_Xy, *d_Z;
    const common::gpu_context ctx;
    cu_errchk(cudaSetDevice(ctx.phy_id()));
    cudaStream_t cu_stream;
    cu_errchk(cudaStreamCreate(&cu_stream));
    cu_errchk(cudaMallocAsync(&d_X, X_size, cu_stream));
    cu_errchk(cudaMemcpyAsync(d_X, X, X_size, cudaMemcpyHostToDevice, cu_stream));
    cu_errchk(cudaMallocAsync(&d_Xy, Xy_size, cu_stream));
    cu_errchk(cudaMemcpyAsync(d_Xy, Xy, Xy_size, cudaMemcpyHostToDevice, cu_stream));
    cu_errchk(cudaMallocAsync(&d_Z, Z_size, cu_stream));
    const uint32_t len_TILE_WIDTH = std::ceil(double(lag) / double(CUDA_TILE_WIDTH));
    G_kernel_xy<<<CUDA_THREADS_BLOCKS_2D(X_cols), 0, cu_stream>>>(X_cols, Xy_cols, lag, rows, rows / lag, len_TILE_WIDTH, lambda, d_X, d_Xy, d_Z);
    cu_errchk(cudaFreeAsync(d_X, cu_stream));
    cu_errchk(cudaFreeAsync(d_Xy, cu_stream));
    cu_errchk(cudaMemcpyAsync(Z, d_Z, Z_size, cudaMemcpyDeviceToHost, cu_stream));
    cu_errchk(cudaFreeAsync(d_Z, cu_stream));
    cu_errchk(cudaDeviceSynchronize());
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
    cu_errchk(cudaStreamCreate(&cu_stream));
    cu_errchk(cudaMallocAsync(&d_X, X_size, cu_stream));
    cu_errchk(cudaMemcpyAsync(d_X, X, X_size, cudaMemcpyHostToDevice, cu_stream));
    cu_errchk(cudaMallocAsync(&d_Xy, Xy_size, cu_stream));
    cu_errchk(cudaMemcpyAsync(d_Xy, Xy, Xy_size, cudaMemcpyHostToDevice, cu_stream));
    cu_errchk(cudaMallocAsync(&d_Z, Z_size, cu_stream));
    const uint32_t lag_TILE_WIDTH = std::ceil(double(lag) / double(CUDA_TILE_WIDTH));
    G_kernel_xy<<<CUDA_THREADS_BLOCKS_2D(X_cols), 0, cu_stream>>>(X_cols, Xy_cols, rows, lag, rows / lag, lag_TILE_WIDTH, lambda, 2. * gamma * gamma, d_X, d_Xy, d_Z);
    cu_errchk(cudaFreeAsync(d_X, cu_stream));
    cu_errchk(cudaFreeAsync(d_Xy, cu_stream));
    cu_errchk(cudaMemcpyAsync(Z, d_Z, Z_size, cudaMemcpyDeviceToHost, cu_stream));
    cu_errchk(cudaFreeAsync(d_Z, cu_stream));
    cu_errchk(cudaDeviceSynchronize());
    cu_errchk(cudaStreamDestroy(cu_stream));
}

}
}
