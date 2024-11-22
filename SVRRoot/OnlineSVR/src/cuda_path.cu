#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include "common/defines.h"
#include "cuda_path.cuh"
#include "common/cuda_util.cuh"
#include "common/gpu_handler.hpp"
#include "common/constants.hpp"
#include "onlinesvr.hpp"
#include "model/SVRParameters.hpp"
#include "cuqrsolve.cuh"
#include "common/constants.hpp"

// TODO Reimplement properly the path kernel (or a better choice eg. NTK) according to https://www.csc.kth.se/~fpokorny/static/publications/baisero2013a.pdf

// #define HIFI_PATH // Actually lowers precision when tuning parameters, so keep off for now

#define blockXX(i, j) (X[(i) * rows + (j)])
#define blockYY(i, j) (Y[(i) * rows + (j)])

#define tx threadIdx.x
#define ty threadIdx.y

namespace svr {
namespace kernel::path {


__device__ __forceinline__ double
do_product_sum(const uint32_t rows, const uint32_t lag, const uint32_t dim, const uint32_t lag_TILE_WIDTH, const double lambda, const double tau, CRPTRd X,
               CRPTRd Y, double power_mult[32], double ta[32][32], double tam1[32][32], double tb[32][32], double tbm1[32][32], const uint32_t kk,
               const uint32_t mm, const bool kk_X, const bool mm_Y, const bool do_matrix_product_sum)
{
    double matrix_prod_sum = 0;
UNROLL()
    for (uint32_t jA = 0; jA < dim; ++jA) {
        const auto jA_lag = jA * lag;
UNROLL()
        for (uint32_t kk_internal_big = 0; kk_internal_big < lag_TILE_WIDTH; ++kk_internal_big) {
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
                tb[ty][tx] = blockYY(mm, tx_kk_internal_big_TILE_WIDTH + jA_lag);
                if (tx_kk_internal_big_TILE_WIDTH) tbm1[ty][tx] = tb[ty][tx] - blockYY(mm, tx_kk_internal_big_TILE_WIDTH + jA_lag - 1);
            }
            __syncthreads();

            if (do_matrix_product_sum)
UNROLL(common::C_cu_tile_width)
                for (uint32_t kk_internal_small = 0; kk_internal_small < common::C_cu_tile_width; ++kk_internal_small) {
                    const auto kk_internal = kk_internal_small + kk_internal_big_TILE_WIDTH;
                    if (kk_internal >= lag) continue;
#ifdef HIFI_PATH
                    matrix_prod_sum += (DIST(ta[tx][kk_internal_small] - tb[ty][kk_internal_small]) +
                            (kk_internal ? (C_kernel_path_tau * DIST(tam1[tx][kk_internal_small] - tbm1[ty][kk_internal_small])) : 0.)) *
                                    power_mult[kk_internal_small] / double(dim);
#else
                    matrix_prod_sum += (DIST(ta[tx][kk_internal_small] - tb[ty][kk_internal_small]) +
                            (kk_internal ? (C_kernel_path_tau * DIST(tam1[tx][kk_internal_small] - tbm1[ty][kk_internal_small])) : 0.)) *
                                    power_mult[kk_internal_small];
#endif
                }
            __syncthreads();
        }
    }
    return matrix_prod_sum;
}

__global__  void
G_kernel_xy(const uint32_t X_cols, const uint32_t Y_cols, const uint32_t rows, const uint32_t lag, const uint32_t dim, const uint32_t lag_TILE_WIDTH, const double lambda,
            const double tau, const double *const X, const double *const Y, double *Z)
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
    const auto matrix_prod_sum = do_product_sum(
            rows, lag, dim, lag_TILE_WIDTH, lambda, tau, X, Y, power_mult, ta, tam1, tb, tbm1, kk, mm, kk_X, mm_Y, do_matrix_product_sum);
    if (do_matrix_product_sum)
#ifdef HIFI_PATH
        Z[kk * Y_cols + mm] = matrix_prod_sum;
#else
        Z[kk * Y_cols + mm] = matrix_prod_sum / dim;
#endif
}


__global__  void
G_kernel_xy(const uint32_t X_cols, const uint32_t Y_cols, const uint32_t rows, const uint32_t lag, const uint32_t dim, const uint32_t lag_TILE_WIDTH, const double lambda,
            const double tau, const double gamma, const double *const X, const double *const Y, double *Z)
{
    if (blockIdx.x * blockDim.x >= X_cols || blockIdx.y * blockDim.y >= Y_cols) return;

    __shared__ double power_mult[common::C_cu_tile_width];
    __shared__ double ta[common::C_cu_tile_width][common::C_cu_tile_width];
    __shared__ double tam1[common::C_cu_tile_width][common::C_cu_tile_width];
    __shared__ double tb[common::C_cu_tile_width][common::C_cu_tile_width];
    __shared__ double tbm1[common::C_cu_tile_width][common::C_cu_tile_width];

    const auto kk = threadIdx.x + blockIdx.x * blockDim.x;
    const auto mm = threadIdx.y + blockIdx.y * blockDim.y;
    const bool kk_X = kk < X_cols;
    const bool mm_Y = mm < Y_cols;
    const auto do_matrix_product_sum = mm_Y && kk_X;
    const auto matrix_prod_sum = do_product_sum(rows, lag, dim, lag_TILE_WIDTH, lambda, 0, X, Y, power_mult, ta, tam1, tb, tbm1, kk, mm, kk_X, mm_Y, do_matrix_product_sum);
    if (do_matrix_product_sum)
#ifdef HIFI_PATH
        Z[kk * Y_cols + mm] = matrix_prod_sum;
#else
        Z[kk * Y_cols + mm] = matrix_prod_sum / dim / gamma;
#endif
}

__global__ void G_threshold(RPTR(double) Z, const uint32_t len, const double threshold)
{
    const auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len && abs(Z[i]) < threshold) Z[i] = 0;
}

void cu_threshold(RPTR(double) v, const uint32_t n, const cudaStream_t custream)
{
    return;

    const auto meanabs = solvers::meanabs(v, n, custream);
    G_threshold<<<CU_BLOCKS_THREADS(n), 0, custream>>>(v, n, meanabs / 2);
}

void cu_distances_xx(const uint32_t cols, const uint32_t rows, const uint32_t lag, const double lambda, const double tau, CRPTR(double) X, RPTR(double) Z)
{
    assert(rows % lag == 0);
    const uint32_t X_size = cols * rows * sizeof(double);
    const auto Z_len = cols * cols;
    const uint32_t Z_size = Z_len * sizeof(double);
    double *d_Z, *d_X;
    CTX_CUSTREAM;
    cu_errchk(cudaMallocAsync(&d_X, X_size, custream));
    cu_errchk(cudaMemcpyAsync(d_X, X, X_size, cudaMemcpyHostToDevice, custream));
    cu_errchk(cudaMallocAsync(&d_Z, Z_size, custream));
    G_kernel_xy<<<CU_BLOCKS_THREADS_2D(cols), 0, custream>>>(cols, cols, rows, lag, rows / lag, CDIVI(lag, common::C_cu_tile_width), lambda, 0, d_X, d_X, d_Z);
    cu_threshold(d_Z, Z_len, custream);
    cu_errchk(cudaFreeAsync(d_X, custream));
    cu_errchk(cudaMemcpyAsync(Z, d_Z, Z_size, cudaMemcpyDeviceToHost, custream));
    cu_errchk(cudaFreeAsync(d_Z, custream));
    cusyndestroy(custream);
}


void
cu_distances_xy(const uint32_t X_cols, const uint32_t Xy_cols, const uint32_t rows, const uint32_t lag, const double lambda, const double tau, CRPTRd X, CRPTRd Xy, RPTR(double) Z)
{
    const uint32_t X_size = X_cols * lag * sizeof(double);
    const uint32_t Xy_size = Xy_cols * lag * sizeof(double);
    const auto Z_len = X_cols * Xy_cols;
    const uint32_t Z_size = Z_len * sizeof(double);
    double *d_X, *d_Xy, *d_Z;
    CTX_CUSTREAM;
    cu_errchk(cudaMallocAsync(&d_X, X_size, custream));
    cu_errchk(cudaMemcpyAsync(d_X, X, X_size, cudaMemcpyHostToDevice, custream));
    cu_errchk(cudaMallocAsync(&d_Xy, Xy_size, custream));
    cu_errchk(cudaMemcpyAsync(d_Xy, Xy, Xy_size, cudaMemcpyHostToDevice, custream));
    cu_errchk(cudaMallocAsync(&d_Z, Z_size, custream));
    G_kernel_xy<<<CU_BLOCKS_THREADS_2D(X_cols), 0, custream>>>(X_cols, Xy_cols, lag, rows, rows / lag, CDIVI(lag, common::C_cu_tile_width), lambda, 0, d_X, d_Xy, d_Z);
    cu_threshold(d_Z, Z_len, custream);
    cu_errchk(cudaFreeAsync(d_X, custream));
    cu_errchk(cudaFreeAsync(d_Xy, custream));
    cu_errchk(cudaMemcpyAsync(Z, d_Z, Z_size, cudaMemcpyDeviceToHost, custream));
    cu_errchk(cudaFreeAsync(d_Z, custream));
    cusyndestroy(custream);
}


void cu_kernel_xx(const uint32_t cols, const uint32_t rows, const uint32_t lag, const double lambda, const double tau, const double gamma, CRPTRd X, RPTR(double) K)
{
    LOG4_THROW("Kills precision!");
    const auto K_len = cols * cols;
    const uint32_t K_size = K_len * sizeof(double);
    double *d_K;
    CTX4_CUSTREAM;
    const auto d_X = cumallocopy(X, custream, rows * cols);
    cu_errchk(cudaMallocAsync(&d_K, K_size, custream));
    G_kernel_xy<<<CU_BLOCKS_THREADS_2D(cols), 0, custream>>>(cols, cols, rows, lag, rows / lag, CDIVI(lag, common::C_cu_tile_width), lambda, 0, gamma, d_X, d_X, d_K);
    cu_errchk(cudaFreeAsync(d_X, custream));
    cufreecopy(K, d_K, custream, K_len);
    cusyndestroy(custream);
}


void cu_kernel_xy(const uint32_t X_cols, const uint32_t Xy_cols, const uint32_t rows, const uint32_t lag, const double lambda, const double gamma,
                  CPTRd X, CPTRd Xy, RPTR(double) K)
{
    LOG4_THROW("Kills precision!");
    const auto K_len = X_cols * Xy_cols;
    const auto K_size = K_len * sizeof(double);
    double *d_K;
    CTX4_CUSTREAM;
    const auto d_X = cumallocopy(X, custream, X_cols * rows);
    const auto d_Xy = cumallocopy(Xy, custream, Xy_cols * rows);
    cu_errchk(cudaMallocAsync(&d_K, K_size, custream));
    G_kernel_xy<<<CU_BLOCKS_THREADS_2D(X_cols), 0, custream>>>(X_cols, Xy_cols, rows, lag, rows / lag, CDIVI(lag, common::C_cu_tile_width), lambda, 0, gamma, d_X, d_Xy, d_K);
    cu_errchk(cudaFreeAsync(d_X, custream));
    cu_errchk(cudaFreeAsync(d_Xy, custream));
    cu_errchk(cudaMemcpyAsync(K, d_K, K_size, cudaMemcpyDeviceToHost, custream));
    cu_errchk(cudaFreeAsync(d_K, custream));
    cusyndestroy(custream);
}


}
}
