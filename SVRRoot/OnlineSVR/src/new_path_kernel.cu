//
// Created by zarko on 19/10/2024.
//
#include <iostream>
#include <cuda_runtime.h>
#include <boost/math/special_functions/pow.hpp>
#include "new_path_kernel.cuh"
#include "common/logging.hpp"
#include "common/cuda_util.cuh"
#include "cuda_path.cuh"
#include "model/SVRParameters.hpp"

namespace svr {
namespace kernel {

// Copy upper triangle to lower triangle, on a non-symetric matrix the shorter axis is used for mirroring
__global__ void G_copy_upper_tri(RPTR(double) result, const unsigned rows, const unsigned cols)
{
    const auto row = blockIdx.x * blockDim.x + threadIdx.x;
    const auto col = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= cols || row >= rows || col >= row) return;
    // if (row < 5 && col < 5) printf("G_copy_upper_tri: row %u, col %u, res1 %f, res2 %f\n", row, col, result[row + col * rows], result[row * _MIN(rows, cols) + col]);
    result[row * cols + col] = result[row + col * rows];
}

// Original Path kernel implementation - buggy
constexpr double CHV = 0.5; // Weight for vertical/horizontal steps
constexpr double CD = 0.3; // Weight for diagonal steps

__device__ __forceinline__ double ground_kernel(const double a, const double b)
{
    return fabs(a - b); // (a == b) ? 1.0 : 0.0; // Simple example: returns 1 if equal, else 0
}

__device__ __forceinline__ double path_distances(CRPTRd s, CRPTRd t, const unsigned length, const unsigned length_1, const double lambda)
{
    constexpr unsigned tmp_len = (datamodel::C_default_svrparam_lag_count + 1) * 2;
    constexpr unsigned last_ix = (datamodel::C_default_svrparam_lag_count % 2) * (datamodel::C_default_svrparam_lag_count + 1) + datamodel::C_default_svrparam_lag_count;
    double tmp[tmp_len]; // TODO change to kernel with shared memory of length_1 * 2
    memset(tmp + sizeof(double), 0, length_1 * sizeof(double));
    *tmp = ground_kernel(*s, *t);
    UNROLL(datamodel::C_default_svrparam_lag_count)
    for (unsigned i = 1; i <= length; ++i)
        for (unsigned j = 1; j <= length; ++j) {
            const auto i_1 = i - 1;
            const auto j_1 = j - 1;
            const auto i_1_len = (i_1 % 2) * length_1;
            const auto i_2 = i % 2;
            tmp[i_2 * length_1 + j] = ground_kernel(s[i_1], t[j_1]) + lambda * (tmp[i_1_len + j] + CHV + tmp[i_2 * length_1 + j_1] + CHV + tmp[i_1_len + j_1] + CD);
        }
    return tmp[last_ix];
}

// KDTW kernel
constexpr double C_gk_epsilon = 1e-3;
constexpr double C_gk_div = 3. * (1. + C_gk_epsilon);

__device__ __forceinline__ double ground_kernel_kdtw(const double a, const double b, const double sigma, const double tau)
{
    return (exp(-pow(fabs(a - b), tau) / sigma) + C_gk_epsilon) / C_gk_div;
}

// sigma=1,5..; epsilon = 1e-3
__device__ __forceinline__ double
kdtw_distances(CRPTRd A, CRPTRd B, const uint16_t l, const uint16_t l1, const uint16_t last_ix, const double sigma, const double tau)
{
    constexpr auto DP_len = 2 * (datamodel::C_default_svrparam_lag_count + 1);
    double DP[DP_len], DP1[DP_len], Kjj[datamodel::C_default_svrparam_lag_count];
    memset(DP + 1, 0, l1 * sizeof(double));
    memset(DP1 + 1, 0, l1 * sizeof(double));
    *DP = *DP1 = 1;
    UNROLL(common::C_default_svrparam_lag_count / 10)
    for (uint16_t j = 0; j < l; ++j) Kjj[j] = ground_kernel_kdtw(A[j], B[j], sigma, tau);
    
    UNROLL(common::C_default_svrparam_lag_count / 10)
    for (uint16_t i = 1; i < l1; ++i) {
        const auto i_1 = i - 1;
        const auto K_i_i = ground_kernel_kdtw(A[i_1], B[i_1], sigma, tau);
        for (uint16_t j = 1; j < l1; ++j) {
            const auto j_1 = j - 1;
            const auto K_i_j = ground_kernel_kdtw(A[i_1], B[j_1], sigma, tau);
            const auto j_l = (j % 2) * l1;
	        const auto ijl = i + j_l;
            const auto j1_l = (j_1 % 2) * l1;
            const auto i1_jl = i_1 + j_l;
            const auto i_j1_l = i + j1_l;
            DP[ijl] = (DP[i1_jl] + DP[i_j1_l] + DP[i_1 + j1_l]) * K_i_j;
            DP1[ijl] = DP1[i_j1_l] * Kjj[j_1];
            DP1[ijl] += i == j ? DP1[i_1 + j1_l] * K_i_j + DP1[i1_jl] * K_i_i : DP1[i1_jl] * K_i_i;
        }
    }
    return DP[last_ix] + DP1[last_ix];
}

// n_cols_y is the number of rows of the result matrix, n_cols_x is the number of columns of the result matrix
__global__ void G_path_distances_matrix(double *result, const double *const x, const double *const y, const uint32_t n_rows_x, const uint32_t n_cols_x, const uint32_t dim_x,
                                        const uint32_t n_rows_y, const uint32_t n_cols_y, const uint32_t dim_y, const double dimdiv, const uint32_t lag, const uint32_t lag_1,
                                        const uint32_t lag_1_2, const uint32_t last_ix, const uint32_t mirror_axis, const double lambda, const double tau)
{
    const auto i_x = blockIdx.x * blockDim.x + threadIdx.x;
    const auto col_x = i_x % n_cols_x;
    const auto str_x = i_x / n_cols_x;
    const auto i_y = blockIdx.y * blockDim.y + threadIdx.y;
    const auto col_y = i_y % n_cols_y;
    const auto str_y = i_y / n_cols_y;
    if (col_x >= col_y || str_x >= dim_x || str_y >= dim_y /* || col_x >= n_cols_x || col_y >= n_cols_y */ ) return;
    // const auto xy_lag_1 = threadIdx.y + threadIdx.x * lag_1_2;
    // const auto dist = path_distances(y + col_y * n_rows_y + str_y * lag, x + col_x * n_rows_x + str_x * lag, lag, lag_1, lambda) / dimdiv;
    const auto dist = kdtw_distances(y + col_y * n_rows_y + str_y * lag, x + col_x * n_rows_x + str_x * lag, lag, lag_1, last_ix, lambda, tau) / dimdiv;
    atomicAdd(result + col_x * n_cols_y + col_y, dist);
    atomicAdd(result + col_y * mirror_axis + col_x, dist);
    /* // Debug
    if (col_x < 5 && col_y < 5)
        printf("G_path_distances_matrix: res lo %f, res hi %f, dist %f, col_x %u, col_y %u, str_x %u, str_y %u\n",
        result[col_x * n_cols_y + col_y], result[col_y * _MIN(n_cols_y, n_cols_x) + col_x],
        dist, col_x, col_y, str_x, str_y);
    */
}

double *cu_compute_path_distances(CRPTRd x, CRPTRd y, const unsigned n_rows_x, const unsigned n_rows_y, const unsigned n_cols_x, const unsigned n_cols_y, const unsigned lag,
                                  const double lambda, const double tau, const cudaStream_t custream)
{
    const auto dim_x = n_rows_x / lag;
    const auto dim_y = n_rows_y / lag;
    const double dimdiv = dim_x * dim_y;
    const auto lag_1 = lag + 1;
    const auto lag_1_2 = lag_1 * 2;
    const auto last_ix = (lag % 2) * lag_1 + lag;
    auto d_result = cucalloc<double>(custream, n_cols_x * n_cols_y);
    // cols of y is rows of result matrix, and vice versa
    const auto [blocks, threads] = CU_BLOCKS_THREADS_2D2_t(n_cols_y * dim_y, n_cols_x * dim_x);
    const auto shorter_axis = _MIN(n_cols_y, n_cols_x);
    G_path_distances_matrix<<<blocks, threads, 0 /* threads.x * threads.y * 4 * lag_1 * sizeof(double) */, custream>>>(
            d_result, x, y, n_rows_x, n_cols_x, dim_x, n_rows_y, n_cols_y, dim_y, dimdiv, lag, lag_1, lag_1_2, last_ix, shorter_axis, lambda, tau);
    // G_copy_upper_tri<<<CU_BLOCKS_THREADS_2D2(n_cols_y, n_cols_x), 0, custream>>>(d_result, n_cols_y, shorter_axis);
    return d_result;
}

arma::mat path_distances_t(const arma::mat &x_t, const arma::mat &y_t, const unsigned lag, const double lambda, const double tau)
{
    CTX_CUSTREAM;
    auto d_x = cumallocopy(x_t, custream);
    auto d_y = cumallocopy(y_t, custream);
    const auto d_result = cu_compute_path_distances(d_x, d_y, x_t.n_rows, y_t.n_rows, x_t.n_cols, y_t.n_cols, lag, lambda, 0, custream);
    cu_errchk(cudaStreamSynchronize(custream));
    cu_errchk(cudaFreeAsync(d_x, custream));
    cu_errchk(cudaFreeAsync(d_y, custream));
    arma::mat result(y_t.n_cols, x_t.n_cols, arma::fill::none);
    cu_errchk(cudaMemcpyAsync(result.memptr(), d_result, x_t.n_cols * y_t.n_cols * sizeof(double), cudaMemcpyDeviceToHost, custream));
    cu_errchk(cudaFreeAsync(d_result, custream));
    cusyndestroy(custream);
    LOG4_TRACE("New path K: " << result.submat(0, 0, 5, 5) << ", present " << common::present(result));
    return result;
}

arma::mat path_distances_t(const arma::mat &x_t, const unsigned lag, const double lambda, const double tau)
{
    CTX4_CUSTREAM;
    auto d_x = cumallocopy(x_t, custream);
    const auto d_result = cu_compute_path_distances(d_x, d_x, x_t.n_rows, x_t.n_rows, x_t.n_cols, x_t.n_cols, lag, lambda, 0, custream);
    cu_errchk(cudaFreeAsync(d_x, custream));
    arma::mat result(x_t.n_cols, x_t.n_cols, arma::fill::none);
    cu_errchk(cudaMemcpyAsync(result.memptr(), d_result, x_t.n_cols * x_t.n_cols * sizeof(double), cudaMemcpyDeviceToHost, custream));
    cu_errchk(cudaFreeAsync(d_result, custream));
    cusyndestroy(custream);
    LOG4_TRACE("New path K: " << result.submat(0, 0, 5, 5) << ", present " << common::present(result));
    return result;
}

} // namespace kernel
} // namespace svr

