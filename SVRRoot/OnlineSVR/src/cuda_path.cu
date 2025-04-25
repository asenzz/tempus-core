#include <cuda.h>
#include <cuda_runtime.h>
#include "cuda_path.hpp"
#include "common/gpu_handler.hpp"
#include "common/constants.hpp"
#include "cuqrsolve.cuh"
#include "kernel_base.cuh"
#include "common/cuda_util.cuh"

// TODO Verify the path kernel is implemented according to https://www.csc.kth.se/~fpokorny/static/publications/baisero2013a.pdf @new_path_kernel.cu

#define blockXX(i, j) (X[(i) * rows + (j)])
#define blockYY(i, j) (Y[(i) * rows + (j)])

#define tx threadIdx.x
#define ty threadIdx.y

namespace svr {
namespace kernel::path {

// constexpr T C_overscale = 2;

#define TTILE(x) T (x)[common::C_cu_tile_width][common::C_cu_tile_width]

constexpr float C_max_tau = 50;

template<typename T> inline T frau(const T tau) // Input [0..2]
{
    return tau < 1 ? std::pow(tau, 2) : tau * C_max_tau;
}

template<typename T> __device__ __forceinline__ T
do_product_sum(const uint32_t rows, const uint16_t lag, const uint16_t dim, const uint16_t lag_TILE_WIDTH, const T lambda, const T tau, CRPTR(T) X, CRPTR(T) Y,
               T power_mult[common::C_cu_tile_width], TTILE(ta), TTILE(tam1), TTILE(tb), TTILE(tbm1),
               const uint32_t ix, const uint32_t iy, const bool ix_X, const bool iy_Y, const bool do_matrix_product_sum)
{
    T matrix_prod_sum = 0;
    UNROLL()
    for (DTYPE(dim) jA = 0; jA < dim; ++jA) {
        const auto jA_lag = jA * lag;
        UNROLL()
        for (DTYPE(lag_TILE_WIDTH) internal_big = 0; internal_big < lag_TILE_WIDTH; ++internal_big) {
            const auto internal_big_TILE_WIDTH = internal_big * common::C_cu_tile_width;
            const auto ty_internal_big_TILE_WIDTH = ty + internal_big_TILE_WIDTH;
            if (ty_internal_big_TILE_WIDTH < lag) {
                if (tx == 0) power_mult[ty] = pow(1. / T(lag - ty_internal_big_TILE_WIDTH), lambda);
                if (ix_X) {
                    ta[tx][ty] = blockXX(ix, ty_internal_big_TILE_WIDTH + jA_lag);
                    if (ty_internal_big_TILE_WIDTH) tam1[tx][ty] = ta[tx][ty] - blockXX(ix, ty_internal_big_TILE_WIDTH + jA_lag - 1);
                }
            }

            const auto tx_internal_big_TILE_WIDTH = tx + internal_big_TILE_WIDTH;
            if (iy_Y && tx_internal_big_TILE_WIDTH < lag) {
                tb[ty][tx] = blockYY(iy, tx_internal_big_TILE_WIDTH + jA_lag);
                if (tx_internal_big_TILE_WIDTH) tbm1[ty][tx] = tb[ty][tx] - blockYY(iy, tx_internal_big_TILE_WIDTH + jA_lag - 1);
            }
            __syncthreads();

            if (do_matrix_product_sum) {
                UNROLL(common::C_cu_tile_width)
                for (DTYPE(common::C_cu_tile_width) internal_small = 0; internal_small < common::C_cu_tile_width; ++internal_small) {
                    const auto internal = internal_small + internal_big_TILE_WIDTH;
                    if (internal >= lag) continue;
                    const auto tab_diff = ta[tx][internal_small] - tb[ty][internal_small];
                    const auto tabm_diff = tam1[tx][internal_small] - tbm1[ty][internal_small];
                    matrix_prod_sum += (signum(tab_diff) * DIST(tab_diff) + (internal ? (tau * signum(tabm_diff) * DIST(tabm_diff)) : T(0))) * power_mult[internal_small] / dim;
                }
            }
            __syncthreads();
        }
    }
    return matrix_prod_sum;
}

#define X_OUT iy * X_cols + ix

template<typename T> __global__  void
G_distances_xy(const uint32_t X_cols, const uint32_t Y_cols, const uint32_t rows, const uint16_t lag, const uint16_t dim, const uint16_t lag_TILE_WIDTH, const T lambda,
               const T tau, CRPTR(T) X, CRPTR(T) Y, RPTR(T) Z)
{
    if (blockIdx.x * blockDim.x >= X_cols || blockIdx.y * blockDim.y >= Y_cols) return;

    __shared__ T power_mult[common::C_cu_tile_width], ta[common::C_cu_tile_width][common::C_cu_tile_width], tb[common::C_cu_tile_width][common::C_cu_tile_width];
    __shared__ T tam1[common::C_cu_tile_width][common::C_cu_tile_width], tbm1[common::C_cu_tile_width][common::C_cu_tile_width];

    const auto ix = threadIdx.x + blockIdx.x * blockDim.x;
    const auto iy = threadIdx.y + blockIdx.y * blockDim.y;
    const bool ix_X = ix < X_cols;
    const bool iy_Y = iy < Y_cols;
    const auto do_matrix_product_sum = iy_Y && ix_X;
    const auto matrix_prod_sum = do_product_sum(rows, lag, dim, lag_TILE_WIDTH, lambda, tau, X, Y, power_mult, ta, tam1, tb, tbm1, ix, iy, ix_X, iy_Y, do_matrix_product_sum);
    if (do_matrix_product_sum) Z[X_OUT] = matrix_prod_sum;
}

/*
 * Kernel matrix format, where x is X_cols and y is Y_cols:
 * output size X_cols x Y_cols

 L0 - L0, L0 - L1, L0 - L2, ... L0 - Ly
 L1 - L0, L1 - L1, L1 - L2, ... L1 - Ly
 L2 - L0, L2 - L1, L2 - L2, ... L2 - Ly
 ...
 Lx - L0, Lx - L1, Lx - L2, ... Lx - Ly
*/
template<typename T> __global__  void
G_kernel_xy(const uint32_t X_cols, const uint32_t Y_cols, const uint32_t rows, const uint16_t lag, const uint16_t dim, const uint16_t lag_TILE_WIDTH, const T lambda,
            const T tau, const T gamma, CRPTR(T) X, CRPTR(T) Y, RPTR(T) Z)
{
    if (blockIdx.x * blockDim.x >= X_cols || blockIdx.y * blockDim.y >= Y_cols) return;

    __shared__ T power_mult[common::C_cu_tile_width], ta[common::C_cu_tile_width][common::C_cu_tile_width], tb[common::C_cu_tile_width][common::C_cu_tile_width];
    __shared__ T tam1[common::C_cu_tile_width][common::C_cu_tile_width], tbm1[common::C_cu_tile_width][common::C_cu_tile_width];

    const auto ix = threadIdx.x + blockIdx.x * blockDim.x;
    const auto iy = threadIdx.y + blockIdx.y * blockDim.y;
    const bool ix_X = ix < X_cols;
    const bool iy_Y = iy < Y_cols;
    const auto do_matrix_product_sum = iy_Y && ix_X;
    const auto matrix_prod_sum = do_product_sum(rows, lag, dim, lag_TILE_WIDTH, lambda, 0, X, Y, power_mult, ta, tam1, tb, tbm1, ix, iy, ix_X, iy_Y, do_matrix_product_sum);
    if (do_matrix_product_sum) Z[X_OUT] = matrix_prod_sum / gamma;
}


template<typename T> __global__ void G_threshold(RPTR(T) Z, const uint32_t len, const T threshold)
{
    const auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len && fabs(Z[i]) < threshold) Z[i] = 0;
}

#define T double

#define PATH_BLOCKS(x, y) CU_BLOCKS_THREADS_2D2(_MAX(common::C_cu_tile_width, (x)), _MAX(common::C_cu_tile_width, (y)))

template<> void cu_distances_xy<T>(
        const uint32_t X_cols, const uint32_t Y_cols, const uint32_t rows, const uint16_t lag, const uint16_t dim, const uint16_t lag_TILE_WIDTH, const T lambda,
        const T tau, CRPTR(T) X, CRPTR(T) Y, RPTR(T) Z, const cudaStream_t custream)
{
    G_distances_xy<<<PATH_BLOCKS(X_cols, Y_cols), 0, custream>>>(X_cols, Y_cols, rows, lag, dim, lag_TILE_WIDTH, lambda, frau(tau), X, Y, Z);
}

template<> void cu_kernel_xy<T>(const uint32_t X_cols, const uint32_t Y_cols, const uint32_t rows, const uint16_t lag, const uint16_t dim, const uint16_t lag_TILE_WIDTH,
                                const T lambda, const T tau, const T gamma, const T mean, CRPTR(T) X, CRPTR(T) Y, RPTR(T) Z, const cudaStream_t custream)
{
    // G_kernel_xy<<<CU_BLOCKS_THREADS_2D2(X_cols, Y_cols), 0, custream>>>(X_cols, Y_cols, rows, lag, dim, lag_TILE_WIDTH, lambda, frau(tau), gamma, X, Y, Z);
    G_distances_xy<<<PATH_BLOCKS(X_cols, Y_cols), 0, custream>>>(X_cols, Y_cols, rows, lag, dim, lag_TILE_WIDTH, lambda, frau(tau), X, Y, Z);
    const auto n = X_cols * Y_cols;
    if (gamma != 1 || mean != 0) kernel::G_kernel_from_distances_I<<<CU_BLOCKS_THREADS(n), 0, custream>>>(Z, n, gamma, mean);
}

template<> void
distances_xy<T>(const uint32_t X_cols, const uint32_t Xy_cols, const uint32_t rows, const uint16_t lag, const T lambda, const T tau, CRPTR(T) X, CRPTR(T) Xy, RPTR(T) Z)
{
    const size_t Z_size = X_cols * Xy_cols * sizeof(T);
    T *d_Z;
    CTX4_CUSTREAM;
    const auto d_X = cumallocopy(X, custream, X_cols * rows);
    const auto d_Xy = X == Xy ? d_X : cumallocopy(Xy, custream, Xy_cols * rows);
    cu_errchk(cudaMallocAsync(&d_Z, Z_size, custream));
    cu_distances_xy(X_cols, Xy_cols, lag, rows, rows / lag, CDIVI(lag, common::C_cu_tile_width), lambda, tau, d_X, d_Xy, d_Z, custream);
    cu_errchk(cudaFreeAsync(d_X, custream));
    if (X != Xy) cu_errchk(cudaFreeAsync(d_Xy, custream));
    cu_errchk(cudaMemcpyAsync(Z, d_Z, Z_size, cudaMemcpyDeviceToHost, custream));
    cu_errchk(cudaFreeAsync(d_Z, custream));
    cusyndestroy(custream);
}


template<> void kernel_xy(
        const uint32_t X_cols, const uint32_t Xy_cols, const uint32_t rows, const uint16_t lag,
        const T gamma, const T lambda, const T tau, const T mean,
        CRPTR(T) X, CRPTR(T) Xy, RPTR(T) K)
{
    const auto K_size = X_cols * Xy_cols * sizeof(T);
    T *d_K;
    CTX4_CUSTREAM;
    const auto d_X = cumallocopy(X, custream, X_cols * rows);
    const auto d_Xy = X == Xy ? d_X : cumallocopy(Xy, custream, Xy_cols * rows);
    cu_errchk(cudaMallocAsync(&d_K, K_size, custream));
    cu_kernel_xy(X_cols, Xy_cols, rows, lag, rows / lag, CDIVI(lag, common::C_cu_tile_width), lambda, tau, gamma, mean, d_X, d_Xy, d_K, custream);
    cu_errchk(cudaFreeAsync(d_X, custream));
    if (X != Xy) cu_errchk(cudaFreeAsync(d_Xy, custream));
    cu_errchk(cudaMemcpyAsync(K, d_K, K_size, cudaMemcpyDeviceToHost, custream));
    cu_errchk(cudaFreeAsync(d_K, custream));
    cusyndestroy(custream);
}

template<> void kernel_xy<T>(
        const uint32_t X_cols, const uint32_t Xy_cols, const uint32_t rows, const uint16_t lag,
        const T gamma, const T lambda, const T tau, const T mean,
        CRPTR(T) X, CRPTR(T) Xy, RPTR(T) K, const uint16_t gpu_id)
{
    const auto K_len = X_cols * Xy_cols;
    const auto K_size = K_len * sizeof(T);
    T *d_K;
    DEV_CUSTREAM(gpu_id);
    const auto d_X = cumallocopy(X, custream, X_cols * rows);
    const auto d_Xy = X == Xy ? d_X : cumallocopy(Xy, custream, Xy_cols * rows);
    cu_errchk(cudaMallocAsync(&d_K, K_size, custream));
    cu_kernel_xy(X_cols, Xy_cols, rows, lag, rows / lag, CDIVI(lag, common::C_cu_tile_width), lambda, tau, gamma, mean, d_X, d_Xy, d_K, custream);
    cu_errchk(cudaFreeAsync(d_X, custream));
    if (X != Xy) cu_errchk(cudaFreeAsync(d_Xy, custream));
    cu_errchk(cudaMemcpyAsync(K, d_K, K_size, cudaMemcpyDeviceToHost, custream));
    cu_errchk(cudaFreeAsync(d_K, custream));
    cusyndestroy(custream);
}

#undef T

}
}