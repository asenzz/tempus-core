#include <cuda.h>
#include <cuda_runtime.h>
#include "common/cuda_util.cuh"
#include "common/gpu_handler.hpp"
#include "common/constants.hpp"
#include "kernel_path.hpp"
#include "cuqrsolve.cuh"
#include "kernel_base.cuh"

namespace svr {
namespace kernel {
__global__ void G_powbuf(RPTR(float) powbuf, const uint16_t lag2, const float tau)
{
    const auto i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= lag2) return;
    powbuf[i] = pow(float(i + 1) / lag2, tau);
}

template<typename T> __device__ __forceinline__ T pathdiff(const T *const X, const T *const Xy, const uint16_t i, const uint16_t j, CRPTR(float) powbuf)
{
    return (X[i] - Xy[j]) * powbuf[i + j];
}

template<typename T> __device__ __forceinline__ T path_impl(
    const uint32_t X_cols, const uint32_t Xy_chunk_cols, const uint32_t ix, const uint32_t iy, const uint32_t iy_dp, const uint16_t d, const uint16_t dim, const uint16_t lag, RPTR(T) dpbuf,
    const float H, const float D, const float V, CRPTR(float) powbuf, CRPTR(T) X, CRPTR(T) Xy)
{
    const auto dimlag = dim * lag;
    CRPTR(double) X2 = X + d * lag + ix * dimlag;
    CRPTR(double) Xy2 = Xy + d * lag + iy * dimlag;
    double *const dp = dpbuf + (ix + iy_dp * X_cols + d * X_cols * Xy_chunk_cols) * lag * lag;

#define pdp(i, j) dp[(i) + (j) * lag]
    pdp(0, 0) = pathdiff(X2, Xy2, 0, 0, powbuf);

    UNROLL(1 + common::C_default_svrparam_lag_count / 10)
    for (DTYPE(lag) i = 1; i < lag; ++i) {
        pdp(0, i) = H * pdp(0, i - 1) + pathdiff(X2, Xy2, 0, i, powbuf);
        pdp(i, 0) = V * pdp(i - 1, 0) + pathdiff(X2, Xy2, i, 0, powbuf);
    }

    UNROLL(1 + common::C_default_svrparam_lag_count / 10)
    for (DTYPE(lag) i = 1; i < lag; ++i) {
        for (DTYPE(lag) j = 1; j < lag; ++j) {
            pdp(i, j) = pathdiff(X2, Xy2, i, j, powbuf);
#ifdef PATHS_AVERAGE
            pdp(i, j) += H * pdp(i, j - 1) + D * pdp(i - 1, j - 1) + V * pdp(i - 1, j);
#else
            uint8_t minix = 0;
            auto minv = abs(pdp(i - 1, j));
            auto minv2 = abs(pdp(i, j - 1));
            if (minv2 < minv) {
                minix = 1;
                minv = minv2;
            }
            if (abs(pdp(i - 1, j - 1)) < minv) minix = 2;
            switch (minix) {
                case 0: pdp(i, j) += V * pdp(i - 1, j);
                    break;
                case 1: pdp(i, j) += H * pdp(i, j - 1);
                    break;
                case 2: pdp(i, j) += D * pdp(i - 1, j - 1);
                    break;
            }
#endif
        }
    }
    return pdp(lag - 1, lag - 1) / lag / dim;
}

template<typename T> __global__ void G_distances(
    const uint32_t offsety, const uint32_t X_cols, const uint32_t Xy_cols, const uint32_t Xy_chunk_cols, const uint16_t dim, const uint16_t lag,
    RPTR(T) dpbuf, const float H, const float D, const float V, CRPTR(float) powbuf, CRPTR(T) X, CRPTR(T) Xy, RPTR(T) Z)
{
    const auto ix = threadIdx.x + blockIdx.x * blockDim.x;
    const auto iy_dp = threadIdx.y + blockIdx.y * blockDim.y;
    const auto iy = iy_dp + offsety;
    const auto d = threadIdx.z + blockIdx.z * blockDim.z;
    if (ix >= X_cols || iy >= Xy_cols || d >= dim || iy_dp >= Xy_chunk_cols) return;
    atomicAdd(Z + ix + iy * X_cols, path_impl<T>(X_cols, Xy_chunk_cols, ix, iy, iy_dp, d, dim, lag, dpbuf, H, D, V, powbuf, X, Xy));
    // We are using transposed output therefore it should not be iy + ix * Xy_cols
}

template<typename T> __global__ void G_kernel(
    const uint32_t offsety, const uint32_t X_cols, const uint32_t Xy_cols, const uint32_t Xy_chunk_cols, const uint16_t dim, const uint16_t lag, const T gamma, const T mean,
    const float lambda,
    RPTR(T) dpbuf, const float H, const float D, const float V, CRPTR(float) powbuf, CRPTR(T) X, CRPTR(T) Xy, RPTR(T) K)
{
    const auto ix = threadIdx.x + blockIdx.x * blockDim.x;
    const auto iy_dp = threadIdx.y + blockIdx.y * blockDim.y;
    const auto iy = iy_dp + offsety;
    const auto d = threadIdx.z + blockIdx.z * blockDim.z;
    if (ix >= X_cols || iy >= Xy_cols || d >= dim || iy_dp >= Xy_chunk_cols) return;
    atomicAdd(K + ix + iy * X_cols, kernel::K_from_Z(path_impl<T>(X_cols, Xy_chunk_cols, ix, iy, iy_dp, d, dim, lag, dpbuf, H, D, V, powbuf, X, Xy), gamma, mean, lambda));
}


#define T double // Template specializations follow below

constexpr uint32_t C_free_mem_divisor = 2;

template<typename PATH_KERNEL_WRAPPER> void cu_path_caller(
    const PATH_KERNEL_WRAPPER path_fun, const uint32_t X_cols, const uint32_t Xy_cols, const uint16_t lag, const uint16_t dim, const float tau, RPTR(T) Kz, const cudaStream_t custream)
{
    LOG4_BEGIN();
    float *powbuf;
    const auto lag2 = lag * 2;
    cu_errchk(cudaMallocAsync(&powbuf, lag2 * sizeof(float), custream));
#if 1
    size_t free_mem, total_mem;
    cu_errchk(cudaStreamSynchronize(custream));
    cu_errchk(cudaMemGetInfo(&free_mem, &total_mem));
    free_mem /= C_free_mem_divisor;
#else // faster
    const auto free_mem = common::gpu_handler<4>::get().get_max_gpu_data_chunk_size() / C_free_mem_divisor;
#endif
    const auto X_dim_size = X_cols * dim * lag * lag * sizeof(double);
    const auto kernel_size = Xy_cols * X_dim_size;
    G_powbuf<<<CU_BLOCKS_THREADS(lag2), 0, custream>>>(powbuf, lag2, tau);
    const auto num_mem_chunks = CDIVI(kernel_size, free_mem);
    const auto max_chunky_len = CDIVI(Xy_cols, num_mem_chunks);
    const auto num_chunks = CDIVI(Xy_cols, max_chunky_len);
    const auto chunk_size = max_chunky_len * X_dim_size;
    LOG4_TRACE("X dim size " << X_dim_size << ", kernel size " << kernel_size << ", free mem " << free_mem << ", num chunks " << num_chunks << ", max chunky len " << max_chunky_len
        << ", chunk size " << chunk_size << ", lag2 " << lag2 << ", X cols " << X_cols << ", Xy cols " << Xy_cols << ", dim " << dim << ", num memory chunks " << num_mem_chunks);
    double *dpbuf;
    cu_errchk(cudaMallocAsync(&dpbuf, chunk_size, custream));
    cu_errchk(cudaMemsetAsync(Kz, 0, X_cols * Xy_cols * sizeof(double), custream));
    for (DTYPE(num_chunks) i = 0; i < num_chunks; ++i) {
        const auto starty = i * max_chunky_len;
        const auto endy = std::min<uint32_t>(starty + max_chunky_len, Xy_cols);
        const auto this_chunky_len = endy - starty;
        path_fun(this_chunky_len, starty, dpbuf, powbuf);
    }
    cu_errchk(cudaFreeAsync(dpbuf, custream));
    cu_errchk(cudaFreeAsync(powbuf, custream));
    LOG4_END();
}

template<> void kernel_path<T>::cu_distances_xy(
    const uint32_t X_cols, const uint32_t Xy_cols, const uint16_t lag, const uint16_t dim, const float tau, const float H, const float D, const float V, CRPTR(T) X, CRPTR(T) Xy, RPTR(T) Z,
    const cudaStream_t custream)
{
    LOG4_BEGIN();
    cu_path_caller([X_cols, Xy_cols, lag, dim, H, D, V, X, Xy, Z, custream] (const uint32_t this_chunky_len, const uint32_t starty, RPTR(double) dpbuf, CRPTR(float) powbuf) {
                       G_distances<<<CU_BLOCKS_THREADS_3D3(X_cols, this_chunky_len, dim), 0, custream>>>(
                           starty, X_cols, Xy_cols, this_chunky_len, dim, lag, dpbuf, H, D, V, powbuf, X, Xy, Z);
                   },
                   X_cols, Xy_cols, lag, dim, tau, Z, custream);
    LOG4_END();
}

template<> void kernel_path<T>::cu_kernel_xy(
    const uint32_t X_cols, const uint32_t Xy_cols, const uint16_t lag, const uint16_t dim, const T gamma, const T mean, const float lambda, const float tau, const float H, const float D,
    const float V, CRPTR(T) X, CRPTR(T) Xy, RPTR(T) K, const cudaStream_t custream)
{
    LOG4_BEGIN();
    cu_path_caller([X_cols, Xy_cols, lag, dim, gamma, mean, lambda, H, D, V, X, Xy, K, custream](const uint32_t this_chunky_len, const uint32_t starty, RPTR(double) dpbuf, CRPTR(float) powbuf) {
                       G_kernel<<<CU_BLOCKS_THREADS_3D3(X_cols, this_chunky_len, dim), 0, custream>>>(
                       starty, X_cols, Xy_cols, this_chunky_len, dim, lag, gamma, mean, lambda, dpbuf, H, D, V, powbuf, X, Xy, K);
                   },
                   X_cols, Xy_cols, lag, dim, tau, K, custream);
    LOG4_END();
}

template<> void kernel_path<T>::distances_xy(
    const uint32_t X_cols, const uint32_t Xy_cols, const uint32_t rows, const uint16_t lag, const float tau, const float H, const float D, const float V,
    CRPTR(T) X, CRPTR(T) Xy,RPTR(T) Z)
{
    const auto Z_len = X_cols * Xy_cols;
    CTX4_CUSTREAM;
    const auto d_X = cumallocopy(X, custream, X_cols * rows);
    const auto d_Xy = X == Xy ? d_X : cumallocopy(Xy, custream, Xy_cols * rows);
    double *d_Z;
    cu_errchk(cudaMallocAsync(&d_Z, Z_len * sizeof(double), custream));
    cu_distances_xy(X_cols, Xy_cols, lag, rows / lag, tau, H, D, V, d_X, d_Xy, d_Z, custream);
    cu_errchk(cudaFreeAsync(d_X, custream));
    if (X != Xy) cu_errchk(cudaFreeAsync(d_Xy, custream));
    cufreecopy(Z, d_Z, custream, Z_len);
    cusyndestroy(custream);
}

template<> void kernel_path<T>::kernel_xy(
    const uint32_t X_cols, const uint32_t Xy_cols, const uint32_t rows, const uint16_t lag, const T gamma, const T mean, const float lambda, const float tau, const float H, const float D,
    const float V, CRPTR(T) X, CRPTR(T) Xy, RPTR(T) K)
{
    const auto K_len = X_cols * Xy_cols;
    CTX4_CUSTREAM;
    const auto d_X = cumallocopy(X, custream, X_cols * rows);
    const auto d_Xy = X == Xy ? d_X : cumallocopy(Xy, custream, Xy_cols * rows);
    double *d_K;
    cu_errchk(cudaMallocAsync(&d_K, K_len * sizeof(double), custream));
    cu_kernel_xy(X_cols, Xy_cols, lag, rows / lag, gamma, mean, lambda, tau, H, D, V, d_X, d_Xy, d_K, custream);
    cu_errchk(cudaFreeAsync(d_X, custream));
    if (X != Xy) cu_errchk(cudaFreeAsync(d_Xy, custream));
    cufreecopy(K, d_K, custream, K_len);
    cusyndestroy(custream);
}

#undef T
}
}
