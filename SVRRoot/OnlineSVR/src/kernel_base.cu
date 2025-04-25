//
// Created by zarko on 19/03/2025.
//

#include "kernel_base.hpp"
#include "kernel_base.cuh"
#include "common/cuda_util.cuh"

namespace svr {
namespace kernel {


// Specialization for double
#define T double

template<> __global__ void G_kernel_from_distances_I<T>(RPTR(T) Kz, const uint32_t mn, const T divisor, const T mean)
{
    CU_STRIDED_FOR_i(mn) Kz[i] = z2k(Kz[i] - mean, divisor);
}

template<> __global__ void G_kernel_from_distances<T>(RPTR(T) K, CRPTR(T) Z, const uint32_t mn, const T divisor, const T mean)
{
    CU_STRIDED_FOR_i(mn) K[i] = z2k(Z[i] - mean, divisor);
}

template<> void kernel_from_distances<T>(RPTR(T) K, CRPTR(T) Z, const uint32_t m, const uint32_t n, const T gamma, const T mean)
{
    const auto mn = m * n;
    const auto mat_size = mn * sizeof(T);
    CTX4_CUSTREAM;
    T *d_K;
    const auto d_Z = cumallocopy(Z, custream, mn);
    cu_errchk(cudaMallocAsync((void **) &d_K, mat_size, custream));
    G_kernel_from_distances<<<CU_BLOCKS_THREADS(mn), 0, custream>>>(d_K, d_Z, mn, gamma, mean);
    cu_errchk(cudaFreeAsync(d_Z, custream));
    cufreecopy(K, d_K, custream, mn);
    cusyndestroy(custream);
}

template<> void kernel_from_distances<T>(RPTR(T) Kz, const uint32_t m, const uint32_t n, const T gamma, const T mean)
{
    const auto mn = m * n;
    const auto mat_size = mn * sizeof(T);
    CTX4_CUSTREAM;
    T *d_K;
    const auto d_Z = cumallocopy(Kz, custream, mn);
    cu_errchk(cudaMallocAsync((void **) &d_K, mat_size, custream));
    G_kernel_from_distances<<<CU_BLOCKS_THREADS(mn), 0, custream>>>(d_K, d_Z, mn, gamma, mean);
    cu_errchk(cudaFreeAsync(d_Z, custream));
    cufreecopy(Kz, d_K, custream, mn);
    cusyndestroy(custream);
}

template<> void d_kernel_from_distances<T>(
        RPTR(T) d_K, CRPTR(T) d_Z, const uint32_t m, const uint32_t n, const T gamma, const T mean, const cudaStream_t custream)
{
    const auto mn = m * n;
    G_kernel_from_distances<<<CU_BLOCKS_THREADS(mn), 0, custream>>>(d_K, d_Z, mn, gamma, mean);
}

#undef T

}
}