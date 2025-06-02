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

__global__ void G_kernel_from_distances(RPTR(T) Kz, const uint32_t mn, const T divisor, const T mean, const T degree)
{
    CU_STRIDED_FOR_i(mn) Kz[i] = K_from_Z(Kz[i], divisor, mean, degree);
}

__global__ void G_kernel_from_distances(RPTR(T) K, CRPTR(T) Z, const uint32_t mn, const T divisor, const T mean, const T degree)
{
    CU_STRIDED_FOR_i(mn) K[i] = K_from_Z(Z[i], divisor, mean, degree);
}

__global__ void G_kernel_from_distances(RPTR(T) Kz, const uint32_t mn, const T divisor, const T mean)
{
    CU_STRIDED_FOR_i(mn) Kz[i] = K_from_Z(Kz[i], divisor, mean);
}

__global__ void G_kernel_from_distances(RPTR(T) K, CRPTR(T) Z, const uint32_t mn, const T divisor, const T mean)
{
    CU_STRIDED_FOR_i(mn) K[i] = K_from_Z(Z[i], divisor, mean);
}

__global__ void G_kernel_from_distances_symm(RPTR(T) K, CRPTRd dist, const uint32_t m, const double divisor, const double mean) // Lower triangular
{
    const auto ix = threadIdx.x + blockIdx.x * blockDim.x;
    const auto iy = threadIdx.y + blockIdx.y * blockDim.y;
    if (ix >= m || iy >= m || iy > ix) return;
    if (ix == iy)
        K[ix + iy * m] = K_from_Z(dist[ix + iy * m], divisor, mean);
    else
        K[ix + iy * m] = K[iy + ix * m] = K_from_Z(dist[ix + iy * m], divisor, mean);
}


__global__ void G_kernel_from_distances_symm(RPTR(T) K, CRPTRd dist, const uint32_t m, const double divisor, const double mean, const double degree) // Lower triangular
{
    const auto ix = threadIdx.x + blockIdx.x * blockDim.x;
    const auto iy = threadIdx.y + blockIdx.y * blockDim.y;
    if (ix >= m || iy >= m || iy > ix)
        return;
    if (ix == iy)
        K[ix + iy * m] = K_from_Z(dist[ix + iy * m], divisor, mean, degree);
    else
        K[ix + iy * m] = K[iy + ix * m] = K_from_Z(dist[ix + iy * m], divisor, mean, degree);
}

__global__ void G_kernel_from_distances_symm(RPTR(T) Kz, const uint32_t m, const double divisor, const double mean) // Lower triangular
{
    const auto ix = threadIdx.x + blockIdx.x * blockDim.x;
    const auto iy = threadIdx.y + blockIdx.y * blockDim.y;
    if (ix >= m || iy >= m || iy > ix)
        return;
    if (ix == iy)
        Kz[ix + iy * m] = K_from_Z(Kz[ix + iy * m], divisor, mean);
    else
        Kz[ix + iy * m] = Kz[iy + ix * m] = K_from_Z(Kz[ix + iy * m], divisor, mean);
}


__global__ void G_kernel_from_distances_symm(RPTR(T) Kz, const uint32_t m, const double divisor, const double mean, const double degree) // Lower triangular
{
    const auto ix = threadIdx.x + blockIdx.x * blockDim.x;
    const auto iy = threadIdx.y + blockIdx.y * blockDim.y;
    if (ix >= m || iy >= m || iy > ix) return;
    if (ix == iy)
        Kz[ix + iy * m] = K_from_Z(Kz[ix + iy * m], divisor, mean, degree);
    else
        Kz[ix + iy * m] = Kz[iy + ix * m] = K_from_Z(Kz[ix + iy * m], divisor, mean, degree);
}


template<> void kernel_from_distances<T>(RPTR(T) Kz, const uint32_t m, const uint32_t n, const T gamma, const T mean, const T degree)
{
    const auto mn = m * n;
    CTX4_CUSTREAM;
    auto d_Kz = cumallocopy(Kz, custream, mn);
    if (false /* m == n */) G_kernel_from_distances_symm<<<CU_BLOCKS_THREADS(m), 0, custream>>>(d_Kz, m, gamma, mean, degree);
    else G_kernel_from_distances<<<CU_BLOCKS_THREADS(mn), 0, custream>>>(d_Kz, mn, gamma, mean, degree);
    cufreecopy(Kz, d_Kz, custream, mn);
    cusyndestroy(custream);
}

template<> void kernel_from_distances<T>(RPTR(T) K, CRPTR(T) Z, const uint32_t m, const uint32_t n, const T gamma, const T mean, const T degree)
{
    const auto mn = m * n;
    // const auto mat_size = mn * sizeof(T);
    CTX4_CUSTREAM;
    const auto d_Kz = cumallocopy(Z, custream, mn);
    if (false /* m == n */) G_kernel_from_distances_symm<<<CU_BLOCKS_THREADS(m), 0, custream>>>(d_Kz, m, gamma, mean, degree);
    else G_kernel_from_distances<<<CU_BLOCKS_THREADS(mn), 0, custream>>>(d_Kz, mn, gamma, mean, degree);
    cufreecopy(K, d_Kz, custream, mn);
    LOG4_TRACE("Distances " << m << "x" << n << " " << common::to_string(Z, std::min<uint32_t>(mn, 10)) << ", parameters gamma " << gamma << ", mean " << mean << ", degree " << degree
        << ", to kernel " << common::to_string(K, std::min<uint32_t>(mn, 10)) << ", mean " << common::mean(K, mn));
    cusyndestroy(custream);
}

template<> void d_kernel_from_distances<T>(
        RPTR(T) d_K, CRPTR(T) d_Z, const uint32_t m, const uint32_t n, const T gamma, const T mean, const T degree, const cudaStream_t custream)
{
    if (false /* m == n */) G_kernel_from_distances_symm<<<CU_BLOCKS_THREADS(m), 0, custream>>>(d_K, d_Z, m, gamma, mean, degree);
    else {
        const auto mn = m * n;
        G_kernel_from_distances<<<CU_BLOCKS_THREADS(mn), 0, custream>>>(d_K, d_Z, mn, gamma, mean, degree);
    }
}

#undef T

}
}
