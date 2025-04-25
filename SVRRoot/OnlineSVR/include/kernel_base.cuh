//
// Created by zarko on 19/03/2025.
//

#ifndef SVR_KERNEL_BASE_CUH
#define SVR_KERNEL_BASE_CUH

#include <npp.h>
#include <thrust/sort.h>
#include <cmath>
#include <cublas_v2.h>
#include <magma_types.h>
#include <magma_v2.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "common/compatibility.hpp"
#include "common/gpu_handler.hpp"
#include "common/constants.hpp"
#include "onlinesvr.hpp"
#include "cuda_path.hpp"
#include "common/cuda_util.cuh"


namespace svr {
namespace kernel {

template<typename T> __global__ void G_kernel_from_distances_I(RPTR(T) Kz, const uint32_t mn, const T divisor, const T mean);

template<typename T> __global__ void G_kernel_from_distances(RPTR(T) K, CRPTR(T) Z, const uint32_t mn, const T divisor, const T mean);


// Distances to kernel value
template<typename T> __forceinline__ __host__ __device__ T z2k(const T z, const T gamma)
{
    return z / gamma;
    // return (1. - z) / gamma; // Original SVM kernel (less precise)
}

}
}

#endif //SVR_KERNEL_BASE_CUH
