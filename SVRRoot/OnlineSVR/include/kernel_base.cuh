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
#include "common/cuda_util.cuh"


namespace svr {
namespace kernel {

template<typename T> __device__ __host__ __forceinline__ T K_from_Z(const T z, const float degree)
{
    return copysign(pow(abs(z), degree), z);
}

template<typename T> __device__ __host__ __forceinline__ T K_from_Z(const T z, const T divisor, const T mean)
{
    return common::scale(z, divisor, mean);
}

template<typename T> __device__ __host__ __forceinline__ T K_from_Z(const T z, const T divisor, const T mean, const float degree)
{
    return common::scale(K_from_Z(z, degree), divisor, mean);
}

}
}

#endif //SVR_KERNEL_BASE_CUH
