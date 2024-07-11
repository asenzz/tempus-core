//
// Created by zarko on 2/17/22.
//

#ifndef SVR_CUDA_UTIL_CUH
#define SVR_CUDA_UTIL_CUH

#include <cufft.h>
#include <sstream>
#include <thrust/device_vector.h>
#include "common/logging.hpp"
#include "common/defines.h"
#include "common/constants.hpp"

constexpr unsigned long long C_sign_mask_dbl = 0x7FFFFFFF;

#define tid threadIdx.x

#ifdef PRODUCTION_BUILD
#define CUDA_STRIDED_FOR_i(N) \
    const auto __stride = blockDim.x * gridDim.x; \
    _Pragma("unroll")         \
    for (auto i = blockIdx.x * blockDim.x + threadIdx.x; i < (N); i += __stride)
#else
#define CUDA_STRIDED_FOR_i(N) \
    const auto __stride = blockDim.x * gridDim.x; \
    for (auto i = blockIdx.x * blockDim.x + threadIdx.x; i < (N); i += __stride)
#endif

#define CUDA_THREADS(n) unsigned((n) > common::C_cu_block_size ? common::C_cu_block_size : (n))
#define CUDA_BLOCKS(n) (unsigned) _CEILDIV((n), common::C_cu_block_size)
#define CUDA_THREADS_BLOCKS(n) CUDA_BLOCKS(n), CUDA_THREADS(n)

#define CUDA_THREADS_2D(x_size) unsigned((x_size) > common::C_cu_tile_width ? common::C_cu_tile_width : (x_size))
#define CUDA_BLOCKS_2D(x_size) (unsigned) _CEILDIV((x_size), common::C_cu_tile_width)
#define CUDA_THREADS_BLOCKS_2D(x_size) dim3(CUDA_BLOCKS_2D(x_size), CUDA_BLOCKS_2D(x_size)), dim3(CUDA_THREADS_2D(x_size), CUDA_THREADS_2D(x_size))


constexpr unsigned clamp_n (const unsigned n) { return _MIN(n, svr::common::C_cu_clamp_n); }

template<typename T> T *
cuda_malloccopy(const T *source, const size_t size, const cudaMemcpyKind kind)
{
    T *ptr = nullptr;
    switch (kind) {
        case cudaMemcpyDeviceToHost:
        case cudaMemcpyHostToHost:
            ptr = (T *) malloc(size);
            break;
        case cudaMemcpyDeviceToDevice:
        case cudaMemcpyHostToDevice:
        case cudaMemcpyDefault: cu_errchk(cudaMalloc(&ptr, size));
            break;
    }
    cu_errchk(cudaMemcpy(ptr, source, size, kind));
    return ptr;
}

template<typename T> std::vector<T>
cuda_copy(const T *source, const size_t length)
{
    std::vector<T> res(length);
    cu_errchk(cudaMemcpy(res.data(), source, length * sizeof(T), cudaMemcpyDeviceToHost));
    return res;
}

template<typename T> std::vector<T>
cuda_copy(const thrust::device_vector<T> &source)
{
    std::vector<T> res(source.size());
    cu_errchk(cudaMemcpy(res.data(), thrust::raw_pointer_cast(source.data()), source.size() * sizeof(T), cudaMemcpyDeviceToHost));
    return res;
}

template<typename T> void
cuda_copy(std::vector<T> &res, const T *source, const size_t length)
{
    if (res.size() != length) res.resize(length);
    cu_errchk(cudaMemcpy(res.data(), source, length * sizeof(T), cudaMemcpyDeviceToHost));
}

template<typename T> T *
cuda_malloccopy(const std::vector<T> &v)
{
    T *ptr = nullptr;
    //std::cout << "Allocating " << v.size() * sizeof(T) << std::endl;
    cu_errchk(cudaMalloc(&ptr, v.size() * sizeof(T)));
    cu_errchk(cudaMemcpy(ptr, v.data(), v.size() * sizeof(T), cudaMemcpyKind::cudaMemcpyHostToDevice));
    return ptr;
}

inline void *cuda_calloc(const size_t size, const size_t count)
{
    void *ptr;
    cu_errchk(cudaMalloc(&ptr, count * size));
    cu_errchk(cudaMemset(ptr, 0, count * size));
    return ptr;
}


#endif //SVR_CUDA_UTIL_CUH
