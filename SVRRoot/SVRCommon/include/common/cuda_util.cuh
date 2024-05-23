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

constexpr unsigned long long C_sign_mask_dbl = 0x7FFFFFFF;

#define CUDA_STRIDED_FOR_i(N) \
    const auto __stride = blockDim.x * gridDim.x; \
    _Pragma("unroll")                             \
    for (auto i = blockIdx.x * blockDim.x + threadIdx.x; i < (N); i += __stride)

#define _CUSIGN(X) ((X) > 0. ? 1.: (X) < 0 ? -1. : 0.)
#define _MIN(X, Y) ((X) > (Y) ? (Y) : (Y) > (X) ? (X) : (X))


#define CUDA_THREADS(x_size) unsigned((x_size) > CUDA_BLOCK_SIZE ? CUDA_BLOCK_SIZE : (x_size))
#define CUDA_BLOCKS(x_size) unsigned(std::ceil(double(x_size) / double(CUDA_BLOCK_SIZE)))
#define CUDA_THREADS_BLOCKS(x_size) CUDA_BLOCKS(x_size), CUDA_THREADS(x_size)

#define CUDA_THREADS_2D(x_size) unsigned((x_size) > CUDA_TILE_WIDTH ? CUDA_TILE_WIDTH : (x_size))
#define CUDA_BLOCKS_2D(x_size) unsigned(std::ceil(double(x_size) / double(CUDA_TILE_WIDTH)))
#define CUDA_THREADS_BLOCKS_2D(x_size) dim3(CUDA_BLOCKS_2D(x_size), CUDA_BLOCKS_2D(x_size)), dim3(CUDA_THREADS_2D(x_size), CUDA_THREADS_2D(x_size))


std::string cufft_get_error_string(const cufftResult s);

#ifdef PRODUCTION_BUILD
#define cf_errchk(cmd) (cmd)
#define ma_errchk(cmd) (cmd)
#define cu_errchk(cmd) (cmd)
#define cb_errchk(cmd) (cmd)
#define cs_errchk(cmd) (cmd)
#else
#ifndef cf_errchk
#define cf_errchk(cmd) {               \
    cufftResult __err;                 \
    if ((__err = cmd) != CUFFT_SUCCESS) \
        LOG4_THROW("CuFFT call " #cmd " failed with error " << int(__err) << ", " << cufft_get_error_string(__err)); \
    }
#endif

#ifndef ma_errchk
#define ma_errchk(cmd) {   \
        magma_int_t __err; \
        if ((__err = (cmd)) < MAGMA_SUCCESS) \
            LOG4_THROW("Magma call " #cmd " failed with error " << __err << " " << magma_strerror(__err)); \
    }
#endif

#ifndef cu_errchk
#define cu_errchk(cmd) {               \
    cudaError_t __err;                 \
    if ((__err = cmd) != cudaSuccess) \
        LOG4_THROW("Cuda call " #cmd " failed with error " << int(__err) << " " << cudaGetErrorName(__err) << ", " << cudaGetErrorString(__err)); \
    }
#endif

#ifndef cb_errchk
#define cb_errchk(cmd) {      \
        cublasStatus_t __err; \
        if ((__err = (cmd)) != CUBLAS_STATUS_SUCCESS) \
            LOG4_THROW("Cublas call " #cmd " failed with " << int(__err) << " " << cublasGetStatusName(__err) << ", " << cublasGetStatusString(__err)); \
}
#endif

#ifndef cs_errchk
#define cs_errchk(cmd) {                                             \
        cusolverStatus_t __err;                                      \
        if ((__err = (cmd)) != CUSOLVER_STATUS_SUCCESS)              \
            LOG4_THROW("Cusolver call " #cmd " failed with " << int(__err));  \
}
#endif
#endif

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
