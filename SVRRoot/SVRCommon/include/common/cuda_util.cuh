//
// Created by zarko on 2/17/22.
//
#ifdef __CUDACC__

#ifndef SVR_CUDA_UTIL_CUH
#define SVR_CUDA_UTIL_CUH

#include <cufft.h>
#include <sstream>
#include <thrust/device_vector.h>
#include <nppdefs.h>
#include "common/logging.hpp"
#include "common/defines.h"
#include "common/constants.hpp"
#include "util/math_utils.hpp"

namespace svr {

constexpr unsigned long long C_sign_mask_dbl = 0x7FFFFFFF;

// #define HETEROGENOUS_GPU_HW

#define tid threadIdx.x
#define CRPTR(T) const T *__restrict__ const
#ifdef PRODUCTION_BUILD
#define CU_STRIDED_FOR_i(N)                             \
    const auto __stride = blockDim.x * gridDim.x;       \
    UNROLL()                                            \
    for (auto i = blockIdx.x * blockDim.x + threadIdx.x; i < (N); i += __stride)
#else
#define CU_STRIDED_FOR_i(N)                         \
    const auto __stride = blockDim.x * gridDim.x;   \
    for (auto i = blockIdx.x * blockDim.x + threadIdx.x; i < (N); i += __stride)
#endif

#define CU_THREADS(n) unsigned((n) > common::C_cu_block_size ? common::C_cu_block_size : (n))
#define CU_BLOCKS(n) (unsigned) CDIVI((n), common::C_cu_block_size)
#define CU_BLOCKS_THREADS(n) CU_BLOCKS(n), CU_THREADS(n)
#define CU_BLOCKS_THREADS_t(n) std::pair{CU_BLOCKS_THREADS(n)}

#define CU_THREADS_2D(x_size) unsigned((x_size) > common::C_cu_tile_width ? common::C_cu_tile_width : (x_size))
#define CU_BLOCKS_2D(x_size) (unsigned) CDIV((x_size), common::C_cu_tile_width)
#define CU_BLOCKS_THREADS_2D(x_size) dim3(CU_BLOCKS_2D(x_size), CU_BLOCKS_2D(x_size)), dim3(CU_THREADS_2D(x_size), CU_THREADS_2D(x_size))

template<typename T> inline __device__ __host__ dtype(T::x) cunorm(const T &v) { return v.x * v.x + v.y * v.y; };

template<typename T> __device__ __forceinline__ void warp_reduce_sum(volatile T *sumdata, const unsigned ix, const unsigned n)
{
    assert(ix < 32);

#define _DO_WARP_REDUCE_SUM_n(_N)                           \
    if (n >= (_N)) {                                        \
        const unsigned ix_N_2 = ix + (_N) / 2;              \
        if (ix_N_2 < n) sumdata[ix] += sumdata[ix_N_2];     \
    }

    _DO_WARP_REDUCE_SUM_n(64);
    _DO_WARP_REDUCE_SUM_n(32);
    _DO_WARP_REDUCE_SUM_n(16);
    _DO_WARP_REDUCE_SUM_n(8);
    _DO_WARP_REDUCE_SUM_n(4);
    _DO_WARP_REDUCE_SUM_n(2);
}

template<const unsigned block_size, typename T> __device__ __forceinline__ void warp_reduce_sum(volatile T *sumdata, const unsigned ix, const unsigned n)
{
    assert(ix < 32);

#define _DO_WARP_REDUCE_SUM(N)                              \
    if (block_size >= (N)) {                                \
        const unsigned ix_N_2 = ix + (N) / 2;               \
        if (ix_N_2 < n) sumdata[ix] += sumdata[ix_N_2];     \
    }

    _DO_WARP_REDUCE_SUM(64);
    _DO_WARP_REDUCE_SUM(32);
    _DO_WARP_REDUCE_SUM(16);
    _DO_WARP_REDUCE_SUM(8);
    _DO_WARP_REDUCE_SUM(4);
    _DO_WARP_REDUCE_SUM(2);
}

template<const unsigned block_size, typename T> __device__ __forceinline__ void warp_reduce_sum(volatile T *sumdata1, volatile T *sumdata2, const unsigned ix, const unsigned n)
{
    assert(ix < 32);

#define _DO_WARP_REDUCE_2SUM(N)                  \
    if (block_size >= (N)) {                    \
        const unsigned ix_N_2 = ix + (N) / 2;   \
        if (ix_N_2 < n)              {          \
            sumdata1[ix] += sumdata1[ix_N_2];   \
            sumdata2[ix] += sumdata2[ix_N_2];   \
        }                                       \
    }

    _DO_WARP_REDUCE_2SUM(64);
    _DO_WARP_REDUCE_2SUM(32);
    _DO_WARP_REDUCE_2SUM(16);
    _DO_WARP_REDUCE_2SUM(8);
    _DO_WARP_REDUCE_2SUM(4);
    _DO_WARP_REDUCE_2SUM(2);
}

NppStreamContext get_npp_context(const unsigned gpuid, const cudaStream_t custream);

void copy_submat(CPTR(double) in, double *const out, const unsigned ldin, const unsigned in_start_m, const unsigned in_start_n, const unsigned in_end_m,
                 const unsigned in_end_n, const unsigned ldout, cudaMemcpyKind kind, const cudaStream_t stm);

__host__ __device__ inline constexpr unsigned clamp_n (const unsigned n) { return _MIN(n, svr::common::C_cu_clamp_n); }


template<typename T> T *
cumallocopy(const T *source, const size_t len, const cudaMemcpyKind kind, const cudaStream_t custream = nullptr)
{
    const auto size = len * sizeof(T);
    T *ptr;
    switch (kind) {
        case cudaMemcpyDeviceToHost:
        case cudaMemcpyHostToHost:
            ptr = (T *) malloc(size);
            break;
        case cudaMemcpyDeviceToDevice:
        case cudaMemcpyHostToDevice:
        case cudaMemcpyDefault: cu_errchk(cudaMallocAsync((void **)&ptr, size, custream));
            break;
    }
    cu_errchk(cudaMemcpyAsync(ptr, source, size, kind, custream));
    return ptr;
}

template<typename T> T *
cumallocopy(const T *source, const cudaStream_t custream, const size_t len = 1, const cudaMemcpyKind kind = cudaMemcpyHostToDevice)
{
    return cumallocopy(source, len, kind, custream);
}

template<typename T> void cufreecopy(T *output, const T *source, const cudaStream_t custream = nullptr, const size_t len = 1)
{
    cu_errchk(cudaMemcpyAsync(output, source, len * sizeof(T), cudaMemcpyDeviceToHost, custream));
    cu_errchk(cudaFreeAsync((void *) source, custream));
}

template<typename T> std::vector<T> cufreecopy(const T *source, const cudaStream_t custream = nullptr, const size_t len = 1)
{
    std::vector<T> res(len);
    cu_errchk(cudaMemcpyAsync(res.data(), source, len * sizeof(T), cudaMemcpyDeviceToHost, custream));
    cu_errchk(cudaFreeAsync((void *) source, custream));
    return res;
}

template<typename T> std::vector<T>
cucopy(const T *source, const size_t length, const cudaStream_t custream = nullptr)
{
    std::vector<T> res(length);
    cu_errchk(cudaMemcpyAsync(res.data(), source, length * sizeof(T), cudaMemcpyDeviceToHost, custream));
    return res;
}

template<typename T> std::vector<T>
cucopy(const thrust::device_vector<T> &source, const cudaStream_t custream = nullptr)
{
    std::vector<T> res(source.size());
    cu_errchk(cudaMemcpyAsync(res.data(), thrust::raw_pointer_cast(source.data()), source.size() * sizeof(T), cudaMemcpyDeviceToHost, custream));
    return res;
}

template<typename T> T *
cumallocopy(const std::vector<T> &v, const cudaStream_t custream = nullptr)
{
    T *ptr;
    cu_errchk(cudaMallocAsync((void **)&ptr, v.size() * sizeof(T), custream));
    cu_errchk(cudaMemcpyAsync(ptr, v.data(), v.size() * sizeof(T), cudaMemcpyKind::cudaMemcpyHostToDevice, custream));
    return ptr;
}

template<typename T> inline T *cucalloc(const cudaStream_t custream = nullptr, const size_t count = 1)
{
    T *ptr;
    cu_errchk(cudaMallocAsync((void **) &ptr, count * sizeof(T), custream));
    cu_errchk(cudaMemsetAsync(ptr, 0, count * sizeof(T), custream));
    return ptr;
}


// float atomicMin
__device__ __forceinline__ float atomicMin(float *address, float val)
{
    int ret = __float_as_int(*address);
    while(val < __int_as_float(ret))
    {
        int old = ret;
        if((ret = atomicCAS((int *)address, old, __float_as_int(val))) == old)
            break;
    }
    return __int_as_float(ret);
}

// float atomicMax
__device__ __forceinline__ float atomicMax(float *address, float val)
{
    int ret = __float_as_int(*address);
    while(val > __int_as_float(ret))
    {
        int old = ret;
        if((ret = atomicCAS((int *)address, old, __float_as_int(val))) == old)
            break;
    }
    return __int_as_float(ret);
}

// double atomicMin
__device__ __forceinline__ double atomicMin(double *address, double val)
{
    unsigned long long ret = __double_as_longlong(*address);
    while(val < __longlong_as_double(ret))
    {
        unsigned long long old = ret;
        if((ret = atomicCAS((unsigned long long *)address, old, __double_as_longlong(val))) == old)
            break;
    }
    return __longlong_as_double(ret);
}

// double atomicMax
__device__ __forceinline__ double atomicMax(double *address, double val)
{
    unsigned long long ret = __double_as_longlong(*address);
    while(val > __longlong_as_double(ret))
    {
        unsigned long long old = ret;
        if((ret = atomicCAS((unsigned long long *)address, old, __double_as_longlong(val))) == old)
            break;
    }
    return __longlong_as_double(ret);
}

}

#endif //SVR_CUDA_UTIL_CUH

#else

#define CRPTR(T) CPTR(T)

#endif //__CUDACC__
