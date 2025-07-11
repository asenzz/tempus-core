//
// Created by zarko on 2/17/22.
//
#ifdef __CUDACC__

#ifndef SVR_CUDA_UTIL_CUH
#define SVR_CUDA_UTIL_CUH

#include <cufft.h>
#include <sstream>
#include <thrust/device_vector.h>
#include <thrust/async/for_each.h>
#include <nppdefs.h>
#include "common/logging.hpp"
#include "common/defines.h"
#include "common/constants.hpp"
#include "util/math_utils.hpp"

namespace svr {

constexpr uint64_t C_sign_mask_dbl = 0x7FFFFFFF;
constexpr unsigned C_cufft_input_limit = 64e5;
// #define HETEROGENOUS_GPU_HW

#define tid_ threadIdx.x
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

#define CU_THREADS_2D(x) unsigned((x) > common::C_cu_tile_width ? common::C_cu_tile_width : (x))
#define CU_BLOCKS_2D(x) (unsigned) CDIVI((x), common::C_cu_tile_width)
#define CU_BLOCKS_THREADS_2D(x) dim3(CU_BLOCKS_2D(x), CU_BLOCKS_2D(x)), dim3(CU_THREADS_2D(x), CU_THREADS_2D(x))
#define CU_BLOCKS_THREADS_2D2(x, y) dim3(CU_BLOCKS_2D(x), CU_BLOCKS_2D(y)), dim3(CU_THREADS_2D(x), CU_THREADS_2D(y))
#define CU_BLOCKS_THREADS_2D2_t(x, y) std::pair{CU_BLOCKS_THREADS_2D2((x), (y))}

constexpr uint16_t C_tile_width_3D_xy = 16;
constexpr uint16_t C_tile_width_3D_xy2 = C_tile_width_3D_xy * C_tile_width_3D_xy;
constexpr uint16_t C_tile_width_3D_z = 4;
#define CU_THREADS_3DXY(x) unsigned((x) > C_tile_width_3D_xy ? C_tile_width_3D_xy : (x))
#define CU_BLOCKS_3DXY(x) (unsigned) CDIVI((x), C_tile_width_3D_xy)
#define CU_THREADS_3DZ(x) unsigned((x) > C_tile_width_3D_z ? C_tile_width_3D_z : (x))
#define CU_BLOCKS_3DZ(x) (unsigned) CDIVI((x), C_tile_width_3D_z)
#define CU_BLOCKS_THREADS_3D3(x, y, z) dim3(CU_BLOCKS_3DXY(x), CU_BLOCKS_3DXY(y), CU_BLOCKS_3DZ(z)), dim3(CU_THREADS_3DXY(x), CU_THREADS_3DXY(y), CU_THREADS_3DZ(z))
#define CU_BLOCKS_THREADS_3D3_t(x, y, z) std::pair{CU_BLOCKS_THREADS_3D3((x), (y), (z))}

constexpr dim3 C_cu_tile_dim(common::C_cu_tile_width, common::C_cu_tile_width);

uint32_t total(const dim3 &d);

template<typename T> inline __device__ __host__ DTYPE(T::x) cunorm(const T &v)
{ return v.x * v.x + v.y * v.y; };

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

template<typename T> void cu_fill(T *const data, const unsigned n, const T value, const cudaStream_t custream = nullptr)
{
    (void) thrust::for_each(thrust::cuda::par.on(custream), data, data + n,[value]
            __device__(T & d)
    { d = value; });
}

constexpr uint32_t C_cu_default_stream_flags = cudaStreamDefault; // Do not set to cudaStreamNonBlocking

#ifdef NDEBUG

#define DEV_CUSTREAM(x)                             \
    cu_errchk(cudaSetDevice((x)));                  \
    cudaStream_t custream;                          \
    cu_errchk(cudaStreamCreateWithFlags(&custream, C_cu_default_stream_flags));         \

#define CTX_CUSTREAM_(x)                                                                \
    common::gpu_context_<(x)> ctx;                                                      \
    cu_errchk(cudaSetDevice(ctx.phy_id()));                                             \
    cudaStream_t custream;                                                              \
    cu_errchk(cudaStreamCreateWithFlags(&custream, C_cu_default_stream_flags));         \

#else

#define DEV_CUSTREAM(x)                             \
    cu_errchk(cudaSetDevice((x)));                  \
    cudaStream_t custream;                          \
    cu_errchk(cudaStreamCreateWithFlags(&custream, C_cu_default_stream_flags));         \
    if (!custream) LOG4_THROW("CUDA stream handle not initialized.");                   \
    int devid, stream_devid;                                                            \
    cu_errchk(cudaGetDevice(&devid));                                                   \
    if (devid != (x)) LOG4_THROW("CUDA device id mismatch " << devid << " should be " << (x)); \
    cu_errchk(cudaStreamGetDevice(custream, &stream_devid));                            \
    if (stream_devid != (x)) LOG4_THROW("CUDA stream device id mismatch " << stream_devid << " should be " << (x)); \
    LOG4_TRACE("CUDA stream device id " << stream_devid << " created on device " << devid);

#define CTX_CUSTREAM_(x)                                                                \
    const common::gpu_context_<(x)> ctx;                                                      \
    cu_errchk(cudaSetDevice(ctx.phy_id()));                                             \
    cudaStream_t custream;                                                              \
    cu_errchk(cudaStreamCreateWithFlags(&custream, C_cu_default_stream_flags));         \
    if (!custream) LOG4_THROW("CUDA stream handle not initialized.");                   \
    int devid, stream_devid;                                                            \
    cu_errchk(cudaGetDevice(&devid));                                                   \
    if (devid != ctx.phy_id()) LOG4_THROW("CUDA device id mismatch " << devid << " should be " << ctx.phy_id()); \
    cu_errchk(cudaStreamGetDevice(custream, &stream_devid));                            \
    if (stream_devid != ctx.phy_id()) LOG4_THROW("CUDA stream device id mismatch " << stream_devid << " should be " << ctx.phy_id()); \
    LOG4_TRACE("CUDA stream device id " << stream_devid << " created on device " << devid);

#endif

#define CTX_CUSTREAM CTX_CUSTREAM_(CTX_PER_GPU)
#define CTX4_CUSTREAM CTX_CUSTREAM_(4)

void cusyndestroy(const cudaStream_t strm);

NppStreamContext get_npp_context(const unsigned gpuid, const cudaStream_t custream);

void copy_submat(CRPTRd in, RPTR(double) out, const uint32_t ldin, const uint32_t in_start_m, const uint32_t in_start_n, const uint32_t in_end_m,
                 const uint32_t in_end_n, const uint32_t ldout, cudaMemcpyKind kind, const cudaStream_t stm);

__host__ __device__ inline constexpr unsigned clamp_n(const unsigned n)
{ return _MIN(n, svr::common::C_cu_clamp_n); }


template<typename T> inline T *
cumallocopy(const std::vector<T> &v, const cudaStream_t custream = nullptr)
{
    T *ptr;
    cu_errchk(cudaMallocAsync((void **) &ptr, v.size() * sizeof(T), custream));
    cu_errchk(cudaMemcpyAsync(ptr, v.data(), v.size() * sizeof(T), cudaMemcpyKind::cudaMemcpyHostToDevice, custream));
    return ptr;
}

template<typename T> inline T *
cumallocopy(const std::span<T> &v, const cudaStream_t custream = nullptr)
{
    std::decay_t<T> *ptr;
    cu_errchk(cudaMallocAsync((void **) &ptr, v.size() * sizeof(T), custream));
    cu_errchk(cudaMemcpyAsync(ptr, v.data(), v.size() * sizeof(T), cudaMemcpyKind::cudaMemcpyHostToDevice, custream));
    return ptr;
}

template<typename T> inline T *
cumallocopy(const arma::Mat<T> &v, const cudaStream_t custream = nullptr)
{
    T *ptr;
    cu_errchk(cudaMallocAsync((void **) &ptr, v.n_elem * sizeof(T), custream));
    cu_errchk(cudaMemcpyAsync(ptr, v.mem, v.n_elem * sizeof(T), cudaMemcpyKind::cudaMemcpyHostToDevice, custream));
    return ptr;
}

template<typename I, typename T = typename I::value_type> inline T *
cumallocopy(const I &begin, const I &end, const cudaStream_t custream = nullptr)
{
    T *ptr;
    const auto size = std::distance(begin, end) * sizeof(T);
    cu_errchk(cudaMallocAsync((void **) &ptr, size, custream));
    cu_errchk(cudaMemcpyAsync(ptr, &*begin, size, cudaMemcpyKind::cudaMemcpyHostToDevice, custream));
    return ptr;
}

template<typename T> inline T *cucalloc(const cudaStream_t custream = nullptr, const size_t count = 1)
{
    T *ptr;
    cu_errchk(cudaMallocAsync((void **) &ptr, count * sizeof(T), custream));
    cu_errchk(cudaMemsetAsync(ptr, 0, count * sizeof(T), custream));
    return ptr;
}

template<typename T> inline T *
cumallocopy(const T *source, const cudaStream_t custream = nullptr, const unsigned len = 1, const cudaMemcpyKind kind = cudaMemcpyHostToDevice)
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
        case cudaMemcpyDefault: cu_errchk(cudaMallocAsync((void **) &ptr, size, custream));
            break;
    }
    cu_errchk(cudaMemcpyAsync(ptr, source, size, kind, custream));
    return ptr;
}

template<typename T> void cufreecopy(T *output, const T *source, const cudaStream_t custream = nullptr, const size_t len = 1)
{
    cu_errchk(cudaMemcpyAsync(output, source, len * sizeof(T), cudaMemcpyDeviceToHost, custream));
    cu_errchk(cudaFreeAsync((void *) source, custream));
    cu_errchk(cudaStreamSynchronize(custream));
}

template<typename T> std::vector<T> cufreecopy(const T *source, const cudaStream_t custream = nullptr, const size_t len = 1)
{
    std::vector<T> res(len);
    cu_errchk(cudaMemcpyAsync(res.data(), source, len * sizeof(T), cudaMemcpyDeviceToHost, custream));
    cu_errchk(cudaFreeAsync((void *) source, custream));
    return res;
}

template<typename T> std::vector<T> cucopy(const T *source, const size_t length, const cudaStream_t custream = nullptr)
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

template<typename T> inline void
cucopy(RPTR(T) dest, CRPTR(T) src, const size_t length = 1, const cudaStream_t custream = nullptr, const cudaMemcpyKind kind = cudaMemcpyDeviceToDevice, const unsigned stride = 1)
{
    cu_errchk(cudaMemcpy2DAsync(dest, sizeof(T), src, stride * sizeof(T), sizeof(T), length / stride, kind));
}


// float atomicMin
__device__ __forceinline__ float atomicMin(float *address, float val)
{
    int ret = __float_as_int(*address);
    while (val < __int_as_float(ret)) {
        int old = ret;
        if ((ret = atomicCAS((int *) address, old, __float_as_int(val))) == old)
            break;
    }
    return __int_as_float(ret);
}

// float atomicMax
__device__ __forceinline__ float atomicMax(float *address, float val)
{
    int ret = __float_as_int(*address);
    while (val > __int_as_float(ret)) {
        int old = ret;
        if ((ret = atomicCAS((int *) address, old, __float_as_int(val))) == old)
            break;
    }
    return __int_as_float(ret);
}

// double atomicMin
__device__ __forceinline__ double atomicMin(double *address, double val)
{
    unsigned long long ret = __double_as_longlong(*address);
    while (val < __longlong_as_double(ret)) {
        unsigned long long old = ret;
        if ((ret = atomicCAS((unsigned long long *) address, old, __double_as_longlong(val))) == old)
            break;
    }
    return __longlong_as_double(ret);
}

// double atomicMax
__device__ __forceinline__ double atomicMax(double *address, double val)
{
    unsigned long long ret = __double_as_longlong(*address);
    while (val > __longlong_as_double(ret)) {
        unsigned long long old = ret;
        if ((ret = atomicCAS((unsigned long long *) address, old, __double_as_longlong(val))) == old)
            break;
    }
    return __longlong_as_double(ret);
}


__device__ __forceinline__ double atomicMul(double *const address, double val)
{
    auto const address_as_ull = (unsigned long long int *) address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val * __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

__device__ __forceinline__ float atomicMul(float *const address, float val)
{
    auto const address_as_int = (int *) address;
    int old = *address_as_int, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_int, assumed, __float_as_int(val * __float_as_int(assumed)));
    } while (assumed != old);
    return __int_as_float(old);
}


template<typename T> __host__ __device__ __forceinline__ int8_t signum(const T val)
{
    return (T(0) < val) - (val < T(0));
}

// ICPX bug forced to move this out of cuvalidate
uint8_t get_streams_per_gpu(const uint32_t n_rows);

template<typename T> __device__ inline T min(const T a, const T b, const T c)
{
    T m = a;
    if (m > b) m = b;
    if (m > c) m = c;
    return m;
}

}

#endif //SVR_CUDA_UTIL_CUH

#endif //__CUDACC__
