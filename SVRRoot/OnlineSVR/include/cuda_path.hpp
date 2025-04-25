//
// Created by zarko on 12/22/22.
//

#ifndef SVR_CUDA_PATH_HPP
#define SVR_CUDA_PATH_HPP

#include "common/compatibility.hpp"
#include "cuqrsolve.cuh"

constexpr uint8_t C_dist_pow = 4;
#define DIST(x) pow(abs(x), C_dist_pow)

namespace svr::kernel::path {

template<typename T>
void cu_kernel_xy(const uint32_t X_cols, const uint32_t Y_cols, const uint32_t rows, const uint16_t lag, const uint16_t dim, const uint16_t lag_TILE_WIDTH, const T lambda,
                  const T tau, const T gamma, const T mean, CRPTR(T) X, CRPTR(T) Y, RPTR(T) Z, const cudaStream_t custream);

template<typename T>
void cu_distances_xy(const uint32_t X_cols, const uint32_t Y_cols, const uint32_t rows, const uint16_t lag, const uint16_t dim, const uint16_t lag_TILE_WIDTH, const T lambda,
                     const T tau, CRPTR(T) X, CRPTR(T) Y, RPTR(T) Z, const cudaStream_t custream);

template<typename T>
void distances_xy(const uint32_t X_cols, const uint32_t Xy_cols, const uint32_t rows, const uint16_t lag, const double lambda, const T tau, CRPTR(T) X, CRPTR(T) Xy, RPTR(T) Z);

template<typename T>
void kernel_xy(const uint32_t X_cols, const uint32_t Xy_cols, const uint32_t rows, const uint16_t lag, const T gamma, const T lambda, const T tau, const T mean,
               CRPTR(T) X, CRPTR(T) Xy, RPTR(T) Z);

template<typename T>
void kernel_xy(const uint32_t X_cols, const uint32_t Xy_cols, const uint32_t rows, const uint16_t lag, const T gamma, const T lambda, const T tau, const T mean,
               CRPTR(T) X, CRPTR(T) Xy, RPTR(T) K, const uint16_t gpu_id);

}

#endif
