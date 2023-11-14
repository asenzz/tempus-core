//
// Created by jiashuai on 17-9-14.
//

#ifndef THUNDERSVM_THUNDERSVM_H
#define THUNDERSVM_THUNDERSVM_H
#include <cstdlib>
#include <string>
#include <vector>

#define DATASET_DIR "/home/zarko/pub/thundersvm/dataset/"

#define USE_DOUBLE

#include "math.h"
#include "common/Logging.hpp"

#ifndef max

template<class T>
static inline T max(T x, T y) { return (x > y) ? x : y; }

#endif
#ifndef min

template<class T>
static inline T min(T x, T y) { return (x < y) ? x : y; }

#endif

template<typename T>
inline T max2power(T n) {
    return T(pow(2, floor(log2f(float(n)))));
}
const int BLOCK_SIZE = 512;

const int NUM_BLOCKS = 32 * 56;


#ifdef USE_CUDA

#include "cuda_runtime_api.h"

#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  { \
    cudaError_t error = condition; \
    if (error == cudaErrorMemoryAllocation) throw std::bad_alloc(); \
    if (error != cudaSuccess) LOG4_ERROR("CUDA error " << cudaGetErrorString(error)); \
  }

#define SAFE_KERNEL_LAUNCH(kernel_name, ...) \
    kernel_name<<<NUM_BLOCKS,BLOCK_SIZE>>>(__VA_ARGS__);\
    CUDA_CHECK(cudaPeekAtLastError())

#define KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)
#else
#define __host__
#define __device__
#define KERNEL_LOOP(i, n) \
  for (int i = 0; \
       i < (n); \
       i++)
#endif


using std::string;
using std::vector;
typedef double float_type;


#ifdef USE_DOUBLE
typedef double kernel_type;
#else
typedef float kernel_type;
#endif
#endif //THUNDERSVM_THUNDERSVM_H
