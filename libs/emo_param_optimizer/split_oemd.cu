#include <vector>
#include <algorithm>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cassert>

#include <cublasLt.h>
#include <cusolverDn.h>
#include <cublas_v2.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cufft.h>

#include <unistd.h>
#include <fcntl.h>
#include <sys/time.h>
#include <mutex>
#include "fast_functions.hpp"


#define BLOCK_SIZE 32

// Example with cublasLt library to execute single precision 
// gemm with cublasLtMatmul. This is almost a drop-in replacement for 
// cublasSgemm, with the addition of the workspace to support 
// split-K algorithms.
//
// Additional notes: Pointer mode is always host. To change it,
// configure the appropriate matmul descriptor attribute.
//
// Matmul here does not use cuBLAS handle's configuration of math
// mode. Also, here tensor ops are implicitly allowed; to change
// this, configure appropriate attribute in the preference handle.




#ifndef cublasSafeCall
#define cublasSafeCall(err)     {if (err) std::cout << " Abort "<<  __FILE__<< __LINE__ << std::endl;}
#endif

#define gpu_errchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#define cufft_errchk(ans) { cufftAssert((ans), __FILE__, __LINE__); }

inline void cufftAssert(cufftResult_t code, const char *file, int line, bool abort = true)
{
    if (code != CUFFT_SUCCESS) {
        fprintf(stderr, "cufftAssert: error code %i file %s line %d\n", (int) code, file, line);
        if (abort) exit(code);
    }
}


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}


void expand_the_mask(const size_t mask_size, const size_t input_size, const double *dev_mask, double *dev_expanded_mask)
{
    gpu_errchk(cudaMemset(dev_expanded_mask, 0, sizeof(double) * input_size));
    // gpu_errchk(cudaDeviceSynchronize());
    gpu_errchk(cudaMemcpy(dev_expanded_mask, dev_mask, sizeof(double) * mask_size, cudaMemcpyDeviceToDevice));
}


__global__ void gpu_multiply_complex(const size_t input_size,
                                     const cufftDoubleComplex *multiplier,
                                     cufftDoubleComplex *output)
{
    const size_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total_block_size = blockDim.x * gridDim.x;

    for (size_t j = thread_idx; j < (input_size / 2 + 1); j += total_block_size) {//because it is D2Z transform
        cufftDoubleComplex new_output;
        new_output.x = output[j].x * multiplier[j].x - output[j].y * multiplier[j].y;
        new_output.y = output[j].x * multiplier[j].y + output[j].y * multiplier[j].x;
        output[j].x = new_output.x / (double) input_size;//because of inverse fft
        output[j].y = new_output.y / (double) input_size;
    }
}

__global__ void
vec_power(
        cufftDoubleComplex *x,
        const size_t x_size, const int siftings)
{
    const size_t ix = blockIdx.x * blockDim.x + threadIdx.x;
    for (size_t j = ix; j < x_size / 2 + 1; j += blockDim.x * gridDim.x) {
        double px = (1. - x[j].x);
        double py = -x[j].y;

        double px_out, py_out;
        for (int i = 1; i < siftings; i++) {
            px_out = px * (1. - x[j].x) - py * (-x[j].y);
            py_out = px * (-x[j].y) + py * (1. - x[j].x);
            px = px_out;
            py = py_out;
        }
        x[j].x = px;
        x[j].y = py;
    }
}


__global__ void
vec_subtract_inplace(
        double *x,
        const double *__restrict__ y,
        const size_t x_size)
{
    const size_t ix = blockIdx.x * blockDim.x + threadIdx.x;
    for (int j = ix; j < x_size; j += blockDim.x * gridDim.x) {
        x[j] = x[j] - y[j];
    }
}


const size_t oemd_block_size = 1024;
#define CUDA_THREADS_BLOCKS(x_size) ((x_size) + oemd_block_size - 1) / oemd_block_size, oemd_block_size

int split_oemd(const std::vector<double> &cvmd, const std::vector <std::vector<double>> &mask, const std::vector<int> &siftings, std::vector <std::vector<double>> &oemd_levels)
{
//WARNING - creation of plans is not thread safe  perhaps!

    std::vector <thrust::device_vector<double>> d_mask(mask.size());
    for (int i = 0; i < mask.size(); i++) {
        d_mask[i].resize(mask[i].size());
    }
    oemd_levels.resize(mask.size() + 1);
    size_t full_input_size = cvmd.size();
    thrust::device_vector<double> d_zm(full_input_size);
    double *d_zm_ptr = thrust::raw_pointer_cast(d_zm.data());
    thrust::device_vector <cufftDoubleComplex> d_zm_fft(full_input_size);
    cufftDoubleComplex *d_zm_fft_ptr = thrust::raw_pointer_cast(d_zm_fft.data());
    thrust::device_vector<double> d_rem(full_input_size);
    thrust::device_vector<double> d_imf(full_input_size);
    double *d_rem_ptr = thrust::raw_pointer_cast(d_rem.data());
    double *d_imf_ptr = thrust::raw_pointer_cast(d_imf.data());
    thrust::device_vector <cufftDoubleComplex> d_rem_fft(full_input_size);
    cufftDoubleComplex *d_rem_fft_ptr = thrust::raw_pointer_cast(d_rem_fft.data());
    gpu_errchk(cudaMemcpy(d_rem_ptr, cvmd.data(), sizeof(double) * full_input_size, cudaMemcpyHostToDevice));
    d_imf = d_rem;
    int n_batch = 1;
    std::cout << full_input_size << std::endl;
    cufftHandle plan_full_forward, plan_full_backward;
    cufft_errchk(cufftPlan1d(&plan_full_forward, full_input_size, CUFFT_D2Z, n_batch));
    cufft_errchk(cufftPlan1d(&plan_full_backward, full_input_size, CUFFT_Z2D, n_batch));
    for (int i = 0; i < mask.size(); i++) {
        std::cout << "Doing level " << i << std::endl;

        oemd_levels[i].resize(full_input_size);
        size_t mask_size = mask[i].size();
        thrust::host_vector<double> h_mask(mask_size);
        std::memcpy(h_mask.data(), mask[i].data(), sizeof(double) * mask_size);
        d_mask[i] = h_mask;

        double *d_mask_ptr = thrust::raw_pointer_cast(d_mask[i].data());
        expand_the_mask(mask_size, full_input_size, d_mask_ptr, d_zm_ptr);
        cufft_errchk(cufftExecD2Z(plan_full_forward, d_zm_ptr, d_zm_fft_ptr));
        cufft_errchk(cufftExecD2Z(plan_full_forward, d_rem_ptr, d_rem_fft_ptr));
        vec_power<<<CUDA_THREADS_BLOCKS(full_input_size / 2 + 1)>>>(d_zm_fft_ptr, full_input_size, siftings[i]);
        gpu_multiply_complex<<<CUDA_THREADS_BLOCKS(full_input_size / 2 + 1) >>>(full_input_size, d_zm_fft_ptr, d_rem_fft_ptr);
        cufft_errchk(cufftExecZ2D(plan_full_backward, d_rem_fft_ptr, d_rem_ptr));

        vec_subtract_inplace<<<CUDA_THREADS_BLOCKS(full_input_size)>>>(d_imf_ptr, d_rem_ptr, full_input_size);
        gpu_errchk(cudaMemcpy(oemd_levels[i].data(), d_imf_ptr, sizeof(double) * full_input_size, cudaMemcpyDeviceToHost));
        d_imf = d_rem;
    }

    oemd_levels[mask.size()].resize(full_input_size);
    gpu_errchk(cudaMemcpy(oemd_levels[mask.size()].data(), d_rem_ptr, sizeof(double) * full_input_size, cudaMemcpyDeviceToHost));
    cufftDestroy(plan_full_forward);
    cufftDestroy(plan_full_backward);
    return 0;
}

