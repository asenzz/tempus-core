#if 0
//
// Created by zarko on 1/18/23.
//

#include <vector>
#include <algorithm>
#include <unistd.h>
#include <fcntl.h>
#include <sys/time.h>
#include <mutex>
#include <cassert>
//#include <helper_functions.h>
//#include <helper_cuda.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublasLt.h>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cufft.h>
#include "common/cuda_util.cuh"
#include "online_emd_impl.cuh"


namespace svr::cuoemd {


#define MASK_TILE_SIZE 2048


void __global__ cuda_oemd_fast(const size_t N, const size_t M, const double *input, const double *masks, double *output, const double stretch)
{
    __shared__ double t_mask[MASK_TILE_SIZE];

    const size_t tx = threadIdx.x;
    const size_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total_block_size = blockDim.x * gridDim.x;

    //read masks into shared memory
    for (int mask_idx = tx; mask_idx < M; mask_idx += blockDim.x) {
        t_mask[mask_idx] = masks[mask_idx];
    }
    for (int mask_idx = tx; mask_idx < M; mask_idx += blockDim.x) {
        t_input[mask_idx] = masks[mask_idx];
    }
    __syncthreads();
    for (ssize_t real_idx = thread_idx; real_idx < N; real_idx += total_block_size) {
        double sum = 0;
        if (real_idx - M * stretch + 1 < 0) {
            for (int j = 0; j < M; j++) {
                const ssize_t back_index = real_idx - (M - j) * stretch + 1;//ssize_t!
                if (back_index + stretch - 1 >= 0) {
                    const double mask_val = t_mask[j];
                    double s = 0;
                    for (int k = 0; k < stretch; k++) {
                        if (back_index + k >= 0) {
                            s += input[back_index + k];
                            //s+=t_input[j];
                        }
                    }
                    sum += mask_val * s;
                }
            }
        } else {
            for (int j = 0; j < M; j++) {
                const ssize_t back_index = real_idx - (M - j) * stretch + 1;//ssize_t!
                const double mask_val = t_mask[j];
                double s = 0;
                for (int k = 0; k < stretch; k++) {
                    s += input[back_index + k];
                    //s+=t_input[j];
                }
                sum += mask_val * s;
            }
        }
        output[real_idx] = sum;
    }
}

void do_oemd_partial(const int gpu_id, std::vector<double> &inputs, std::vector<double> &smask, std::vector<std::vector<double>> &oemd_results, const double stretch)
{
    const size_t N = inputs.size();
    const size_t M = smask.size();//fits always
    oemd_results.resize(2);
    oemd_results[0].resize(N);
    oemd_results[1].resize(N);
    thrust::device_vector<double> d_inputs(N);
    thrust::device_vector<double> d_mask(M);
    double *d_input_ptr = thrust::raw_pointer_cast(d_inputs.data());
    double *d_mask_ptr = thrust::raw_pointer_cast(d_mask.data());
    cudaStream_t stream;

    gpu_errchk(cudaHostRegister(inputs.data(), sizeof(double) * N, cudaHostRegisterPortable));
    gpu_errchk(cudaHostRegister(smask.data(), sizeof(double) * M, cudaHostRegisterPortable));
    gpu_errchk(cudaStreamCreate(&stream));
    gpu_errchk(cudaMemcpyAsync(d_input_ptr, inputs.data(), sizeof(double) * N, cudaMemcpyHostToDevice, stream));
    gpu_errchk(cudaMemcpyAsync(d_mask_ptr, smask.data(), sizeof(double) * M, cudaMemcpyHostToDevice, stream));

    thrust::device_vector<double> d_output(N);
    double *d_output_ptr = thrust::raw_pointer_cast(d_output.data());
    dim3 blockDims(128 * 128, 1);
    dim3 threadDims(256, 1);
    // assert(MASK_TILE_SIZE >= M);
    cuda_oemd_fast<<<blockDims, threadDims, 0, stream>>>(N, M, d_input_ptr, d_mask_ptr, d_output_ptr, stretch);
    gpu_errchk(cudaHostRegister(oemd_results[0].data(), sizeof(double) * N, cudaHostRegisterPortable));
    gpu_errchk(cudaMemcpyAsync(oemd_results[0].data(), d_output_ptr, sizeof(double) * N, cudaMemcpyDeviceToHost, stream));
    gpu_errchk(cudaStreamSynchronize(stream));
#pragma omp parallel for
    for (size_t j = 0; j < N; ++j) {
        gpu_errchk(cudaSetDevice(gpu_id));
        oemd_results[1][j] = inputs[j] - oemd_results[0][j];
    }
    cudaHostUnregister(oemd_results[0].data());
    cudaHostUnregister(inputs.data());
    cudaHostUnregister(smask.data());
}

int do_oemd_fast(const int gpu_id, std::vector<double> &inputs, std::vector<double> &smask, std::vector <std::vector<double>> &oemd_results, const double stretch)
{
    std::vector<double> smask1 = smask;
    std::vector<double> smask2 = smask;
    const size_t N = inputs.size();
    const int num_paral = 4;
    std::vector<std::vector<double>> inputs1(num_paral);
    std::vector<std::vector<std::vector<double>>> oemd_results1(num_paral);
    std::vector<size_t> sizes(num_paral);
#pragma omp parallel for
    for (int i = 0; i < num_paral; ++i) {
        gpu_errchk(cudaSetDevice(gpu_id));
        std::vector<double> temp(
                inputs.begin() + N / num_paral * i + (i == 0 ? 0 : -(int) smask.size() + 1),
                (i == num_paral - 1 ? inputs.end() : inputs.begin() + N / num_paral * (i + 1) + ((i > 0) ? -(int) smask.size() + 1 : 0)));
        inputs1[i] = temp;
        sizes[i] = temp.size();
    }
    for (int i = 0; i < num_paral; i++) {
        std::vector<double> temp_smask(smask);
        //std::cout << inputs1[i].size() << std::endl;
        do_oemd_partial(inputs1[i], temp_smask, oemd_results1[i], stretch);
    }
    oemd_results.resize(2);
    oemd_results[0].resize(N);
    oemd_results[1].resize(N);
    //std::cout << oemd_results[0].size() << " " << std::endl;
    //std::cout << oemd_results1[0].size() << " " << N / 2 << std::endl;
    oemd_results.resize(2);
#pragma omp parallel for collapse(2)
    for (int j = 0; j < 2; ++j) {
        for (int i = 0; i < num_paral; ++i) {
            gpu_errchk(cudaSetDevice(gpu_id));
            if (oemd_results[j].size() != N) oemd_results[j].resize(N);
            if (i == 0) {
                std::memcpy(oemd_results[j].data(), oemd_results1[i][j].data(), sizes[i] * sizeof(double));
            } else {
                std::memcpy(oemd_results[j].data() + N / num_paral * i, oemd_results1[i][j].data() + smask.size() - 1, (sizes[i] - smask.size() + 1) * sizeof(double));
            }
        }
    }
}



void transform(
        const std::vector<double> &input,
        std::vector <std::vector<double>> &decon,
        const int gpu_id,
        const std::vector <size_t> &siftings,
        const std::vector <std::vector<double>> &masks,
        const double stretch_coef,
        const size_t levels)
{
    gpu_errchk(cudaSetDevice(gpu_id));
    do_oemd_fast(gpu_id, input, std::vector<double> &smask, std::vector <std::vector<double>> &oemd_results, stretch);
}


}
#endif