//
// Created by zarko on 05/12/2024.
//

#if 0

#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include "common/compatibility.hpp"

namespace svr {
namespace kernel {
namespace js {


// CUDA kernel for Jensen-Shannon divergence
__global__ void G_js_divergence(CRPTRd P, CRPTRd Q, RPTR(double) result, uint32_t size)
{
    extern __shared__ double sharedMem[]; // Shared memory for reduction
    const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
    const auto lane = threadIdx.x;

    // Initialize shared memory
    sharedMem[lane] = 0;

    // Compute D_KL terms for the thread's assigned elements
    if (tid < size) {
        const double p = P[tid];
        const double q = Q[tid];
        const double m = .5 * (p + q);
        // Avoid log(0) or division by 0
        if (p > 0) sharedMem[lane] += 0.5f * p * log(p / m);
        if (q > 0) sharedMem[lane] += 0.5f * q * log(q / m);
    }

    // Synchronize threads within the block
    __syncthreads();

    // Perform reduction within the block
    for (auto stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (lane < stride) sharedMem[lane] += sharedMem[lane + stride];
        __syncthreads();
    }

    // Write block result to global memory
    if (!lane) atomicAdd(result, sharedMem[0]);
}

// Function to normalize a vector to make it a probability distribution
std::vector<double> normalize(const std::vector<double>& vec) {
    double sum = std::accumulate(vec.begin(), vec.end(), 0.0);
    std::vector<double> normalized(vec.size());
    for (size_t i = 0; i < vec.size(); ++i) {
        normalized[i] = vec[i] / sum;
    }
    return normalized;
}

int main()
{
    const int size = 8;
    const int blockSize = 256;

      // Normalize P and Q
    std::vector<double> P_norm = normalize(P);
    std::vector<double> Q_norm = normalize(Q);

    // Example probability distributions
    double h_P[size] = {0.1, 0.2, 0.15, 0.25, 0.1, 0.05, 0.05, 0.1};
    double h_Q[size] = {0.15, 0.25, 0.1, 0.2, 0.1, 0.1, 0.05, 0.05};

    // Allocate device memory
    double *d_P, *d_Q, *d_result;
    cudaMalloc(&d_P, size * sizeof(double));
    cudaMalloc(&d_Q, size * sizeof(double));
    cudaMalloc(&d_result, sizeof(double));

    // Copy data to device
    cudaMemcpy(d_P, h_P, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Q, h_Q, size * sizeof(double), cudaMemcpyHostToDevice);

    // Initialize result on device
    double h_result = 0.0f;
    cudaMemcpy(d_result, &h_result, sizeof(double), cudaMemcpyHostToDevice);

    // Launch kernel
    int gridSize = (size + blockSize - 1) / blockSize;
    js_divergence<<<gridSize, blockSize, blockSize * sizeof(double)>>>(d_P, d_Q, d_result, size);

    // Copy result back to host
    cudaMemcpy(&h_result, d_result, sizeof(double), cudaMemcpyDeviceToHost);

    // Print result
    std::cout << "Jensen-Shannon Divergence: " << h_result << std::endl;

    // Free device memory
    cudaFree(d_P);
    cudaFree(d_Q);
    cudaFree(d_result);

    return 0;
}


} // namespace svr
} // namespace kernel
} // namespace js

#endif