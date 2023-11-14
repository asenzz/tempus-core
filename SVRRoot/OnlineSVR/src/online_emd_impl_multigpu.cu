//
// Created by zarko on 2/16/22.
//

#include "online_emd_impl.cuh"

#if 0 // Buggy

#include <vector>
#include <thread>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cassert>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "common/cuda_util.cuh"


const size_t oemd_block_size = 1024;
#define CUDA_THREADS_BLOCKS(x_size) ((x_size) + oemd_block_size - 1) / oemd_block_size, oemd_block_size

namespace svr {
namespace cuoemd {

#if 0
__global __ void reduce_mask_trim(
        const size_t apply_ix,
        const double * masks,
        const double * rx,
        const double * x,
)
{
    rx[ix] += masks[size_t(mask_size - 1 - ix / stretch_coef + j / stretch_coef)] * x[j];
    sum += masks[size_t(mask_size - 1 - ix / stretch_coef + j / stretch_coef)];
}

__global__ void reduce_mask(
        double * rx,
        const double * masks,
        const double * x,
        const size_t apply_ix,
        const size_t mask_size,
        const double stretch_coef)
{
    const size_t m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m < mask_size * stretch_coef)
        rx[apply_ix] += masks[size_t(m / stretch_coef)] * x[size_t(apply_ix - mask_size + m / stretch_coef + 1)] / stretch_coef;
}

#endif

__global__ void apply_mask(
    const double stretch_coef,
    const double * __restrict__ x,
    const size_t x_size,
    const double * __restrict__ masks,
    const size_t mask_size,
    double * __restrict__ rx,
    const size_t x_offset,
    const size_t total_size)
{
    const size_t ix = blockIdx.x * blockDim.x + threadIdx.x;

    if (ix >= x_size or ix + x_offset >= total_size) return;

    rx[ix] = 0;
    if (x_offset + ix >= stretch_coef * mask_size - 1) {
        for (size_t m = 0; m < mask_size * stretch_coef; ++m)
            rx[ix] += masks[size_t(m / stretch_coef)] * x[size_t(x_offset + ix - stretch_coef * mask_size + m + 1)] / stretch_coef;
    } else {
        double sum = 0;
        for (size_t j = 0; j <= x_offset + ix; ++j) {
            rx[ix] += masks[size_t(mask_size - 1 - (x_offset + ix) / stretch_coef + j / stretch_coef)] * x[j] / stretch_coef;
            sum += masks[size_t(mask_size - 1 - (x_offset + ix) / stretch_coef + j / stretch_coef)] / stretch_coef;
        }
        rx[ix] /= sum;
    }
}


__global__ void
vec_subtract_inplace(
    double * __restrict__ x,
    const double * __restrict__ y,
    const size_t y_size,
    const size_t x_offset,
    const size_t y_offset,
    const size_t total_size)
{
    const size_t ix = blockIdx.x * blockDim.x + threadIdx.x;
    if (ix + x_offset < total_size and ix + y_offset < total_size and ix < y_size) x[ix + x_offset] -= y[ix + y_offset];
}


void transform(
        const std::vector<double> &input,
        std::vector <std::vector<double>> &decon,
        const std::vector<int> &gpu_ids,
        const std::vector<size_t> &siftings,
        const std::vector<std::vector<double>> &masks,
        const double stretch_coef,
        const size_t levels)
{
    //for (const auto gpu_id: gpu_ids) if (gpu_id != gpu_ids[0]) cu_errchk(cudaDeviceEnablePeerAccess(gpu_id, 0));
    const auto run_gpus = [&](const std::function<void(const size_t)> &gpu_fun) {
        std::vector<std::thread> gpu_threads;
        for (size_t gpu_ix = 0; gpu_ix < gpu_ids.size(); ++gpu_ix) {
            gpu_threads.emplace_back([gpu_fun, gpu_ix, &gpu_ids]() {
                cu_errchk(cudaSetDevice(gpu_ids[gpu_ix]));
                gpu_fun(gpu_ix);
                cu_errchk(cudaDeviceSynchronize());
            });
        }
        for (auto &thr: gpu_threads) thr.join();
        gpu_threads.clear();
    };

    const auto gpu_chunk_ct = input.size() / gpu_ids.size();
    std::vector<double *> p_remainder(gpu_ids.size());
    std::vector<double *> p_rx(gpu_ids.size());
    std::vector<double *> p_rx2(gpu_ids.size());
    run_gpus([&](const size_t gpu_ix) {
        cu_errchk(cudaMalloc(&(p_remainder.data()[gpu_ix]), gpu_chunk_ct * sizeof(double)));
        cu_errchk(cudaMemcpy(p_remainder[gpu_ix], input.data() + gpu_ix * gpu_chunk_ct, gpu_chunk_ct * sizeof(double), cudaMemcpyHostToDevice));

        const auto this_dev_sz = (gpu_ix + 1) * gpu_chunk_ct * sizeof(double);
        cu_errchk(cudaMalloc(&(p_rx.data()[gpu_ix]), this_dev_sz));
        cu_errchk(cudaMemcpy(p_rx[gpu_ix], input.data(), this_dev_sz, cudaMemcpyHostToDevice));

        cu_errchk(cudaMalloc(&(p_rx2.data()[gpu_ix]), gpu_chunk_ct * sizeof(double)));
        cu_errchk(cudaMemset(p_rx2[gpu_ix], 0, gpu_chunk_ct * sizeof(double)));
    });

    if (decon.size() != input.size()) decon.resize(input.size());
    for (size_t l = 0; l < levels - 1; l++) {
        std::vector<double *> dev_mask(gpu_ids.size());
        run_gpus([&](const size_t gpu_ix) {
            dev_mask[gpu_ix] = cuda_malloccopy(masks[l]);
        });

        for (size_t j = 0; j < siftings[l]; ++j) {
            run_gpus([&](const size_t gpu_ix) {
                const auto this_dev_offset = gpu_ix * gpu_chunk_ct;
                apply_mask<<<CUDA_THREADS_BLOCKS(gpu_chunk_ct)>>>(stretch_coef, p_rx[gpu_ix], gpu_chunk_ct, dev_mask[gpu_ix], masks[l].size(), p_rx2[gpu_ix], this_dev_offset, input.size());
                vec_subtract_inplace<<<CUDA_THREADS_BLOCKS(gpu_chunk_ct)>>>(p_rx[gpu_ix], p_rx2[gpu_ix], gpu_chunk_ct, this_dev_offset, 0, input.size());
            });
        }

        auto host_rx = (double *) malloc(input.size() * sizeof(double));
        run_gpus([&](const size_t gpu_ix) {
            cu_errchk(cudaFree(dev_mask[gpu_ix]));
            cu_errchk(cudaMemcpy(host_rx + gpu_ix * gpu_chunk_ct, p_rx[gpu_ix] + gpu_ix * gpu_chunk_ct, gpu_chunk_ct * sizeof(double), cudaMemcpyDeviceToHost))
        });

        // TODO Parallelize
        for (size_t t = 0; t < input.size(); ++t) {
            if (decon[t].size() != levels) decon[t].resize(levels);
            decon[t][levels - l - 1] = host_rx[t];
        }

        run_gpus([&](const size_t gpu_ix) {
            const auto this_dev_sz = (gpu_ix + 1) * gpu_chunk_ct * sizeof(double);
            cu_errchk(cudaMemcpy(p_rx[gpu_ix], host_rx, this_dev_sz, cudaMemcpyHostToDevice));

            vec_subtract_inplace<<<CUDA_THREADS_BLOCKS(gpu_chunk_ct)>>>(p_remainder[gpu_ix], p_rx[gpu_ix], gpu_chunk_ct, 0, gpu_ix * gpu_chunk_ct, input.size());
            cu_errchk(cudaMemcpy(p_rx[gpu_ix] + gpu_ix * gpu_chunk_ct, p_remainder[gpu_ix], gpu_chunk_ct * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToDevice));
        });
        free(host_rx);
    }
    {
        double *p_host_rx = (double *) malloc(input.size() * sizeof(double));
        run_gpus([&](const size_t gpu_ix) {
            cu_errchk(cudaMemcpy(p_host_rx + gpu_ix * gpu_chunk_ct, p_rx[gpu_ix], gpu_chunk_ct * sizeof(double), cudaMemcpyDeviceToHost));
        });

        for (size_t t = 0; t < input.size(); ++t) decon[t][0] = p_host_rx[t];
        free(p_host_rx);
    }
    run_gpus([&](const size_t gpu_ix) {
        cu_errchk(cudaFree(p_rx2[gpu_ix]));
        cu_errchk(cudaFree(p_rx[gpu_ix]));
        cu_errchk(cudaFree(p_remainder[gpu_ix]));
    });
}


} // namespace cuoemd
} // namespace svr

#endif // #ifdef CUDA_OEMD_MULTIGPU
