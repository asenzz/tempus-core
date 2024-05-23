//
// Created by zarko on 2/16/22.
//

#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cufft.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <common/cuda_util.cuh>
#include "model/DeconQueue.hpp"
#include "common/gpu_handler.tpp"
#include "online_emd.hpp"

#ifdef OEMDFFT
#define CUFFT_INPUT_LIMIT // 64e5 // TODO Implement chunking and multi GPU support
#endif

namespace svr {
namespace oemd {


__global__ void
G_gpu_multiply_complex(
        const size_t input_size,
        const cufftDoubleComplex *__restrict__ multiplier,
        cufftDoubleComplex *__restrict__ output)
{
    const auto thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const auto total_block_size = blockDim.x * gridDim.x;

    cufftDoubleComplex new_output;
    for (auto j = thread_idx; j < input_size / 2 + 1; j += total_block_size) {//because it is D2Z transform
        new_output.x = output[j].x * multiplier[j].x - output[j].y * multiplier[j].y;
        new_output.y = output[j].x * multiplier[j].y + output[j].y * multiplier[j].x;
        output[j].x = new_output.x / double(input_size); // because of inverse FFT
        output[j].y = new_output.y / double(input_size);
    }
}


__global__ void
vec_power(
        cufftDoubleComplex *__restrict__ x,
        const size_t x_size,
        const int siftings)
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
G_vec_power(
        const cufftDoubleComplex *__restrict__ x,
        cufftDoubleComplex *__restrict__ y,
        const size_t x_size,
        const size_t siftings)
{
    const auto ix = blockIdx.x * blockDim.x + threadIdx.x;
    double px, py;
    for (auto j = ix; j < x_size / 2 + 1; j += blockDim.x * gridDim.x) {
        px = 1. - x[j].x;
        py = -x[j].y;
        for (size_t i = 1; i < siftings; ++i) {
            px = px * (1. - x[j].x) - py * (-x[j].y);
            py = px * (-x[j].y) + py * (1. - x[j].x);
        }
        y[j].x = px;
        y[j].y = py;
    }
}


__global__ void
G_vec_subtract_inplace(
        double *__restrict__ x,
        const double *__restrict__ y,
        const size_t x_size)
{
    CUDA_STRIDED_FOR_i(x_size) x[i] -= y[i];
}

__global__ void
apply_mask(
        const double stretch_coef,
        double *__restrict__ rx,
        const size_t x_size,
        const double *__restrict__ mask,
        const size_t mask_size,
        const size_t stretched_mask_size,
        double *__restrict__ rx2,
        const size_t start_x)
{
    const size_t ix = start_x + blockIdx.x * blockDim.x + threadIdx.x;
    if (ix >= x_size) return;

    rx2[ix] = 0;
    if (ix >= stretched_mask_size - 1) {
        for (size_t m = 0; m < stretched_mask_size; ++m)
            rx2[ix] += mask[size_t(m / stretch_coef)] * rx[size_t(ix - stretched_mask_size + m + 1)] / stretch_coef;
    } else {
        double sum = 0;
        for (size_t j = 0; j <= ix; ++j) {
            rx2[ix] += mask[size_t(mask_size - 1 - ix / stretch_coef + j / stretch_coef)] * rx[j] / stretch_coef;
            sum += mask[size_t(mask_size - 1 - ix / stretch_coef + j / stretch_coef)] / stretch_coef;
        }
        rx2[ix] /= sum;
    }
}

#ifdef OEMDFFT

void transform_fft(
        datamodel::datarow_range &inout,
        const std::vector<double> &tail,
        const std::deque<size_t> &siftings,
        const std::deque<std::vector<double>> &masks,
        const double stretch_coef)
{
    const size_t levels = inout.levels() / 4;
    const size_t in_colix = levels * 2;
    const auto full_input_size = inout.distance() + tail.size();
    cu_errchk(cudaSetDevice(0));
    const auto max_gpus = common::gpu_handler_hid::get().get_gpu_devices_count();
    std::deque<size_t> gpuids(max_gpus);
    for (size_t d = 0; d < max_gpus; ++d) gpuids[d] = d;
    std::vector<double> h_rx(full_input_size);
#pragma omp parallel for schedule(static, 1 + full_input_size / std::thread::hardware_concurrency()) num_threads(adj_threads(full_input_size))
    for (size_t t = 0; t < full_input_size; ++t)
        h_rx[t] = t < tail.size() ? tail[t] : inout[t - tail.size()]->get_value(in_colix);

    std::deque<size_t> start_ix(max_gpus), job_len(max_gpus);
#pragma omp parallel for num_threads(adj_threads(max_gpus)) schedule(static, 1)
    for (size_t d = 0; d < max_gpus; ++d) {
        cu_errchk(cudaSetDevice(gpuids[d]));
        start_ix[d] = d * full_input_size / max_gpus;
        job_len[d] = d == max_gpus - 1 ? full_input_size - start_ix[d] : full_input_size / max_gpus;
    }

    std::deque<thrust::device_vector<double>> d_mask(masks.size());
    for (size_t i = 0; i < masks.size(); i++) d_mask[i].resize(masks[i].size());

    thrust::device_vector<double> d_zm(full_input_size);
    double *d_zm_ptr = thrust::raw_pointer_cast(d_zm.data());
    thrust::device_vector<cufftDoubleComplex> d_zm_fft(full_input_size);
    cufftDoubleComplex *d_zm_fft_ptr = thrust::raw_pointer_cast(d_zm_fft.data());
    thrust::device_vector<double> d_rem(full_input_size);
    thrust::device_vector<double> d_imf(full_input_size);
    double *d_rem_ptr = thrust::raw_pointer_cast(d_rem.data());
    double *d_imf_ptr = thrust::raw_pointer_cast(d_imf.data());
    thrust::device_vector<cufftDoubleComplex> d_rem_fft(full_input_size);
    cufftDoubleComplex *d_rem_fft_ptr = thrust::raw_pointer_cast(d_rem_fft.data());
    cu_errchk(cudaMemcpy(d_rem_ptr, h_rx.data(), sizeof(double) * full_input_size, cudaMemcpyHostToDevice));
    d_imf = d_rem;
    int n_batch = 1;
    LOG4_DEBUG("Deconstructing " << full_input_size << " values.");

    cufftHandle plan_full_forward, plan_full_backward;
    cufft_errchk(cufftPlan1d(&plan_full_forward, full_input_size, CUFFT_D2Z, n_batch));
    cufft_errchk(cufftPlan1d(&plan_full_backward, full_input_size, CUFFT_Z2D, n_batch));
    for (unsigned i = 0; i < masks.size(); i++) {
        LOG4_DEBUG("Doing level " << i);

        size_t mask_size = masks[i].size();
        thrust::host_vector<double> h_mask(mask_size);
        std::memcpy(h_mask.data(), masks[i].data(), sizeof(double) * mask_size);
        d_mask[i] = h_mask;

        double *d_mask_ptr = thrust::raw_pointer_cast(d_mask[i].data());
        online_emd::expand_the_mask(mask_size, full_input_size, d_mask_ptr, d_zm_ptr);
        cufft_errchk(cufftExecD2Z(plan_full_forward, d_zm_ptr, d_zm_fft_ptr));
        cufft_errchk(cufftExecD2Z(plan_full_forward, d_rem_ptr, d_rem_fft_ptr));
        vec_power<<<CUDA_THREADS_BLOCKS(full_input_size / 2 + 1)>>>(d_zm_fft_ptr, full_input_size, siftings[i]);
        gpu_multiply_complex<<<CUDA_THREADS_BLOCKS(full_input_size / 2 + 1) >>>(full_input_size, d_zm_fft_ptr, d_rem_fft_ptr);
        cufft_errchk(cufftExecZ2D(plan_full_backward, d_rem_fft_ptr, d_rem_ptr));

        std::vector<double> h_tmp(full_input_size);
        vec_subtract_inplace<<<CUDA_THREADS_BLOCKS(full_input_size)>>>(d_imf_ptr, d_rem_ptr, full_input_size);
        cu_errchk(cudaMemcpy(h_tmp.data(), d_imf_ptr, sizeof(double) * full_input_size, cudaMemcpyDeviceToHost));
        for (size_t t = 0; t < full_input_size; ++t)
            if (t >= tail.size()) inout[t - tail.size()]->set_value(2 * i, h_tmp[t]);
        d_imf = d_rem;
    }

    cu_errchk(cudaMemcpy(h_rx.data(), d_rem_ptr, sizeof(double) * full_input_size, cudaMemcpyDeviceToHost));
#pragma omp parallel for schedule(static, 1 + full_input_size / std::thread::hardware_concurrency()) num_threads(adj_threads(full_input_size))
    for (size_t t = 0; t < full_input_size; ++t)
        if (t >= tail.size()) inout[t - tail.size()]->set_value(masks.size() * 2, h_rx[t]);
    cufftDestroy(plan_full_forward);
    cufftDestroy(plan_full_backward);
}

#else

void transform_fir(
        datamodel::datarow_range &inout,
        const std::vector<double> &tail,
        const std::deque<size_t> &siftings,
        const std::deque<std::vector<double>> &mask,
        const double stretch_coef)
{
    const size_t levels = inout.levels() / 4;
    const size_t in_colix = levels * 2;
    const auto full_input_size = inout.distance() + tail.size();
    const auto max_gpus = common::gpu_handler_hid::get().get_gpu_devices_count();
    std::deque<size_t> gpuids(max_gpus);
    for (size_t d = 0; d < max_gpus; ++d) gpuids[d] = d;
    std::vector<double> h_rx(full_input_size);
#pragma omp parallel for schedule(static, 1 + full_input_size / std::thread::hardware_concurrency()) num_threads(adj_threads(full_input_size))
    for (size_t t = 0; t < full_input_size; ++t)
        h_rx[t] = t < tail.size() ? tail[t] : inout[t - tail.size()]->get_value(in_colix);
    std::deque<double *> d_remainder_ptr(max_gpus), d_rx_ptr(max_gpus), d_rx2_ptr(max_gpus);
    std::deque<size_t> start_ix(max_gpus), job_len(max_gpus);
#pragma omp parallel for num_threads(adj_threads(max_gpus)) schedule(static, 1)
    for (size_t d = 0; d < max_gpus; ++d) {
        cu_errchk(cudaSetDevice(gpuids[d]));
        start_ix[d] = d * full_input_size / max_gpus;
        job_len[d] = d == max_gpus - 1 ? full_input_size - start_ix[d] : full_input_size / max_gpus;
        d_remainder_ptr[d] = cuda_malloccopy(h_rx);
        d_rx_ptr[d] = cuda_malloccopy(h_rx);
        cu_errchk(cudaMalloc(&d_rx2_ptr[d], full_input_size * sizeof(double)));
    }
    for (size_t l = 0; l < levels - 1; l++) {
        const size_t actual_l = levels - l - 1;
        const size_t stretched_mask_size = mask[l].size() * stretch_coef;

        std::deque<double *> d_mask_ptr(max_gpus);
#pragma omp parallel for num_threads(adj_threads(max_gpus)) schedule(static, 1)
        for (size_t d = 0; d < max_gpus; ++d) {
            cu_errchk(cudaSetDevice(gpuids[d]));
            d_mask_ptr[d] = cuda_malloccopy(mask[l]);
        }

        for (size_t s = 0; s < siftings[l]; ++s) {
#pragma omp parallel for num_threads(adj_threads(max_gpus)) schedule(static, 1)
            for (size_t d = 0; d < max_gpus; ++d) {
                cu_errchk(cudaSetDevice(gpuids[d]));
                apply_mask<<<CUDA_THREADS_BLOCKS(job_len[d])>>>(stretch_coef, d_rx_ptr[d], full_input_size, d_mask_ptr[d], mask[l].size(), stretched_mask_size, d_rx2_ptr[d], start_ix[d]);
                oemd::G_vec_subtract_inplace<<<CUDA_THREADS_BLOCKS(job_len[d])>>>(d_rx_ptr[d] + start_ix[d], d_rx2_ptr[d] + start_ix[d], job_len[d]);
            }
        }

        std::vector<double> h_tmp(full_input_size);
#pragma omp parallel for num_threads(adj_threads(max_gpus)) schedule(static, 1)
        for (size_t d = 0; d < max_gpus; ++d) {
            cu_errchk(cudaSetDevice(gpuids[d]));
            cu_errchk(cudaMemcpy(h_tmp.data() + start_ix[d], d_rx_ptr[d] + start_ix[d], job_len[d] * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost));
            cu_errchk(cudaDeviceSynchronize());
#pragma omp parallel for schedule(static, 1 + job_len[d] / std::thread::hardware_concurrency()) num_threads(adj_threads(job_len[d]))
            for (size_t i = start_ix[d]; i < start_ix[d] + job_len[d]; ++i)
                if (i >= tail.size()) inout[i - tail.size()]->set_value(2 * actual_l, h_tmp[i]);

            oemd::G_vec_subtract_inplace<<<CUDA_THREADS_BLOCKS(job_len[d])>>>(d_remainder_ptr[d] + start_ix[d], d_rx_ptr[d] + start_ix[d], job_len[d]);
            cu_errchk(cudaMemcpy(d_rx_ptr[d] + start_ix[d], d_remainder_ptr[d] + start_ix[d], job_len[d] * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToDevice));
            cu_errchk(cudaFree(d_mask_ptr[d]));
        }
#pragma omp parallel for num_threads(adj_threads(max_gpus)) schedule(static, 1)
        for (size_t d = 0; d < max_gpus; ++d) {
            cu_errchk(cudaSetDevice(gpuids[d]));
            cu_errchk(cudaMemcpy(h_tmp.data() + start_ix[d], d_rx_ptr[d] + start_ix[d], job_len[d] * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost));
        }
#pragma omp parallel for num_threads(adj_threads(max_gpus)) schedule(static, 1)
        for (size_t d = 0; d < max_gpus; ++d) {
            cu_errchk(cudaSetDevice(gpuids[d]));
            cu_errchk(cudaMemcpy(d_rx_ptr[d], h_tmp.data(), full_input_size * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice));
        }
    }
#pragma omp parallel for num_threads(adj_threads(max_gpus)) schedule(static, 1)
    for (size_t d = 0; d < max_gpus; ++d) {
        cu_errchk(cudaSetDevice(gpuids[d]));
        cu_errchk(cudaMemcpy(h_rx.data() + start_ix[d], d_rx_ptr[d] + start_ix[d], job_len[d] * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost));
    }
    cu_errchk(cudaDeviceSynchronize());
#pragma omp parallel for schedule(static, 1 + full_input_size / std::thread::hardware_concurrency()) num_threads(adj_threads(full_input_size))
    for (size_t t = 0; t < full_input_size; ++t)
        if (t >= tail.size()) inout[t - tail.size()]->set_value(0, h_rx[t]);
#pragma omp parallel for num_threads(adj_threads(max_gpus)) schedule(static, 1)
    for (size_t d = 0; d < max_gpus; ++d) {
        cu_errchk(cudaSetDevice(gpuids[d]));
        cu_errchk(cudaFree(d_rx2_ptr[d]));
        cu_errchk(cudaFree(d_rx_ptr[d]));
        cu_errchk(cudaFree(d_remainder_ptr[d]));
    }
}

#endif

void online_emd::expand_the_mask(const size_t mask_size, const size_t input_size, const double *dev_mask, double *dev_expanded_mask)
{
    cu_errchk(cudaMemset(dev_expanded_mask + mask_size, 0, sizeof(double) * std::max<size_t>(0, input_size - mask_size)));
    cu_errchk(cudaMemcpy(dev_expanded_mask, dev_mask, sizeof(double) * mask_size, cudaMemcpyDeviceToDevice));
}

void online_emd::transform(
        datamodel::datarow_range &inout,
        const std::vector<double> &tail,
        const std::deque<size_t> &siftings,
        const std::deque<std::vector<double>> &mask,
        const double stretch_coef)
{
#ifdef OEMDFFT
    transform_fft(inout, tail, siftings, mask, stretch_coef); // TODO Test latest implementation
#else
    transform_fir(inout, tail, siftings, mask, stretch_coef);
#endif
}

} // namespace oemd
} // namespace svr