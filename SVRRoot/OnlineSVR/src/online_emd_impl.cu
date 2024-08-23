//
// Created by zarko on 2/16/22.
//

#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "common/cuda_util.cuh"
#include "model/DeconQueue.hpp"
#include "common/gpu_handler.tpp"
#include "online_emd.hpp"
#include "common/barrier.hpp"

#ifdef OEMDFFT

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cufft.h>

#define CUFFT_INPUT_LIMIT // 64e5 // TODO Implement chunking and multi GPU support

#endif

namespace svr {
namespace oemd {


__global__ void G_subtract_inplace(double *__restrict__ const x, CRPTR(double) y, const unsigned n)
{
    CU_STRIDED_FOR_i(n) x[i] -= y[i];
}

__global__ void
G_apply_mask(
        const double stretch_coef,
        CRPTR(double) rx,
        const unsigned x_size,
        CRPTR(double) mask,
        const unsigned mask_size,
        const unsigned stretched_mask_size,
        double *__restrict__ const rx2,
        const unsigned start_x)
{
    const auto ix = start_x + blockIdx.x * blockDim.x + tid;
    if (ix >= x_size) return;

    rx2[ix] = 0;
    if (ix >= stretched_mask_size - 1) {
UNROLL()
        for (unsigned m = 0; m < stretched_mask_size; ++m)
            rx2[ix] += mask[unsigned(m / stretch_coef)] * rx[unsigned(ix - stretched_mask_size + m + 1)] / stretch_coef;
    } else {
        double sum = 0;
UNROLL()
        for (unsigned j = 0; j <= ix; ++j) {
            const auto mask_ix = unsigned(mask_size - 1 - ix / stretch_coef + j / stretch_coef);
            rx2[ix] += mask[mask_ix] * rx[j] / stretch_coef;
            sum += mask[mask_ix] / stretch_coef;
        }
        rx2[ix] /= sum;
    }
}

#ifdef OEMDFFT

void transform_fft(
        datamodel::datarow_range &inout,
        const std::vector<double> &tail,
        const std::deque<unsigned> &siftings,
        const std::deque<std::vector<double>> &masks,
        const double stretch_coef)
{
    const unsigned levels = inout.levels() / 4;
    const unsigned in_colix = levels * 2;
    const auto full_input_size = inout.distance() + tail.size();
    cu_errchk(cudaSetDevice(0));
    const auto max_gpus = common::gpu_handler_hid::get().get_gpu_devices_count();
    std::deque<unsigned> gpuids(max_gpus);
    for (unsigned d = 0; d < max_gpus; ++d) gpuids[d] = d;
    std::vector<double> h_rx(full_input_size);
#pragma omp parallel for schedule(static, 1 + full_input_size / C_n_cpu) num_threads(adj_threads(full_input_size))
    for (unsigned t = 0; t < full_input_size; ++t)
        h_rx[t] = t < tail.size() ? tail[t] : inout[t - tail.size()]->get_value(in_colix);

    std::deque<unsigned> start_ix(max_gpus), job_len(max_gpus);
#pragma omp parallel for num_threads(adj_threads(max_gpus)) schedule(static, 1)
    for (unsigned d = 0; d < max_gpus; ++d) {
        cu_errchk(cudaSetDevice(gpuids[d]));
        start_ix[d] = d * full_input_size / max_gpus;
        job_len[d] = d == max_gpus - 1 ? full_input_size - start_ix[d] : full_input_size / max_gpus;
    }

    std::deque<thrust::device_vector<double>> d_mask(masks.size());
    for (unsigned i = 0; i < masks.size(); i++) d_mask[i].resize(masks[i].size());

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

        unsigned mask_size = masks[i].size();
        thrust::host_vector<double> h_mask(mask_size);
        std::memcpy(h_mask.data(), masks[i].data(), sizeof(double) * mask_size);
        d_mask[i] = h_mask;

        double *d_mask_ptr = thrust::raw_pointer_cast(d_mask[i].data());
        online_emd::expand_the_mask(mask_size, full_input_size, d_mask_ptr, d_zm_ptr);
        cufft_errchk(cufftExecD2Z(plan_full_forward, d_zm_ptr, d_zm_fft_ptr));
        cufft_errchk(cufftExecD2Z(plan_full_forward, d_rem_ptr, d_rem_fft_ptr));
        vec_power<<<CUDA_THREADS_BLOCKS(full_input_size / 2 + 1)>>>(d_zm_fft_ptr, full_input_size, siftings[i]);
        G_multiply_complex<<<CUDA_THREADS_BLOCKS(full_input_size / 2 + 1) >>>(full_input_size, d_zm_fft_ptr, d_rem_fft_ptr);
        cufft_errchk(cufftExecZ2D(plan_full_backward, d_rem_fft_ptr, d_rem_ptr));

        std::vector<double> h_tmp(full_input_size);
        vec_subtract_inplace<<<CUDA_THREADS_BLOCKS(full_input_size)>>>(d_imf_ptr, d_rem_ptr, full_input_size);
        cu_errchk(cudaMemcpy(h_tmp.data(), d_imf_ptr, sizeof(double) * full_input_size, cudaMemcpyDeviceToHost));
        for (unsigned t = 0; t < full_input_size; ++t)
            if (t >= tail.size()) inout[t - tail.size()]->set_value(2 * i, h_tmp[t]);
        d_imf = d_rem;
    }

    cu_errchk(cudaMemcpy(h_rx.data(), d_rem_ptr, sizeof(double) * full_input_size, cudaMemcpyDeviceToHost));
#pragma omp parallel for schedule(static, 1 + full_input_size / C_n_cpu) num_threads(adj_threads(full_input_size))
    for (unsigned t = 0; t < full_input_size; ++t)
        if (t >= tail.size()) inout[t - tail.size()]->set_value(masks.size() * 2, h_rx[t]);
    cufftDestroy(plan_full_forward);
    cufftDestroy(plan_full_backward);
}

#else

void transform_fir(
        datamodel::datarow_range &inout,
        const std::vector<double> &tail,
        const std::deque<unsigned> &siftings,
        const std::deque<std::vector<double>> &mask,
        const double stretch_coef)
{
    const unsigned levels = inout.levels() / 4;
    const unsigned in_colix = levels * 2;
    const auto full_input_size = inout.distance() + tail.size();
    const auto max_gpus = common::gpu_handler_hid::get().get_gpu_devices_count();
    std::deque<unsigned> gpuids(max_gpus);
    std::deque<cudaStream_t> custreams(max_gpus);

#define FOR_MAX_GPU_d \
    OMP_FOR_(max_gpus,) \
    for (unsigned d = 0; d < max_gpus; ++d)

    FOR_MAX_GPU_d {
        gpuids[d] = d;
        cu_errchk(cudaSetDevice(d));
        cudaStream_t s;
        cu_errchk(cudaStreamCreate(&s));
        custreams[d] = s;
    }
    std::vector<double> h_rx(full_input_size);
    OMP_FOR(full_input_size)
    for (unsigned t = 0; t < full_input_size; ++t)
        h_rx[t] = t < tail.size() ? tail[t] : inout[t - tail.size()]->at(in_colix);
    std::deque<double *> d_remainder_ptr(max_gpus), d_rx_ptr(max_gpus), d_rx2_ptr(max_gpus);
    std::deque<unsigned> start_ix(max_gpus), job_len(max_gpus);

    FOR_MAX_GPU_d {
        cu_errchk(cudaSetDevice(gpuids[d]));
        start_ix[d] = d * full_input_size / max_gpus;
        job_len[d] = d == max_gpus - 1 ? full_input_size - start_ix[d] : full_input_size / max_gpus;
        d_remainder_ptr[d] = cumallocopy(h_rx, custreams[d]);
        d_rx_ptr[d] = cumallocopy(h_rx, custreams[d]);
        cu_errchk(cudaMallocAsync((void **) &d_rx2_ptr[d], full_input_size * sizeof(double), custreams[d]));
    }

UNROLL()
    for (unsigned l = 0; l < levels - 1; ++l) {
        const unsigned actual_l = levels - l - 1;
        const unsigned stretched_mask_size = mask[l].size() * stretch_coef;

        std::deque<double *> d_mask_ptr(max_gpus);
        FOR_MAX_GPU_d {
            cu_errchk(cudaSetDevice(gpuids[d]));
            d_mask_ptr[d] = cumallocopy(mask[l], custreams[d]);
        }

UNROLL()
        for (unsigned s = 0; s < siftings[l]; ++s) {
            FOR_MAX_GPU_d {
                cu_errchk(cudaSetDevice(gpuids[d]));
                G_apply_mask<<<CU_BLOCKS_THREADS(job_len[d]), 0, custreams[d]>>>(stretch_coef, d_rx_ptr[d], full_input_size, d_mask_ptr[d], mask[l].size(),
                                                                                 stretched_mask_size, d_rx2_ptr[d], start_ix[d]);
                G_subtract_inplace<<<CU_BLOCKS_THREADS(job_len[d]), 0, custreams[d]>>>(d_rx_ptr[d] + start_ix[d], d_rx2_ptr[d] + start_ix[d], job_len[d]);
            }
        }

        common::barrier bar(max_gpus);
        std::vector<double> h_tmp(full_input_size);
        FOR_MAX_GPU_d {
            cu_errchk(cudaSetDevice(gpuids[d]));
            cu_errchk(cudaMemcpyAsync(h_tmp.data() + start_ix[d], d_rx_ptr[d] + start_ix[d], job_len[d] * sizeof(double), cudaMemcpyDeviceToHost, custreams[d]));
            cu_errchk(cudaStreamSynchronize(custreams[d]));
            const auto start_ix_d = std::max<unsigned>(tail.size(), start_ix[d]);
            const auto end_ix_d = start_ix[d] + job_len[d];
            for (unsigned i = start_ix_d; i < end_ix_d; ++i) inout[i - tail.size()]->set_value(2 * actual_l, h_tmp[i]);
            G_subtract_inplace<<<CU_BLOCKS_THREADS(job_len[d]), 0, custreams[d]>>>(d_remainder_ptr[d] + start_ix[d], d_rx_ptr[d] + start_ix[d], job_len[d]);
            cu_errchk(cudaMemcpyAsync(d_rx_ptr[d] + start_ix[d], d_remainder_ptr[d] + start_ix[d], job_len[d] * sizeof(double), cudaMemcpyDeviceToDevice, custreams[d]));
            cu_errchk(cudaFreeAsync(d_mask_ptr[d], custreams[d]));
            cu_errchk(cudaMemcpyAsync(h_tmp.data() + start_ix[d], d_rx_ptr[d] + start_ix[d], job_len[d] * sizeof(double), cudaMemcpyDeviceToHost, custreams[d]));
            cu_errchk(cudaStreamSynchronize(custreams[d]));
            bar.wait();
            cu_errchk(cudaMemcpyAsync(d_rx_ptr[d], h_tmp.data(), full_input_size * sizeof(double), cudaMemcpyHostToDevice, custreams[d]));
        }
    }
    FOR_MAX_GPU_d {
        cu_errchk(cudaSetDevice(gpuids[d]));
        cu_errchk(cudaMemcpyAsync(h_rx.data() + start_ix[d], d_rx_ptr[d] + start_ix[d], job_len[d] * sizeof(double), cudaMemcpyDeviceToHost, custreams[d]));
        cu_errchk(cudaFreeAsync(d_rx2_ptr[d], custreams[d]));
        cu_errchk(cudaFreeAsync(d_rx_ptr[d], custreams[d]));
        cu_errchk(cudaFreeAsync(d_remainder_ptr[d], custreams[d]));
        cu_errchk(cudaStreamSynchronize(custreams[d]));
    }
    OMP_FOR(full_input_size - tail.size())
    for (unsigned t = tail.size(); t < full_input_size; ++t) inout[t - tail.size()]->set_value(0, h_rx[t]);
}

#endif

void online_emd::expand_the_mask(const unsigned mask_size, const unsigned input_size, const double *dev_mask, double *dev_expanded_mask, const cudaStream_t custream)
{
    if (input_size > mask_size) cu_errchk(cudaMemsetAsync(dev_expanded_mask + mask_size, 0, sizeof(double) * (input_size - mask_size), custream));
    cu_errchk(cudaMemcpyAsync(dev_expanded_mask, dev_mask, sizeof(double) * mask_size, cudaMemcpyDeviceToDevice, custream));
}

void online_emd::transform(
        datamodel::datarow_range &inout,
        const std::vector<double> &tail,
        const std::deque<unsigned> &siftings,
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