//
// Created by zarko on 2/16/22.
//

#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "model/DeconQueue.hpp"
#include "common/gpu_handler.hpp"
#include "online_emd.hpp"
#include "common/barrier.hpp"
#include "SVRParametersService.hpp"
#include "IQScalingFactorService.hpp"
#include "common/cuda_util.cuh"

#ifdef OEMDFFT
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cufft.h>
#endif

namespace svr {
namespace oemd {

__global__ void G_subtract_I(RPTR(double) x, const double y, const uint32_t n)
{
    CU_STRIDED_FOR_i(n) x[i] -= y;
}

__global__ void G_subtract_I(RPTR(double) x, CRPTRd y, const uint32_t n)
{
    CU_STRIDED_FOR_i(n) x[i] -= y[i];
}

__global__ void G_subtract_inplace2(CRPTRd x, RPTR(double) y, const uint32_t n)
{
    CU_STRIDED_FOR_i(n) y[i] = x[i] - y[i];
}

__global__ void
G_apply_fir(
        const double stretch_coef,
        CRPTRd in,
        const uint32_t len,
        CRPTRd mask,
        const uint32_t mask_len,
        const uint32_t stretched_mask_len,
        RPTR(double) out,
        const uint32_t in_start)
{
    const auto i = blockIdx.x * blockDim.x + tid;
    const auto in_i = in_start + i;
    if (in_i >= len) return;

    out[i] = 0;
    if (in_i >= stretched_mask_len - 1) {
        const auto in_i_mask_start = in_i + 1 - stretched_mask_len;
        UNROLL()
        for (DTYPE(stretched_mask_len) j = 0; j < stretched_mask_len; ++j)
            out[i] += mask[uint32_t(j / stretch_coef)] * in[in_i_mask_start + j] / stretch_coef;
    } else {
        const double mask_start_stretched = mask_len - 1 - in_i / stretch_coef;
        double sum = 0;
        UNROLL()
        for (DTYPE(in_i) j = 0; j <= in_i; ++j) {
            const auto stretched_mask_coef = mask[uint32_t(mask_start_stretched + j / stretch_coef)] / stretch_coef;
            out[i] += in[j] * stretched_mask_coef;
            sum += stretched_mask_coef;
        }
        out[i] /= sum;
    }
}

#ifdef OEMDFFT

void transform_fft(
        datamodel::datarow_range &inout,
        const std::vector<double> &tail,
        const std::deque<uint16_t> &siftings,
        const std::deque<std::vector<double>> &masks,
        const double stretch_coef)
{
    const uint16_t levels = inout.levels() / 4;
    const uint16_t in_colix = levels * 2;
    const auto full_input_size = inout.distance() + tail.size();
    cu_errchk(cudaSetDevice(0));
    const auto max_gpus = common::gpu_handler_1::get().get_gpu_devices_count();
    std::deque<uint16_t> gpuids(max_gpus);
    for (uint16_t d = 0; d < max_gpus; ++d) gpuids[d] = d;
    std::vector<double> h_rx(full_input_size);
#pragma omp parallel for schedule(static, 1 + full_input_size / C_n_cpu) num_threads(adj_threads(full_input_size))
    for (uint32_t t = 0; t < full_input_size; ++t)
        h_rx[t] = t < tail.size() ? tail[t] : inout[t - tail.size()]->get_value(in_colix);

    std::deque<uint32_t> start_ix(max_gpus), job_len(max_gpus);
#pragma omp parallel for num_threads(adj_threads(max_gpus)) schedule(static, 1)
    for (uint16_t d = 0; d < max_gpus; ++d) {
        cu_errchk(cudaSetDevice(gpuids[d]));
        start_ix[d] = d * full_input_size / max_gpus;
        job_len[d] = d == max_gpus - 1 ? full_input_size - start_ix[d] : full_input_size / max_gpus;
    }

    std::deque<thrust::device_vector<double>> d_mask(masks.size());
    for (uint16_t i = 0; i < masks.size(); i++) d_mask[i].resize(masks[i].size());

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
    for (uint16_t i = 0; i < masks.size(); i++) {
        LOG4_DEBUG("Doing level " << i);

        uint32_t mask_size = masks[i].size();
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
        vec_subtract_I<<<CUDA_THREADS_BLOCKS(full_input_size)>>>(d_imf_ptr, d_rem_ptr, full_input_size);
        cu_errchk(cudaMemcpy(h_tmp.data(), d_imf_ptr, sizeof(double) * full_input_size, cudaMemcpyDeviceToHost));
        for (uint32_t t = 0; t < full_input_size; ++t)
            if (t >= tail.size()) inout[t - tail.size()]->set_value(2 * i, h_tmp[t]);
        d_imf = d_rem;
    }

    cu_errchk(cudaMemcpy(h_rx.data(), d_rem_ptr, sizeof(double) * full_input_size, cudaMemcpyDeviceToHost));
#pragma omp parallel for schedule(static, 1 + full_input_size / C_n_cpu) num_threads(adj_threads(full_input_size))
    for (uint32_t t = 0; t < full_input_size; ++t)
        if (t >= tail.size()) inout[t - tail.size()]->set_value(masks.size() * 2, h_rx[t]);
    cufftDestroy(plan_full_forward);
    cufftDestroy(plan_full_backward);
}

#else

void transform_fir(
        const datamodel::datarow_crange &in,
        datamodel::datarow_range &out,
        const std::vector<double> &tail,
        const std::deque<uint16_t> &siftings,
        const std::deque<std::vector<double>> &mask,
        const double stretch_coef,
        const uint16_t oemd_levels,
        const uint16_t in_col,
        const datamodel::t_iqscaler &scaler)
{

#define DEV_FOR_d_begin {                                                           \
    PRAGMASTR(omp parallel ADJ_THREADS(max_gpus))                                   \
    PRAGMASTR(omp single)                                                           \
    {                                                                               \
        PRAGMASTR(omp taskloop mergeable default(shared) grainsize(1))              \
        for (uint16_t d = 0; d < max_gpus; ++d) try { cu_errchk(cudaSetDevice(d));


#define DEV_FOR_d_end } catch (const std::exception &e) {                           \
        LOG4_ERROR("Error in FOR_MAX_GPU_" << d << ", " << e.what()); } } }

    const auto full_input_size = in.distance() + tail.size();
    const auto max_gpus = 1; // common::gpu_handler_1::get().get_gpu_devices_count(); // TODO Buggy when using multiple GPUs, fix

    std::deque<cudaStream_t> custreams(max_gpus);
    DEV_FOR_d_begin
                    cu_errchk(cudaStreamCreateWithFlags(&custreams[d], C_cu_default_stream_flags));
    DEV_FOR_d_end

    std::vector<double> h_input(full_input_size);
    OMP_FOR_i(full_input_size)h_input[i] = scaler(i < tail.size() ? tail[i] : in[i - tail.size()]->at(in_col));

    std::deque<double *> d_imf(max_gpus), d_work(max_gpus);
    std::deque<uint32_t> start_ix(max_gpus), job_len(max_gpus);
    const auto chunk_len = full_input_size / max_gpus;
    DEV_FOR_d_begin
                    start_ix[d] = d * chunk_len;
                    job_len[d] = d == max_gpus - 1 ? full_input_size - start_ix[d] : chunk_len;
                    d_imf[d] = cumallocopy(h_input, custreams[d]);
                    cu_errchk(cudaMallocAsync((void **) &d_work[d], job_len[d] * sizeof(double), custreams[d]));
    DEV_FOR_d_end

    UNROLL()
    for (DTYPE(oemd_levels) l = 0; l < oemd_levels - 1; ++l) {
        const auto actual_l = oemd_levels - l - 1;
        const uint32_t stretched_mask_size = mask[l].size() * stretch_coef;
        std::deque<double *> d_mask_ptr(max_gpus);
        DEV_FOR_d_begin
                        d_mask_ptr[d] = cumallocopy(mask[l], custreams[d]);
        DEV_FOR_d_end

        std::vector<double> h_imf(full_input_size);
        UNROLL(oemd_coefficients::C_default_siftings)
        for (uint32_t s = 0; s < siftings[l]; ++s) {
            DEV_FOR_d_begin
                            G_apply_fir<<<CU_BLOCKS_THREADS(job_len[d]), 0, custreams[d]>>>(
                                    stretch_coef, d_imf[d], full_input_size, d_mask_ptr[d], mask[l].size(), stretched_mask_size, d_work[d], start_ix[d]);
                            G_subtract_I<<<CU_BLOCKS_THREADS(job_len[d]), 0, custreams[d]>>>(d_imf[d] + start_ix[d], d_work[d], job_len[d]);
                            cu_errchk(cudaMemcpyAsync(h_imf.data() + start_ix[d], d_imf[d] + start_ix[d], job_len[d] * sizeof(double), cudaMemcpyDeviceToHost, custreams[d]));
                            cu_errchk(cudaStreamSynchronize(custreams[d]));
            DEV_FOR_d_end

            DEV_FOR_d_begin
                            cu_errchk(cudaMemcpyAsync(d_imf[d], h_imf.data(), full_input_size * sizeof(double), cudaMemcpyHostToDevice, custreams[d]));
                            cu_errchk(cudaStreamSynchronize(custreams[d]));
            DEV_FOR_d_end
        }
#ifdef EMD_ONLY
        const auto actual_l_2 = actual_l;
#else
        const auto actual_l_2 = 2 * actual_l;
#endif
        if (false /* !l */) {
            OMP_FOR_i(out.distance()) out[i]->at(actual_l_2) = h_imf[i + tail.size()] / oemd_levels;
        } else {
            OMP_FOR_i(out.distance()) out[i]->at(actual_l_2) = h_imf[i + tail.size()];
        }
        RELEASE_CONT(h_imf);

        DEV_FOR_d_begin
                        cu_errchk(cudaFreeAsync(d_mask_ptr[d], custreams[d]));
                        cu_errchk(cudaMemcpyAsync(d_work[d], h_input.data() + start_ix[d], job_len[d] * sizeof(double), cudaMemcpyHostToDevice, custreams[d]));
                        G_subtract_inplace2<<<CU_BLOCKS_THREADS(job_len[d]), 0, custreams[d]>>>(d_work[d], d_imf[d] + start_ix[d], job_len[d]);
                        cu_errchk(cudaMemcpyAsync(h_input.data() + start_ix[d], d_imf[d] + start_ix[d], job_len[d] * sizeof(double), cudaMemcpyDeviceToHost, custreams[d]));
        DEV_FOR_d_end
    }

    DEV_FOR_d_begin
                    cu_errchk(cudaFreeAsync(d_work[d], custreams[d]));
                    cu_errchk(cudaFreeAsync(d_imf[d], custreams[d]));
                    cu_errchk(cudaStreamSynchronize(custreams[d]));
                    cu_errchk(cudaStreamDestroy(custreams[d]));
    DEV_FOR_d_end

    OMP_FOR_i(out.distance()) **out[i] = h_input[i + tail.size()];
}

#endif

void online_emd::expand_the_mask(const uint32_t mask_size, const uint32_t input_size, CPTRd dev_mask, double *const dev_expanded_mask, const cudaStream_t custream)
{
    if (input_size > mask_size) cu_errchk(cudaMemsetAsync(dev_expanded_mask + mask_size, 0, sizeof(double) * (input_size - mask_size), custream));
    cu_errchk(cudaMemcpyAsync(dev_expanded_mask, dev_mask, sizeof(double) * mask_size, cudaMemcpyDeviceToDevice, custream));
}

void online_emd::transform(
        const datamodel::datarow_crange &in,
        datamodel::datarow_range &out,
        const std::vector<double> &tail,
        const std::deque<uint16_t> &siftings,
        const std::deque<std::vector<double>> &mask,
        const double stretch_coef,
        const uint16_t oemd_levels,
        const uint16_t in_colix,
        const datamodel::t_iqscaler &scaler)
{
#ifdef OEMDFFT
    transform_fft(inout, tail, siftings, mask, stretch_coef); // TODO Test latest implementation
#else
    transform_fir(in, out, tail, siftings, mask, stretch_coef, oemd_levels, in_colix, scaler);
#endif
}

} // namespace oemd
} // namespace svr
