//
// Created by zarko on 10/3/22.
//
#include <cstdlib>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>
#include <queue>
#include <thrust/device_vector.h>
#include <thrust/logical.h>
#include <iomanip>
#include <thrust/execution_policy.h>
#include "pprune.hpp"
#include "common/compatibility.hpp"
#include "oemd_coefficients_search.hpp"
#include "online_emd.hpp"
#include "../../SVRCommon/include/common/cuda_util.cuh"
#include "util/time_utils.hpp"
#include "firefly.hpp"
#include "common/logging.hpp"
#include "cuqrsolve.cuh"
#include "onlinesvr.hpp"

// #define USE_FIREFLY // else use BITEOPT

namespace svr {
namespace oemd {

__global__ void
G_multiply_complex(
        const double input_len_div,
        const unsigned fft_len,
        CRPTR(cufftDoubleComplex) multiplier,
        cufftDoubleComplex *__restrict__ output)
{
    cufftDoubleComplex new_output;
    CU_STRIDED_FOR_i(fft_len) {
        new_output.x = output[i].x * multiplier[i].x - output[i].y * multiplier[i].y;
        new_output.y = output[i].x * multiplier[i].y + output[i].y * multiplier[i].x;
        output[i].x = new_output.x / input_len_div; // because of inverse FFT
        output[i].y = new_output.y / input_len_div;
    }
}

__global__ void G_vec_power_inplace(
        cufftDoubleComplex *__restrict__ x,
        const unsigned x_size_2_1,
        const unsigned siftings)
{
    const auto ix = blockIdx.x * blockDim.x + tid;
    const auto stride = blockDim.x * gridDim.x;
    double px_out, py_out, px, py;
    UNROLL()
    for (auto j = ix; j < x_size_2_1; j += stride) {
        px = 1. - x[j].x;
        py = -x[j].y;
        UNROLL()
        for (unsigned i = 1; i < siftings; i++) {
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
        CRPTR(cufftDoubleComplex) x,
        cufftDoubleComplex *__restrict__ y,
        const unsigned n,
        const unsigned siftings)
{
    double px, py;
    CU_STRIDED_FOR_i(n) {
        px = 1. - x[i].x;
        py = -x[i].y;
        for (unsigned j = 1; j < siftings; ++j) {
            px = px * (1. - x[j].x) - py * (-x[j].y);
            py = px * (-x[j].y) + py * (1. - x[j].x);
        }
        y[i].x = px;
        y[i].y = py;
    }
}

__global__ void G_gpu_multiply_smooth(
        const unsigned input_size,
        const double coeff,
        cufftDoubleComplex *__restrict__ output)
{
    CU_STRIDED_FOR_i(input_size / 2 + 1) {
        const double mult = exp(-coeff * double(i) / double(input_size));
        output[i].x *= mult;
        output[i].y *= mult;
    }
}


__global__ void G_vec_sift(
        const unsigned fft_size,
        const unsigned siftings,
        const cufftDoubleComplex *__restrict__ x,
        cufftDoubleComplex *__restrict__ imf,
        cufftDoubleComplex *__restrict__ rem)
{
    double px, py;
    CU_STRIDED_FOR_i(fft_size) {
        px = 1. - x[i].x;
        py = -x[i].y;
        UNROLL()
        for (unsigned j = 1; j < siftings; ++j) {
            px = px * (1. - x[i].x) - py * (-x[i].y);
            py = px * (-x[i].y) + py * (1. - x[i].x);
        }
        imf[i].x = px;
        imf[i].y = py;
        rem[i].x = 1. - px;
        rem[i].y = -py;
    }
}

__global__ void G_sum_expanded(
        double *__restrict__ d_sum_imf,
        double *__restrict__ d_sum_rem,
        double *__restrict__ d_sum_corr,
        const double *__restrict__ d_imf_mask,
        const double *__restrict__ d_rem_mask,
        const unsigned expand_size,
        const double *__restrict__ d_global_sift_matrix)
{
    const double expand_size_2 = expand_size * expand_size;

    __shared__ double _sh_sum_imf[common::C_cu_block_size];
    __shared__ double _sh_sum_rem[common::C_cu_block_size];
    __shared__ double _sh_sum_corr[common::C_cu_block_size];
    _sh_sum_imf[tid] = 0;
    _sh_sum_rem[tid] = 0;
    _sh_sum_corr[tid] = 0;
    CU_STRIDED_FOR_i(expand_size) {
        double sum1 = 0, sum2 = 0;
        UNROLL()
        for (int j = 0; j < expand_size; ++j) {
            const auto abs_i_j = labs(i - j);
            sum1 += d_global_sift_matrix[abs_i_j] * d_imf_mask[j];
            sum2 += d_global_sift_matrix[abs_i_j] * d_rem_mask[j];
        }
        _sh_sum_imf[tid] += sum1 * d_imf_mask[i] / expand_size_2;
        _sh_sum_rem[tid] += sum2 * d_rem_mask[i] / expand_size_2;
        _sh_sum_corr[tid] += sum1 * d_rem_mask[i] / expand_size_2;
    }
    __syncthreads();

    UNROLL()
    for (auto size = common::C_cu_block_size / 2; size > 0; size /= 2) { // uniform
        if (tid >= size) continue;
        _sh_sum_imf[tid] += _sh_sum_imf[tid + size];
        _sh_sum_rem[tid] += _sh_sum_rem[tid + size];
        _sh_sum_corr[tid] += _sh_sum_corr[tid + size];
        __syncthreads();
    }
    if (tid) return;
    atomicAdd(d_sum_imf, *_sh_sum_imf);
    atomicAdd(d_sum_rem, *_sh_sum_rem);
    atomicAdd(d_sum_corr, *_sh_sum_corr);
}


void oemd_coefficients_search::transform(
        double *d_values, CPTR(double) d_mask, const unsigned input_len, const unsigned mask_len,
        const unsigned siftings, double *d_temp, const cudaStream_t custream) const
{
    auto d_imf = cumallocopyl(d_values, input_len, cudaMemcpyDeviceToDevice, custream);
    sift(siftings, input_len, mask_len, custream, d_mask, d_imf, d_temp);
    oemd::G_subtract_inplace<<<CU_BLOCKS_THREADS(input_len), 0, custream>>>(d_values, d_imf, input_len);
    cu_errchk(cudaFreeAsync(d_imf, custream));
}


std::tuple<double, double, double, double>
oemd_coefficients_search::sift_the_mask(
        const unsigned mask_size,
        const unsigned siftings,
        const double *d_mask,
        const cufftHandle plan_sift_forward,
        const cufftHandle plan_sift_backward,
        const double *d_expanded_mask,
        const cufftDoubleComplex *d_expanded_mask_fft,
        const double *d_global_sift_matrix_ptr,
        const unsigned gpu_id)
{
    cu_errchk(cudaSetDevice(gpu_id));
    cudaStream_t custream;
    cu_errchk(cudaStreamCreate(&custream));
    double sum_full, sum_imf, sum_rem, sum_corr;
    const unsigned expand_size = siftings * mask_size;
    thrust::device_vector<double> d_zm_mask(expand_size);
    thrust::device_vector<double> d_imf_mask(expand_size);
    thrust::device_vector<double> d_rem_mask(expand_size);

    double *d_expanded_mask_ptr = thrust::raw_pointer_cast(d_zm_mask.data());

    online_emd::expand_the_mask(mask_size, expand_size, d_mask, d_expanded_mask_ptr, custream);
    const auto fft_size = common::to_fft_len(expand_size);
    thrust::device_vector<cufftDoubleComplex> d_fzm_mask(fft_size);
    thrust::device_vector<cufftDoubleComplex> d_mask_imf_fft(fft_size);
    thrust::device_vector<cufftDoubleComplex> d_mask_rem_fft(fft_size);
    // cufftDoubleComplex *d_expanded_mask_fft = thrust::raw_pointer_cast(d_fzm_mask.data());
    // cf_errchk(cufftSetStream(plan_sift_forward, custream));
    // cufft_errchk(cufftExecD2Z(plan_sift_forward, d_expanded_mask_ptr, d_expanded_mask_fft));
    G_vec_sift<<<CU_BLOCKS_THREADS(fft_size), 0, custream>>>(fft_size, siftings, d_expanded_mask_fft, thrust::raw_pointer_cast(d_mask_imf_fft.data()),
                                                             thrust::raw_pointer_cast(d_mask_rem_fft.data()));
    cf_errchk(cufftSetStream(plan_sift_backward, custream));
    cf_errchk(cufftExecZ2D(plan_sift_backward, thrust::raw_pointer_cast(d_mask_imf_fft.data()), thrust::raw_pointer_cast(d_imf_mask.data())));
    cf_errchk(cufftExecZ2D(plan_sift_backward, thrust::raw_pointer_cast(d_mask_rem_fft.data()), thrust::raw_pointer_cast(d_rem_mask.data())));

    double *d_sum_imf, *d_sum_rem, *d_sum_corr;
    cu_errchk(cudaMallocAsync((void **) &d_sum_imf, sizeof(double), custream));
    cu_errchk(cudaMallocAsync((void **) &d_sum_rem, sizeof(double), custream));
    cu_errchk(cudaMallocAsync((void **) &d_sum_corr, sizeof(double), custream));
    G_sum_expanded<<<CU_BLOCKS_THREADS(expand_size), 0, custream>>>(
            d_sum_imf, d_sum_rem, d_sum_corr, thrust::raw_pointer_cast(d_imf_mask.data()), thrust::raw_pointer_cast(d_rem_mask.data()),
            expand_size, d_global_sift_matrix_ptr);
    cu_errchk(cudaMemcpyAsync(&sum_imf, d_sum_imf, sizeof(double), cudaMemcpyDeviceToHost, custream));
    cu_errchk(cudaMemcpyAsync(&sum_rem, d_sum_rem, sizeof(double), cudaMemcpyDeviceToHost, custream));
    cu_errchk(cudaMemcpyAsync(&sum_corr, d_sum_corr, sizeof(double), cudaMemcpyDeviceToHost, custream));
    cu_errchk(cudaFreeAsync(d_sum_imf, custream));
    cu_errchk(cudaFreeAsync(d_sum_rem, custream));
    cu_errchk(cudaFreeAsync(d_sum_corr, custream));
    cu_errchk(cudaMemcpyAsync(&sum_full, d_global_sift_matrix_ptr, sizeof(double), cudaMemcpyDeviceToHost, custream));
    cu_errchk(cudaStreamSynchronize(custream));
    cu_errchk(cudaStreamDestroy(custream));
    return {sum_full, sum_imf, sum_rem, sum_corr};
}


__global__ void G_do_quality(
        CRPTR(cuDoubleComplex) mask_fft, const unsigned siftings, const unsigned n, const double coeff, const unsigned end_i, const double mask_fft_coef, double *result)
{
    __shared__ double shared[common::C_cu_block_size];
    constexpr cuDoubleComplex cplx_one{1, 0};
    CU_STRIDED_FOR_i(n) {
        cuDoubleComplex p{1, 0};
        if (i < end_i) {
            const auto zz = make_cuDoubleComplex(1. - mask_fft[i].x, -mask_fft[i].y);
            UNROLL()
            for (unsigned k = 0; k < siftings; ++k) p = cuCmul(p, zz);
            shared[tid] = cunorm(p) + abs(1. - cunorm(cuCsub(cplx_one, p)));
        } else {
            const cuDoubleComplex zz = mask_fft[i];
            UNROLL()
            for (unsigned k = 0; k < siftings; ++k) p = cuCmul(p, zz);
            shared[tid] = i < mask_fft_coef ? cunorm(p) : oemd_coefficients_search::C_smooth_factor * cunorm(p);
        }
        const double norm_zz = cunorm(mask_fft[i]);
        if (norm_zz > 1) shared[tid] += norm_zz;
    }
    __syncthreads();

    const auto sh_limit = _MIN(n, common::C_cu_block_size);
#define stride_reduce_sum(block_low_)                        \
        if (common::C_cu_block_size >= block_low_) {         \
            constexpr unsigned stride2 = block_low_ / 2;     \
            const auto tid_stride2 = tid + stride2;          \
            if (tid < stride2 && tid_stride2 < sh_limit)     \
                shared[tid] += shared[tid_stride2];          \
            __syncthreads();                                 \
        }

    stride_reduce_sum(1024);
    stride_reduce_sum(512);
    stride_reduce_sum(256);
    stride_reduce_sum(128);

    if (tid >= 32) return;
    warp_reduce_sum<common::C_cu_block_size>(shared, tid, sh_limit);

    if (tid) return;
    atomicAdd(result, shared[0]);
}

double oemd_coefficients_search::cu_quality(const cufftDoubleComplex *mask_fft, const unsigned mask_size, const unsigned siftings, const cudaStream_t custream)
{
    const double coeff = mask_size / 250.;
    const unsigned end_i = mask_size * 2. * lambda1 / coeff;
    const auto mask_fft_coef = mask_size * 2. * lambda2 / coeff;
    double result, *d_result = cucalloc<double>(custream);
    G_do_quality<<<CU_BLOCKS_THREADS(mask_size), 0, custream>>>(mask_fft, siftings, mask_size, coeff, end_i, mask_fft_coef, d_result);
    cu_errchk(cudaMemcpyAsync(&result, d_result, sizeof(double), cudaMemcpyDeviceToHost, custream));
    cu_errchk(cudaFreeAsync(d_result, custream));
    cu_errchk(cudaStreamSynchronize(custream));
    return result / mask_size;
}


void
oemd_coefficients_search::gauss_smoothen_mask(
        const unsigned mask_size,
        std::vector<double> &mask,
        common::t_drand48_data_ptr buffer,
        cufftHandle plan_mask_forward,
        cufftHandle plan_mask_backward,
        const unsigned gpu_id)
{
    const auto full_size = 2 * mask_size;
    cu_errchk(cudaSetDevice(gpu_id));
    cudaStream_t custream;
    cu_errchk(cudaStreamCreate(&custream));
    double *d_mask_zm;
    cu_errchk(cudaMallocAsync((void **) &d_mask_zm, full_size, custream));
    cufftDoubleComplex *d_mask_zm_fft;
    const auto full_fft = common::to_fft_len(full_size);
    cu_errchk(cudaMallocAsync((void **) &d_mask_zm_fft, full_fft, custream));
    cu_errchk(cudaMemsetAsync(d_mask_zm + mask_size, 0, mask_size * sizeof(double), custream));
    cu_errchk(cudaMemcpyAsync(d_mask_zm, mask.data(), sizeof(double) * mask_size, cudaMemcpyKind::cudaMemcpyHostToDevice, custream));
    cf_errchk(cufftSetStream(plan_mask_forward, custream));
    cf_errchk(cufftExecD2Z(plan_mask_forward, d_mask_zm, d_mask_zm_fft));
    G_gpu_multiply_smooth<<<CU_BLOCKS_THREADS(full_fft), 0, custream>>>(full_size, 5. * -log(common::drander(buffer)), d_mask_zm_fft);
    cf_errchk(cufftSetStream(plan_mask_backward, custream));
    cf_errchk(cufftExecZ2D(plan_mask_backward, d_mask_zm_fft, d_mask_zm));
    thrust::transform(thrust::cuda::par.on(custream), d_mask_zm, d_mask_zm + mask_size, d_mask_zm,
    [mask_size]
            __device__(
    const double &iter) -> double{return iter > 0 ? iter / double(mask_size) : 0;} );
    if (mask.size() != full_size) mask.resize(full_size);
    cu_errchk(cudaMemcpyAsync(mask.data(), d_mask_zm, full_size * sizeof(*d_mask_zm), cudaMemcpyDeviceToHost, custream));
    cu_errchk(cudaFreeAsync(d_mask_zm, custream));
    cu_errchk(cudaFreeAsync(d_mask_zm_fft, custream));
    cu_errchk(cudaStreamSynchronize(custream));
    cu_errchk(cudaStreamDestroy(custream));
}


void
oemd_coefficients_search::create_random_mask(
        const unsigned position, double step, const unsigned mask_size, std::vector<double> &mask, const double *start_mask,
        common::t_drand48_data_ptr buffer, cufftHandle plan_mask_forward, cufftHandle plan_mask_backward, const unsigned gpu_id)
{
    step *= common::drander(buffer);
    if (!start_mask) {
UNROLL()
        for (unsigned i = 0; i < mask_size; ++i) mask[i] = common::drander(buffer);
    } else {
#pragma omp parallel for default(shared) num_threads(adj_threads(mask_size))
        for (unsigned i = 0; i < mask_size; ++i) {
            if (common::drander(buffer) > .25) {
                if (common::drander(buffer) > .05) {
                    if (common::drander(buffer) > .5) {
                        //masks[i]=start_mask[i]+step*(-log(drand48()));
                        mask[i] = start_mask[i] + step * common::drander(buffer);
                    } else {
                        mask[i] = std::max<double>(0., start_mask[i] - step * common::drander(buffer));
                    }
                } else {
                    mask[i] = start_mask[i] * (1. + step * (2 * common::drander(buffer) - 1));
                }
            } else {
                mask[i] = start_mask[i];
            }
        }
    }

    if (common::drander(buffer) > .01) gauss_smoothen_mask(mask_size, mask, buffer, plan_mask_forward, plan_mask_backward, gpu_id);

//    fix_mask(mask);
}


__global__ void compute_power_spectrum(CRPTR(cufftDoubleComplex) d_freq, double *const d_psd, double *const d_psd_sum, const unsigned N, const unsigned N_2_1)
{
    const auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_2_1) return;
    d_psd[i] = (d_freq[i].x * d_freq[i].x + d_freq[i].y * d_freq[i].y) / N;
    atomicAdd(d_psd_sum, d_psd[i]);
}

__global__ void normalize_psd(double *const d_psd, CRPTR(double) d_psd_sum, const unsigned N_2_1)
{
    const auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_2_1) return;
    d_psd[i] /= *d_psd_sum;
}

__global__ void compute_spectral_entropy(CRPTR(double) d_psd, double *const d_entropy, const unsigned N_2_1)
{
    const auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_2_1) return;
    const double value = d_psd[i];
    if (value > 0) atomicAdd(d_entropy, -value * log2f(value));
}

double oemd_coefficients_search::compute_spectral_entropy_cufft(const double *d_signal, const unsigned N, const cudaStream_t custream)
{
    cufftDoubleComplex *d_freq;
    double *d_psd;
    const auto N_2_1 = common::to_fft_len(N);

    // Allocate memory on the device
    cu_errchk(cudaMallocAsync((void **) &d_freq, N_2_1 * sizeof(cufftDoubleComplex), custream));
    cu_errchk(cudaMallocAsync((void **) &d_psd, N_2_1 * sizeof(double), custream));
    auto d_psd_sum = cucalloc<double>(custream);
    auto d_entropy = cucalloc<double>(custream);

    // Create a CUFFT plan
    cufftHandle plan;
    cf_errchk(cufftPlan1d(&plan, N, CUFFT_D2Z, 1));
    cf_errchk(cufftSetStream(plan, custream));
    // Execute the FFT
    cf_errchk(cufftExecD2Z(plan, (double *) d_signal, d_freq));

    // Compute the Power Spectral Density (PSD)
    compute_power_spectrum<<<CU_BLOCKS_THREADS(N_2_1), 0, custream>>>(d_freq, d_psd, d_psd_sum, N, N_2_1);

    // Normalize the PSD
    normalize_psd<<<CU_BLOCKS_THREADS(N_2_1), 0, custream>>>(d_psd, d_psd_sum, N_2_1);

    // Compute the Spectral Entropy
    compute_spectral_entropy<<<CU_BLOCKS_THREADS(N_2_1), 0, custream>>>(d_psd, d_entropy, N_2_1);

    // Copy the result back to the host
    double h_entropy;
    cu_errchk(cudaMemcpyAsync((void **) &h_entropy, d_entropy, sizeof(double), cudaMemcpyDeviceToHost, custream));

    // Clean up
    cufftDestroy(plan);
    cu_errchk(cudaFreeAsync((void *) d_freq, custream));
    cu_errchk(cudaFreeAsync((void *) d_psd, custream));
    cu_errchk(cudaFreeAsync((void *) d_entropy, custream));
    cu_errchk(cudaStreamSynchronize(custream));

    return h_entropy;
}

bool cu_fix_mask(double *const d_mask, const unsigned mask_len, const cudaStream_t custream)
{
    const auto mask_sum = solvers::sum(d_mask, mask_len, custream);
    if (mask_sum <= 0) return false;
    datamodel::G_div_inplace<<<CU_BLOCKS_THREADS(mask_len), 0, custream>>>(d_mask, mask_sum, mask_len);
    return true;
}


double
oemd_coefficients_search::evaluate_mask(
        const std::vector<double> &h_mask, const std::vector<double> &h_workspace,
        const size_t validate_start_ix, const size_t validation_len, const unsigned siftings,
        const unsigned current_level, const std::deque<unsigned> &head_tail_sizes, const double meanabs_input,
        const unsigned gpu_id) const
{
    const auto full_input_len = h_workspace.size();
    const auto mask_len = h_mask.size();
    cu_errchk(cudaSetDevice(gpu_id));
    cudaStream_t custream;
    cu_errchk(cudaStreamCreate(&custream));
    auto d_mask = cumallocopy(h_mask, custream);
    if (!cu_fix_mask(d_mask, mask_len, custream)) {
        cu_errchk(cudaFreeAsync(d_mask, custream));
        cu_errchk(cudaStreamDestroy(custream));
        return common::C_bad_validation;
    }
    auto d_rx = cumallocopy(h_workspace, custream);
    double *d_rx2;
    cu_errchk(cudaMallocAsync((void **) &d_rx2, full_input_len * sizeof(double), custream));
    sift(siftings, full_input_len, mask_len, custream, d_mask, d_rx, d_rx2);
    cu_errchk(cudaFreeAsync(d_rx2, custream));
    cu_errchk(cudaFreeAsync(d_mask, custream));
    const auto d_imf = d_rx + validate_start_ix;
    const auto meanabs_imf = solvers::meanabs(d_imf, validation_len, custream);
    if (!std::isnormal(meanabs_imf)) {
        LOG4_WARN("Bad IMF " << meanabs_imf);
        cu_errchk(cudaFreeAsync(d_rx, custream));
        return common::C_bad_validation;
    }
#if 0 // Evaluate output?
    double *d_out;
    cu_errchk(cudaMallocAsync((void **) &d_out, validation_len * sizeof(*d_out), custream));
    cu_errchk(cudaMemcpyAsync(d_out, h_workspace.data() + validate_start_ix, validation_len * sizeof(*d_out), cudaMemcpyHostToDevice, custream));
    G_subtract_inplace<<<CU_BLOCKS_THREADS(validation_len), 0, custream>>>(d_out, d_imf, validation_len);
    const auto meanabs_out = solvers::meanabs(d_out, validation_len, custream);
    cu_errchk(cudaFreeAsync(d_out, custream));
#endif
    const auto rel_pow = std::abs(meanabs_input / meanabs_imf - levels / 2.);
    double autocor = 0;
    UNROLL()
    for (const auto i: head_tail_sizes)
        autocor += solvers::unscaled_distance(d_imf, d_imf + validation_len - i, 1, i, 1, i, custream);
    autocor = (autocor / head_tail_sizes.size()) / meanabs_imf;
    const auto inv_entropy = 1. / compute_spectral_entropy_cufft(d_imf, validation_len, custream);
    cu_errchk(cudaFreeAsync(d_rx, custream)); // d_imf is a part of d_rx
    cu_errchk(cudaStreamSynchronize(custream));
    cu_errchk(cudaStreamDestroy(custream));
    constexpr double autocor_w = 10; // Weights
    constexpr double rel_pow_w = 5;
    constexpr double inv_entropy_w = 1;
    const auto score = std::pow(rel_pow, rel_pow_w) * std::pow(autocor, autocor_w) * std::pow(inv_entropy, inv_entropy_w);
    LOG4_TRACE("Returning autocorrelation " << autocor << ", relative power " << rel_pow << ", score " << score << ", inv entropy " << inv_entropy << ", meanabs imf " <<
                                            meanabs_imf << ", meanabs input " << meanabs_input);

    return score;
}


void oemd_coefficients_search::sift(
        const unsigned siftings, const unsigned full_input_len, const unsigned mask_len, const cudaStream_t custream, CPTR(double) d_mask, double *const d_rx,
        double *const d_rx2) const noexcept
{
    UNROLL()
    for (unsigned s = 0; s < siftings; ++s) {
        oemd::G_apply_mask<<<CU_BLOCKS_THREADS(full_input_len), 0, custream>>>(
                stretch_coef, d_rx, full_input_len, d_mask, mask_len, mask_len * stretch_coef, d_rx2, 0);
        oemd::G_subtract_inplace<<<CU_BLOCKS_THREADS(full_input_len), 0, custream>>>(d_rx, d_rx2, full_input_len);
    }
}


double
oemd_coefficients_search::find_good_mask(
        const unsigned siftings, const unsigned valid_start_ix, const std::vector<double> &workspace, std::vector<double> &mask, const unsigned level) const
{
    const unsigned mask_len = mask.size();
    const auto inside_window_start = valid_start_ix + mask_len * siftings;
    const auto in_window_len = workspace.size() - inside_window_start;
    LOG4_DEBUG(
            "Optimizing mask with " << mask_len << " elements, " << siftings << " siftings, " << in_window_len << " window len, " << workspace.size() << " workspace len");
    std::deque<unsigned> ht_sizes;
    UNROLL()
    for (float i = in_window_len / 2; i < in_window_len - 1; i = std::max(i + 1., i * 1.001)) ht_sizes.emplace_back(i);
    const double meanabs_input = std::reduce(C_default_exec_policy, workspace.cbegin() + inside_window_start, workspace.cend(), 0.,
                                             [](const double sum, const double val) { return sum + std::abs(val); }) / workspace.size();
#ifdef USE_FIREFLY
    const auto loss_function = [&](const std::vector<double> &x) {
#else
    const auto loss_function = [&, mask_len, in_window_len, siftings, level, meanabs_input](const double *x, double *const f) {
#endif
        static tbb::mutex mx_incr;
        static unsigned gl_incr;
        tbb::mutex::scoped_lock ul(mx_incr);
        const unsigned l_incr = gl_incr;
        gl_incr = (gl_incr + 1) % parallelism;
        const auto devix = gpuids[l_incr % max_gpus];
        ul.release();

        static std::deque<tbb::mutex> mxs(parallelism);
        const tbb::mutex::scoped_lock lg(mxs[l_incr]);
#ifdef USE_FIREFLY
        return evaluate_mask(x,
#else
        *f = evaluate_mask(common::wrap_vector<double>((double *) x, mask_len),
#endif
                           workspace, inside_window_start, in_window_len, siftings, level, ht_sizes, meanabs_input, devix);
    };

    arma::mat bounds(mask_len, 2, arma::fill::none);
    bounds.col(0).fill(0);
    bounds.col(1).fill(1);
#ifdef USE_FIREFLY
    double score;
    std::tie(score, h_mask) = optimizer::firefly(
            h_mask.size(), particles, iterations, common::C_FFA_alpha, common::C_FFA_betamin, common::C_FFA_gamma, bounds,
            arma::vec(h_mask.size(), arma::fill::ones), loss_function).operator std::pair<double, std::vector<double>>();
    fix_mask(res.best_parameters, mask);
    return score;
#else
    const optimizer::t_pprune_res res = optimizer::pprune(0, particles, bounds, loss_function, iterations);
    fix_mask(res.best_parameters, mask);
    return res.best_score;
#endif
}


void
oemd_coefficients_search::optimize_levels(
        const datamodel::datarow_range &input,
        const std::vector<double> &tail,
        std::deque<std::vector<double>> &masks,
        std::deque<unsigned> &siftings,
        const unsigned window_start,
        const unsigned window_end,
        const std::string &queue_name) const
{
    if (gpuids.empty()) LOG4_THROW("No GPUs found, aborting.");
    const auto gpu_id = gpuids.front();
    const auto window_len = window_end - window_start;
    const auto window_size = window_len * sizeof(double);
    const auto in_colix = input.front()->get_values().size() / 2;

    std::vector<double> h_workspace(window_len);
    OMP_FOR(window_len)
    for (unsigned i = window_start; i < window_end; ++i)
        h_workspace[i - window_start] = i < tail.size() ? tail[i] : input[i - tail.size()]->at(in_colix);
    LOG4_DEBUG(
            "Optimizing " << masks.size() + 1 << " levels for queue " << queue_name << " with " << gpuids.size() << " GPUs, tail len " << tail.size() << ", window len "
                          << window_len << ", window start " <<
                          window_start << ", window end " << window_end);

    cu_errchk(cudaSetDevice(gpu_id));
    cudaStream_t custream;
    unsigned validation_start_ix = C_fir_mask_end_len;
    cu_errchk(cudaStreamCreate(&custream));
    UNROLL()
    for (unsigned i = 0; i < masks.size(); ++i) {
        const auto mask_len = masks[i].size();
        double result;
        PROFILE_EXEC_TIME(result = find_good_mask(siftings[i], validation_start_ix, h_workspace, masks[i], i), "Optimizing mask " << i);
        LOG4_DEBUG("Level " << i << ", queue " << queue_name << ", score " << result);
        save_mask(masks[i], queue_name, i, masks.size() + 1);
        cu_errchk(cudaSetDevice(gpu_id));
        auto d_level_imf = cumallocopy(h_workspace, custream);
        const auto d_mask = cumallocopy(masks[i], custream);
        double *d_workspace;
        cu_errchk(cudaMallocAsync((void **) &d_workspace, window_size, custream));
        transform(d_level_imf, d_mask, window_len, mask_len, siftings[i], d_workspace, custream);
        cu_errchk(cudaFreeAsync(d_mask, custream));
        cu_errchk(cudaMemcpyAsync(d_workspace, h_workspace.data(), window_size, cudaMemcpyHostToDevice, custream));
        G_subtract_inplace<<<CU_BLOCKS_THREADS(window_len), 0, custream>>>(d_workspace, d_level_imf, window_len);
        cu_errchk(cudaFreeAsync(d_level_imf, custream));
        validation_start_ix += siftings[i] * mask_len;
        cu_errchk(cudaMemcpyAsync(h_workspace.data(), d_workspace, window_size, cudaMemcpyDeviceToHost, custream));
        cu_errchk(cudaFreeAsync(d_workspace, custream));
        cu_errchk(cudaStreamSynchronize(custream));
    }
    cu_errchk(cudaStreamSynchronize(custream));
    cu_errchk(cudaStreamDestroy(custream));
}

} // oemd_search
} // svr
