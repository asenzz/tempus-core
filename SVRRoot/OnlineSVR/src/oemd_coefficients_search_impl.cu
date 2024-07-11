//
// Created by zarko on 10/3/22.
//
#include <cstdlib>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>
#include <queue>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
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


namespace svr {
namespace oemd {


#define CUDA_SET_DEVICE(__i) cu_errchk(cudaSetDevice(gpuids[(__i) % max_gpus]))


__global__ void
gpu_multiply_smooth(
        const size_t input_size,
        const double coeff,
        cufftDoubleComplex *__restrict__ output)
{
    const auto thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const auto total_block_size = blockDim.x * gridDim.x;
#ifndef __GNUC__
#pragma unroll
#endif
    for (auto j = thread_idx; j < input_size / 2 + 1; j += total_block_size) { // Because it is D2Z transform
        const double mult = std::exp(-coeff * double(j) / double(input_size));
        output[j].x *= mult;
        output[j].y *= mult;
    }
}


__global__ void
vec_sift(
        const size_t fft_size,
        const size_t siftings,
        const cufftDoubleComplex *__restrict__ x,
        cufftDoubleComplex *__restrict__ imf,
        cufftDoubleComplex *__restrict__ rem)
{
    double px, py;
#ifndef __GNUC__
#pragma unroll
#endif
    for (size_t j = blockIdx.x * blockDim.x + threadIdx.x; j < fft_size; j += blockDim.x * gridDim.x) {
        px = 1. - x[j].x;
        py = -x[j].y;
#ifndef __GNUC__
#pragma unroll
#endif
        for (size_t i = 1; i < siftings; ++i) {
            px = px * (1. - x[j].x) - py * (-x[j].y);
            py = px * (-x[j].y) + py * (1. - x[j].x);
        }
        imf[j].x = px;
        imf[j].y = py;
        rem[j].x = 1. - px;
        rem[j].y = -py;
    }
}

__global__ void
sum_expanded(
        double *__restrict__ d_sum_imf,
        double *__restrict__ d_sum_rem,
        double *__restrict__ d_sum_corr,
        const double *__restrict__ d_imf_mask,
        const double *__restrict__ d_rem_mask,
        const size_t expand_size,
        const double *__restrict__ d_global_sift_matrix)
{
    const size_t thr_ix = threadIdx.x;
    const size_t g_thr_ix = thr_ix + blockIdx.x * common::C_cu_block_size;
    const size_t grid_size = common::C_cu_block_size * gridDim.x;
    const auto d_expand_size = double(expand_size);

    double _sum_imf = 0;
    double _sum_rem = 0;
    double _sum_corr = 0;
    // TODO Optimize code here!
#ifndef __GNUC__
#pragma unroll
#endif
    for (size_t i = g_thr_ix; i < expand_size; i += grid_size) {
        double sum1 = 0, sum2 = 0;
#ifndef __GNUC__
#pragma unroll
#endif
        for (size_t j = 0; j < expand_size; ++j) {
            sum1 += d_global_sift_matrix[labs(i - j)] * d_imf_mask[j];
            sum2 += d_global_sift_matrix[labs(i - j)] * d_rem_mask[j];
        }
        _sum_imf += sum1 * d_imf_mask[i] / d_expand_size / d_expand_size;
        _sum_rem += sum2 * d_rem_mask[i] / d_expand_size / d_expand_size;
        _sum_corr += sum1 * d_rem_mask[i] / d_expand_size / d_expand_size;
    }
    __shared__ double _sh_sum_imf[common::C_cu_block_size];
    __shared__ double _sh_sum_rem[common::C_cu_block_size];
    __shared__ double _sh_sum_corr[common::C_cu_block_size];
    _sh_sum_imf[thr_ix] = _sum_imf;
    _sh_sum_rem[thr_ix] = _sum_rem;
    _sh_sum_corr[thr_ix] = _sum_corr;

    __syncthreads();

#ifndef __GNUC__
#pragma unroll
#endif
    for (size_t size = common::C_cu_block_size / 2; size > 0; size /= 2) { // uniform
        if (thr_ix >= size) continue;
        _sh_sum_imf[thr_ix] += _sh_sum_imf[thr_ix + size];
        _sh_sum_rem[thr_ix] += _sh_sum_rem[thr_ix + size];
        _sh_sum_corr[thr_ix] += _sh_sum_corr[thr_ix + size];
        __syncthreads();
    }
    if (thr_ix == 0) {
        *d_sum_imf = _sh_sum_imf[0];
        *d_sum_rem = _sh_sum_rem[0];
        *d_sum_corr = _sh_sum_corr[0];
    }
}


void
oemd_coefficients_search::transform(
        double *d_values, double *h_mask, const size_t input_size, const size_t mask_size,
        const size_t siftings, double *d_temp, const size_t gpu_id)
{
    cudaSetDevice(gpu_id);
    cufftHandle plan_forward;
    cufftHandle plan_backward;
    thrust::device_vector<double> d_mask(mask_size);
    thrust::device_vector<double> d_expanded_mask(input_size);
    cudaMemcpy(thrust::raw_pointer_cast(d_mask.data()), h_mask, sizeof(double) * mask_size, cudaMemcpyHostToDevice);
    online_emd::expand_the_mask(mask_size, input_size, thrust::raw_pointer_cast(d_mask.data()), thrust::raw_pointer_cast(d_expanded_mask.data()));
    thrust::device_vector<cufftDoubleComplex> d_expanded_mask_fft(to_fft_size(input_size));
    thrust::device_vector<cufftDoubleComplex> d_input_fft(to_fft_size(input_size));
    cufftDoubleComplex *dev_expanded_mask_fft = thrust::raw_pointer_cast(d_expanded_mask_fft.data());
    const int n_batch = 1;
    fft_acquire();
    cf_errchk(cufftPlan1d(&plan_forward, input_size, CUFFT_D2Z, n_batch));
    cf_errchk(cufftPlan1d(&plan_backward, input_size, CUFFT_Z2D, n_batch));
    cf_errchk(cufftExecD2Z(plan_forward, thrust::raw_pointer_cast(d_expanded_mask.data()), dev_expanded_mask_fft));
    fft_release();

    cufftDoubleComplex *dev_input_fft = thrust::raw_pointer_cast(d_input_fft.data());
#pragma omp unroll
    for (size_t i = 0; i < siftings; ++i) {
        fft_acquire();
        cf_errchk(cufftExecD2Z(plan_forward, d_values, dev_input_fft));
        fft_release();
        G_gpu_multiply_complex<<<CUDA_THREADS_BLOCKS(to_fft_size(input_size)) >>>(input_size, dev_expanded_mask_fft, dev_input_fft);
        cu_errchk(cudaPeekAtLastError());
        fft_acquire();
        cf_errchk(cufftExecZ2D(plan_backward, dev_input_fft, d_temp));
        fft_release();
        G_vec_subtract_inplace<<<CUDA_THREADS_BLOCKS(input_size)>>>(d_values, d_temp, input_size);
        cu_errchk(cudaPeekAtLastError());
    }

    cufftDestroy(plan_forward);
    cufftDestroy(plan_backward);
}


std::tuple<double, double, double, double>
oemd_coefficients_search::sift_the_mask(
        const size_t mask_size,
        const size_t siftings,
        const double *d_mask,
        const cufftHandle plan_sift_forward,
        const cufftHandle plan_sift_backward,
        const double *d_expanded_mask,
        const cufftDoubleComplex *d_expanded_mask_fft,
        const double *d_global_sift_matrix_ptr,
        const size_t gpu_id)
{
    cudaSetDevice(gpu_id);
    double sum_full, sum_imf, sum_rem, sum_corr;
    const size_t expand_size = siftings * mask_size;
    thrust::device_vector<double> d_zm_mask(expand_size);
    thrust::device_vector<double> d_imf_mask(expand_size);
    thrust::device_vector<double> d_rem_mask(expand_size);

    double *d_expanded_mask_ptr = thrust::raw_pointer_cast(d_zm_mask.data());

    online_emd::expand_the_mask(mask_size, expand_size, d_mask, d_expanded_mask_ptr);
    const auto fft_size = to_fft_size(expand_size);
    thrust::device_vector<cufftDoubleComplex> d_fzm_mask(fft_size);
    thrust::device_vector<cufftDoubleComplex> d_mask_imf_fft(fft_size);
    thrust::device_vector<cufftDoubleComplex> d_mask_rem_fft(fft_size);
    //cufftDoubleComplex *d_expanded_mask_fft = thrust::raw_pointer_cast(d_fzm_mask.data());
    //cufft_errchk(cufftExecD2Z(plan_sift_forward, d_expanded_mask_ptr, d_expanded_mask_fft));
    vec_sift<<<CUDA_THREADS_BLOCKS(fft_size)>>>(
            fft_size, siftings, d_expanded_mask_fft, thrust::raw_pointer_cast(d_mask_imf_fft.data()), thrust::raw_pointer_cast(d_mask_rem_fft.data()));
    cf_errchk(cufftExecZ2D(plan_sift_backward, thrust::raw_pointer_cast(d_mask_imf_fft.data()), thrust::raw_pointer_cast(d_imf_mask.data())));
    cf_errchk(cufftExecZ2D(plan_sift_backward, thrust::raw_pointer_cast(d_mask_rem_fft.data()), thrust::raw_pointer_cast(d_rem_mask.data())));

    auto d_sum_imf = (double *) cuda_calloc(sizeof(double), 1);
    auto d_sum_rem = (double *) cuda_calloc(sizeof(double), 1);
    auto d_sum_corr = (double *) cuda_calloc(sizeof(double), 1);
    sum_expanded<<<CUDA_THREADS_BLOCKS(expand_size)>>>(
            d_sum_imf, d_sum_rem, d_sum_corr, thrust::raw_pointer_cast(d_imf_mask.data()), thrust::raw_pointer_cast(d_rem_mask.data()),
            expand_size, d_global_sift_matrix_ptr);
    cudaMemcpy(&sum_imf, d_sum_imf, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&sum_rem, d_sum_rem, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&sum_corr, d_sum_corr, sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_sum_imf);
    cudaFree(d_sum_rem);
    cudaFree(d_sum_corr);
    cudaMemcpy(&sum_full, d_global_sift_matrix_ptr, sizeof(double), cudaMemcpyDeviceToHost);
    return {sum_full, sum_imf, sum_rem, sum_corr};
}


double
oemd_coefficients_search::evaluate_mask(
        const size_t siftings,
        const std::vector<double> &h_mask, const std::vector<double> &h_workspace,
        const std::vector<cufftDoubleComplex> &values_fft, const size_t val_start,
        const cufftHandle plan_expanded_forward, const cufftHandle plan_expanded_backward,
        const cufftHandle plan_mask_forward, const cufftHandle plan_sift_forward, const cufftHandle plan_sift_backward,
        const size_t current_level, const size_t gpu_id)
{
    const auto full_input_size = h_workspace.size();
    double result = 0;
    cu_errchk(cudaSetDevice(gpu_id));
    auto dev_values_fft = cuda_malloccopy(values_fft);
    auto d_mask_ptr = cuda_malloccopy(h_mask);
    //auto d_global_sift_matrix = cuda_malloccopy(global_sift_matrix);
    const auto d_workspace = cuda_malloccopy(h_workspace);
    const size_t expanded_size = h_mask.size() * C_mask_expander;

    double *d_expanded_mask;
    cu_errchk(cudaMalloc(&d_expanded_mask, expanded_size * sizeof(*d_expanded_mask)));
    online_emd::expand_the_mask(h_mask.size(), expanded_size, d_mask_ptr, d_expanded_mask);
    cufftDoubleComplex *d_expanded_mask_fft;
    cu_errchk(cudaMalloc(&d_expanded_mask_fft, to_fft_size(expanded_size) * sizeof(*d_expanded_mask_fft)));
    cf_errchk(cufftExecD2Z(plan_mask_forward, d_expanded_mask, d_expanded_mask_fft));
    auto h_mask_fft = cuda_copy(d_expanded_mask_fft, to_fft_size(expanded_size));

#ifdef FILTEROUT_BAD_MASKS
    if (do_filter(h_mask_fft)) {
        cu_errchk(cudaFree(d_expanded_mask));
        cu_errchk(cudaFree(d_expanded_mask_fft));`
        return 0;
    }
#endif
    cu_errchk(cudaDeviceSynchronize());
    const auto quality = do_quality(h_mask_fft, siftings);
    result += quality;
#ifdef FILTEROUT_BAD_MASKS
    if (filter && quality > 2 * best_quality.load(std::memory_order_relaxed)) {
        cu_errchk(cudaFree(d_expanded_mask));
        cu_errchk(cudaFree(d_expanded_mask_fft));`
        filter = false;
        return 0;
    }
#endif

    //[[maybe_unused]] const auto [sum_full, sum_imf, sum_rem, sum_corr] =
    //        sift_the_mask(h_mask.size(), siftings, d_mask_ptr, plan_sift_forward, plan_sift_backward, d_expanded_mask, d_expanded_mask_fft, d_global_sift_matrix, gpu_id);
    // cu_errchk(cudaFree(d_global_sift_matrix));
    cu_errchk(cudaFree(d_expanded_mask));
    cu_errchk(cudaFree(d_expanded_mask_fft));

    double *d_zm_ptr;
    cu_errchk(cudaMalloc(&d_zm_ptr, full_input_size * sizeof(*d_zm_ptr)));
    online_emd::expand_the_mask(h_mask.size(), full_input_size, d_mask_ptr, d_zm_ptr);
    cu_errchk(cudaFree(d_mask_ptr));

    cufftDoubleComplex *d_zm_fft_ptr, *d_zm_convert_ptr;
    cu_errchk(cudaMalloc(&d_zm_fft_ptr, to_fft_size(full_input_size) * sizeof(*d_zm_fft_ptr)));
    cu_errchk(cudaMalloc(&d_zm_convert_ptr, to_fft_size(full_input_size) * sizeof(*d_zm_convert_ptr)));
    cf_errchk(cufftExecD2Z(plan_expanded_forward, d_zm_ptr, d_zm_fft_ptr));
    G_vec_power<<<CUDA_THREADS_BLOCKS(to_fft_size(full_input_size))>>>(d_zm_fft_ptr, d_zm_convert_ptr, full_input_size, siftings);
    cu_errchk(cudaFree(d_zm_fft_ptr));

    G_gpu_multiply_complex<<<CUDA_THREADS_BLOCKS(to_fft_size(full_input_size))>>>(full_input_size, dev_values_fft, d_zm_convert_ptr);
    cu_errchk(cudaFree(dev_values_fft));
    cf_errchk(cufftExecZ2D(plan_expanded_backward, d_zm_convert_ptr, d_zm_ptr));
    cu_errchk(cudaFree(d_zm_convert_ptr));

    const size_t inside_window_start = val_start + h_mask.size() * siftings;
    const size_t inside_window_end = full_input_size;
    const size_t in_window_len = inside_window_end - inside_window_start;
    auto h_imf_temp = cuda_copy(d_zm_ptr + inside_window_start, in_window_len);
    double *d_values_copy;
    cu_errchk(cudaMalloc(&d_values_copy, in_window_len * sizeof(*d_values_copy)));
    cu_errchk(cudaMemcpy(d_values_copy, d_workspace + inside_window_start, in_window_len * sizeof(*d_values_copy), cudaMemcpyDeviceToDevice));
    G_vec_subtract_inplace<<<CUDA_THREADS_BLOCKS(in_window_len)>>>(d_values_copy, d_zm_ptr + inside_window_start, in_window_len);
    auto h_rem_temp = cuda_copy(d_values_copy, in_window_len);
    cu_errchk(cudaFree(d_zm_ptr));
    cu_errchk(cudaFree(d_values_copy));
    cu_errchk(cudaFree(d_workspace));
#if 1
    const arma::vec input((double *)h_workspace.data() + inside_window_start, in_window_len, false, true);
    const arma::vec output((double *)h_rem_temp.data(), in_window_len, false, true);
    double loscore = 0;
    const unsigned half_in_window_len = in_window_len / 2;
    OMP_FOR_(half_in_window_len, simd reduction(+:loscore))
    for (size_t i = half_in_window_len; i < in_window_len; ++i) {
        const double score = arma::mean(arma::abs(output.head(i) - output.tail(i)));
        loscore += score;
    }
    loscore /= double(half_in_window_len);
    const auto meanabs_input = common::meanabs(input);
    const auto meanabs_output = common::meanabs(output);
    LOG4_TRACE("Loscore " << loscore << ", meanabs input " << meanabs_input << ", meanabs output " << meanabs_output);
    return 2. * loscore / meanabs_output + (1. - std::min(1., meanabs_output / meanabs_input));
#endif

    double sum1 = 0;
    double sum2 = 0;
    double big_sum1 = 0;
    double big_sum2 = 0;
    double big_sum3 = 0;
    double prod = 0;
    const size_t partial = 1000;
    size_t cntr = 0;
    double corr = 0;
    size_t corr_count = 0;
    const size_t half_window_len = in_window_len / 2;
#pragma omp unroll // Non-parallelissabile!
    for (size_t i = half_window_len; i < in_window_len; ++i) {
        sum1 += pow(h_imf_temp[i] - h_imf_temp[i - 1], 2);
        sum2 += pow(h_rem_temp[i] - h_rem_temp[i - 1], 2);
        big_sum1 += pow(h_imf_temp[i] - h_imf_temp[i - 1], 2);
        big_sum2 += pow(h_rem_temp[i] - h_rem_temp[i - 1], 2);
        big_sum3 += pow(h_rem_temp[i] + h_imf_temp[i] - h_rem_temp[i - 1] - h_imf_temp[i - 1], 2);
        prod += (h_imf_temp[i] - h_imf_temp[i - 1]) * (h_rem_temp[i] - h_rem_temp[i - 1]);
        ++cntr;
        if (cntr >= partial) {
            corr += sum1 * sum2 != 0 ? fabs(prod) / sqrt(sum1 * sum2) : 1.;
            ++corr_count;
            sum1 = 0;
            sum2 = 0;
            prod = 0;
            cntr = 0;
        }
    }
    corr = corr_count ? corr / double(corr_count) : corr + fabs(prod) / sqrt(sum1 * sum2);

    if (big_sum2 > big_sum3) result = 1000. + (result + sqrt(big_sum2 / half_window_len)) * 1000.;
    if (big_sum1 > big_sum3) result = 10. + (result + sqrt(big_sum1 / half_window_len)) * 100.;

    result = result * (1 + corr);
    if (!current_level) {
        if (corr > .05) result = 1. + result * (10. + corr);
    } else {
        if (corr > .20) result = 1. + result * (10. + corr);
    }

    if (big_sum2 >= 0.95 * big_sum3) result = 30 + (result + sqrt(big_sum2 / half_window_len)) * 10.;
    if (big_sum2 >= 0.60 * big_sum3) result = result + 5 + (sqrt(big_sum2 / half_window_len)) * 1.;
    if (big_sum2 >= 0.80 * big_sum3) result = result + 3 + (sqrt(big_sum2 / half_window_len)) * 1.;
    if (big_sum1 > big_sum2) result = result + 4 + (big_sum1 / big_sum3);
    if (big_sum1 > 0.5 * big_sum3) result = 10 + (result + sqrt(big_sum1 / half_window_len)) * 10.;
    if (big_sum2 >= 0.99 * big_sum3) result += 1. + big_sum2 / big_sum3;
    if (big_sum2 >= 0.98 * big_sum3) result += 1. + big_sum2 / big_sum3;
    if (big_sum2 >= 0.97 * big_sum3) result += 1. + big_sum2 / big_sum3;
    if (big_sum2 >= 0.96 * big_sum3) result += 1. + big_sum2 / big_sum3;

    result += sqrt((big_sum1 + big_sum2) / big_sum3);
#if 0
    if (result < best_result.load(std::memory_order_relaxed) && quality <= best_quality.load(std::memory_order_relaxed)) {
        LOG4_DEBUG("Found global best score " << result << " and best quality " << quality);
        best_result.store(result, std::memory_order_relaxed);
        best_quality.store(quality, std::memory_order_relaxed);
    }
#endif
    return result;
}


void
oemd_coefficients_search::gauss_smoothen_mask(
        const size_t mask_size,
        std::vector<double> &mask,
        common::t_drand48_data_ptr buffer,
        cufftHandle plan_mask_forward,
        cufftHandle plan_mask_backward,
        const size_t gpu_id)
{
    const size_t full_size = 2 * mask_size;
    cu_errchk(cudaSetDevice(gpu_id));
    thrust::device_vector<double> d_mask_zm(full_size);
    thrust::device_vector<cufftDoubleComplex> d_mask_zm_fft(to_fft_size(full_size));
    cu_errchk(cudaMemset(thrust::raw_pointer_cast(d_mask_zm.data() + mask_size), 0, mask_size * sizeof(double)));
    cu_errchk(cudaMemcpy(thrust::raw_pointer_cast(d_mask_zm.data()), mask.data(), sizeof(double) * mask_size, cudaMemcpyKind::cudaMemcpyHostToDevice));
    cf_errchk(cufftExecD2Z(plan_mask_forward, thrust::raw_pointer_cast(d_mask_zm.data()), thrust::raw_pointer_cast(d_mask_zm_fft.data())));
    gpu_multiply_smooth<<<CUDA_THREADS_BLOCKS(to_fft_size(full_size))>>>(full_size, 5. * -log(common::drander(buffer)), thrust::raw_pointer_cast(d_mask_zm_fft.data()));
    cf_errchk(cufftExecZ2D(plan_mask_backward, thrust::raw_pointer_cast(d_mask_zm_fft.data()), thrust::raw_pointer_cast(d_mask_zm.data())));
    thrust::transform(thrust::device,
                      d_mask_zm.begin(), d_mask_zm.begin() + mask_size, d_mask_zm.begin(),
    [mask_size] __device__ (const double &iter) -> double { return (iter > 0 ? iter : 0) / double(mask_size); } );
    cuda_copy(mask, thrust::raw_pointer_cast(d_mask_zm.data()), d_mask_zm.size());
}


void
oemd_coefficients_search::create_random_mask(
        const size_t position, double step, const size_t mask_size, std::vector<double> &mask, const double *start_mask,
        common::t_drand48_data_ptr buffer, cufftHandle plan_mask_forward, cufftHandle plan_mask_backward, const size_t gpu_id)
{
    step *= common::drander(buffer);
    if (!start_mask) {
#pragma omp unroll
        for (size_t i = 0; i < mask_size; ++i) mask[i] = common::drander(buffer);
    } else {
#pragma omp parallel for default(shared) num_threads(adj_threads(mask_size))
        for (size_t i = 0; i < mask_size; ++i) {
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

    fix_mask(mask);
}


double
oemd_coefficients_search::find_good_mask_ffly(
        const size_t siftings, const size_t valid_start_index,
        const std::vector<double> &h_workspace, const std::vector<cufftDoubleComplex> &h_workspace_fft, std::vector<double> &h_mask,
        std::deque<cufftHandle> &plan_full_forward, std::deque<cufftHandle> &plan_full_backward, std::deque<cufftHandle> &plan_mask_forward,
        std::deque<cufftHandle> &plan_sift_forward, std::deque<cufftHandle> &plan_sift_backward,
        const size_t current_level)
{
    const auto loss_function = [&](const double *x, double *const f) {
        static std::mutex mx_incr;
        static size_t gl_incr;
        std::unique_lock<std::mutex> ul(mx_incr);
        const size_t l_incr = gl_incr;
        gl_incr = (gl_incr + 1) % C_parallelism;
        const auto devix = gpuids[l_incr % max_gpus];
        ul.unlock();

        static std::deque<std::mutex> mxs(C_parallelism);
        const std::scoped_lock lg(mxs[l_incr]);
        *f = evaluate_mask(
                siftings, common::wrap_vector<double>((double *)x, h_mask.size()), h_workspace, h_workspace_fft, valid_start_index,
                plan_full_forward[l_incr], plan_full_backward[l_incr], plan_mask_forward[l_incr], plan_sift_forward[l_incr], plan_sift_backward[l_incr],
                current_level, devix);
    };

    double score;
    /*
    std::tie(score, h_mask) = svr::optimizer::firefly(
            h_mask.size(), FIREFLY_PARTICLES, FIREFLY_ITERATIONS, common::C_FFA_alpha, common::C_FFA_betamin, common::C_FFA_gamma,
            arma::vec(h_mask.size()), arma::vec(h_mask.size(), arma::fill::value(1. / h_mask.size())), arma::vec(h_mask.size(), arma::fill::ones),
            loss_function).operator std::pair<double, std::vector<double>>();
    */
    arma::mat bounds(h_mask.size(), 2);
    bounds.col(0).zeros();
    bounds.col(1).fill(1. / h_mask.size());
    const optimizer::t_pprune_res res = optimizer::pprune(prima_algorithm_t::PRIMA_LINCOA, 50, bounds, loss_function, 200, .25, 1e-11);
    h_mask = arma::conv_to<std::vector<double>>::from(res.best_parameters);
    score = res.best_score;
    fix_mask(h_mask);
    return score;
}


void
oemd_coefficients_search::optimize_levels(
        const datamodel::datarow_range &input,
        const std::vector<double> &tail,
        std::deque<std::vector<double>> &masks,
        std::deque<size_t> &siftings,
        const size_t window_start,
        const size_t window_end,
        const std::string &queue_name)
{
    if (gpuids.empty()) LOG4_THROW("No GPUs found, aborting.");
    const size_t gpu_id = gpuids.front();
    const auto window_len = window_end - window_start;
    const auto in_colix = input.begin()->get()->get_values().size() / 2;

    std::vector<double> h_workspace(window_len);
#pragma omp unroll // parallel for schedule(static, 1 + window_len / std::thread::hardware_concurrency()) num_threads(adj_threads(window_len))
    for (size_t i = window_start; i < window_end; ++i)
        h_workspace[i - window_start] = i < tail.size() ? tail[i] : input[i - tail.size()]->get_value(in_colix);

    cu_errchk(cudaSetDevice(gpu_id));
//   thrust::host_vector<double> h_temp(window_len);
    thrust::device_vector<double> d_workspace(window_len);
    thrust::device_vector<double> d_temp(window_len);
    double *d_temp_ptr = thrust::raw_pointer_cast(d_temp.data());
    d_workspace = h_workspace;
    std::deque<thrust::host_vector<double>> h_imf(masks.size());
    cu_errchk(cudaSetDevice(gpu_id));
    thrust::device_vector<cufftDoubleComplex> d_values_fft(to_fft_size(window_len));
    cufftDoubleComplex *d_values_fft_ptr = thrust::raw_pointer_cast(d_values_fft.data());
    std::deque<thrust::device_vector<double>> d_imf(masks.size());
    double *d_workspace_ptr = thrust::raw_pointer_cast(d_workspace.data());
    std::deque<cufftHandle> plan_full_forward(C_parallelism);
    std::deque<cufftHandle> plan_full_backward(C_parallelism);
    size_t start_valid_ix = 0; // first correct values start from this position inside d_workspace

    constexpr int n_batch = 1;
#pragma omp parallel for num_threads(adj_threads(C_parallelism)) schedule(static, 1)
    for (size_t i = 0; i < C_parallelism; ++i) {
        CUDA_SET_DEVICE(i);
        cf_errchk(cufftPlan1d(&plan_full_forward[i], window_len, CUFFT_D2Z, n_batch));
        cf_errchk(cufftPlan1d(&plan_full_backward[i], window_len, CUFFT_Z2D, n_batch));
    }
#pragma omp unroll
    for (size_t i = 0; i < masks.size(); ++i) {
        std::deque<cufftHandle> plan_sift_forward(C_parallelism);
        std::deque<cufftHandle> plan_sift_backward(C_parallelism);
#pragma omp parallel for num_threads(adj_threads(C_parallelism))
        for (size_t j = 0; j < C_parallelism; ++j) {
            CUDA_SET_DEVICE(j);
            cf_errchk(cufftPlan1d(&plan_sift_forward[j], siftings[i] * masks[i].size(), CUFFT_D2Z, n_batch));
            cf_errchk(cufftPlan1d(&plan_sift_backward[j], siftings[i] * masks[i].size(), CUFFT_Z2D, n_batch));
        }
        //auto global_sift_matrix = fill_auto_matrix(
        //        masks[i].size(), siftings[i], window_len - start_valid_ix, &h_workspace[window_start + start_valid_ix]);
        std::deque<cufftHandle> plan_mask_forward(C_parallelism);
        std::deque<cufftHandle> plan_mask_backward(C_parallelism);
#pragma omp parallel for num_threads(adj_threads(C_parallelism)) schedule(static, 1)
        for (size_t j = 0; j < C_parallelism; ++j) {
            CUDA_SET_DEVICE(j);
            cf_errchk(cufftPlan1d(&plan_mask_forward[j], C_mask_expander * masks[i].size(), CUFFT_D2Z, n_batch));
            cf_errchk(cufftPlan1d(&plan_mask_backward[j], C_mask_expander * masks[i].size(), CUFFT_Z2D, n_batch));
        }
        cu_errchk(cudaSetDevice(gpu_id));
        cf_errchk(cufftExecD2Z(plan_full_forward[0], d_workspace_ptr, d_values_fft_ptr));
        auto h_values_fft = cuda_copy(d_values_fft);
        const double result = find_good_mask_ffly(
                siftings[i], start_valid_ix, h_workspace, h_values_fft,
                masks[i], plan_full_forward, plan_full_backward, plan_mask_forward, plan_sift_forward, plan_sift_backward,
                i);
        LOG4_DEBUG("Level " << i << ", queue " << queue_name << ", FFA score " << result);
        save_mask(masks[i], queue_name, i, masks.size() + 1);
        cu_errchk(cudaSetDevice(gpu_id));
        d_imf[i] = d_workspace; // copy
        double *d_imf_ptr = thrust::raw_pointer_cast(d_imf[i].data());
        transform(d_imf_ptr, masks[i].data(), window_len, masks[i].size(), siftings[i], d_temp_ptr, gpu_id);
        G_vec_subtract_inplace<<<CUDA_THREADS_BLOCKS(window_len)>>>(d_workspace_ptr, d_imf_ptr, window_len);
        start_valid_ix += siftings[i] * (masks[i].size() - 1); // is it -1 really?
/*
        h_imf[i] = d_imf[i];
        h_temp = d_workspace;
#pragma omp parallel for
        for (size_t j = 0; j < window_len; ++j) h_temp[j] += h_imf[i][j];
*/
        h_workspace = cuda_copy(d_workspace_ptr, d_workspace.size());
#if 0
        best_result.store(std::numeric_limits<double>::max(), std::memory_order_relaxed);
        best_quality.store(1, std::memory_order_relaxed);
#endif
#pragma omp parallel for num_threads(adj_threads(C_parallelism)) schedule(static, 1)
        for (size_t j = 0; j < C_parallelism; ++j) {
            CUDA_SET_DEVICE(j);
            cf_errchk(cufftDestroy(plan_mask_forward[j]));
            cf_errchk(cufftDestroy(plan_sift_forward[j]));
            cf_errchk(cufftDestroy(plan_sift_backward[j]));
        }
    }

#pragma omp parallel for num_threads(adj_threads(C_parallelism)) schedule(static, 1)
    for (size_t i = 0; i < C_parallelism; ++i) {
        CUDA_SET_DEVICE(i);
        cf_errchk(cufftDestroy(plan_full_forward[i]));
        cf_errchk(cufftDestroy(plan_full_backward[i]));
    }
}

} // oemd_search
} // svr
