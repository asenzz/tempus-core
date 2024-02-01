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
#include <complex>
#include <iomanip>
#include <filesystem>
#include <thrust/execution_policy.h>

#include "optimizer.hpp"
#include "oemd_coefficients_search.hpp"
#include "common/gpu_handler.hpp"
#include "../../SVRCommon/include/common/cuda_util.cuh"
#include "util/TimeUtils.hpp"
#include "firefly.hpp"
#include "common/Logging.hpp"

namespace svr {
namespace oemd_search {

#define CUDA_SET_DEVICE(__i) cu_errchk(cudaSetDevice(gpuids[(__i) % max_gpus]))

size_t to_fft_size(const size_t input_size)
{
    return input_size / 2 + 1;
}

void
oemd_coefficients_search::save_mask(
        const std::vector<double> &mask,
        const std::string &queue_name,
        const size_t level,
        const size_t levels)
{
    size_t ctr = 0;
    while (file_exists(svr::common::formatter() << C_oemd_fir_coefs_dir << mask_file_name(ctr, level, levels, queue_name))) { ++ctr; }
    std::ofstream myfile(svr::common::formatter() << C_oemd_fir_coefs_dir << mask_file_name(ctr, level, levels, queue_name));
    if (myfile.is_open()) {
        LOG4_DEBUG("Saving mask " << level << " to " << C_oemd_fir_coefs_dir << mask_file_name(ctr, level, levels, queue_name));
        myfile << std::setprecision(std::numeric_limits<double>::max_digits10);
        for (auto it = mask.cbegin(); it != mask.cend(); ++it)
            if (it != std::prev(mask.cend()))
                myfile << *it << ",";
            else
                myfile << *it;
        myfile.close();
    } else
        LOG4_ERROR(
                "Aborting saving! Unable to open file " << std::filesystem::current_path() << "/" << mask_file_name(ctr, level, levels, queue_name) << " for writing.");
}

#if 0
double get_std(double *x, const size_t input_size)
{
    double sum = 0.;
    for (size_t i = 0; i < input_size - 1; ++i) {
        sum += pow(x[i] - x[i + 1], 2);
    }
    return sqrt(sum / (double) input_size);
}
#endif

std::vector<double>
oemd_coefficients_search::fill_auto_matrix(const size_t M, const size_t siftings, const size_t N, const double *x)
{
    std::vector<double> diff(N - 1);
#pragma omp parallel for
    for (size_t i = 0; i < N - 1; ++i)
        diff[i] = x[i + 1] - x[i];

    const size_t Msift = M * siftings;
    std::vector<double> global_sift_matrix(Msift);
#pragma omp parallel for
    for (size_t i = 0; i < Msift; ++i) {
        double sum = 0;
        for (size_t j = N / 2; j < N - 1 - i; ++j)
            sum += diff[j] * diff[j + i];
        global_sift_matrix[i] = sum / double(N - 1. - i - N / 2.);
    }
    return global_sift_matrix;
}


void
oemd_coefficients_search::expand_the_mask(const size_t mask_size, const size_t input_size, const double *dev_mask, double *dev_expanded_mask)
{
    cu_errchk(cudaMemset(dev_expanded_mask + mask_size, 0, sizeof(double) * std::max<size_t>(0, input_size - mask_size)));
    cu_errchk(cudaMemcpy(dev_expanded_mask, dev_mask, sizeof(double) * mask_size, cudaMemcpyDeviceToDevice));
}


__global__ void
gpu_multiply_complex(
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
gpu_multiply_smooth(
        const size_t input_size,
        const double coeff,
        cufftDoubleComplex *__restrict__ output)
{
    const auto thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const auto total_block_size = blockDim.x * gridDim.x;
    for (auto j = thread_idx; j < input_size / 2 + 1; j += total_block_size) { // Because it is D2Z transform
        const double mult = std::exp(-coeff * double(j) / double(input_size));
        output[j].x *= mult;
        output[j].y *= mult;
    }
}


__global__ void
vec_subtract_inplace(
        double *__restrict__ x,
        const double *__restrict__ y,
        const size_t x_size)
{
    for (auto j = blockIdx.x * blockDim.x + threadIdx.x; j < x_size; j += blockDim.x * gridDim.x) x[j] -= y[j];
}


__global__ void
vec_power(
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
vec_sift(
        const size_t fft_size,
        const size_t siftings,
        const cufftDoubleComplex *__restrict__ x,
        cufftDoubleComplex *__restrict__ imf,
        cufftDoubleComplex *__restrict__ rem)
{
    double px, py;
    for (size_t j = blockIdx.x * blockDim.x + threadIdx.x; j < fft_size; j += blockDim.x * gridDim.x) {
        px = 1. - x[j].x;
        py = -x[j].y;
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
    const size_t g_thr_ix = thr_ix + blockIdx.x * CUDA_BLOCK_SIZE;
    const size_t grid_size = CUDA_BLOCK_SIZE * gridDim.x;
    const auto d_expand_size = double(expand_size);

    double _sum_imf = 0;
    double _sum_rem = 0;
    double _sum_corr = 0;
    // TODO Optimize code here!
    for (size_t i = g_thr_ix; i < expand_size; i += grid_size) {
        double sum1 = 0, sum2 = 0;
        for (size_t j = 0; j < expand_size; ++j) {
            sum1 += d_global_sift_matrix[labs(i - j)] * d_imf_mask[j];
            sum2 += d_global_sift_matrix[labs(i - j)] * d_rem_mask[j];
        }
        _sum_imf += sum1 * d_imf_mask[i] / d_expand_size / d_expand_size;
        _sum_rem += sum2 * d_rem_mask[i] / d_expand_size / d_expand_size;
        _sum_corr += sum1 * d_rem_mask[i] / d_expand_size / d_expand_size;
    }
    __shared__ double _sh_sum_imf[CUDA_BLOCK_SIZE];
    __shared__ double _sh_sum_rem[CUDA_BLOCK_SIZE];
    __shared__ double _sh_sum_corr[CUDA_BLOCK_SIZE];
    _sh_sum_imf[thr_ix] = _sum_imf;
    _sh_sum_rem[thr_ix] = _sum_rem;
    _sh_sum_corr[thr_ix] = _sum_corr;

    __syncthreads();

    for (size_t size = CUDA_BLOCK_SIZE / 2; size > 0; size /= 2) { // uniform
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
    //probably these sizes are not important
    cudaSetDevice(gpu_id);
    cufftHandle plan_forward;
    cufftHandle plan_backward;
    thrust::device_vector<double> d_mask(mask_size);
    thrust::device_vector<double> d_expanded_mask(input_size);
    cudaMemcpy(thrust::raw_pointer_cast(d_mask.data()), h_mask, sizeof(double) * mask_size, cudaMemcpyHostToDevice);
    expand_the_mask(mask_size, input_size, thrust::raw_pointer_cast(d_mask.data()), thrust::raw_pointer_cast(d_expanded_mask.data()));
    thrust::device_vector<cufftDoubleComplex> d_expanded_mask_fft(to_fft_size(input_size));
    thrust::device_vector<cufftDoubleComplex> d_input_fft(to_fft_size(input_size));
    cufftDoubleComplex *dev_expanded_mask_fft = thrust::raw_pointer_cast(d_expanded_mask_fft.data());
    const int n_batch = 1;
    fft_acquire();
    cufft_errchk(cufftPlan1d(&plan_forward, input_size, CUFFT_D2Z, n_batch));
    cufft_errchk(cufftPlan1d(&plan_backward, input_size, CUFFT_Z2D, n_batch));
    cufft_errchk(cufftExecD2Z(plan_forward, thrust::raw_pointer_cast(d_expanded_mask.data()), dev_expanded_mask_fft));
    fft_release();

    cufftDoubleComplex *dev_input_fft = thrust::raw_pointer_cast(d_input_fft.data());
    //cudaDeviceSynchronize();
    for (size_t i = 0; i < siftings; ++i) {
//        cudaSetDevice(gpu_id);
        fft_acquire();
        cufft_errchk(cufftExecD2Z(plan_forward, d_values, dev_input_fft));
        fft_release();
        gpu_multiply_complex<<<CUDA_THREADS_BLOCKS(to_fft_size(input_size)) >>>(input_size, dev_expanded_mask_fft, dev_input_fft);
        cu_errchk(cudaPeekAtLastError());
        //cu_errchk(cudaDeviceSynchronize());
        fft_acquire();
        cufft_errchk(cufftExecZ2D(plan_backward, dev_input_fft, d_temp));
        fft_release();
        vec_subtract_inplace<<<CUDA_THREADS_BLOCKS(input_size)>>>(d_values, d_temp, input_size);
        cu_errchk(cudaPeekAtLastError());
        //cu_errchk(cudaDeviceSynchronize());
    }

    cufftDestroy(plan_forward);
    cufftDestroy(plan_backward);
}


int oemd_coefficients_search::do_filter(const std::vector<cufftDoubleComplex> &h_mask_fft)
{
    for (auto i: h_mask_fft)
        if (std::norm<double>({i.x, i.y}) > norm_thresh)
            return 1;
    return 0;
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

    expand_the_mask(mask_size, expand_size, d_mask, d_expanded_mask_ptr);
    const auto fft_size = to_fft_size(expand_size);
    thrust::device_vector<cufftDoubleComplex> d_fzm_mask(fft_size);
    thrust::device_vector<cufftDoubleComplex> d_mask_imf_fft(fft_size);
    thrust::device_vector<cufftDoubleComplex> d_mask_rem_fft(fft_size);
    //cufftDoubleComplex *d_expanded_mask_fft = thrust::raw_pointer_cast(d_fzm_mask.data());
    //cufft_errchk(cufftExecD2Z(plan_sift_forward, d_expanded_mask_ptr, d_expanded_mask_fft));
    vec_sift<<<CUDA_THREADS_BLOCKS(fft_size)>>>(
            fft_size, siftings, d_expanded_mask_fft, thrust::raw_pointer_cast(d_mask_imf_fft.data()), thrust::raw_pointer_cast(d_mask_rem_fft.data()));
    cufft_errchk(cufftExecZ2D(plan_sift_backward, thrust::raw_pointer_cast(d_mask_imf_fft.data()), thrust::raw_pointer_cast(d_imf_mask.data())));
    cufft_errchk(cufftExecZ2D(plan_sift_backward, thrust::raw_pointer_cast(d_mask_rem_fft.data()), thrust::raw_pointer_cast(d_rem_mask.data())));

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
        const std::vector<double> &global_sift_matrix, const size_t current_level, const size_t gpu_id)
{
    const auto full_input_size = h_workspace.size();
    double result = 0;
    cu_errchk(cudaSetDevice(gpu_id));
    auto dev_values_fft = cuda_malloccopy(values_fft);
    auto d_mask_ptr = cuda_malloccopy(h_mask);
    auto d_global_sift_matrix = cuda_malloccopy(global_sift_matrix);
    const auto d_workspace = cuda_malloccopy(h_workspace);
    const size_t expanded_size = h_mask.size() * C_mask_expander;


    double *d_expanded_mask;
    cu_errchk(cudaMalloc(&d_expanded_mask, expanded_size * sizeof(*d_expanded_mask)));
    expand_the_mask(h_mask.size(), expanded_size, d_mask_ptr, d_expanded_mask);
    cufftDoubleComplex *d_expanded_mask_fft;
    cu_errchk(cudaMalloc(&d_expanded_mask_fft, to_fft_size(expanded_size) * sizeof(*d_expanded_mask_fft)));
    cufft_errchk(cufftExecD2Z(plan_mask_forward, d_expanded_mask, d_expanded_mask_fft));
    auto h_mask_fft = cuda_copy(d_expanded_mask_fft, to_fft_size(expanded_size));

#ifdef FILTEROUT_BAD_MASKS
    if (do_filter(h_mask_fft)) {
        cu_errchk(cudaFree(d_expanded_mask));
        cu_errchk(cudaFree(d_expanded_mask_fft));`
        return 0;
    }
#endif
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

    [[maybe_unused]] const auto [sum_full, sum_imf, sum_rem, sum_corr] =
            sift_the_mask(h_mask.size(), siftings, d_mask_ptr, plan_sift_forward, plan_sift_backward, d_expanded_mask, d_expanded_mask_fft, d_global_sift_matrix, gpu_id);
    cu_errchk(cudaFree(d_global_sift_matrix));
    cu_errchk(cudaFree(d_expanded_mask));
    cu_errchk(cudaFree(d_expanded_mask_fft));

    double *d_zm_ptr;
    cu_errchk(cudaMalloc(&d_zm_ptr, full_input_size * sizeof(*d_zm_ptr)));
    expand_the_mask(h_mask.size(), full_input_size, d_mask_ptr, d_zm_ptr);
    cu_errchk(cudaFree(d_mask_ptr));

    cufftDoubleComplex *d_zm_fft_ptr;
    cu_errchk(cudaMalloc(&d_zm_fft_ptr, to_fft_size(full_input_size) * sizeof(*d_zm_fft_ptr)));
    cufftDoubleComplex *d_zm_convert_ptr;
    cu_errchk(cudaMalloc(&d_zm_convert_ptr, to_fft_size(full_input_size) * sizeof(*d_zm_convert_ptr)))
    cufft_errchk(cufftExecD2Z(plan_expanded_forward, d_zm_ptr, d_zm_fft_ptr));
    vec_power<<<CUDA_THREADS_BLOCKS(to_fft_size(full_input_size))>>>(d_zm_fft_ptr, d_zm_convert_ptr, full_input_size, siftings);
    cu_errchk(cudaFree(d_zm_fft_ptr));
    // auto h_zm = cuda_copy(d_zm_convert_ptr, to_fft_size(full_input_size) * sizeof(*d_zm_convert_ptr));

    gpu_multiply_complex<<<CUDA_THREADS_BLOCKS(to_fft_size(full_input_size))>>>(full_input_size, dev_values_fft, d_zm_convert_ptr);
    cu_errchk(cudaFree(dev_values_fft));
    cufft_errchk(cufftExecZ2D(plan_expanded_backward, d_zm_convert_ptr, d_zm_ptr));
    cu_errchk(cudaFree(d_zm_convert_ptr));

    const size_t inside_window_start = val_start + h_mask.size() * siftings;
    const size_t inside_window_end = full_input_size;
    const size_t inside_window_len = inside_window_end - inside_window_start;
    auto h_imf_temp = cuda_copy(d_zm_ptr + inside_window_start, inside_window_len);
    double *d_values_copy;
    cu_errchk(cudaMalloc(&d_values_copy, inside_window_len * sizeof(*d_values_copy)));
    cu_errchk(cudaMemcpy(d_values_copy, d_workspace + inside_window_start, inside_window_len * sizeof(*d_values_copy), cudaMemcpyDeviceToDevice));
    vec_subtract_inplace<<<CUDA_THREADS_BLOCKS(inside_window_len)>>>(d_values_copy, d_zm_ptr + inside_window_start, inside_window_len);
    auto h_rem_temp = cuda_copy(d_values_copy, inside_window_len);
    cu_errchk(cudaFree(d_zm_ptr));
    cu_errchk(cudaFree(d_values_copy));
    cu_errchk(cudaFree(d_workspace));


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
    const size_t full_count = inside_window_len / 2;
    for (size_t i = full_count; i < inside_window_len; ++i) { // Non-parallelissabile!
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

    if (big_sum2 > big_sum3) result = 1000. + (result + sqrt(big_sum2 / full_count)) * 1000.;
    if (big_sum1 > big_sum3) result = 10. + (result + sqrt(big_sum1 / full_count)) * 100.;

    result = result * (1 + corr);
    if (!current_level) {
        if (corr > .05) result = 1. + result * (10. + corr);
    } else {
        if (corr > .20) result = 1. + result * (10. + corr);
    }

    if (big_sum2 >= 0.95 * big_sum3) result = 30 + (result + sqrt(big_sum2 / full_count)) * 10.;
    if (big_sum2 >= 0.60 * big_sum3) result = result + 5 + (sqrt(big_sum2 / full_count)) * 1.;
    if (big_sum2 >= 0.80 * big_sum3) result = result + 3 + (sqrt(big_sum2 / full_count)) * 1.;
    if (big_sum1 > big_sum2) result = result + 4 + (big_sum1 / big_sum3);
    if (big_sum1 > 0.5 * big_sum3) result = 10 + (result + sqrt(big_sum1 / full_count)) * 10.;
    if (big_sum2 >= 0.99 * big_sum3) result += 1. + big_sum2 / big_sum3;
    if (big_sum2 >= 0.98 * big_sum3) result += 1. + big_sum2 / big_sum3;
    if (big_sum2 >= 0.97 * big_sum3) result += 1. + big_sum2 / big_sum3;
    if (big_sum2 >= 0.96 * big_sum3) result += 1. + big_sum2 / big_sum3;

    result += sqrt((big_sum1 + big_sum2) / big_sum3);
    if (result < best_result.load() && quality <= best_quality.load()) {
        LOG4_DEBUG("Found global best score " << result << " and best quality " << quality);
        best_result.store(result);
        best_quality.store(quality);
    }
    return result;
}


void oemd_coefficients_search::fix_mask(std::vector<double> &mask)
{
    double sum = 0;
    const double padding_value = 1. / double(mask.size());
#pragma omp parallel for reduction(+:sum) default(shared)
    for (size_t i = 0; i < mask.size(); ++i) {
        if (mask[i] < 0) mask[i] = 0;
        if (isnan(mask[i]) || isinf(mask[i])) {
            LOG4_DEBUG("Very bad masks!");
            mask[i] = padding_value;
        }
        sum += mask[i];
    }
    if (sum == 0) {
        LOG4_ERROR("Bad masks!");
#pragma omp parallel for schedule(static, 1 + mask.size() / std::thread::hardware_concurrency())
        for (auto &m: mask) m = padding_value;
    } else {
#pragma omp parallel for schedule(static, 1 + mask.size() / std::thread::hardware_concurrency())
        for (auto &m: mask) m /= sum;
    }
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
    cufft_errchk(cufftExecD2Z(plan_mask_forward, thrust::raw_pointer_cast(d_mask_zm.data()), thrust::raw_pointer_cast(d_mask_zm_fft.data())));
    gpu_multiply_smooth<<<CUDA_THREADS_BLOCKS(to_fft_size(full_size))>>>(full_size, 5. * -log(common::drander(buffer)), thrust::raw_pointer_cast(d_mask_zm_fft.data()));
    cufft_errchk(cufftExecZ2D(plan_mask_backward, thrust::raw_pointer_cast(d_mask_zm_fft.data()), thrust::raw_pointer_cast(d_mask_zm.data())));
    thrust::transform(thrust::device,
                      d_mask_zm.begin(), d_mask_zm.begin() + mask_size, d_mask_zm.begin(),
                      [mask_size] __device__ (const double &iter) -> double { return (iter > 0 ? iter : 0) / double(mask_size); } );
    cuda_copy(mask, thrust::raw_pointer_cast(d_mask_zm.data()), d_mask_zm.size());
}


void
oemd_coefficients_search::smoothen_mask(std::vector<double> &mask, common::t_drand48_data_ptr buffer)
{
    const size_t window_size = 3 + 2. * (mask.size() * common::drander(buffer) / 10.);
    const auto mask_size = mask.size();

    std::vector<double> weights(window_size);
    double wsum = 0;
#pragma omp simd reduction(+:wsum)
    for (size_t i = 0; i < window_size; ++i) {
        weights[i] = exp(-pow((3. * ((double) i - (window_size / 2))) / (double) (window_size / 2), 2) / 2.);
        wsum += weights[i];
    }

    std::vector<double> nmask(mask_size);
#pragma omp simd
    for (size_t i = 0; i < mask_size; ++i) {
        double sum = 0;
        for (size_t j = std::max<size_t>(0, i - window_size / 2); j <= std::min<size_t>(i + window_size / 2, mask_size - 1); ++j)
            sum += weights[window_size / 2 + i - j] * mask[j];
        nmask[i] = sum / wsum;
    }
    mask = nmask;
}


void
oemd_coefficients_search::create_random_mask(
        const size_t position, double step, const size_t mask_size, std::vector<double> &mask, const double *start_mask,
        common::t_drand48_data_ptr buffer, cufftHandle plan_mask_forward, cufftHandle plan_mask_backward, const size_t gpu_id)
{
    step *= common::drander(buffer);
    if (!start_mask) {
//#pragma omp simd
        for (size_t i = 0; i < mask_size; ++i) mask[i] = common::drander(buffer);
    } else {
#pragma omp parallel for default(shared)
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
        const std::vector<double> &global_sift_matrix, const size_t current_level)
{
    svr::optimizer::loss_callback_t loss_function = [&](std::vector<double> &x) -> double {
#ifdef FIX_MASK
        auto y = x;
        fix_mask(y);
#endif

        static std::mutex mx_incr;
        static size_t gl_incr;
        std::unique_lock<std::mutex> ul(mx_incr);
        const size_t l_incr = gl_incr;
        gl_incr = (gl_incr + 1) % C_parallelism;
        const auto devix = gpuids[l_incr % max_gpus];
        ul.unlock();

        static std::deque<std::mutex> mxs(C_parallelism);
        const std::scoped_lock lg(mxs[l_incr]);
        return evaluate_mask(
                siftings, x, h_workspace, h_workspace_fft, valid_start_index,
                plan_full_forward[l_incr], plan_full_backward[l_incr], plan_mask_forward[l_incr], plan_sift_forward[l_incr], plan_sift_backward[l_incr],
                global_sift_matrix, current_level, devix);
    };

    double score;
    std::tie(score, h_mask) = svr::optimizer::firefly(
            h_mask.size(), FIREFLY_PARTICLES, FIREFLY_ITERATIONS, FFA_ALPHA, FFA_BETAMIN, FFA_GAMMA,
            std::vector(h_mask.size(), 0.), std::vector(h_mask.size(), 1.), std::vector(h_mask.size(), 1.),
            loss_function).operator std::tuple<double, std::vector<double>>();
    fix_mask(h_mask);
    return score;
}


// TODO Pass a vector of all available GPU-IDs at the time of call and use them all up
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
#pragma omp simd
    for (size_t i = window_start; i < window_end; ++i)
        h_workspace[i - window_start] = i < tail.size() ? tail[i] : input[i - tail.size()]->get_value(in_colix);

    cu_errchk(cudaSetDevice(gpu_id));
// thrust::host_vector<double> h_temp(window_len);
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
#pragma omp parallel for
    for (size_t i = 0; i < C_parallelism; ++i) {
        CUDA_SET_DEVICE(i);
        cufft_errchk(cufftPlan1d(&plan_full_forward[i], window_len, CUFFT_D2Z, n_batch));
        cufft_errchk(cufftPlan1d(&plan_full_backward[i], window_len, CUFFT_Z2D, n_batch));
    }
    for (size_t i = 0; i < masks.size(); ++i) {
        std::deque<cufftHandle> plan_sift_forward(C_parallelism);
        std::deque<cufftHandle> plan_sift_backward(C_parallelism);
#pragma omp parallel for
        for (size_t j = 0; j < C_parallelism; ++j) {
            CUDA_SET_DEVICE(j);
            cufft_errchk(cufftPlan1d(&plan_sift_forward[j], siftings[i] * masks[i].size(), CUFFT_D2Z, n_batch));
            cufft_errchk(cufftPlan1d(&plan_sift_backward[j], siftings[i] * masks[i].size(), CUFFT_Z2D, n_batch));
        }
        auto global_sift_matrix = fill_auto_matrix(
                masks[i].size(), siftings[i], window_len - start_valid_ix, &h_workspace[window_start + start_valid_ix]);
        std::deque<cufftHandle> plan_mask_forward(C_parallelism);
        std::deque<cufftHandle> plan_mask_backward(C_parallelism);
#pragma omp parallel for
        for (size_t j = 0; j < C_parallelism; ++j) {
            CUDA_SET_DEVICE(j);
            cufft_errchk(cufftPlan1d(&plan_mask_forward[j], C_mask_expander * masks[i].size(), CUFFT_D2Z, n_batch));
            cufft_errchk(cufftPlan1d(&plan_mask_backward[j], C_mask_expander * masks[i].size(), CUFFT_Z2D, n_batch));
        }
        cu_errchk(cudaSetDevice(gpu_id));
        cufft_errchk(cufftExecD2Z(plan_full_forward[0], d_workspace_ptr, d_values_fft_ptr));
        auto h_values_fft = cuda_copy(d_values_fft);
        const double result = find_good_mask_ffly(
                siftings[i], start_valid_ix, h_workspace, h_values_fft,
                masks[i], plan_full_forward, plan_full_backward, plan_mask_forward, plan_sift_forward, plan_sift_backward,
                global_sift_matrix, i);
        LOG4_DEBUG("Level " << i << ", queue " << queue_name << ", FFA score " << result);
        save_mask(masks[i], queue_name, i, masks.size() + 1);
        cu_errchk(cudaSetDevice(gpu_id));
        d_imf[i] = d_workspace; // copy
        double *d_imf_ptr = thrust::raw_pointer_cast(d_imf[i].data());
        transform(d_imf_ptr, masks[i].data(), window_len, masks[i].size(), siftings[i], d_temp_ptr, gpu_id);
        vec_subtract_inplace<<<CUDA_THREADS_BLOCKS(window_len)>>>(d_workspace_ptr, d_imf_ptr, window_len);
        start_valid_ix += siftings[i] * (masks[i].size() - 1); // is it -1 really?
        h_imf[i] = d_imf[i];
/*
        h_temp = d_workspace;
#pragma omp parallel for
        for (size_t j = 0; j < window_len; ++j) h_temp[j] += h_imf[i][j];
*/
        h_workspace = cuda_copy(d_workspace_ptr, d_workspace.size());
        best_result.store(std::numeric_limits<double>::max(), std::memory_order_relaxed);
        best_quality.store(1, std::memory_order_relaxed);
#pragma omp parallel for
        for (size_t j = 0; j < C_parallelism; ++j) {
            CUDA_SET_DEVICE(j);
            cufft_errchk(cufftDestroy(plan_mask_forward[j]));
            cufft_errchk(cufftDestroy(plan_sift_forward[j]));
            cufft_errchk(cufftDestroy(plan_sift_backward[j]));
        }
    }

#pragma omp parallel for
    for (size_t i = 0; i < C_parallelism; ++i) {
        CUDA_SET_DEVICE(i);
        cufft_errchk(cufftDestroy(plan_full_forward[i]));
        cufft_errchk(cufftDestroy(plan_full_backward[i]));
    }
}

} // oemd_search
} // svr
