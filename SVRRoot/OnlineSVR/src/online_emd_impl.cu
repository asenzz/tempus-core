//
// Created by zarko on 2/16/22.
//

#include <vector>
#include <cassert>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <common/cuda_util.cuh>
#include <cufft.h>

#include "online_emd_impl.cuh"



namespace svr {
namespace cuoemd {

#if 0

__global __ void
reduce_mask_trim(
        const size_t apply_ix,
        const double *masks,
        const double *rx,
        const double *x,
)
{
    rx[ix] += masks[size_t(mask_size - 1 - ix / stretch_coef + j / stretch_coef)] * x[j];
    sum += masks[size_t(mask_size - 1 - ix / stretch_coef + j / stretch_coef)];
}

__global__ void
reduce_mask(
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

__global__ void
apply_mask(
        const double stretch_coef,
        double *__restrict__ x,
        const size_t x_size,
        const double *__restrict__ mask,
        const size_t mask_size,
        double *rx)
{
    const size_t ix = blockIdx.x * blockDim.x + threadIdx.x;

    if (ix >= x_size) return;

    rx[ix] = 0;
    if (ix >= stretch_coef * mask_size - 1) {
        for (size_t m = 0; m < mask_size * stretch_coef; ++m)
            rx[ix] += mask[size_t(m / stretch_coef)] * x[size_t(ix - stretch_coef * mask_size + m + 1)] / stretch_coef;
    } else {
        double sum = 0;
        for (size_t j = 0; j <= ix; ++j) {
            rx[ix] += mask[size_t(mask_size - 1 - ix / stretch_coef + j / stretch_coef)] * x[j] / stretch_coef;
            sum += mask[size_t(mask_size - 1 - ix / stretch_coef + j / stretch_coef)] / stretch_coef;
        }
        rx[ix] /= sum;
    }
}


//modelled along the previous, but using expanded masks and cumsum
//order of expanded masks is inverted to order of masks
__global__ void
apply_mask_initial(
        double *__restrict__ x,
        const size_t x_size,
        const double *__restrict__ expanded_mask,
        const double *__restrict__ expanded_mask_cumsum,
        const size_t mask_size,
        double *__restrict__ rx)
{
    const size_t ix = blockIdx.x * blockDim.x + threadIdx.x;
    if (ix >= x_size) return;

    rx[ix] = 0;
    if (ix >= mask_size - 1) {
        const double mask_normalizer = expanded_mask_cumsum[mask_size - 1];
        double sum = 0;
        for (size_t m = 0; m < mask_size; ++m)
            sum += expanded_mask[m] * x[size_t(ix - m)] / mask_normalizer;
        rx[ix] += sum;
    } else {
        for (size_t j = 0; j <= ix; ++j)
            rx[ix] += expanded_mask[ix - j] * x[j];
        rx[ix] /= expanded_mask_cumsum[ix];
    }
}


__global__ void
vec_subtract_inplace(
        double *__restrict__ x,
        const double *__restrict__ y,
        const size_t x_size)
{
    const size_t ix = blockIdx.x * blockDim.x + threadIdx.x;
    if (ix < x_size) x[ix] -= y[ix];
}


__global__ void
gpu_expand_mask(
        const size_t mask_size,
        const double *__restrict__ mask,
        const int stretch_coef,
        double *expanded_mask)
{
    const size_t all_work = (size_t) (mask_size * stretch_coef);
    const size_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total_block_size = blockDim.x * gridDim.x;

    for (size_t j = thread_idx; j < all_work; j += total_block_size) {
        const size_t mask_idx = size_t(j) / size_t(stretch_coef);
#ifdef USE_VAR_STRETCH
        size_t stretch_idx =  j % stretch_coef;
                        const double lambda = 1.;
                        double this_stretch_multiplier = exp(-lambda * (double)stretch_idx/(double)stretch_coef);
                        expanded_mask[j] = masks[mask_size - 1 - mask_idx]*this_stretch_multiplier;
#else
        expanded_mask[j] = mask[mask_size - 1 - mask_idx] / double(stretch_coef); //could be without stretch_coef, using the cumsum
#endif
    }
}

__global__ void
gpu_cumsum_mask(
        const size_t expanded_mask_size,
        const double *__restrict__ expanded_mask,
        double *expanded_mask_cumsum)
{
    //cumsum using only 1 block!
    __shared__ double s_cumsum_data[CUDA_BLOCK_SIZE];
    const size_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    double carry_sum = 0;
    for (size_t i = thread_idx; i < expanded_mask_size; i += CUDA_BLOCK_SIZE) {
        s_cumsum_data[thread_idx] = expanded_mask[i];
        __syncthreads();
        for (unsigned s = 1; s < blockDim.x; s *= 2) {
            if (thread_idx & s)
                s_cumsum_data[threadIdx.x] += s_cumsum_data[(threadIdx.x & ~(s - 1)) - 1];
            __syncthreads();
        }
        expanded_mask_cumsum[i] = carry_sum + s_cumsum_data[thread_idx];
        carry_sum += s_cumsum_data[CUDA_BLOCK_SIZE - 1];
    }
}


void
expand_the_mask(
        const size_t stretch_coef,
        const size_t mask_size,
        const size_t input_size,
        const double *dev_mask,
        double *&expanded_mask)
{
    const size_t expanded_size = std::max(input_size, mask_size * stretch_coef);////assume input_size bigger than mask_size*stretch_coef
    expanded_mask = (double *) cuda_calloc(expanded_size, sizeof(double));
    cu_errchk(cudaMemset(expanded_mask, 0, sizeof(double) * expanded_size));
    //probably exact fixed sizes not important
    gpu_expand_mask<<<CUDA_THREADS_BLOCKS(expanded_size)>>>(mask_size, dev_mask, stretch_coef, expanded_mask);
    cu_errchk(cudaPeekAtLastError());
    //cu_errchk(cudaDeviceSynchronize());
}


void
cumsum_the_mask(
        const size_t stretch_coef,
        const size_t mask_size,
        const double *expanded_mask,
        double *&expanded_mask_cumsum)
{
    expanded_mask_cumsum = (double *) cuda_calloc(mask_size * stretch_coef, sizeof(double));
    gpu_cumsum_mask<<<1 /* Must be 1! */, CUDA_BLOCK_SIZE>>>(mask_size * stretch_coef, expanded_mask, expanded_mask_cumsum);
    cu_errchk(cudaPeekAtLastError());
    //cu_errchk(cudaDeviceSynchronize());
}


__global__ void
gpu_multiply_complex(
    const size_t input_size,
    const size_t expanded_mask_size,
    const cufftDoubleComplex *__restrict__ multiplier,
    cufftDoubleComplex *__restrict__ output,
    const double *__restrict__ dev_expanded_mask_cumsum)
{
    const size_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total_block_size = blockDim.x * gridDim.x;
    for (size_t j = thread_idx; j < (input_size / 2 + 1); j += total_block_size) { // because it is D2Z transform
        cufftDoubleComplex new_output;
        new_output.x = output[j].x * multiplier[j].x - output[j].y * multiplier[j].y;
        new_output.y = output[j].x * multiplier[j].y + output[j].y * multiplier[j].x;
        output[j].x = new_output.x / double(input_size) / dev_expanded_mask_cumsum[expanded_mask_size - 1]; // because of inverse fft
        output[j].y = new_output.y / double(input_size) / dev_expanded_mask_cumsum[expanded_mask_size - 1];
    }
}


void do_multiply(
    const size_t input_size,
    const size_t expanded_mask_size,
    const cufftDoubleComplex *dev_expanded_mask_fft,
    cufftDoubleComplex *dev_input_fft,
    double *dev_expanded_mask_cumsum)
{
    gpu_multiply_complex<<<(input_size + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE, CUDA_BLOCK_SIZE>>>(input_size, expanded_mask_size, dev_expanded_mask_fft, dev_input_fft, dev_expanded_mask_cumsum);
    cu_errchk(cudaPeekAtLastError());
    //cu_errchk(cudaDeviceSynchronize());

}


void prepare_mask(
    const int stretch_coef,
    const size_t input_size,
    const size_t mask_size,
    const double *dev_mask,
    double *&dev_expanded_mask,
    double *&dev_expanded_mask_cumsum)
{
    expand_the_mask(stretch_coef, mask_size, input_size, dev_mask, dev_expanded_mask);
    cumsum_the_mask(stretch_coef, mask_size, dev_expanded_mask, dev_expanded_mask_cumsum);
}

void apply_mask_via_fft(
    const int stretch_coef,
    const size_t input_size,
    const size_t mask_size,
    const double *dev_mask,
    double *dev_input,
    double *dev_output,
    double *dev_expanded_mask,
    double *dev_expanded_mask_cumsum)
{

    const size_t expanded_size = mask_size * stretch_coef;

    cufftHandle plan_forward;
    cufftHandle plan_backward;

    cufftDoubleComplex *dev_expanded_mask_fft, *dev_input_fft;
    cu_errchk(cudaMalloc((void **) &dev_expanded_mask_fft, sizeof(cufftDoubleComplex) * (input_size / 2 + 1)));
    cu_errchk(cudaMalloc((void **) &dev_input_fft, sizeof(cufftDoubleComplex) * (input_size / 2 + 1)));
    const int n_batch = 1;


    cufft_errchk(cufftPlan1d(&plan_forward, input_size, CUFFT_D2Z, n_batch));
    cufft_errchk(cufftPlan1d(&plan_backward, input_size, CUFFT_Z2D, n_batch));

    //cudaDeviceSynchronize();
    cufft_errchk(cufftExecD2Z(plan_forward, dev_expanded_mask, dev_expanded_mask_fft));
    cufft_errchk(cufftExecD2Z(plan_forward, dev_input, dev_input_fft));
    do_multiply(input_size, expanded_size, dev_expanded_mask_fft, dev_input_fft, dev_expanded_mask_cumsum);

    cufft_errchk(cufftExecZ2D(plan_backward, dev_input_fft, dev_output));
    //TODO probably multiplied by the wrong constant now. must be corrected somewhere, maybe even on the host

    //cudaDeviceSynchronize();

    cufftDestroy(plan_forward);
    cufftDestroy(plan_backward);
    cu_errchk(cudaFree(dev_expanded_mask_fft));
    cu_errchk(cudaFree(dev_input_fft))
}


double get_skewed_stretch_coef(const double actual_l, const double last_mask_size, const double mask_size, const double stretch_coef)
{
    const auto skewed_stretch_coef = stretch_coef; // * std::max(1., 1. + (MAX_SKEW_STRETCH_COEF / actual_l) * (last_mask_size / mask_size - 1.));
    std::cout << "Level " << actual_l << ", masks size " << mask_size << ", stretch coefficient " << stretch_coef << ", skewed stretch coefficient (not used) " << skewed_stretch_coef << " stretched masks size " << mask_size * stretch_coef << std::endl;
    return stretch_coef;
}


void transform_fft(
    const std::vector<double> &input,
    std::vector<std::vector<double>> &decon,
    const std::vector<size_t> &siftings,
    const std::vector<std::vector<double>> &mask,
    const double stretch_coef,
    const size_t levels)
{
    auto p_remainder = cuda_malloccopy(input);
    auto p_rx = cuda_malloccopy(input);
    auto p_rx2 = (double *) cuda_calloc(input.size(), sizeof(double));
    if (decon.size() != input.size()) decon.resize(input.size());
    for (size_t l = 0; l < levels - 1; l++) {
        const size_t actual_l = levels - l - 1;
        double *dev_mask = cuda_malloccopy(mask[l]);
        double *dev_expanded_mask;
        double *dev_expanded_mask_cumsum;
        prepare_mask(stretch_coef, input.size(), mask[l].size(), dev_mask, dev_expanded_mask, dev_expanded_mask_cumsum);
        for (size_t s = 0; s < siftings[l]; ++s) {
            cu_errchk(cudaMemset(p_rx2, 0, sizeof(double) * input.size()));
            apply_mask_via_fft(get_skewed_stretch_coef(actual_l, mask.back().size(), mask[l].size(), stretch_coef), input.size(),  mask[l].size(), dev_mask, p_rx, p_rx2, dev_expanded_mask, dev_expanded_mask_cumsum);
                                        //fix the starting mask_size*stretch_coef elements using the slower algorithm
                                        cu_errchk(cudaMemset (p_rx2, 0, sizeof(double) * mask[l].size() * stretch_coef));
                                        //apply_mask_middle<<<CUDA_THREADS_BLOCKS_MIDDLE(masks[l].size()*stretch_coef)>>>(p_rx,  masks[l].size()*stretch_coef, dev_expanded_mask,dev_expanded_mask_cumsum, masks[l].size()*stretch_coef, p_rx2);
                                        apply_mask_initial<<<CUDA_THREADS_BLOCKS(mask[l].size()*stretch_coef)>>>(p_rx, mask[l].size()*stretch_coef, dev_expanded_mask,dev_expanded_mask_cumsum,mask[l].size(), p_rx2);

#ifdef CUDA_OUTPUT_FIR
            if (s == 0) {
//                if (stretch_coef > 600) std::this_thread::sleep_for(std::chrono::minutes(10)); // Sleep to cool down GPU
                std::vector<double> p_host_rx = cuda_copy(p_rx, input.size(), cudaMemcpyDeviceToHost);
                CU_LOG4_FILE(SAVE_OUTPUT_LOCATION << "oemd_level_" << actual_l << "_in.csv", deep_to_string(p_host_rx));
                std::vector<double> p_host_rx2 = cuda_copy(p_rx2, input.size(), cudaMemcpyDeviceToHost);
                CU_LOG4_FILE(SAVE_OUTPUT_LOCATION << "oemd_level_" << actual_l << "_out.csv", deep_to_string(p_host_rx2));
            }
#endif
            vec_subtract_inplace<<<CUDA_THREADS_BLOCKS(input.size())>>>(p_rx, p_rx2, input.size());
        }
        cu_errchk(cudaFree(dev_expanded_mask));
        cu_errchk(cudaFree(dev_expanded_mask_cumsum));

        auto host_rx = cuda_malloccopy(p_rx, input.size() * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost);
        for (size_t t = 0; t < input.size(); ++t) {
            if (decon[t].size() != levels) decon[t].resize(levels);
            decon[t][actual_l] = host_rx[t];
        }
        free(host_rx);
        vec_subtract_inplace<<<CUDA_THREADS_BLOCKS(input.size())>>>(p_remainder, p_rx, input.size());
        cu_errchk(cudaMemcpy(p_rx, p_remainder, input.size() * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToDevice));
        cu_errchk(cudaFree(dev_mask));
    }
    {

        double *p_host_rx = cuda_malloccopy(p_rx, input.size() * sizeof(double), cudaMemcpyDeviceToHost);
        for (size_t t = 0; t < input.size(); ++t) decon[t][0] = p_host_rx[t];
        free(p_host_rx);
    }
    cu_errchk(cudaFree(p_rx2));
    cu_errchk(cudaFree(p_rx));
    cu_errchk(cudaFree(p_remainder));
}


void transform_fir(
        const size_t gpu_id,
        const std::vector<double> &input,
        std::vector<std::vector<double>> &decon,
        const std::vector<size_t> &siftings,
        const std::vector<std::vector<double>> &mask,
        const double stretch_coef,
        const size_t levels)
{
    cu_errchk(cudaSetDevice(gpu_id));
    auto p_remainder = cuda_malloccopy(input);
    auto p_rx = cuda_malloccopy(input);
    auto p_rx2 = (double *) cuda_calloc(input.size(), sizeof(double));
    if (decon.size() != input.size()) decon.resize(input.size());
    for (size_t l = 0; l < levels - 1; l++) {
        const size_t actual_l = levels - l - 1;
        double *dev_mask = cuda_malloccopy(mask[l]);
        for (size_t s = 0; s < siftings[l]; ++s) {
            cu_errchk(cudaMemset(p_rx2, 0, sizeof(double) * input.size()));
            apply_mask<<<CUDA_THREADS_BLOCKS(input.size())>>>(get_skewed_stretch_coef(actual_l, mask.back().size(), mask[l].size(), stretch_coef), p_rx, input.size(), dev_mask, mask[l].size(), p_rx2);

#ifdef CUDA_OUTPUT_FIR
            if (s == 0) {
//                if (stretch_coef > 600) std::this_thread::sleep_for(std::chrono::minutes(10)); // Sleep to cool down GPU
                std::vector<double> p_host_rx = cuda_copy(p_rx, input.size(), cudaMemcpyDeviceToHost);
                CU_LOG4_FILE(SAVE_OUTPUT_LOCATION << "oemd_level_" << actual_l << "_in.csv", deep_to_string(p_host_rx));
                std::vector<double> p_host_rx2 = cuda_copy(p_rx2, input.size(), cudaMemcpyDeviceToHost);
                CU_LOG4_FILE(SAVE_OUTPUT_LOCATION << "oemd_level_" << actual_l << "_out.csv", deep_to_string(p_host_rx2));
            }
#endif
            vec_subtract_inplace<<<CUDA_THREADS_BLOCKS(input.size())>>>(p_rx, p_rx2, input.size());
        }

        auto host_rx = cuda_malloccopy(p_rx, input.size() * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToHost);
        for (size_t t = 0; t < input.size(); ++t) {
            if (decon[t].size() != levels) decon[t].resize(levels);
            decon[t][actual_l] = host_rx[t];
        }
        free(host_rx);
        vec_subtract_inplace<<<CUDA_THREADS_BLOCKS(input.size())>>>(p_remainder, p_rx, input.size());
        cu_errchk(cudaMemcpy(p_rx, p_remainder, input.size() * sizeof(double), cudaMemcpyKind::cudaMemcpyDeviceToDevice));
        cu_errchk(cudaFree(dev_mask));
    }
    {

        double *p_host_rx = cuda_malloccopy(p_rx, input.size() * sizeof(double), cudaMemcpyDeviceToHost);
        for (size_t t = 0; t < input.size(); ++t) decon[t][0] = p_host_rx[t];
        free(p_host_rx);
    }
    cu_errchk(cudaFree(p_rx2));
    cu_errchk(cudaFree(p_rx));
    cu_errchk(cudaFree(p_remainder));
}


void transform(
        const std::vector<double> &input,
        std::vector<std::vector<double>> &decon,
        const int gpu_id,
        const std::vector<size_t> &siftings,
        const std::vector<std::vector<double>> &mask,
        const double stretch_coef,
        const size_t levels)
{
    transform_fir(gpu_id, input, decon, siftings, mask, stretch_coef, levels);
}


} // namespace cuoemd
} // namespace svr