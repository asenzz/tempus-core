//
// Created by zarko on 10/3/22.
//
#include <cstdlib>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>
#include <queue>
#include <iomanip>
#include <thrust/device_vector.h>
#include <thrust/logical.h>
#include <thrust/execution_policy.h>
#include <cub/cub.cuh>
#include "pprune.hpp"
#include "common/compatibility.hpp"
#include "oemd_coefficient_search.hpp"
#include "online_emd.hpp"
#include "../../SVRCommon/include/common/cuda_util.cuh"
#include "util/time_utils.hpp"
#include "firefly.hpp"
#include "common/logging.hpp"
#include "cuqrsolve.cuh"
#include "align_features.cuh"
#include "onlinesvr.hpp"
#include "ModelService.hpp"

// #define USE_FIREFLY // else use BITEOPT

namespace svr {
namespace oemd {

bool cu_fix_mask(double *const d_mask, const unsigned mask_len, const cudaStream_t custream)
{
    const auto mask_sum = solvers::sum(d_mask, mask_len, custream);
    if (mask_sum == 0.) {
        LOG4_WARN("Zero mask sum.");
        return false;
    }
    datamodel::G_div_inplace<<<CU_BLOCKS_THREADS(mask_len), 0, custream>>>(d_mask, mask_sum, mask_len);
    return true;
}

void fix_mask(CPTR(double) h_in, double *const h_out, const unsigned mask_len, const cudaStream_t custream)
{
    auto d_in = cumallocopy(h_in, mask_len, cudaMemcpyHostToDevice, custream);
    cu_fix_mask(d_in, mask_len, custream);
    cufreecopy(h_out, d_in, custream, mask_len);
}

// This function should be equivalent to align_features.cu:cu_align_features
template<const unsigned block_size> __global__ void G_autocorrelation_sum(
        double *__restrict__ const d_sum, CRPTR(double) x, CRPTR(double) y, const unsigned n_min, const unsigned qt, const unsigned n_qt)
{
    static __shared__ double sh_dist[block_size];
    sh_dist[tid] = 0;
    CU_STRIDED_FOR_i(n_qt) {
        double y_qi = 0;
        const auto to_q = (i + 1) * qt;
#pragma unroll
        for (unsigned qi = i * qt; qi < to_q; ++qi) y_qi += y[qi];
        y_qi /= qt;
        sh_dist[tid] += fabs(x[i] - y_qi) / (fabs(x[i]) + fabs(y_qi));
    }

    __syncthreads();

#define stride_reduce_dist(block_low_)                                                  \
        if (block_size >= block_low_) {                                                 \
            constexpr unsigned stride2 = block_low_ / 2;                                \
            const auto tid_stride2 = tid + stride2;                                     \
            if (tid < stride2 && tid_stride2 < n_min)                                   \
                sh_dist[tid] += sh_dist[tid_stride2];                                   \
            __syncthreads();                                                            \
        }

    stride_reduce_dist(1024);
    stride_reduce_dist(512);
    stride_reduce_dist(256);
    stride_reduce_dist(128);
    if (tid >= 32) return;
    warp_reduce_sum<block_size>(sh_dist, tid, n_min);
    if (tid) return;
    atomicAdd(d_sum, *sh_dist);
}

double autocorrelation(const double *d_x, const double *d_y, const unsigned n, const unsigned n_labels, const unsigned offset, const cudaStream_t stm)
{
    double sum, *d_sum;
    cu_errchk(cudaMallocAsync((void **) &d_sum, sizeof(sum), stm));
    auto best_sum = std::numeric_limits<double>::infinity();
    UNROLL(business::ModelService::C_num_quantizations)
    for (const auto qt: business::ModelService::C_quantizations) {
        cu_errchk(cudaMemsetAsync(d_sum, 0, sizeof(sum), stm));
        const auto q_offset = offset * qt;
        if (n / 2 < q_offset) {
            LOG4_WARN("n " << n << " < q_offset " << q_offset);
            continue;
        }
        const auto d_y_offset = d_y + q_offset;
        const auto n_qt = std::min<unsigned>(n_labels, (n - q_offset) / qt);
        const auto [blocks, threads] = CU_BLOCKS_THREADS_t(n_qt);
        G_autocorrelation_sum<common::C_cu_block_size><<<blocks, threads, 0, stm>>>(d_sum, d_x, d_y_offset, threads, qt, n_qt);
        cu_errchk(cudaMemcpyAsync(&sum, d_sum, sizeof(sum), cudaMemcpyDeviceToHost, stm));
        if (qt == business::ModelService::C_quantizations.back()) cu_errchk(cudaFreeAsync(d_sum, stm));
        cu_errchk(cudaStreamSynchronize(stm));
        sum /= n_qt;
        if (sum < best_sum) best_sum = sum;
    }
    return best_sum;
}

template<const unsigned block_size> __global__ void G_autocorrelation_block(
        double *__restrict__ const d_sum, CRPTR(double) x, CRPTR(double) y, const unsigned n, const unsigned n_min, const float st)
{
    constexpr float sk = 1; // Skip is disabled for now
    static __shared__ double sh_dist[block_size];
    sh_dist[tid] = 0;
    CU_STRIDED_FOR_i(n) {
        const auto y_i = y[STRETCHSKIP_(i)];
        sh_dist[tid] += fabs(x[i] - y_i) / (fabs(x[i]) + fabs(y_i));
    }

    __syncthreads();

    stride_reduce_dist(1024);
    stride_reduce_dist(512);
    stride_reduce_dist(256);
    stride_reduce_dist(128);
    if (tid >= 32) return;
    warp_reduce_sum<block_size>(sh_dist, tid, n_min);
    if (tid) return;
    d_sum[blockIdx.x] = *sh_dist;
}

__global__ void G_autocorr_driver(CRPTR(double) in, CRPTR(double) in_n, CRPTR(unsigned) offsets, double *__restrict__ const res, const unsigned n_offsets)
{
    CU_STRIDED_FOR_i(n_offsets) {
        const auto off = offsets[i];
        const auto blocks = CU_BLOCKS(clamp_n(off));
        const auto threads = CU_THREADS(off);
        auto res_i = (double *) malloc(blocks * sizeof(double));
        UNROLL()
        for (float st = 1; st > C_stretch_limit; st *= C_stretch_multiplier) {
            G_autocorrelation_block < common::C_cu_block_size ><<<blocks, threads>>>(res_i, in, in_n - off, off, threads, st);
            const auto this_res = thrust::reduce(thrust::seq, res_i, res_i + blocks) / off;
            if (this_res < res[i]) res[i] = this_res;
        }
        free(res_i);
    }
}

struct t_op_less {
    template<typename KeyT> __device__ bool operator()(const KeyT lhs, const KeyT rhs) const
    { return lhs < rhs; };
};

double autocorrelation_n(CPTR(double) d_in, const unsigned n, const std::vector<unsigned> &offsets, const cudaStream_t &stm)
{
    const auto d_offsets = cumallocopy(offsets, stm);
    const unsigned n_offsets = offsets.size();
    double *d_res;
    cu_errchk(cudaMallocAsync((void **) &d_res, n_offsets * sizeof(*d_res), stm));
    G_autocorr_driver<<<CU_BLOCKS_THREADS(n_offsets), 0, stm>>>(d_in, d_in + n, d_offsets, d_res, n_offsets);
    cu_errchk(cudaFreeAsync(d_offsets, stm));
#if 0
    cu_errchk(cudaStreamSynchronize(stm));
    thrust::sort(thrust::device, d_res, d_res + n_offsets);
#else
    t_op_less custom_op;
    void *d_temp_storage = nullptr;
    std::size_t temp_storage_bytes = 0;
    cub::DeviceMergeSort::SortKeys(d_temp_storage, temp_storage_bytes, d_res, n_offsets, custom_op, stm);
    cu_errchk(cudaMallocAsync((void **) &d_temp_storage, temp_storage_bytes, stm));
    cub::DeviceMergeSort::SortKeys(d_temp_storage, temp_storage_bytes, d_res, n_offsets, custom_op, stm);
    cu_errchk(cudaFreeAsync(d_temp_storage, stm));
#endif
    const auto n_offsets_2 = n_offsets;
    const auto res = solvers::sum(d_res, n_offsets_2, stm);
    cu_errchk(cudaFreeAsync(d_res, stm));
    return res / n_offsets_2;
}

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
    auto d_imf = cumallocopy(d_values, input_len, cudaMemcpyDeviceToDevice, custream);
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
    const auto fft_size = common::fft_len(expand_size);
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
    const auto full_fft = common::fft_len(full_size);
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

    // fix_mask(mask.data(), mask.data(), mask.size());
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

double oemd_coefficients_search::compute_spectral_entropy_cufft(CPTR(double) d_signal, unsigned N, const cudaStream_t custream)
{
    if (N % sizeof(double)) --N;
    cufftDoubleComplex *d_freq;
    double *d_psd;
    const auto N_2_1 = common::fft_len(N);

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
    double entropy;
    cu_errchk(cudaMemcpyAsync((void **) &entropy, d_entropy, sizeof(double), cudaMemcpyDeviceToHost, custream));

    // Clean up
    cufftDestroy(plan);
    cu_errchk(cudaFreeAsync((void *) d_freq, custream));
    cu_errchk(cudaFreeAsync((void *) d_psd, custream));
    cu_errchk(cudaFreeAsync((void *) d_entropy, custream));
    cu_errchk(cudaStreamSynchronize(custream));

    return entropy < .1 ? common::C_bad_validation : entropy;
}


void cu_normalize(double *const d_in, const unsigned n, const cudaStream_t custream)
{
    const auto mean = solvers::mean(d_in, n, custream);
    if (mean != 0) oemd::G_subtract_inplace<<<CU_BLOCKS_THREADS(n), 0, custream>>>(d_in, mean, n);
    const auto meanabs = solvers::meanabs(d_in, n, custream);
    if (meanabs != 1) datamodel::G_div_inplace<<<CU_BLOCKS_THREADS(n), 0, custream>>>(d_in, meanabs, n);
}

template<typename T> __device__ inline T sinc(const T x)
{
    return x == T(0) ? 1 : sin(M_PI * x) / (M_PI * x);
}

__global__ void G_generate_fir_mask(double *__restrict__ const d_mask, const double f, const unsigned len, const double len_2, const double len_1, const double f_2)
{
    constexpr double alpha0 = .42;
    constexpr double alpha1 = .5;
    constexpr double alpha2 = .08;
    constexpr auto pi_2 = 2 * M_PI;
    constexpr auto pi_4 = 4 * M_PI;

    CU_STRIDED_FOR_i(len)d_mask[i] = /* low-pass */ f_2 * sinc(f_2 * (i - len_2)) * /* blackman */ (alpha0 - alpha1 * cos(pi_2 * i / len_1) - alpha2 * cos(pi_4 * i / len_1));
}

double *generate_fir_mask(const unsigned len, const double f, const cudaStream_t custream)
{
    double *d_mask;
    cu_errchk(cudaMallocAsync(&d_mask, len * sizeof(*d_mask), custream));
    G_generate_fir_mask<<<CU_BLOCKS_THREADS(len), 0, custream>>>(d_mask, f, len, len * .5, len - 1, 2 * f);
    return d_mask;
}

__global__ void G_quantise(CRPTR(double) d_imf, double *const __restrict__ d_imf_q, const unsigned validate_len, const unsigned q_validate_len, const unsigned label_len)
{
    CU_STRIDED_FOR_i(q_validate_len) {
        const auto start = i * label_len;
        const auto end = start + label_len;
        d_imf_q[i] = 0;
        for (auto j = start; j < end; ++j) d_imf_q[i] += d_imf[j];
        d_imf_q[i] /= label_len;
    }
}

double
oemd_coefficients_search::evaluate_mask(
        const double freq, const unsigned mask_len, const std::vector<double> &workspace, const size_t validate_start_ix, const size_t validate_len,
        const unsigned siftings, const std::vector<unsigned int> &offsets, const double meanabs_input, const unsigned gpu_id) const
{
    const auto full_input_len = workspace.size();
    cu_errchk(cudaSetDevice(gpu_id));
    cudaStream_t custream;
    cu_errchk(cudaStreamCreate(&custream));
    auto d_mask = generate_fir_mask(mask_len, freq, custream);
    /*
    if (!cu_fix_mask(d_mask, mask_len, custream)) {
        LOG4_WARN("Zero mask sum.");
        cu_errchk(cudaFreeAsync(d_mask, custream));
        cu_errchk(cudaStreamSynchronize(custream));
        cu_errchk(cudaStreamDestroy(custream));
        return common::C_bad_validation;
    }
     */
    auto d_workspace = cumallocopy(workspace, custream);
    double *d_tmp;
    cu_errchk(cudaMallocAsync((void **) &d_tmp, full_input_len * sizeof(double), custream));
    sift(siftings, full_input_len, mask_len, custream, d_mask, d_workspace, d_tmp);
    cu_errchk(cudaFreeAsync(d_tmp, custream));
    cu_errchk(cudaFreeAsync(d_mask, custream));
    auto d_imf = d_workspace + validate_start_ix;
    const auto meanabs_imf = solvers::meanabs(d_imf, validate_len, custream);
    if (!std::isnormal(meanabs_imf)) {
        LOG4_WARN("Bad IMF " << meanabs_imf);
        cu_errchk(cudaFreeAsync(d_workspace, custream));
        return common::C_bad_validation;
    }
    const auto rel_pow = std::abs(meanabs_input / meanabs_imf - levels + 1.);
#if 0
    const auto autocor = autocorrelation_n(d_imf, validate_len, offsets, custream);
#elif 1
    const unsigned n_offsets = offsets.size();
    std::set<double> correlations;
    cu_errchk(cudaStreamSynchronize(custream));
    const auto n_threads = std::min<unsigned>(n_offsets, cdiv(C_n_cpu, common::gpu_handler_hid::get().get_max_gpu_threads()));
    const auto n_iter = cdiv(n_offsets, n_threads);
    double *d_imf_q;
    const auto q_validate_len = validate_len / label_len;
    cu_errchk(cudaMallocAsync(&d_imf_q, q_validate_len * sizeof(*d_imf_q), custream));
    G_quantise<<<CU_BLOCKS_THREADS(q_validate_len), 0, custream>>>(d_imf, d_imf_q, validate_len, q_validate_len, label_len);
    cu_errchk(cudaStreamSynchronize(custream));
    t_omp_lock ac_l;
#pragma omp parallel for num_threads(n_threads) schedule(static, n_iter) firstprivate(d_imf, d_imf_q, q_validate_len) default(shared)
    for (unsigned i = 0; i < n_offsets; ++i) {
        cu_errchk(cudaSetDevice(gpu_id));
        cudaStream_t custream2;
        cu_errchk(cudaStreamCreate(&custream2));
        const auto ac = autocorrelation(d_imf_q, d_imf, validate_len, q_validate_len, offsets[i], custream2);
        cu_errchk(cudaStreamDestroy(custream2));
        ac_l.set();
        correlations.insert(ac);
        ac_l.unset();
    }
    cu_errchk(cudaFreeAsync(d_imf_q, custream));
    const auto n_offsets_2 = n_offsets;
    const double autocor = std::reduce(C_default_exec_policy, correlations.cbegin(), std::next(correlations.cbegin(), n_offsets_2), 0.,
                                       [](const double sum, const double val) { return sum + val; }) / n_offsets_2;
#else
    const auto autocor = 1;
#endif
    const auto inv_entropy = 1.; // compute_spectral_entropy_cufft(d_imf, validate_len, custream);
    cu_errchk(cudaFreeAsync(d_workspace, custream)); // d_imf is a part of d_workspace
    cu_errchk(cudaStreamSynchronize(custream));
    cu_errchk(cudaStreamDestroy(custream));
    // Weights
    constexpr double autocor_w = 1;
    constexpr double rel_pow_w = 1;
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
        oemd::G_apply_fir<<<CU_BLOCKS_THREADS(full_input_len), 0, custream>>>(
                stretch_coef, d_rx, full_input_len, d_mask, mask_len, mask_len * stretch_coef, d_rx2, 0);
        oemd::G_subtract_inplace<<<CU_BLOCKS_THREADS(full_input_len), 0, custream>>>(d_rx, d_rx2, full_input_len);
    }
}

// Function to calculate the magnitude of complex numbers
__global__ void G_calculate_magnitude(CRPTR(cufftDoubleComplex) freq_domain, double *const magnitudes, const unsigned N)
{
    CU_STRIDED_FOR_i(N) magnitudes[i] = fabs(freq_domain[i].x) + fabs(freq_domain[i].y);
}

// Find n-th broadest and tallest peak in the vector
unsigned find_nth_peak(const std::vector<double> &data, const unsigned n)
{
    LOG4_TRACE("Input  " << common::present(arma::vec(data)));
    std::map<double, size_t, common::safe_double_less> peaks;
    t_omp_lock peak_l;
    // Find all peaks in the vector
    OMP_FOR_i((unsigned) data.size()) {
        double peak_width = 0;
        // Check m neighbors
        for (unsigned j = 1; j < std::abs<int>(i - data.size()); ++j)
            if ((i < j || data[i] > data[i - j]) && (i + j >= data.size() || data[i] > data[i + j])) peak_width += 1;
            else break;
        // If it is a peak, add to the list
        if (peak_width > 0) {
            peak_l.set();
            peaks.emplace(peak_width + data[i], i);
            peak_l.unset();
        }
    }
    if (peaks.empty()) LOG4_THROW("No peaks found, expecting " << n);
    const auto res = n < peaks.size() ? *std::next(peaks.crbegin(), n) : *peaks.crbegin();
    LOG4_DEBUG("Found " << peaks.size() << " peaks, starting " << *peaks.begin() << ", ending " << *peaks.rbegin() << " returning " << n << " before last element " << res);
    return res.second;
}

double dominant_frequency(const std::vector<double> &input, const unsigned level, const cudaStream_t custream)
{
    constexpr double sampling_freq_hz = 1; // 1 Hz sampling frequency
    const unsigned n = input.size() % 2 ? input.size() - 1 : input.size();
    const unsigned fft_n = common::fft_len(n);

    // Allocate device memory
    double *d_signal, *d_magnitudes;
    cufftDoubleComplex *d_freq_domain;
    cu_errchk(cudaMallocAsync(&d_signal, n * sizeof(*d_signal), custream));
    cu_errchk(cudaMallocAsync(&d_magnitudes, fft_n * sizeof(*d_magnitudes), custream));
    cu_errchk(cudaMallocAsync(&d_freq_domain, fft_n * sizeof(*d_freq_domain), custream));

    // Copy signal to device
    cu_errchk(cudaMemcpyAsync(d_signal, input.data(), n * sizeof(*d_signal), cudaMemcpyHostToDevice, custream));

    // Create CUFFT plan
    cufftHandle plan;
    cf_errchk(cufftPlan1d(&plan, n, CUFFT_D2Z, 1));
    cf_errchk(cufftSetStream(plan, custream));
    // Execute the plan
    cf_errchk(cufftExecD2Z(plan, d_signal, d_freq_domain));

    // Calculate magnitudes of the frequency components
    G_calculate_magnitude<<<CU_BLOCKS_THREADS(fft_n), 0, custream>>>(d_freq_domain, d_magnitudes, fft_n);

    // Copy magnitudes back to host
    std::vector<double> magnitudes(fft_n);
    cu_errchk(cudaMemcpyAsync(magnitudes.data(), d_magnitudes, fft_n * sizeof(*d_magnitudes), cudaMemcpyDeviceToHost, custream));
    // Clean up
    cu_errchk(cudaFreeAsync(d_signal, custream));
    cu_errchk(cudaFreeAsync(d_freq_domain, custream));
    cu_errchk(cudaFreeAsync(d_magnitudes, custream));
    cu_errchk(cudaStreamSynchronize(custream));
    cf_errchk(cufftDestroy(plan));
    return find_nth_peak(magnitudes, 0) * sampling_freq_hz / n;
}


double
oemd_coefficients_search::find_mask(
        const unsigned siftings, const unsigned valid_start_ix, const std::vector<double> &workspace, std::vector<double> &mask, const unsigned level) const
{
    const unsigned mask_len = mask.size();
    const unsigned in_window_len = workspace.size() - valid_start_ix;
    const unsigned mask_size = mask_len * sizeof(double);
    const double meanabs_input = std::reduce(C_default_exec_policy, workspace.cbegin() + valid_start_ix, workspace.cend(), 0.,
                                             [](const double sum, const double val) { return sum + std::abs(val); }) / workspace.size();
    std::vector<unsigned> offsets;
    UNROLL()
    for (unsigned i = 1;
        i < in_window_len / business::ModelService::C_max_quantisation / 2 && offsets.size() < common::C_default_kernel_max_chunk_len;
        i += std::max<dtype(i) >(1, float(i) / 2e2 /* ideally should be datamodel::C_default_svrparam_lag_count */))
            offsets.emplace_back(i);
    LOG4_DEBUG("Optimizing mask with " << mask_len << " elements, " << siftings << " siftings, " << in_window_len << " window len, " << workspace.size() <<
               " workspace len, valid start ix " << valid_start_ix << ", level " << level << ", meanabs input " << meanabs_input <<
               ", offsets " << offsets.size() << ", last offset " << offsets.back() << ", max quantisation " << business::ModelService::C_max_quantisation);
    const auto loss_function = [&, mask_len, mask_size, valid_start_ix, siftings, level, meanabs_input]
#ifdef USE_FIREFLY
            (const std::vector<double> &x) {
#else
            (const double *x, double *const f) {
#endif

        common::gpu_context ctx;
#ifdef USE_FIREFLY
        return evaluate_mask(x.front(),
#else
        *f = evaluate_mask(*x, mask_len, workspace, valid_start_ix, in_window_len, siftings, offsets, meanabs_input, ctx.phy_id());
#endif
    };

    common::gpu_context ctx;
    cu_errchk(cudaSetDevice(ctx.phy_id()));
    cudaStream_t custream;
    cu_errchk(cudaStreamCreate(&custream));
    auto freq = dominant_frequency(workspace, 1, custream);
    MAXAS(freq, 1. / in_window_len);
    MINAS(freq, (in_window_len - 1.) / in_window_len);
    arma::vec x0(1, arma::fill::value(freq));
    constexpr double freq_range = 1.5;
    arma::mat bounds(1, 2, arma::fill::none);
    bounds.col(0).fill(std::max(freq / freq_range, 1. / in_window_len));
    bounds.col(1).fill(std::min(freq * freq_range, 1.));
#ifdef USE_FIREFLY
    double score;
    std::tie(score, h_mask) = optimizer::firefly(
            h_mask.size(), particles, iterations, common::C_FFA_alpha, common::C_FFA_betamin, common::C_FFA_gamma, bounds,
            arma::vec(h_mask.size(), arma::fill::ones), loss_function).operator std::pair<double, std::vector<double>>();
    fix_mask(res.best_parameters, mask);
    return score;
#else
    optimizer::pprune opt(0, particles, bounds, loss_function, 20, 0, 0, x0);
    const optimizer::t_pprune_res res = opt;
    cu_errchk(cudaSetDevice(ctx.phy_id()));
    auto d_mask = generate_fir_mask(mask_len, res.best_parameters.front(), custream);
    // cu_fix_mask(d_mask, mask_len, custream);
    cu_errchk(cudaMemcpyAsync(mask.data(), d_mask, mask_size, cudaMemcpyDeviceToHost, custream));
    cu_errchk(cudaFreeAsync(d_mask, custream));
    cu_errchk(cudaStreamSynchronize(custream));
    cu_errchk(cudaStreamDestroy(custream));
    return res.best_score;
#endif
}

void
oemd_coefficients_search::optimize_levels(
        const datamodel::datarow_crange &input,
        const std::vector<double> &tail,
        std::deque<std::vector<double>> &masks,
        std::deque<unsigned> &siftings,
        const unsigned window_start,
        const unsigned window_end,
        const std::string &queue_name,
        const unsigned in_colix,
        const datamodel::t_iqscaler &scaler) const
{
    if (gpuids.empty()) LOG4_THROW("No GPUs found, aborting.");
    const auto gpu_id = gpuids.front();
    const auto window_len = window_end - window_start;
    const auto window_size = window_len * sizeof(double);
    assert(masks.size() == levels - 1);
    assert(in_colix == levels * 2);

    std::vector<double> workspace(window_len);
    OMP_FOR(window_len)
    for (unsigned i = window_start; i < window_end; ++i)
        workspace[i - window_start] = i < tail.size() ? scaler(tail[i]) : scaler(input[i - tail.size()]->at(in_colix));
    LOG4_DEBUG(
            "Optimizing " << masks.size() << " masks for queue " << queue_name << " using " << gpuids.size() << " GPUs, tail len " << tail.size() << ", window len "
              << window_len << ", window start " << window_start << ", window end " << window_end << ", levels " << levels << ", input column index " << in_colix);

    cu_errchk(cudaSetDevice(gpu_id));
    cudaStream_t custream;
    unsigned validation_start_ix = C_fir_mask_end_len;
    cu_errchk(cudaStreamCreate(&custream));
    UNROLL()
    for (unsigned i = 0; i < masks.size(); ++i) {
        const auto mask_len = masks[i].size();
        validation_start_ix += std::min<unsigned>(mask_len * siftings[i], window_size / 2);
        double result;
        const auto level = levels - i - 1;
        PROFILE_EXEC_TIME(result = find_mask(siftings[i], validation_start_ix, workspace, masks[i], level), "Optimizing mask " << i << " for level " << level);
        LOG4_DEBUG("Level " << level << ", mask " << i << ", queue " << queue_name << ", score " << result);
        save_mask(masks[i], queue_name, i, masks.size() + 1);
        cu_errchk(cudaSetDevice(gpu_id));
        auto d_level_imf = cumallocopy(workspace, custream);
        const auto d_mask = cumallocopy(masks[i], custream);
        double *d_workspace;
        cu_errchk(cudaMallocAsync((void **) &d_workspace, window_size, custream));
        transform(d_level_imf, d_mask, window_len, mask_len, siftings[i], d_workspace, custream);
        cu_errchk(cudaFreeAsync(d_mask, custream));
        cu_errchk(cudaMemcpyAsync(d_workspace, workspace.data(), window_size, cudaMemcpyHostToDevice, custream));
        G_subtract_inplace<<<CU_BLOCKS_THREADS(window_len), 0, custream>>>(d_workspace, d_level_imf, window_len);
        cu_errchk(cudaFreeAsync(d_level_imf, custream));
        validation_start_ix += siftings[i] * mask_len;
        cu_errchk(cudaMemcpyAsync(workspace.data(), d_workspace, window_size, cudaMemcpyDeviceToHost, custream));
        cu_errchk(cudaFreeAsync(d_workspace, custream));
        cu_errchk(cudaStreamSynchronize(custream));
    }
    cu_errchk(cudaStreamSynchronize(custream));
    cu_errchk(cudaStreamDestroy(custream));
}

} // oemd_search
} // svr
