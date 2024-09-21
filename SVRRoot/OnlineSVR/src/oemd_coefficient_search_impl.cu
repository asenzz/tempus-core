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
#include <thrust/sort.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <boost/math/special_functions/bessel.hpp>
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
#include "appcontext.hpp"

// #define USE_FIREFLY // else use BITEOPT

namespace svr {
namespace oemd {


namespace {

constexpr unsigned C_column_interleave = datamodel::C_features_superset_coef;

constexpr unsigned C_quantisation_interleave = 3;
#ifdef PRODUCTION_BUILD
constexpr unsigned unroll_ct = CDIVI(datamodel::C_default_svrparam_lag_count * datamodel::C_features_superset_coef / C_column_interleave, 10);
#endif
}

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
        RPTR(double) d_sum, CRPTR(double) x, CRPTR(double) y, const unsigned n_min, const unsigned qt, const unsigned n_qt)
{
    static __shared__ double sh_dist[block_size];
    sh_dist[tid] = 0;
    CU_STRIDED_FOR_i(n_qt) {
        double y_qi = 0;
        const auto to_q = (i + 1) * qt;
        UNROLL()
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

template<const unsigned block_size> __global__ void G_autocorrelation_block(
        RPTR(double) d_sum, CRPTR(double) x, CRPTR(double) y, const unsigned n, const unsigned n_min, const float st)
{
//    constexpr float sk = 1; // Skip is disabled for now
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

__global__ void G_autocorr_driver(CRPTR(double) in, CRPTR(double) in_n, CRPTR(unsigned) offsets, RPTR(double) res, const unsigned n_offsets)
{
    CU_STRIDED_FOR_i(n_offsets) {
        const auto off = offsets[i];
        const auto blocks = CU_BLOCKS(clamp_n(off));
        const auto threads = CU_THREADS(off);
        auto res_i = (double *) malloc(blocks * sizeof(double));
        UNROLL()
        for (float st = 1; st > C_stretch_limit; st *= C_stretch_multiplier) {
            G_autocorrelation_block<common::C_cu_block_size><<<blocks, threads>>>(res_i, in, in_n - off, off, threads, st);
            const auto this_res = thrust::reduce(thrust::seq, res_i, res_i + blocks) / off;
            if (this_res < res[i]) res[i] = this_res;
        }
        free(res_i);
    }
}

double autocorrelation_n(CPTR(double) d_in, const unsigned n, const std::vector<unsigned> &offsets, const cudaStream_t &stm)
{
    const auto d_offsets = cumallocopy(offsets, stm);
    const unsigned n_offsets = offsets.size();
    double *d_res;
    cu_errchk(cudaMallocAsync((void **) &d_res, n_offsets * sizeof(*d_res), stm));
    G_autocorr_driver<<<CU_BLOCKS_THREADS(n_offsets), 0, stm>>>(d_in, d_in + n, d_offsets, d_res, n_offsets);
    cu_errchk(cudaFreeAsync(d_offsets, stm));
    thrust::sort(thrust::cuda::par.on(stm), d_res, d_res + n_offsets);
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
        RPTR(double) d_sum_imf, RPTR(double) d_sum_rem, RPTR(double) d_sum_corr, CRPTR(double) d_imf_mask, CRPTR(double) d_rem_mask,
        const unsigned expand_size, CRPTR(double) d_global_sift_matrix)
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
        CPTR(double) d_mask,
        const cufftHandle plan_sift_forward,
        const cufftHandle plan_sift_backward,
        CPTR(double) d_expanded_mask,
        const cufftDoubleComplex *d_expanded_mask_fft,
        CPTR(double) d_global_sift_matrix_ptr,
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
        const unsigned position, double step, const unsigned mask_size, std::vector<double> &mask, CPTR(double) start_mask,
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
    CU_STRIDED_FOR_i(N_2_1) {
        d_psd[i] = fabs(d_freq[i].x) + fabs(d_freq[i].y);
        atomicAdd(d_psd_sum, d_psd[i]);
    }
}

__global__ void normalize_psd(double *const d_psd, CRPTR(double) d_psd_sum, const unsigned N_2_1)
{
    CU_STRIDED_FOR_i(N_2_1) d_psd[i] /= *d_psd_sum;
}

__global__ void compute_spectral_entropy(CRPTR(double) d_psd, double *const d_entropy, const unsigned N_2_1)
{
    CU_STRIDED_FOR_i(N_2_1) if (d_psd[i] > 0) atomicAdd(d_entropy, -d_psd[i] * log2f(d_psd[i]));
}

double oemd_coefficients_search::compute_spectral_entropy_cufft(double *d_signal, unsigned N, const cudaStream_t custream)
{
    if (N > C_cufft_input_limit) {
        d_signal += N - C_cufft_input_limit;
        N = C_cufft_input_limit;
    } else if (N % 2) {
        ++d_signal;
        --N;
    }
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
    cu_errchk(cudaFreeAsync((void *) d_freq, custream));
    cu_errchk(cudaFreeAsync((void *) d_psd, custream));
    cu_errchk(cudaFreeAsync((void *) d_entropy, custream));
    cu_errchk(cudaStreamSynchronize(custream));
    cufftDestroy(plan);

    return entropy;
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

__global__ void G_generate_fir_mask(RPTR(double) d_mask, const double f, const unsigned len, const double len_2, const double len_1, const double f_2)
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

// TODO Port to CUDA
std::vector<double> lbp_fir(const double As_, const double fp, const double fs_, const double Fs)
{
    constexpr bool Kaiser = false;
    const auto fs = std::min<double>(Fs, fp + fs_);
    const auto As = _MIN(1, As_) * 74;

    // Cut-off frequency
    const double fc = (fs + fp) / 2;

    // Transition band (rad/sec)
    const double Tb = C_pi_2 * (fs - fp) / Fs;
    std::vector<double> w;
    unsigned N;

    // Choice of window function based on stopband attenuation
    if (Kaiser) {
        // Beta estimation
        double beta;
        if (As > 50)
            beta = 0.1102 * (As - 8.7);
        else if (21 < As && As <= 50)
            beta = 0.5842 * std::pow(As - 21, 0.4) + 0.07886 * (As - 21);
        else
            beta = 0;

        // Filter Order
        N = cdiv(As - 8, 2.285 * Tb);

        // Kaiser window function, w[n]
        w.resize(N);
        const auto I0_beta = boost::math::cyl_bessel_i(0., beta);
        const auto N_1 = N + 1;
        OMP_FOR_i(N) w[i] = boost::math::cyl_bessel_i(0., beta * std::sqrt(1. - std::pow((2. * i - N_1) / N, 2))) / I0_beta;
    } else {
        if (As <= 21) { // Rectangular
            N = cdiv(1.8 * M_PI, Tb);
            w = std::vector<double>(N, 1.0); // Rectangular window
        } else if (As > 21 && As <= 26) { // Bartlett
            N = cdiv(6.1 * M_PI, Tb);
            w.resize(N);
            OMP_FOR_i(N) w[i] = 1.0 - std::abs(2.0 * i / (N - 1) - 1.0); // Bartlett window
        } else if (As > 26 && As <= 44) { // Hann
            N = cdiv(6.2 * M_PI, Tb);
            w.resize(N);
            OMP_FOR_i(N) w[i] = 0.5 * (1 - std::cos(C_pi_2 * i / (N - 1))); // Hann window
        } else if (As > 44 && As <= 53) { // Hamming
            N = cdiv(6.6 * M_PI, Tb);
            w.resize(N);
            OMP_FOR_i(N) w[i] = 0.54 - 0.46 * std::cos(C_pi_2 * i / (N - 1)); // Hamming window
        } else if (As > 53) { // Blackman  // && As <= 74
            N = cdiv(11 * M_PI, Tb);
            w.resize(N);
            OMP_FOR_i(N) w[i] = 0.42 - 0.5 * std::cos(C_pi_2 * i / (N - 1)) + 0.08 * std::cos(4 * M_PI * i / (N - 1)); // Blackman window
        }
    }
    if (N > oemd_coefficients_search::C_fir_max_len) {
        LOG4_WARN("FIR mask too long " << N);
        return {};
    }

    // Ideal impulse response of lowpass filter
    const auto alpha = N / 2;
    const auto fc_2 = 2 * fc;
    const auto fc_2_pi = fc * C_pi_2;

    // Multiplying the ideal filter response to window function
    std::vector<double> FIR_lowpass(N);
    OMP_FOR_i(N) FIR_lowpass[i] = (i == alpha ? fc_2 / Fs : std::sin(fc_2_pi / Fs * (i - alpha)) / (M_PI * (i - alpha))) * w[i];

    LOG4_TRACE("Returning " << N << " FIR coefficients.");
    return FIR_lowpass;
}


__global__ void G_quantise_labels_quick(
        CRPTR(double) d_imf, RPTR(double) d_labels, const unsigned validate_len, const unsigned q_validate_len, const unsigned label_len,
        CRPTR(t_label_ix) d_label_ixs, CRPTR(unsigned) ix_end_F)
{
    CU_STRIDED_FOR_i(q_validate_len) {
        double v = 0;
        UNROLL(C_max_label_ixs / 100)
        for (auto j = 0; j < d_label_ixs[i].n_ixs; ++j) v += d_imf[d_label_ixs[i].label_ixs[j]];
        d_labels[i] = v / d_label_ixs[i].n_ixs;
#ifdef EMO_DIFF
        d_labels[i] -= d_imf[ix_end_F[i] - 1];
#endif
    }
}


double
oemd_coefficients_search::evaluate_mask(
        const double att, const double fp, const double fs, const std::vector<double> &workspace, const unsigned validate_start_ix,
        const unsigned validate_len, const unsigned siftings, const double meanabs_input,
        const std::vector<unsigned> &times, const std::vector<t_label_ix> &label_ixs, const std::deque<t_feat_params> &feat_params) const
{
    const auto mask = lbp_fir(att, fp, fs, sample_rate);
    if (mask.empty()) {
        LOG4_WARN("Bad mask for attenuation " << att << ", freq pass " << fp << ", freq stop " << fs);
        return common::C_bad_validation;
    }

    const auto full_input_len = workspace.size();
    const unsigned mask_len = mask.size();
    CTX_CUSTREAM;
    const auto d_mask = cumallocopy(mask, custream);
    const auto d_workspace = cumallocopy(workspace, custream);
    double *d_tmp;
    cu_errchk(cudaMallocAsync((void **) &d_tmp, full_input_len * sizeof(double), custream));
    sift(siftings, full_input_len, mask_len, custream, d_mask, d_workspace, d_tmp);
    cu_errchk(cudaFreeAsync(d_tmp, custream));
    cu_errchk(cudaFreeAsync(d_mask, custream));

    const auto d_imf = d_workspace + validate_start_ix;
#if 1 // Component power
    const auto meanabs_imf = solvers::meanabs(d_imf, validate_len, custream);
    if (!std::isnormal(meanabs_imf)) {
        LOG4_WARN("Bad IMF " << meanabs_imf);
        cu_errchk(cudaFreeAsync(d_workspace, custream));
        cu_sync_destroy(custream);
        return common::C_bad_validation;
    }
    const auto rel_pow = std::abs(meanabs_input / meanabs_imf - levels + 1.); // It was -1 for meanabs_input
#else
    constexpr double meanabs_imf = 1;
    constexpr double rel_pow = 1;
#endif

    double *d_labels, *d_features, *d_scores;
    const unsigned validate_rows = label_ixs.size();
    cu_errchk(cudaMallocAsync(&d_labels, validate_rows * sizeof(*d_labels), custream));
    const auto d_label_ixs = cumallocopy(label_ixs, custream);
    std::vector<unsigned> ix_end_F(validate_rows);
    OMP_FOR_i(validate_rows) ix_end_F[i] = feat_params[i].ix_end;
    const auto d_ix_end_F = cumallocopy(ix_end_F, custream);
    G_quantise_labels_quick<<<CU_BLOCKS_THREADS(validate_rows), 0, custream>>>(d_imf, d_labels, validate_len, validate_rows, label_len, d_label_ixs, d_ix_end_F);
    cu_errchk(cudaFreeAsync(d_label_ixs, custream));
    cu_errchk(cudaFreeAsync(d_ix_end_F, custream));

    constexpr auto full_feat_cols = datamodel::C_features_superset_coef * datamodel::C_default_svrparam_lag_count;
    constexpr auto feat_cols_ileave = full_feat_cols / C_column_interleave;

    auto autocor = common::C_bad_validation;
    const unsigned cols_rows_q = validate_rows * feat_cols_ileave;
    const auto features_size = cols_rows_q * sizeof(*d_features);
    cu_errchk(cudaMallocAsync((void **) &d_features, features_size, custream));
    cu_errchk(cudaMallocAsync((void **) &d_scores, feat_cols_ileave * sizeof(*d_scores), custream));
    std::vector<t_feat_params> feat_params_q(feat_params.size());
    OMP_FOR_i(feat_params.size()) {
        feat_params_q[i].end_time = feat_params[i].end_time;
        feat_params_q[i].ix_end = feat_params[i].ix_end;
    }
    const auto d_times = cumallocopy(times.cbegin() + validate_start_ix, times.cend(), custream);

    UNROLL(business::ModelService::C_num_quantisations / C_quantisation_interleave)
    for (unsigned q = 0; q < business::ModelService::C_num_quantisations; q += C_quantisation_interleave) {
        const auto qt = business::ModelService::C_quantisations[q];
        OMP_FOR_i(feat_params.size()) {
            feat_params_q[i].time_start = feat_params[i].end_time - full_feat_cols * qt;
            feat_params_q[i].ix_start = before_bound(times.cbegin(), times.cend(), feat_params_q[i].time_start) - times.cbegin() - validate_start_ix;
        }
        const auto d_feat_params_q = cumallocopy(feat_params_q, custream);
        const auto interleave_qt = C_column_interleave * qt;
        cu_errchk(cudaMemsetAsync(d_features, 0, features_size, custream));
        G_quantise_features<<<CU_BLOCKS_THREADS(validate_rows), 0, custream>>>(
                d_features, d_imf, d_times, d_feat_params_q, validate_rows, feat_cols_ileave, qt, interleave_qt, interleave_qt * .5);
        cu_errchk(cudaFreeAsync(d_feat_params_q, custream));
        cu_align_features<<<CU_BLOCKS_THREADS(feat_cols_ileave), 0, custream>>>(d_features, d_labels, d_scores, nullptr, nullptr, nullptr, validate_rows, feat_cols_ileave);
        double score;
        if (feat_cols_ileave > datamodel::C_default_svrparam_lag_count) {
            thrust::sort(thrust::cuda::par.on(custream), d_scores, d_scores + feat_cols_ileave);
            score = solvers::sum(d_scores, datamodel::C_default_svrparam_lag_count, custream);
        } else
            score = solvers::sum(d_scores, feat_cols_ileave, custream);
        if (score < autocor) autocor = score;
    }
    cu_errchk(cudaFreeAsync(d_times, custream));
    cu_errchk(cudaFreeAsync(d_scores, custream));
    cu_errchk(cudaFreeAsync(d_features, custream));
    cu_errchk(cudaFreeAsync(d_labels, custream));

    // Spectral entropy
    const auto inv_entropy = 1.; // compute_spectral_entropy_cufft(d_imf, validate_len, custream);
    cu_errchk(cudaFreeAsync(d_workspace, custream)); // d_imf is a part of d_workspace
    cu_sync_destroy(custream);

    // Weights and final score
    constexpr double autocor_w = 1;
    constexpr double rel_pow_w = 2;
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
__global__ void G_calculate_magnitude(CRPTR(cufftDoubleComplex) freq_domain, RPTR(double) magnitudes, const unsigned N)
{
    CU_STRIDED_FOR_i(N) magnitudes[i] = fabs(freq_domain[i].x) + fabs(freq_domain[i].y);
}

// Find n-th percentile broadest and tallest peak in the vector
unsigned find_nth_peak(const std::vector<double> &data, const double n)
{
    // LOG4_DEBUG("Input magnitudes " << common::present(arma::vec(data)));
    std::map<double, size_t, common::safe_double_less> peaks;
    t_omp_lock peak_l;
    // Find all peaks in the vector
    OMP_FOR_i((unsigned) data.size()) {
        double peak_width = 0;
        // Check m neighbors
        for (unsigned j = 1; j < std::abs<int>(i - data.size()); ++j) {
            bool peak_left = false;
            bool peak_right = false;
            if (i < j)
                peak_left = true;
            else if (data[i] > data[i - j]) {
                peak_width += 1;
                peak_left = true;
            }
            if (i + j >= data.size())
                peak_right = true;
            else if (data[i] > data[i + j]) {
                peak_width += 1;
                peak_right = true;
            }
            if (!peak_right || !peak_left) break;
        }
        // If it is a peak, add to the list
        if (peak_width > 0) {
            peak_l.set();
            peaks.emplace(peak_width + data[i], i);
            peak_l.unset();
        }
    }
    if (peaks.empty()) LOG4_THROW("No peaks found.");
    const double size_1 = peaks.size() - 1;
    const auto res = *std::next(peaks.cbegin(), n * size_1);
    LOG4_DEBUG("Found " << peaks.size() << " peaks, starting " << *peaks.cbegin() << ", ending " << *peaks.rbegin() << ", returning " << n << " percentile, " << res);
    return res.second;
}

double oemd_coefficients_search::dominant_frequency(const std::vector<double> &input, const double percentile_greatest_peak, const cudaStream_t custream) const
{
    const unsigned n = input.size() - input.size() % 2;
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
    return find_nth_peak(magnitudes, percentile_greatest_peak) * sample_rate / n;
}


void
oemd_coefficients_search::run(
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
#ifndef EMD_ONLY
    assert(in_colix == levels * 2);
#endif
    std::vector<double> workspace(window_len);
    std::deque<bpt::ptime> times(window_len);
    std::vector<unsigned> times_i(window_len);
    const auto first_time = input.front()->get_value_time();
    const auto first_time_t = bpt::to_time_t(first_time);
    OMP_FOR(window_len)
    for (unsigned i = window_start; i < window_end; ++i) {
        double value;
        bpt::ptime time;
        if (i < tail.size()) {
            value = tail[i];
            time = first_time - resolution * (tail.size() - i);
        } else {
            const auto p_row = input[i - tail.size()];
            value = p_row->at(in_colix);
            time = p_row->get_value_time();
        }
        workspace[i - window_start] = scaler(value);
        times[i - window_start] = time;
        times_i[i - window_start] = bpt::to_time_t(time) - first_time_t;
    }
    // LOG4_TRACE("Tail " << common::present(arma::vec(tail)) << ", workspace " << common::present(arma::vec(workspace)));

    const auto label_duration = label_len * resolution;
    const auto label_times = [&] {
        std::deque<bpt::ptime> r;
        UNROLL(16)
        for (boost::posix_time::ptime it_time(times.front().date(), bpt::hours(times.front().time_of_day().hours()) + onehour);
             it_time < times.back();
             it_time += label_duration)
            r.emplace_back(it_time);
        return r;
    }();
    const auto label_half_duration = label_duration * .5;
    const auto horizon_duration = label_duration * PROPS.get_prediction_horizon();
    std::deque<t_label_ix> label_ixs;
    std::deque<t_feat_params> feat_params;
    const unsigned horizon_samples_1 = label_len * PROPS.get_prediction_horizon() + 1;
    OMP_FOR_(label_times.size(), ordered)
    for (const auto &it_time: label_times) {
        const auto L_start_it = std::lower_bound(times.cbegin(), times.cend(), it_time);
        if (L_start_it == times.cend() || *L_start_it - it_time > label_half_duration) continue;
        const unsigned L_start_ix = L_start_it - times.cbegin();
        if (L_start_ix < max_row_len) continue;

        const auto L_end_time = it_time + label_duration;
        const auto L_end_it = std::lower_bound(L_start_it, std::min(L_start_it + label_len + 1, times.cend()), L_end_time);
        const unsigned L_end_ix = L_end_it - times.cbegin();

        const auto F_time = it_time - horizon_duration;
        auto F_end_it = lower_bound(std::max(L_start_it - horizon_samples_1, times.cbegin()), L_start_it, F_time);
        if (F_end_it == times.cend() || F_end_it == times.cbegin()) continue;
        const unsigned F_end_ix = F_end_it - times.cbegin();
        if (F_end_ix < max_row_len) continue;

        t_label_ix label_ix{label_len};
        const auto this_label_ixs = generate_twap_indexes(times.cbegin(), L_start_it, L_end_it, it_time, L_end_time, resolution, label_len);
        memcpy(label_ix.label_ixs, this_label_ixs.data(), label_len * sizeof(unsigned));
#pragma omp ordered
        {
            label_ixs.emplace_back(label_ix);
            feat_params.emplace_back(t_feat_params{.end_time = unsigned(bpt::to_time_t(F_time) - first_time_t), .ix_end = F_end_ix});
        };
    }
    assert(label_ixs.size() == feat_params.size());
    release_cont(times);

    LOG4_DEBUG(
            "Optimizing " << masks.size() << " masks for queue " << queue_name << " using " << gpuids.size() << " GPUs, tail len " << tail.size() << ", window len "
                          << window_len << ", window start " << window_start << ", window end " << window_end << ", levels " << levels << ", input column index " << in_colix <<
                          ", label ixs " << label_ixs.size() << ", first label last feature ix " << feat_params.front().ix_end);

    cu_errchk(cudaSetDevice(gpu_id));
    cudaStream_t custream;
    cu_errchk(cudaStreamCreateWithFlags(&custream, C_cu_default_stream_flags));
    unsigned validation_start_ix = std::max<unsigned>(tail.size(), C_fir_max_len);
    UNROLL()
    for (unsigned m = 0; m < masks.size(); ++m) {
        const auto level = levels - m - 1;

        const unsigned in_window_len = workspace.size() - validation_start_ix;
        const auto meanabs_input = common::meanabs<double>(workspace.cbegin() + validation_start_ix, workspace.cend());

        const auto min_feat_ix = validation_start_ix + max_row_len;
        unsigned l_start;
        for (l_start = 0; l_start < feat_params.size(); ++l_start)
            if (feat_params[l_start].ix_end >= min_feat_ix) break;
        const auto feat_params_level_len = label_ixs.size() - l_start;
        std::vector<t_label_ix> label_ixs_level(feat_params_level_len);
        std::deque<t_feat_params> feat_params_level(feat_params_level_len);
        OMP_FOR_i(feat_params_level_len) {
            const auto i_l_start = i + l_start;
            t_label_ix label_ix = label_ixs[i_l_start];
            UNROLL(C_max_label_ixs / 100)
            for (auto &lix: label_ix.label_ixs) lix -= validation_start_ix;
            label_ixs_level[i] = label_ix;

            feat_params_level[i] = feat_params[i_l_start];
            feat_params_level[i].ix_end -= validation_start_ix;
        }

        LOG4_DEBUG(
                "Optimizing " << siftings[m] << " siftings, " << in_window_len << " window len, " << workspace.size() << " workspace len, valid start ix " << validation_start_ix <<
              ", level " << level << ", meanabs input " << meanabs_input << ", max quantisation " << business::ModelService::C_max_quantisation << ", level label ixs " <<
              label_ixs_level.size() << ", latest label last feature ix " << feat_params.back().ix_end << ", max row len " << max_row_len);

        const auto loss_function = [&, validation_start_ix, in_window_len, siftings, meanabs_input]
#ifdef USE_FIREFLY
                (const std::vector<double> &x) {
            return
#else
                (const double *x, double *const f) {
            *f =
#endif
                    evaluate_mask(x[0], x[1], x[2], workspace, validation_start_ix, in_window_len, siftings[m], meanabs_input, times_i, label_ixs_level, feat_params_level);
        };
        auto freq = common::constrain(dominant_frequency(workspace, .95, custream), 1. / in_window_len, 1.);
        arma::vec x0(3, arma::fill::none);
        x0[0] = .5;
        x0[1] = freq;
        x0[2] = .01;
        constexpr double freq_range = 1.25;
        arma::mat bounds(3, 2, arma::fill::none);
        bounds(0, 0) = 1e-1; // Min attenuation
        bounds(0, 1) = 1; // Max attenuation
        bounds(1, 0) = std::max(freq / freq_range, 1. / in_window_len); // Min pass frequency
        bounds(1, 1) = std::min(freq * freq_range, 1.); // Max pass frequency
        bounds(2, 0) = 5e-4; // Min frequency stop band
        bounds(2, 1) = 1; // Max frequency stop band
#ifdef USE_FIREFLY
        double score;
        std::tie(score, h_mask) = optimizer::firefly(
                h_mask.size(), particles, iterations, common::C_FFA_alpha, common::C_FFA_betamin, common::C_FFA_gamma, bounds,
                arma::vec(h_mask.size(), arma::fill::ones), loss_function).operator std::pair<double, std::vector<double>>();
        return score;
#else
        optimizer::pprune opt(0, particles, bounds, loss_function, 1, 0, 0, x0);
        const optimizer::t_pprune_res res = opt;
        masks[m] = lbp_fir(res.best_parameters[0], res.best_parameters[1], res.best_parameters[2], sample_rate);
        if (masks[m].empty()) LOG4_THROW("Bad mask for parameters " << res.best_parameters);
#endif
        const auto mask_len = masks[m].size();
        LOG4_DEBUG("Level " << level << ", mask " << m << ", queue " << queue_name << ", score " << res.best_score);
        save_mask(masks[m], queue_name, m, masks.size() + 1);
        cu_errchk(cudaSetDevice(gpu_id));
        auto d_level_imf = cumallocopy(workspace, custream);
        const auto d_mask = cumallocopy(masks[m], custream);
        double *d_workspace;
        cu_errchk(cudaMallocAsync((void **) &d_workspace, window_size, custream));
        transform(d_level_imf, d_mask, window_len, mask_len, siftings[m], d_workspace, custream);
        cu_errchk(cudaFreeAsync(d_mask, custream));
        cu_errchk(cudaMemcpyAsync(d_workspace, workspace.data(), window_size, cudaMemcpyHostToDevice, custream));
        G_subtract_inplace<<<CU_BLOCKS_THREADS(window_len), 0, custream>>>(d_workspace, d_level_imf, window_len);
        cu_errchk(cudaFreeAsync(d_level_imf, custream));
        validation_start_ix += siftings[m] * mask_len;
        cu_errchk(cudaMemcpyAsync(workspace.data(), d_workspace, window_size, cudaMemcpyDeviceToHost, custream));
        cu_errchk(cudaFreeAsync(d_workspace, custream));
        cu_errchk(cudaStreamSynchronize(custream));
    }
    cu_errchk(cudaStreamDestroy(custream));
}

} // oemd_search
} // svr
