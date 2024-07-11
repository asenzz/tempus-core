#include <thrust/async/reduce.h>
#include <npp.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/async/reduce.h>

#include <cmath>
#include <thread>
#include <cublas_v2.h>
#include <magma_types.h>
#include <magma_v2.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cusolverDn.h>
#include "common/compatibility.hpp"
#include "common/gpu_handler.tpp"
#include "common/cuda_util.cuh"
#include "cuqrsolve.cuh"
#include "common/constants.hpp"
#include "onlinesvr.hpp"
#include "thrust/detail/extrema.inl"
#include "cuda_path.hpp"

namespace svr {
namespace solvers {


void __global__
G_score_kernel(
        double *score,
        const double *__restrict kernel, const double *__restrict ref,
        const size_t M, const double norm_ker, const double norm_ref)
{
    const size_t thr_ix = threadIdx.x;
    const size_t g_thr_ix = thr_ix + blockIdx.x * common::C_cu_block_size;
    const size_t grid_size = common::C_cu_block_size * gridDim.x;

    double sum = 0;
    for (size_t i = g_thr_ix; i < M; i += grid_size) sum += kernel[i] * ref[i];

    __shared__ double _sh_sum[common::C_cu_block_size];
    _sh_sum[thr_ix] = sum;
    __syncthreads();

    for (size_t size = common::C_cu_block_size / 2; size > 0; size /= 2) {
        if (thr_ix >= size) continue;
        _sh_sum[thr_ix] += _sh_sum[thr_ix + size];
        __syncthreads();
    }

    if (thr_ix == 0) *score = _sh_sum[0] / (norm_ker * norm_ref);
}

template<const unsigned k_block_size> __global__ void
G_kernel_from_distances_symm(double *__restrict K, const double *__restrict dist, const size_t mm, const size_t m, const double divisor)
{
    const size_t g_thr_ix = threadIdx.x + blockIdx.x * k_block_size;
    if (g_thr_ix >= mm) return;
    const size_t col = g_thr_ix / m;
    K[g_thr_ix + col] = K[((g_thr_ix + col) % m) * m + col] = 1. - dist[g_thr_ix] / divisor;
}

__global__ void
G_kernel_from_distances_symm(double *__restrict K, const double *__restrict dist, const size_t mm, const size_t m, const double divisor)
{
    const size_t g_thr_ix = threadIdx.x + blockIdx.x * blockDim.x;
    if (g_thr_ix >= mm) return;
    const size_t col = g_thr_ix / m;
    K[g_thr_ix + col] = K[((g_thr_ix + col) % m) * m + col] = 1. - dist[g_thr_ix] / divisor;
}

void kernel_from_distances_symm(double *K, const double *Z, const size_t m, const double gamma)
{
    LOG4_THROW("Buggy.");

    double *d_K, *d_Z;
    const auto mm = m * m;
    const auto mat_size = mm * sizeof(double);
    const common::gpu_context ctx;
    cu_errchk(cudaSetDevice(ctx.phy_id()));
    cu_errchk(cudaMalloc((void **) &d_K, mat_size));
    cu_errchk(cudaMalloc((void **) &d_Z, mat_size));
    cu_errchk(cudaMemcpy(d_Z, Z, mat_size, cudaMemcpyHostToDevice));
    const auto half_mm = mm / 2;
    G_kernel_from_distances_symm<<<CUDA_THREADS_BLOCKS(half_mm)>>>(d_K, d_Z, mm / 2, m, gamma);
    cu_errchk(cudaDeviceSynchronize());
    cu_errchk(cudaMemcpy(K, d_K, mat_size, cudaMemcpyDeviceToHost));
    cu_errchk(cudaFree(d_K));
    cu_errchk(cudaFree(d_Z));
}

__global__ void
G_kernel_from_distances(double *__restrict__ K, const double *__restrict__ Z, const size_t mn, const double divisor)
{
    CUDA_STRIDED_FOR_i(mn) K[i] = 1. - Z[i] / divisor;
}

__global__ void
G_kernel_from_distances_inplace(double *__restrict__ Kz, const size_t mn, const double divisor)
{
    CUDA_STRIDED_FOR_i(mn) Kz[i] = 1. - Kz[i] / divisor;
}

// K = 1 - Z / (2 * gamma * gamma)
void kernel_from_distances(double *K, const double *Z, const size_t m, const size_t n, const double gamma)
{
    double *d_K, *d_Z;
    const size_t mn = m * n;
    const size_t mat_size = mn * sizeof(double);
    const common::gpu_context ctx;
    cu_errchk(cudaSetDevice(ctx.phy_id()));
    cudaStream_t cu_stream;
    cu_errchk(cudaStreamCreateWithFlags(&cu_stream, cudaStreamNonBlocking));
    cu_errchk(cudaMallocAsync((void **) &d_Z, mat_size, cu_stream));
    cu_errchk(cudaMemcpyAsync(d_Z, Z, mat_size, cudaMemcpyHostToDevice, cu_stream));
    cu_errchk(cudaMallocAsync((void **) &d_K, mat_size, cu_stream));
    G_kernel_from_distances<<<CUDA_THREADS_BLOCKS(mn), 0, cu_stream>>>(d_K, d_Z, mn, DIST(gamma));
    cu_errchk(cudaFreeAsync(d_Z, cu_stream));
    cu_errchk(cudaMemcpyAsync(K, d_K, mat_size, cudaMemcpyDeviceToHost, cu_stream));
    cu_errchk(cudaFreeAsync(d_K, cu_stream));
    cu_errchk(cudaStreamSynchronize(cu_stream));
    cu_errchk(cudaStreamDestroy(cu_stream));
}

void kernel_from_distances_inplace(double *Kz, const size_t m, const size_t n, const double gamma)
{
    double *d_Kz;
    const size_t mn = m * n;
    const size_t mat_size = mn * sizeof(double);
    const common::gpu_context ctx;
    cu_errchk(cudaSetDevice(ctx.phy_id()));
    cudaStream_t cu_stream;
    cu_errchk(cudaStreamCreateWithFlags(&cu_stream, cudaStreamNonBlocking));
    cu_errchk(cudaMallocAsync((void **) &d_Kz, mat_size, cu_stream));
    cu_errchk(cudaMemcpyAsync(d_Kz, Kz, mat_size, cudaMemcpyHostToDevice, cu_stream));
    G_kernel_from_distances_inplace<<<CUDA_THREADS_BLOCKS(mn), 0, cu_stream>>>(d_Kz, mn, DIST(gamma));
    cu_errchk(cudaMemcpyAsync(Kz, d_Kz, mat_size, cudaMemcpyDeviceToHost, cu_stream));
    cu_errchk(cudaFreeAsync(d_Kz, cu_stream));
    cu_errchk(cudaStreamSynchronize(cu_stream));
    cu_errchk(cudaStreamDestroy(cu_stream));
}

#if 0
void kernel_from_distances_inplace(double *Kz, const size_t m, const size_t n, const double gamma)
{
    double *d_Kz;
    const size_t mn = m * n;
    const size_t mat_size = mn * sizeof(double);
    const common::gpu_context ctx;
    cudaStream_t cu_stream;
    cu_errchk(cudaSetDevice(ctx.phy_id()));
    cu_errchk(cudaStreamCreateWithFlags(&cu_stream, cudaStreamNonBlocking));
    cu_errchk(cudaMallocAsync((void **) &d_Kz, mat_size, cu_stream));
    cu_errchk(cudaMemcpyAsync(d_Kz, Kz, mat_size, cudaMemcpyHostToDevice, cu_stream));
    G_kernel_from_distances_inplace<<<CUDA_THREADS_BLOCKS(mn), 0, cu_stream>>>(d_Kz, mn, 2. * gamma * gamma);
    cu_errchk(cudaMemcpyAsync(Kz, d_Kz, mat_size, cudaMemcpyDeviceToHost, cu_stream));
    cu_errchk(cudaFreeAsync(d_Kz, cu_stream));
    cu_errchk(cudaStreamSynchronize(cu_stream));
    cu_errchk(cudaStreamDestroy(cu_stream));
}
#endif

double
score_kernel(
        const double *ref_kernel /* colmaj order */,
        const double norm_ref,
        const double *Z /* colmaj order */,
        const size_t m,
        const double gamma)
{
#if 0
    double *d_K, *d_Z, *d_ref;
    const size_t mm = m * m;
    const size_t mat_size = mm * sizeof(double);

    const common::gpu_context ctx;
    cu_errchk(cudaSetDevice(ctx.phy_id()));
    cu_errchk(cudaMalloc(&d_K, mat_size));
    cu_errchk(cudaMalloc(&d_Z, mat_size));
    cu_errchk(cudaMemcpy(d_Z, Z, mat_size, cudaMemcpyHostToDevice));
    G_kernel_from_distances_symm<CUDA_THREADS(mm)><<<CUDA_THREADS_BLOCKS(mm)>>>(d_K, d_Z, mm, m, 2. * gamma * gamma);
    cu_errchk(cudaFree(d_Z));
    cu_errchk(cudaDeviceSynchronize());
    cublasHandle_t cublasH;
    cublas_safe_call(cublasCreate(&cublasH));
    double norm_K;
    cublas_safe_call(cublasDnrm2(cublasH, mm, d_K, m, &norm_K));
    cu_errchk(cudaDeviceSynchronize());
    double *d_score;
    cu_errchk(cudaMalloc(&d_score, sizeof(double)));
    cu_errchk(cudaMalloc(&d_ref, mat_size));
    cu_errchk(cudaMemcpy(d_ref, ref_kernel, mat_size, cudaMemcpyHostToDevice));
    G_score_kernel<<<CUDA_THREADS_BLOCKS(m)>>>(d_score, d_K, d_ref, mm, norm_K, norm_ref);
    double score;
    cu_errchk(cudaMemcpy(&score, d_score, sizeof(double), cudaMemcpyDeviceToHost));
    cu_errchk(cudaFree(d_score));
    cu_errchk(cudaFree(d_ref));
    cu_errchk(cudaFree(d_K));
    cublas_safe_call(cublasDestroy(cublasH));
    return 2. - score;
#endif
    return 0;
}


void __global__
gpu_copy_upper_submatrix(
        const double *__restrict d_in,
        double *__restrict d_ou,
        const size_t M, const size_t N, const size_t subM)
{
    const size_t i = threadIdx.x + blockIdx.x * blockDim.x;
    const size_t j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= subM || j >= N) return;
    d_ou[j * subM + i] = d_in[j * M + i];
}


std::tuple<cusolverDnHandle_t, double *, double *, double *, int *, int *>
init_cusolver(const size_t gpu_id, const size_t m, const size_t n)
{
    cu_errchk(cudaSetDevice(gpu_id));

    cusolverDnHandle_t cusolverH;
    cublasHandle_t cublasH;
    int lwork;
    double *d_Ainput, *d_B, *d_work;
    int *d_Ipiv, *d_devInfo;

    cs_errchk(cusolverDnCreate(&cusolverH));
    cb_errchk(cublasCreate(&cublasH));
    cu_errchk(cudaMalloc((void **) &d_Ainput, m * m * sizeof(double)));
    cu_errchk(cudaMalloc((void **) &d_B, m * n * sizeof(double)));
    cs_errchk(cusolverDnDgetrf_bufferSize(cusolverH, m, m, d_Ainput, m /* lda */, &lwork));
    cu_errchk(cudaMalloc((void **) &d_work, sizeof(double) * lwork));
    cu_errchk(cudaMalloc((void **) &d_Ipiv, m * sizeof(int)));
    cu_errchk(cudaMalloc((void **) &d_devInfo, sizeof(int)));

    return {cusolverH, d_Ainput, d_B, d_work, d_Ipiv, d_devInfo};
}


void uninit_cusolver(const size_t gpu_id, const cusolverDnHandle_t cusolverH, double *d_Ainput, double *d_B, double *d_work, int *d_Ipiv, int *d_devInfo)
{
    cu_errchk(cudaSetDevice(gpu_id));

    if (d_Ainput) cu_errchk(cudaFree(d_Ainput));
    if (d_B) cu_errchk(cudaFree(d_B));
    if (d_work) cu_errchk(cudaFree(d_work));
    if (d_Ipiv) cu_errchk(cudaFree(d_Ipiv));
    if (d_devInfo) cu_errchk(cudaFree(d_devInfo));

    if (cusolverH) cs_errchk(cusolverDnDestroy(cusolverH));
}

void dyn_gpu_solve(const cusolverDnHandle_t cusolver_H, const size_t m, const size_t n, const double *d_a, double *d_b, double *d_work, int *d_piv, int *d_info)
{
    cs_errchk(cusolverDnDgetrf(cusolver_H, m, m, (double *) d_a, m, d_work, d_piv, d_info));
    cs_errchk(cusolverDnDgetrs(cusolver_H, CUBLAS_OP_N, m, n, d_a, m, d_piv, d_b, m, d_info));
}

void h_dyn_gpu_solve(
        const size_t gpu_id, const size_t m, const size_t n, const double *h_K, const double *h_L, double *h_weights, cusolverDnHandle_t cusolver_H,
        double *d_a, double *d_b, double *d_work, int *d_piv, int *d_info)
{
    cu_errchk(cudaSetDevice(gpu_id));
    cu_errchk(cudaMemcpy(d_a, h_K, sizeof(double) * m * m, cudaMemcpyHostToDevice));
    cu_errchk(cudaMemcpy(d_b, h_L, sizeof(double) * m * n, cudaMemcpyHostToDevice));
    dyn_gpu_solve(cusolver_H, m, n, d_a, d_b, d_work, d_piv, d_info);
    cu_errchk(cudaMemcpy(h_weights, d_b, sizeof(double) * m * n, cudaMemcpyDeviceToHost));
}


std::tuple<magma_queue_t, magmaDouble_ptr, magmaDouble_ptr, magmaDouble_ptr, magmaDouble_ptr, magmaFloat_ptr, magmaInt_ptr>
init_magma_solver(const size_t m, const size_t b_n, const bool psd, const size_t gpu_id)
{
    cu_errchk(cudaSetDevice(gpu_id));
    magma_queue_t magma_queue;
    magma_queue_create(gpu_id, &magma_queue);
    if (!magma_queue) LOG4_THROW("Failed creating MAGMA queue.");

    magmaDouble_ptr d_a, d_b, d_x, d_wd;
    magmaFloat_ptr d_ws;
    auto piv = (magmaInt_ptr) malloc(m * sizeof(magma_int_t)); // host mem.
    ma_errchk(magma_dmalloc(&d_a, m * m));
    ma_errchk(magma_dmalloc(&d_b, m * b_n));
    if (psd) {
        ma_errchk(magma_dmalloc(&d_x, m * b_n));
        ma_errchk(magma_dmalloc(&d_wd, m * (m + b_n) + m));
        ma_errchk(magma_smalloc(&d_ws, m * (m + b_n) + m));
    } else {
        d_x = nullptr;
        d_wd = nullptr;
        d_ws = nullptr;
    }
    return {magma_queue, d_a, d_b, d_x, d_wd, d_ws, piv};
}

std::tuple<std::vector<magmaDouble_ptr>, std::vector<magmaDouble_ptr>>
init_magma_batch_solver(const size_t batch_size, const size_t m, const size_t n)
{
    std::vector<magmaDouble_ptr> d_a(batch_size, nullptr), d_b(batch_size, nullptr);
#pragma omp unroll
    for (size_t i = 0; i < batch_size; ++i) {
        ma_errchk(magma_dmalloc(&d_a[i], m * m));
        ma_errchk(magma_dmalloc(&d_b[i], m * n));
    }
    return {d_a, d_b};
}


void uninit_magma_solver(
        const magma_queue_t &magma_queue,
        const magmaDouble_ptr d_a, const magmaDouble_ptr d_b, const magmaDouble_ptr d_x, const magmaDouble_ptr d_wd, const magmaFloat_ptr d_ws, const magmaInt_ptr piv,
        const size_t gpu_id)
{
    cu_errchk(cudaSetDevice(gpu_id));
    if (d_a) ma_errchk(magma_free(d_a));
    if (d_b) ma_errchk(magma_free(d_b));
    if (d_x) ma_errchk(magma_free(d_x));
    if (d_wd) ma_errchk(magma_free(d_wd));
    if (d_ws) ma_errchk(magma_free(d_ws));
    if (piv) free(piv);

    if (magma_queue) magma_queue_destroy(magma_queue);
}


void uninit_magma_batch_solver(std::vector<magmaDouble_ptr> &d_a, std::vector<magmaDouble_ptr> &d_b)
{
#pragma omp unroll
    for (size_t i = 0; i < d_a.size(); ++i) {
        if (d_a[i]) ma_errchk(magma_free(d_a[i]));
        if (d_b[i]) ma_errchk(magma_free(d_b[i]));
    }
}


void iter_magma_solve(
        const int m, const int b_n, const double *a, const double *b, double *output, const magma_queue_t magma_queue,
        const magmaDouble_ptr d_a, const magmaDouble_ptr d_b, const magmaDouble_ptr d_x, const magmaDouble_ptr d_workd,
        const magmaFloat_ptr d_works, const bool psd, const size_t gpu_id)
{
    cu_errchk(cudaSetDevice(gpu_id));
    magma_int_t err, iter, info;

    magma_dsetmatrix(m, m, a, m, d_a, m, magma_queue); // copy a -> d_a
    magma_dsetmatrix(m, b_n, b, m, d_b, m, magma_queue); // copy b -> d_b

    if (!psd) goto __solve_dgesv;

    if ((err = magma_dshposv_gpu_expert(magma_uplo_t::MagmaLower, m, b_n, d_a, m, d_b, m, d_x, m, d_workd, d_works, &iter, magma_mode_t::MagmaHybrid, 1, 0, 0, 0,
                                        &info)) < MAGMA_SUCCESS || info != 0) {
        LOG4_WARN("Call to magma_dshposv_gpu_expert failed with error " << err << ", info " << info << ". Trying magma_dgesv_rbt.");
        if (iter < 0) {
            switch (iter) {
                case -1:
                    LOG4_DEBUG(
                            "Iterative magma_dshposv_gpu_expert returned -1 : the routine fell back to full precision for implementation - or machine-specific reasons");
                    break;
                case -2:
                    LOG4_DEBUG("Iterative magma_dshposv_gpu_expert returned -2 : narrowing the precision induced an overflow, the routine fell back to full precision");
                    break;
                case -3:
                    LOG4_DEBUG("Iterative magma_dshposv_gpu_expert returned -3 : failure of SPOTRF");
                    break;
                case -31:
                    LOG4_DEBUG("Iterative magma_dshposv_gpu_expert returned -31: stop the iterative refinement after the 30th iteration");
                    break;
                default:
                    LOG4_ERROR("Iterative refinement magma_dshposv_gpu_expert has failed, double precision factorization has been performed");
            }
        }
    } else {
        LOG4_TRACE("Call to magma_dshposv_gpu_expert triunfo.");
        magma_dgetmatrix(m, b_n, d_x, m, output, m, magma_queue);
        return;
    }

    __solve_dgesv:
    ma_errchk(magma_dgesv_rbt_q(magma_bool_t::MagmaTrue, m, b_n, d_a, m, d_b, m, &info, datamodel::C_rbt_iter, datamodel::C_rbt_threshold, magma_queue));
    if (psd) LOG4_DEBUG("Call to magma_dgesv_rbt triunfo.");
    magma_dgetmatrix(m, b_n, d_b, m, output, m, magma_queue); // copy solution d_b -> output
}

void iter_magma_solve(
        const int m, const int n, const double *a, const double *b, double *output, const magma_queue_t &magma_queue,
        const magmaDouble_ptr d_a, const magmaDouble_ptr d_b)
{
    magma_int_t info;
    magma_dsetmatrix(m, m, a, m, d_a, m, magma_queue);
    magma_dsetmatrix(m, n, b, m, d_b, m, magma_queue);
    cu_errchk(cudaDeviceSynchronize());
    ma_errchk(magma_dgesv_rbt_q(magma_bool_t::MagmaTrue, m, n, d_a, m, d_b, m, &info, datamodel::C_rbt_iter, datamodel::C_rbt_threshold, magma_queue));
    cu_errchk(cudaDeviceSynchronize());
    magma_dgetmatrix(m, n, d_b, m, output, m, magma_queue);
    cu_errchk(cudaDeviceSynchronize());
}

void iter_magma_batch_solve(
        const int m, const int n, const std::deque<arma::mat> &a, const std::deque<arma::mat> &b, std::deque<arma::mat> &output,
        const magma_queue_t magma_queue, std::vector<magmaDouble_ptr> &d_a, std::vector<magmaDouble_ptr> &d_b, const size_t gpu_id)
{
    const auto batch_size = a.size();
    LOG4_DEBUG("m " << magma_int_t(m) << ", n " << magma_int_t(n) << ", batch size " << batch_size);
// #pragma omp parallel for schedule(static, 1) num_threads(adj_threads(batch_size))
    cu_errchk(cudaSetDevice(gpu_id));
#pragma omp unroll
    for (size_t i = 0; i < batch_size; ++i) {
        magma_dsetmatrix(m, m, a[i].mem, m, d_a[i], m, magma_queue);
        magma_dsetmatrix(m, n, b[i].mem, m, d_b[i], m, magma_queue);
    }
    std::vector<magma_int_t> info(batch_size);
    cu_errchk(cudaSetDevice(gpu_id));
    auto da_data = &d_a[0];
    auto db_data = &d_b[0];
    ma_errchk(magma_dgesv_rbt_batched(magma_int_t(m), magma_int_t(n), da_data, m, db_data, m, info.data(), magma_int_t(batch_size), magma_queue));

#pragma omp parallel for schedule(static, 1) num_threads(adj_threads(batch_size))
    for (size_t i = 0; i < batch_size; ++i) {
        cu_errchk(cudaSetDevice(gpu_id));
        magma_dgetmatrix(m, n, d_b[i], m, output[i].memptr(), m, magma_queue); // copy solution d_b -> output
    }
}

// Doesn't work with NVidia CuSolver 12.1, leaks memory
void dyn_magma_solve(const int m, const int b_n, const double *a, const double *b, double *output, magma_queue_t magma_queue,
                     const magmaInt_ptr piv, const magmaDouble_ptr d_a, const magmaDouble_ptr d_b, const size_t gpu_id)
{
    cu_errchk(cudaSetDevice(gpu_id));
    magma_int_t info, err;
    magma_dsetmatrix(m, m, a, m, d_a, m, magma_queue);

    // find the inverse matrix: d_a*X=I using the LU factorization
    // with partial pivoting and row interchanges computed by
    // magma_dgetrf_gpu; row i is interchanged with row piv(i);
    // d_a -mxm matrix; d_a is overwritten by the inverse
    const magma_int_t nb = magma_get_dgetrf_native_nb(m, b_n);
    if ((err = magma_dgetrf_gpu_expert(m, m, d_a, m, piv, &info, nb, magma_mode_t::MagmaNative)) < MAGMA_SUCCESS)
        LOG4_THROW("Failed calling magma_dgetrf_gpu with error code " << err << ", info " << info);

    magma_dsetmatrix(m, b_n, b, m, d_b, m, magma_queue);
    if ((err = magma_dgetrs_gpu(magma_trans_t::MagmaNoTrans, m, b_n, d_a, m /* lda */, piv, d_b, m /* ldb */, &info)) < MAGMA_SUCCESS)
        LOG4_THROW("Failed calling magma_dgetrs_gpu with error code " << err << ", info " << info);

    magma_dgetmatrix(m, b_n, d_b, m, output, m, magma_queue); // copy solution d_b -> output
}


void
qrsolve_over(const size_t Nrows, const size_t Ncols, const size_t Nrhs, double *d_Ainput, double *d_b, double *d_output)
{
    // define handles
    cusolverDnHandle_t cusolverH = NULL;
    cublasHandle_t cublasH = NULL;
    const size_t M = Nrows;
    const size_t N = Ncols;
    const size_t K = Nrhs;

    cs_errchk(cusolverDnCreate(&cusolverH));
    cb_errchk(cublasCreate(&cublasH));

    int *d_devInfo;
    double *d_tau;
    double *d_work;
    double *d_work2;
    cu_errchk(cudaMalloc(&d_tau, sizeof(double) * M));
    cu_errchk(cudaMalloc(&d_devInfo, sizeof(int)));
    int bufSize, bufSize2;

    // in-place A = QR
    cs_errchk(
            cusolverDnDgeqrf_bufferSize(
                    cusolverH,
                    M,
                    N,
                    d_Ainput,
                    M,
                    &bufSize
            )
    );
    cu_errchk(cudaMalloc(&d_work, sizeof(double) * bufSize));
    cs_errchk(
            cusolverDnDgeqrf(
                    cusolverH,
                    M,
                    N,
                    d_Ainput,
                    M,
                    d_tau,
                    d_work,
                    bufSize,
                    d_devInfo
            )
    );

    // Q^T*b
    cs_errchk(
            cusolverDnDormqr_bufferSize(
                    cusolverH,
                    CUBLAS_SIDE_LEFT,
                    CUBLAS_OP_T,
                    M,
                    K,
                    N,
                    d_Ainput,
                    M,
                    d_tau,
                    d_b,
                    M,
                    &bufSize2
            )
    );

    cu_errchk(cudaMalloc(&d_work2, sizeof(double) * bufSize2));
    cs_errchk(
            cusolverDnDormqr(
                    cusolverH,
                    CUBLAS_SIDE_LEFT,
                    CUBLAS_OP_T,
                    M,
                    K,
                    std::min<size_t>(M, N),
                    d_Ainput,
                    M,
                    d_tau,
                    d_b,
                    M,
                    d_work2,
                    bufSize2,
                    d_devInfo
            )
    );

    // need to explicitly copy submatrix for the triangular solve
    double *d_R;
    cu_errchk(cudaMalloc(&d_R, sizeof(double) * N * N));
    dim3 thd_size(32, 32);
    dim3 blk_size((N + thd_size.x - 1) / thd_size.x, (N + thd_size.y - 1) / thd_size.y);
    gpu_copy_upper_submatrix<<<blk_size, thd_size>>>(d_Ainput, d_R, M, N, N);
    blk_size = dim3((N + thd_size.x - 1) / thd_size.x, (K + thd_size.y - 1) / thd_size.y);
    gpu_copy_upper_submatrix<<<blk_size, thd_size>>>(d_b, d_output, M, K, N);

    // solve x = R \ (Q^T*B)
    const double one = 1;
    cb_errchk(
            cublasDtrsm(
                    cublasH,
                    CUBLAS_SIDE_LEFT,
                    CUBLAS_FILL_MODE_UPPER,
                    CUBLAS_OP_N,
                    CUBLAS_DIAG_NON_UNIT,
                    N,
                    K,
                    &one,
                    d_R,
                    N,
                    d_output,
                    N
            )
    );

    cu_errchk(cudaFree(d_R));
    cu_errchk(cudaFree(d_work));
    cu_errchk(cudaFree(d_work2));
    cu_errchk(cudaFree(d_tau));
    cb_errchk(cublasDestroy(cublasH));
    cs_errchk(cusolverDnDestroy(cusolverH));
}


void
call_gpu_overdetermined(
        const size_t Nrows, const size_t Ncols, const size_t Nrhs, const double *cpu_matrix,
        const double *cpu_rhs, double *cpu_output)
{
    const svr::common::gpu_context gtx;
    cudaSetDevice(gtx.phy_id());

    thrust::device_vector<double> gpu_matrix(Nrows * Ncols);
    thrust::device_vector<double> gpu_rhs(Nrows * Nrhs);
    thrust::device_vector<double> gpu_output(Ncols * Nrhs);
    cu_errchk(cudaMemcpy(thrust::raw_pointer_cast(gpu_matrix.data()), cpu_matrix, sizeof(double) * Nrows * Ncols,
                         cudaMemcpyHostToDevice));
    cu_errchk(cudaMemcpy(thrust::raw_pointer_cast(gpu_rhs.data()), cpu_rhs, sizeof(double) * Nrows * Nrhs,
                         cudaMemcpyHostToDevice));
    qrsolve_over(Nrows, Ncols, Nrhs, thrust::raw_pointer_cast(gpu_matrix.data()),
                 thrust::raw_pointer_cast(gpu_rhs.data()), thrust::raw_pointer_cast(gpu_output.data()));
    cu_errchk(cudaMemcpy(cpu_output, thrust::raw_pointer_cast(gpu_output.data()), sizeof(double) * Ncols * Nrhs,
                         cudaMemcpyDeviceToHost));
}


// Adds err + addtive to K and solved labels
__global__ void G_irwls_op2(
        const double *__restrict__ err,
        const double *__restrict__ K,
        const double *__restrict__ labels,
        double *__restrict__ out_K,
        double *__restrict__ solved,
        const double additive,
        const unsigned m,
        const unsigned mn,
        const unsigned mm)
{
    double sum_err_i;
    CUDA_STRIDED_FOR_i(mm) {
        if (i < mn) solved[i] = (err[i] + additive) * labels[i];
        out_K[i] = K[i];
        sum_err_i = 0;
#ifdef PRODUCTION_BUILD
#pragma unroll
#endif
        for (unsigned j = i % m; j < mn; j += m) sum_err_i += err[j] + additive;
        out_K[i] *= sum_err_i;
    }
}

/* LDA version
__global__ void G_irwls_op2(
        const double *__restrict__ err,
        const double *__restrict__ K,
        const unsigned ldK,
        const double *__restrict__ labels,
        double *__restrict__ out_K,
        double *__restrict__ solved,
        const double additive,
        const unsigned m,
        const unsigned mn,
        const unsigned mm)
{
    double sum_err_i;
    CUDA_STRIDED_FOR_i(mm) {
        const unsigned row = i % m;
        const unsigned in_i = (i / m) * ldK + row;
        if (i < mn) solved[i] = (err[i] + additive) * labels[in_i];
        sum_err_i = 0;
#ifdef PRODUCTION_BUILD
#pragma unroll
#endif
        for (unsigned j = row; j < mn; j += m) sum_err_i += err[j] + additive;
        out_K[i] = K[in_i] * sum_err_i;
    }
}
*/

void solve_hybrid(
        const double *const j_K_epsco, const unsigned n, const unsigned train_len, double *const j_solved, const unsigned magma_iters, const double magma_threshold,
        const magma_queue_t ma_queue, const unsigned irwls_iters, const double *const j_train_labels, const size_t train_n_size,
        double *const j_train_error, const cudaStream_t custream, const cublasHandle_t cublas_H, const double *const j_K_tune, const double labels_factor,
        const size_t train_len_n, double &best_solve_score, unsigned &best_iter, double *const d_best_weights, const size_t K_train_len, double *const j_left,
        magma_int_t &info, const double iters_mul)
{
    constexpr double one = 1, oneneg = -1;
    cu_errchk(cudaMemcpyAsync(j_solved, j_train_labels, train_n_size, cudaMemcpyDeviceToDevice, custream));
    ma_errchk(magma_dgesv_rbt_q(MagmaTrue, train_len, n, (double *)j_K_epsco, train_len, j_solved, train_len, &info, magma_iters, magma_threshold, ma_queue));
#pragma unroll common::C_default_online_iter_limit
    for (size_t i = 1; i < irwls_iters + 1; ++i) {
        cu_errchk(cudaMemcpyAsync(j_train_error, j_train_labels, train_n_size, cudaMemcpyDeviceToDevice, custream));
        cb_errchk(cublasDgemm(cublas_H, CUBLAS_OP_N, CUBLAS_OP_N,
                              train_len, n, train_len, &one, (double *)j_K_tune, train_len, j_solved, train_len, &oneneg, j_train_error, train_len));
        const auto solve_score = labels_factor * solvers::irwls_op1(j_train_error, train_len_n, custream);
        if (!std::isnormal(solve_score))
            LOG4_THROW("Score not normal " << solve_score << ", iteration " << i << ", train len " << train_len);
        else if (solve_score < best_solve_score) {
            /* LOG4_TRACE("Try " << j << ", IRWLS iteration " << i << ", kernel dimensions " << train_len << "x" << train_len << ", former best score " <<
                            best_solve_score << ", new best score " << solve_score << ", improvement " << 100. * (1. - solve_score / best_solve_score) << " pct."); */
            best_solve_score = solve_score;
            best_iter = i;
            cu_errchk(cudaMemcpyAsync(d_best_weights, j_solved, train_n_size, cudaMemcpyDeviceToDevice, custream));
            cu_errchk(cudaStreamSynchronize(custream)); // TODO Remove
        }
        if (i == irwls_iters) break;
        G_irwls_op2<<<CUDA_THREADS_BLOCKS(K_train_len), 0, custream>>>(
                j_train_error, j_K_epsco, j_train_labels, j_left, j_solved, common::C_itersolve_delta / (double(i) * iters_mul), train_len, train_len_n, K_train_len);
        /*
        G_irwls_op2<<<CUDA_THREADS_BLOCKS(K_train_len), 0, custream>>>(
        j_train_error, d_K_epsco + train_start, m, d_tune_labels + train_start, j_left, j_solved, common::C_itersolve_delta / (double(i) * iters_mul),
        train_len, train_len_n, K_train_len);*/
        ma_errchk(magma_dgesv_rbt_q(MagmaTrue, train_len, n, j_left, train_len, j_solved, train_len, &info, magma_iters, magma_threshold, ma_queue));
    }
}


__global__ void G_abs(double *__restrict__ inout, const size_t N)
{
    CUDA_STRIDED_FOR_i(N)inout[i] = _ABS(inout[i]);
}


template<const unsigned block_size> __device__ inline void
warp_reduce_sumabs(volatile double *sumdata, const unsigned ix, const unsigned n)
{
#define _DO_WARP_REDUCE_SUMABS(N)               \
    if (block_size >= (N)) {                    \
        const unsigned ix_N_2 = ix + (N) / 2;   \
        if (ix_N_2 < n)                         \
            sumdata[ix] += sumdata[ix_N_2];     \
    }

    _DO_WARP_REDUCE_SUMABS(64);
    _DO_WARP_REDUCE_SUMABS(32);
    _DO_WARP_REDUCE_SUMABS(16);
    _DO_WARP_REDUCE_SUMABS(8);
    _DO_WARP_REDUCE_SUMABS(4);
    _DO_WARP_REDUCE_SUMABS(2);
}

template<const unsigned block_size> __global__ void
G_sumabs(const double *__restrict__ d_input, double *__restrict__ d_result_sum, const size_t n)
{
    extern __shared__ double sumdata[];
    auto i = blockIdx.x * block_size + tid;
    if (i < n) {
        sumdata[tid] = fabs(d_input[i]);
        const auto stride1 = blockDim.x * gridDim.x;
#pragma unroll
        for (i += stride1; i < n; i += stride1) sumdata[tid] += fabs(d_input[i]);
    } else
        sumdata[tid] = 0;

    __syncthreads();
    const auto sh_limit = _MIN(n, block_size);
#define stride_reduce_sum(block_low_)                        \
        if (block_size >= block_low_) {                      \
            constexpr unsigned stride2 = block_low_ / 2;     \
            const auto tid_stride2 = tid + stride2;          \
            if (tid < stride2 && tid_stride2 < sh_limit)     \
                sumdata[tid] += sumdata[tid_stride2];        \
            __syncthreads();                                 \
        }

    stride_reduce_sum(1024);
    stride_reduce_sum(512);
    stride_reduce_sum(256);
    stride_reduce_sum(128);
    if (tid >= 32) return;
    warp_reduce_sumabs<block_size>(sumdata, tid, sh_limit);
    if (tid) return;
    d_result_sum[blockIdx.x] = sumdata[0];
}

double sumabs(const double *d_in, const size_t n, const cudaStream_t &stm)
{
    double sum;
    const auto clamped_n = clamp_n(n);
    const auto grid_len = CUDA_BLOCKS(clamped_n);
    const auto threads = CUDA_THREADS(clamped_n);
    const auto grid_size = grid_len * sizeof(double);
    double *d_sum;
    cu_errchk(cudaMallocAsync((void **) &d_sum, grid_size, stm));
    G_sumabs<common::C_cu_block_size><<<grid_len, threads, threads * sizeof(double), stm>>>(d_in, d_sum, n);
    if (grid_len > 1)
        sum = thrust::async::reduce(thrust::cuda::par.on(stm), d_sum, d_sum + grid_len, double(0), thrust::plus<double>()).get();
    else
        cu_errchk(cudaMemcpyAsync(&sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost, stm));
    cu_errchk(cudaFreeAsync(d_sum, stm));
    cu_errchk(cudaStreamSynchronize(stm));
    return sum;
}

template<const unsigned block_size> __global__ void
G_irwls_op1(double *__restrict__ d_input, double *__restrict__ d_result_sum, const size_t n)
{
    __shared__ double sumdata[block_size];
    auto i = blockIdx.x * block_size + tid;
    if (i < n) {
        d_input[i] = fabs(d_input[i]);
        sumdata[tid] = d_input[i];
        const auto stride1 = blockDim.x * gridDim.x;
#pragma unroll
        for (i += stride1; i < n; i += stride1) {
            d_input[i] = fabs(d_input[i]);
            sumdata[tid] += d_input[i];
        }
    } else
        sumdata[tid] = 0;

    __syncthreads();
    const auto sh_limit = _MIN(n, block_size);
#define stride_reduce_sum(block_low_)                        \
        if (block_size >= block_low_) {                      \
            constexpr unsigned stride2 = block_low_ / 2;     \
            const auto tid_stride2 = tid + stride2;          \
            if (tid < stride2 && tid_stride2 < sh_limit)     \
                sumdata[tid] += sumdata[tid_stride2];        \
            __syncthreads();                                 \
        }

    stride_reduce_sum(1024);
    stride_reduce_sum(512);
    stride_reduce_sum(256);
    stride_reduce_sum(128);
    if (tid >= 32) return;
    warp_reduce_sumabs<block_size>(sumdata, tid, sh_limit);
    if (tid) return;
    d_result_sum[blockIdx.x] = sumdata[0];
}

// Returns meanabs of input, input = abs(input)
double irwls_op1(double *d_in, const size_t n, const cudaStream_t &stm)
{
    double sum;
    const auto clamped_n = clamp_n(n);
    const auto grid_len = CUDA_BLOCKS(clamped_n);
    const auto threads = CUDA_THREADS(clamped_n);
    const auto grid_size = grid_len * sizeof(double);
    double *d_sum;
    cu_errchk(cudaMallocAsync((void **) &d_sum, grid_size, stm));
    G_irwls_op1 < common::C_cu_block_size ><<<grid_len, threads, 0, stm>>>(d_in, d_sum, n);
    if (grid_len > 1)
        sum = thrust::async::reduce(thrust::cuda::par.on(stm), d_sum, d_sum + grid_len, double(0), thrust::plus<double>()).get();
    else
        cu_errchk(cudaMemcpyAsync(&sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost, stm));
    cu_errchk(cudaFreeAsync(d_sum, stm));
    cu_errchk(cudaStreamSynchronize(stm));
    return sum / double(n); // Return mean
}

double meanabs(const double *d_in, const size_t n, const cudaStream_t &stm)
{
    return sumabs(d_in, n, stm) / double(n);
}

#define _SMM_OP(X1, X2, X3, Y1, Y2, Y3) {       \
        (X1) += (Y1);                           \
        _MINAS((X2), (Y2));                     \
        _MAXAS((X3), (Y3));                     \
    }

template<const unsigned block_size> __device__ inline void
warp_reduce_suminmax(volatile double *sumdata, volatile double *mindata, volatile double *maxdata, const unsigned ix, const unsigned n)
{
#define _DO_WARP_REDUCE(N)                      \
    if (block_size >= (N)) {                    \
        const unsigned ix_N_2 = ix + (N) / 2;   \
        if (ix_N_2 < n)                         \
            _SMM_OP(sumdata[ix], mindata[ix], maxdata[ix], sumdata[ix_N_2], mindata[ix_N_2], maxdata[ix_N_2]); \
    }

    _DO_WARP_REDUCE(64);
    _DO_WARP_REDUCE(32);
    _DO_WARP_REDUCE(16);
    _DO_WARP_REDUCE(8);
    _DO_WARP_REDUCE(4);
    _DO_WARP_REDUCE(2);
}

template<const unsigned block_size> __global__ void
G_suminmax(const double *__restrict__ d_input, double *__restrict__ d_result_sum, double *__restrict__ d_result_min, double *__restrict__ d_result_max, const size_t n)
{
    __shared__ double sumdata[block_size], mindata[block_size], maxdata[block_size];
    auto i = blockIdx.x * block_size + tid;
    if (i < n) {
        sumdata[tid] = mindata[tid] = maxdata[tid] = d_input[i];
        const auto stride1 = blockDim.x * gridDim.x;
#pragma unroll
        for (i += stride1; i < n; i += stride1) _SMM_OP(sumdata[tid], mindata[tid], maxdata[tid], d_input[i], d_input[i], d_input[i]);
    } else {
        sumdata[tid] = 0;
        mindata[tid] = std::numeric_limits<double>::max();
        maxdata[tid] = std::numeric_limits<double>::min();
    }
    __syncthreads();
    const auto sh_limit = _MIN(n, block_size);
#define stride_reduce_suminmax(block_low_)                  \
        if (block_size >= block_low_) {                     \
            constexpr unsigned stride2 = block_low_ / 2;    \
            const auto tid_stride2 = tid + stride2;         \
            if (tid < stride2 && tid_stride2 < sh_limit)           \
                _SMM_OP(sumdata[tid], mindata[tid], maxdata[tid], sumdata[tid_stride2], mindata[tid_stride2], maxdata[tid_stride2]); \
            __syncthreads();                                \
        }

    stride_reduce_suminmax(1024);
    stride_reduce_suminmax(512);
    stride_reduce_suminmax(256);
    stride_reduce_suminmax(128);
    if (tid >= 32) return;
    warp_reduce_suminmax<block_size>(sumdata, mindata, maxdata, tid, sh_limit);
    if (tid) return;
    d_result_sum[blockIdx.x] = sumdata[0];
    d_result_min[blockIdx.x] = mindata[0];
    d_result_max[blockIdx.x] = maxdata[0];
}

template<const unsigned block_size> __global__ void
G_suminmax(const double *__restrict__ d_in_sum, const double *__restrict__ d_in_min, const double *__restrict__ d_in_max,
           double *__restrict__ d_result_sum, double *__restrict__ d_result_min, double *__restrict__ d_result_max,
           const size_t n)
{
    __shared__ double sumdata[block_size], mindata[block_size], maxdata[block_size];
    if (tid < n) {
        sumdata[tid] = d_in_sum[tid];
        mindata[tid] = d_in_min[tid];
        maxdata[tid] = d_in_max[tid];
    } else {
        sumdata[tid] = 0;
        mindata[tid] = std::numeric_limits<double>::max();
        maxdata[tid] = std::numeric_limits<double>::min();
    }
    __syncthreads();
    const auto sh_limit = _MIN(n, block_size);
    stride_reduce_suminmax(1024);
    stride_reduce_suminmax(512);
    stride_reduce_suminmax(256);
    stride_reduce_suminmax(128);
    if (tid >= 32) return;
    warp_reduce_suminmax<block_size>(sumdata, mindata, maxdata, tid, sh_limit);
    if (tid) return;
    d_result_sum[0] = sumdata[0];
    d_result_min[0] = mindata[0];
    d_result_max[0] = maxdata[0];
}

std::tuple<double, double, double> suminmax(const double *d_in, const size_t n, const cudaStream_t &stm)
{
    double sum, min, max;
    const auto clamped_n = clamp_n(n);
    const auto grid_len = CUDA_BLOCKS(clamped_n);
    const auto threads = CUDA_THREADS(clamped_n);
    const auto grid_size = grid_len * sizeof(double);
    double *d_sum, *d_min, *d_max;
    cu_errchk(cudaMallocAsync((void **) &d_sum, sizeof(double), stm));
    cu_errchk(cudaMallocAsync((void **) &d_min, sizeof(double), stm));
    cu_errchk(cudaMallocAsync((void **) &d_max, sizeof(double), stm));
    if (grid_len > 1) {
        double *d_block_sums, *d_block_mins, *d_block_maxs;
        cu_errchk(cudaMallocAsync((void **) &d_block_sums, grid_size, stm));
        cu_errchk(cudaMallocAsync((void **) &d_block_mins, grid_size, stm));
        cu_errchk(cudaMallocAsync((void **) &d_block_maxs, grid_size, stm));
        G_suminmax < common::C_cu_block_size ><<<grid_len, threads, 0, stm>>>(d_in, d_block_sums, d_block_mins, d_block_maxs, n);
        G_suminmax < common::C_cu_block_size ><<<1, grid_len, 0, stm>>>(d_block_sums, d_block_mins, d_block_maxs, d_sum, d_min, d_max, grid_len);
        cu_errchk(cudaFreeAsync(d_block_sums, stm));
        cu_errchk(cudaFreeAsync(d_block_mins, stm));
        cu_errchk(cudaFreeAsync(d_block_maxs, stm));
    } else
        G_suminmax < common::C_cu_block_size ><<<grid_len, threads, 0, stm>>>(d_in, d_sum, d_min, d_max, n);
    cu_errchk(cudaMemcpyAsync(&sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost, stm));
    cu_errchk(cudaMemcpyAsync(&min, d_min, sizeof(double), cudaMemcpyDeviceToHost, stm));
    cu_errchk(cudaMemcpyAsync(&max, d_max, sizeof(double), cudaMemcpyDeviceToHost, stm));
    cu_errchk(cudaFreeAsync(d_sum, stm));
    cu_errchk(cudaFreeAsync(d_min, stm));
    cu_errchk(cudaFreeAsync(d_max, stm));
    cu_errchk(cudaStreamSynchronize(stm));
    return {sum, min, max};
}

std::tuple<double, double, double> meanminmax(const double *d_in, const size_t n, const cudaStream_t &stm)
{
    const auto [sum, min, max] = suminmax(d_in, n, stm);
    return {sum / double(n), min, max};
}


template<const unsigned block_size> __device__ inline void
warp_reduce_dist(volatile double *sh_dist, const unsigned ix, const unsigned n)
{
#define _DO_WARP_REDUCE_DIST(N) {               \
    if (block_size >= (N)) {                    \
        const unsigned ix_N_2 = ix + (N) / 2;   \
        if (ix_N_2 < n)                         \
            sh_dist[ix] += sh_dist[ix_N_2];     \
        } }

    _DO_WARP_REDUCE_DIST(64);
    _DO_WARP_REDUCE_DIST(32);
    _DO_WARP_REDUCE_DIST(16);
    _DO_WARP_REDUCE_DIST(8);
    _DO_WARP_REDUCE_DIST(4);
    _DO_WARP_REDUCE_DIST(2);
}

template<const unsigned block_size> __global__ void G_dist_unscaled(
        double *__restrict__ d_mae, const double *__restrict__ d_labels, const double *__restrict__ d_predictions, const size_t n)
{
    __shared__ double sh_dist[block_size];
    auto i = blockIdx.x * block_size + tid;
    if (i < n) {
        sh_dist[tid] = fabs(d_labels[i] - d_predictions[i]);
        const auto stride1 = blockDim.x * gridDim.x;
#ifdef PRODUCTION_BUILD
#pragma unroll
#endif
        for (i += stride1; i < n; i += stride1)
            sh_dist[tid] += fabs(d_labels[i] - d_predictions[i]);
    } else
        sh_dist[tid] = 0;

    __syncthreads();

#define stride_reduce_dist(block_low_)                                  \
        if (block_size >= block_low_) {                                 \
            constexpr unsigned stride2 = block_low_ / 2;                \
            const auto tid_stride2 = tid + stride2;                     \
            if (tid < stride2 && tid_stride2 < n)                       \
                sh_dist[tid] += sh_dist[tid_stride2];                   \
            __syncthreads();                                            \
        }

    stride_reduce_dist(1024);
    stride_reduce_dist(512);
    stride_reduce_dist(256);
    stride_reduce_dist(128);
    if (tid >= 32) return;
    warp_reduce_dist<block_size>(sh_dist, tid, n);
    if (tid) return;
    d_mae[blockIdx.x] = sh_dist[0];
}

double
unscaled_distance(const double *d_labels, const double *d_predictions, const double sf, const size_t n, const cudaStream_t stm)
{
    const auto clamped_n = clamp_n(n);
    const auto grid_len = CUDA_BLOCKS(clamped_n);
    double *d_block_sums;
    cu_errchk(cudaMallocAsync((void **) &d_block_sums, grid_len * sizeof(double), stm));
    G_dist_unscaled < common::C_cu_block_size ><<<CUDA_THREADS_BLOCKS(clamped_n), 0, stm>>>(d_block_sums, d_labels, d_predictions, n);
    double mae;
    if (grid_len > 1)
        mae = thrust::async::reduce(thrust::cuda::par.on(stm), d_block_sums, d_block_sums + grid_len, double(0), thrust::plus<double>()).get();
    else cu_errchk(cudaMemcpyAsync(&mae, d_block_sums, sizeof(double), cudaMemcpyDeviceToHost, stm));
    cu_errchk(cudaFreeAsync(d_block_sums, stm));
    cu_errchk(cudaStreamSynchronize(stm));
    // LOG4_TRACE("Length " << n << ", scaling factor " << sf << ", DC offset " << dc << ", grid len " << grid_len << ", threads " << CUDA_THREADS(clamped_n) << ", sum dist " << res);
    return sf * mae / double(n);
}


double max(const double *d_in, const size_t n, const cudaStream_t stm)
{
    return thrust::async::reduce(thrust::cuda::par.on(stm), d_in, d_in + n, std::numeric_limits<double>::min(), thrust::maximum<double>()).get();
}

double min(const double *d_in, const size_t n, const cudaStream_t stm)
{
    return thrust::async::reduce(thrust::cuda::par.on(stm), d_in, d_in + n, std::numeric_limits<double>::max(), thrust::minimum<double>()).get();
}

double mean(const double *d_in, const size_t n, const cudaStream_t &stm)
{
    return sum(d_in, n, stm) / double(n);
}

double sum(const double *d_in, const size_t n, const cudaStream_t &stm)
{
#if 0
    size_t npp_buffer_size;
    if (nppGetStream() != stm) np_errchk(nppSetStream(stm));
    np_errchk(nppsSumGetBufferSize_64f(n, &npp_buffer_size));
    // auto npp_mean_buf = nppsMalloc_8u(npp_buffer_size);
    Npp8u *npp_sum_buf;
    Npp64f *dres;
    cu_errchk(cudaMallocAsync((void **)&npp_sum_buf, npp_buffer_size, stm));
    cu_errchk(cudaMallocAsync((void **)&dres, sizeof(*dres), stm));
    assert(npp_sum_buf != nullptr);
    np_errchk(nppsSum_64f(d_in, n, dres, npp_sum_buf));
    double res;
    cu_errchk(cudaMemcpyAsync(&res, dres, sizeof(*dres), cudaMemcpyDeviceToHost, stm));
    cu_errchk(cudaStreamSynchronize(stm));
    cu_errchk(cudaFreeAsync(npp_sum_buf, stm));
    cu_errchk(cudaFreeAsync(dres, stm));
    return res;
#else
    return thrust::async::reduce(thrust::cuda::par.on(stm), d_in, d_in + n, double(0), thrust::plus<double>()).get();
#endif
}


__global__ void G_sqrt_add(double *__restrict__ input, const double a, const size_t N)
{
    CUDA_STRIDED_FOR_i(N)input[i] = sqrt(input[i] + a);
}

__global__ void G_matmul_inplace(const double *__restrict__ input, double *__restrict__ output, const size_t N)
{
    CUDA_STRIDED_FOR_i(N)output[i] *= input[i];
}

__global__ void G_eq_matmul(const double *__restrict__ input1, const double *__restrict__ input2, double *__restrict__ output, const size_t N)
{
    CUDA_STRIDED_FOR_i(N)output[i] = input1[i] * input2[i];
}

__global__ void G_abs_subtract(const double *__restrict__ input1, double *__restrict__ input2, const size_t N)
{
    CUDA_STRIDED_FOR_i(N)input2[i] = std::abs(input1[i] - input2[i]);
}

#if 0 // Not used
void solve_irwls(const arma::mat &K_epsco, const arma::mat &K, const arma::mat &rhs, arma::mat &solved, const size_t iters, const size_t gpu_phy_id)
{
    const bool psd = false;
#ifdef USE_MAGMA
    auto [magma_queue, d_K, d_rhs, d_x, d_tmpd, d_tmpf, piv] = solvers::init_magma_solver(K.n_rows, rhs.n_cols, psd, gpu_phy_id);
#else
    auto [cusolverH, d_K, d_rhs, d_tmpd, d_piv, d_devinfo] = solvers::init_cusolver(gpu_phy_id, K.n_rows, rhs.n_cols);
#endif
    if (arma::size(solved) != arma::size(rhs)) solved = rhs; // TODO Should be set_size

#ifdef USE_MAGMA
    solvers::iter_magma_solve(K_epsco.n_rows, rhs.n_cols, K_epsco.mem, rhs.mem, solved.memptr(), magma_queue, d_K, d_rhs, d_x, d_tmpd, d_tmpf, psd, gpu_phy_id);
#else
    solvers::dyn_gpu_solve(gpu_phy_id, K_epsco.n_rows, rhs.n_cols, K_epsco.mem, rhs.mem, solved.memptr(), cusolverH, d_K, d_rhs, d_tmpd, d_piv, d_devinfo);
#endif
    auto best_sae = std::numeric_limits<double>::infinity();
    arma::mat best_solution = solved;
    size_t best_iter = 0;
#pragma omp unroll
    for (size_t i = 1; i < iters; ++i) {
        const arma::mat error_mat = arma::abs(K * solved - rhs);
        const double this_sae = arma::accu(error_mat);
        if (this_sae < best_sae) {
            LOG4_TRACE("IRWLS iteration " << i << ", SAE " << this_sae << ", kernel dimensions " << arma::size(K) << ", best SAE " << best_sae);
            best_sae = this_sae;
            best_solution = solved;
            best_iter = i;
        }
        const arma::mat mult = arma::sqrt(error_mat + common::C_itersolve_delta / (double(i) * common::C_itersolve_range / double(iters)));
        const arma::mat left = (mult * arma::ones(mult.n_cols, K.n_cols)) % K_epsco;
        const arma::mat right = rhs % mult;
#ifdef USE_MAGMA
        solvers::iter_magma_solve(left.n_rows, right.n_cols, left.mem, right.mem, solved.memptr(), magma_queue, d_K, d_rhs, d_x, d_tmpd, d_tmpf, psd, gpu_phy_id);
#else
        solvers::dyn_gpu_solve(gpu_phy_id, left.n_rows, right.n_cols, left.mem, right.mem, solved.memptr(), cusolverH, d_K, d_rhs, d_tmpd, d_piv, d_devinfo);
#endif
    }

#ifdef USE_MAGMA
    solvers::uninit_magma_solver(magma_queue, d_K, d_rhs, d_x, d_tmpd, d_tmpf, piv, gpu_phy_id);
#else
    solvers::uninit_cusolver(gpu_phy_id, cusolverH, d_K, d_rhs, d_tmpd, d_piv, d_devinfo);
#endif
    LOG4_DEBUG("IRWLS best iteration " << best_iter << ", MAE " << best_sae / double(solved.n_elem) << ", kernel dimensions " << arma::size(K) <<
                                       ", delta " << common::C_itersolve_delta << ", range " << common::C_itersolve_range << ", solution " << arma::size(solved));
    solved = best_solution;
}
#endif

// Namespaces
}
}
