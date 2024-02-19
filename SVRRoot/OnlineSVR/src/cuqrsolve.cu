#ifdef USE_CUDA
#include <thread>
#include <magma_types.h>
#include <magma_v2.h>
#include <cassert>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublasLt.h>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include "common/compatibility.hpp"
#include "common/gpu_handler.hpp"
#include "common/cuda_util.cuh"
#include "cuqrsolve.hpp"


namespace svr {
namespace solvers {


void __global__
G_score_kernel(
        double *score,
        const double *__restrict kernel, const double *__restrict ref,
        const size_t M, const double norm_ker, const double norm_ref)
{
    const size_t thr_ix = threadIdx.x;
    const size_t g_thr_ix = thr_ix + blockIdx.x * CUDA_BLOCK_SIZE;
    const size_t grid_size = CUDA_BLOCK_SIZE * gridDim.x;

    double sum = 0;
    for (size_t i = g_thr_ix; i < M; i += grid_size) sum += kernel[i] * ref[i];

    __shared__ double _sh_sum[CUDA_BLOCK_SIZE];
    _sh_sum[thr_ix] = sum;
    __syncthreads();

    for (size_t size = CUDA_BLOCK_SIZE / 2; size > 0; size /= 2) {
        if (thr_ix >= size) continue;
        _sh_sum[thr_ix] += _sh_sum[thr_ix + size];
        __syncthreads();
    }

    if (thr_ix == 0) *score = _sh_sum[0] / (norm_ker * norm_ref);
}


template<unsigned k_block_size> __global__ void
G_kernel_from_distances_symm(double *__restrict K, const double *__restrict dist, const size_t mm, const size_t m, const double divisor)
{
    const auto g_thr_ix = threadIdx.x + blockIdx.x * k_block_size;
    const auto row_ix = g_thr_ix % m;
    const auto col_ix = g_thr_ix / m;
    if (g_thr_ix >= mm || row_ix <= col_ix) return;
    K[m * row_ix + col_ix] = K[g_thr_ix] = 1. - dist[g_thr_ix] / divisor;
}


template<unsigned k_block_size> __global__ void
G_kernel_from_distances(double *__restrict K, const double *__restrict Z, const size_t mn, const double divisor)
{
    const auto g_thr_ix = threadIdx.x + blockIdx.x * k_block_size;
    if (g_thr_ix >= mn) return;
    K[g_thr_ix] = 1. - Z[g_thr_ix] / divisor;
}

// predict_kernel_matrix = 1. - Z / (2. * std::pow<double>(gamma, 2));
void kernel_from_distances(double *K, const double *Z, const size_t m, const size_t n, const double gamma)
{
    double *d_K, *d_Z;
    const size_t mn = m * n;
    const size_t mat_size = mn * sizeof(double);
    const common::gpu_context ctx;
    cu_errchk(cudaSetDevice(ctx.phy_id()));
    cu_errchk(cudaMalloc(&d_K, mat_size));
    cu_errchk(cudaMalloc(&d_Z, mat_size));
    cu_errchk(cudaMemcpy(d_Z, Z, mat_size, cudaMemcpyHostToDevice));
    G_kernel_from_distances<CUDA_BLOCK_SIZE><<<CUDA_THREADS_BLOCKS(mn)>>>(d_K, d_Z, mn, 2. * gamma * gamma);
    cu_errchk(cudaMemcpy(K, d_K, mat_size, cudaMemcpyDeviceToHost));
    cu_errchk(cudaFree(d_K));
    cu_errchk(cudaFree(d_Z));
}


double
score_kernel(
        const double *ref_kernel /* colmaj order */,
        const double norm_ref,
        const double *Z /* colmaj order */,
        const size_t m,
        const double gamma)
{
    double *d_K, *d_Z, *d_ref;
    const size_t mm = m * m;
    const size_t mat_size = mm * sizeof(double);

    common::gpu_context ctx;
    cu_errchk(cudaSetDevice(ctx.phy_id()));
    cu_errchk(cudaMalloc(&d_K, mat_size));
    cu_errchk(cudaMalloc(&d_Z, mat_size));
    cu_errchk(cudaMemcpy(d_Z, Z, mat_size, cudaMemcpyHostToDevice));
    G_kernel_from_distances_symm<CUDA_BLOCK_SIZE><<<CUDA_THREADS_BLOCKS(mm)>>>(d_K, d_Z, mm, m, 2. * gamma * gamma);
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

    cusolver_safe_call(cusolverDnCreate(&cusolverH));
    cublas_safe_call(cublasCreate(&cublasH));
    cu_errchk(cudaMalloc((void **) &d_Ainput, m * m * sizeof(double)));
    cublas_safe_call(cusolverDnDgetrf_bufferSize(cusolverH, m, m, d_Ainput, m /* lda */, &lwork));
    cu_errchk(cudaMalloc((void **) &d_work, sizeof(double) * lwork));
    cu_errchk(cudaMalloc((void **) &d_B, m * n * sizeof(double)));
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
    if (cusolverH) cusolverDnDestroy(cusolverH);
}


void dyn_gpu_solve(
        const size_t gpu_id, const size_t m, const size_t n, const double *Left, const double *Right, double *output, cusolverDnHandle_t cusolverH,
        double *d_Ainput, double *d_B, double *d_work, int *d_Ipiv, int *d_devInfo)
{
    cu_errchk(cudaSetDevice(gpu_id));
    cu_errchk(cudaMemcpy(d_Ainput, Left, sizeof(double) * m * m, cudaMemcpyHostToDevice));
    cu_errchk(cudaMemcpy(d_B, Right, sizeof(double) * m * n, cudaMemcpyHostToDevice));

    if (const auto errc = cusolverDnDgetrf(cusolverH, m, m, d_Ainput, m /* lda */, d_work, d_Ipiv, d_devInfo))
        LOG4_THROW("cusolverDnDgetrf call failed with " << int(errc));
    if (const auto errc = cusolverDnDgetrs(cusolverH, CUBLAS_OP_N, m, n, d_Ainput, m /* lda */, d_Ipiv, d_B, m /* ldb */, d_devInfo))
        LOG4_THROW("cusolverDnDgetrs call failed with " << int(errc));

    cu_errchk(cudaMemcpy(output, d_B, sizeof(double) * m * n, cudaMemcpyDeviceToHost));
}


std::tuple<magma_queue_t, magmaDouble_ptr, magmaDouble_ptr, magmaDouble_ptr, magmaDouble_ptr, magmaFloat_ptr, magmaInt_ptr>
init_magma_solver(const size_t m, const size_t b_n, const bool psd, const size_t gpu_id)
{
    cu_errchk(cudaSetDevice(gpu_id));
    magma_queue_t magma_queue = nullptr;

    magma_init(); // initialize Magma
    magma_queue_create(gpu_id, &magma_queue);
    if (!magma_queue) LOG4_THROW("Failed creating MAGMA queue.");

    magma_int_t err;
    magmaDouble_ptr d_a, d_b, d_x, d_wd;
    magmaFloat_ptr d_ws;
    auto piv = (magmaInt_ptr) malloc(m * sizeof(magma_int_t)); // host mem.
    if ((err = magma_dmalloc(&d_a, m * m)) < MAGMA_SUCCESS) // device memory for a
        LOG4_THROW("Failed calling magma_dmalloc d_a with error code " << err);
    if ((err = magma_dmalloc(&d_b, m * b_n)) < MAGMA_SUCCESS) // device memory for b
        LOG4_THROW("Failed calling magma_dmalloc d_b with error code " << err);
    if (psd) {
        if ((err = magma_dmalloc(&d_x, m * b_n)) < MAGMA_SUCCESS) // device memory for x
            LOG4_ERROR("Failed calling magma_dmalloc d_x with error code " << err);
        if ((err = magma_dmalloc(&d_wd, m * (m + b_n) + m)) < MAGMA_SUCCESS) // device memory for wd
            LOG4_THROW("Failed calling magma_dmalloc d_wd with error code " << err);
        if ((err = magma_smalloc(&d_ws, m * (m + b_n) + m)) < MAGMA_SUCCESS) // device memory for ws
            LOG4_THROW("Failed calling magma_dmalloc d_ws with error code " << err);
    } else {
        d_x = nullptr;
        d_wd = nullptr;
        d_ws = nullptr;
    }
    return {magma_queue, d_a, d_b, d_x, d_wd, d_ws, piv};
}


void uninit_magma_solver(
        const magma_queue_t &magma_queue,
        const magmaDouble_ptr d_a, const magmaDouble_ptr d_b, const magmaDouble_ptr d_x, const magmaDouble_ptr d_wd, const magmaFloat_ptr d_ws, const magmaInt_ptr piv, const size_t gpu_id)
{
    cu_errchk(cudaSetDevice(gpu_id));
    magma_int_t err;
    if (d_a && (err = magma_free(d_a)) < MAGMA_SUCCESS)
        LOG4_THROW("Failed calling magma_free d_a with error code " << err);
    if (d_b && (err = magma_free(d_b)) < MAGMA_SUCCESS)
        LOG4_THROW("Failed calling magma_free d_b with error code " << err);
    if (d_x && (err = magma_free(d_x)) < MAGMA_SUCCESS)
        LOG4_THROW("Failed calling magma_free d_x with error code " << err);
    if (d_wd && (err = magma_free(d_wd)) < MAGMA_SUCCESS)
        LOG4_THROW("Failed calling magma_free d_wd with error code " << err);
    if (d_ws && (err = magma_free(d_ws)) < MAGMA_SUCCESS)
        LOG4_THROW("Failed calling magma_free d_ws with error code " << err);
    if (piv) free(piv);

    if (magma_queue) magma_queue_destroy(magma_queue);

    if ((err = magma_finalize()) < MAGMA_SUCCESS)
        LOG4_THROW("Failed calling magma_finalize with error code " << err);
}


void iter_magma_solve(
        const int m, const int b_n, const double *a, const double *b, double *output, const magma_queue_t magma_queue,
        const magmaDouble_ptr d_a, const magmaDouble_ptr d_b, const magmaDouble_ptr d_x, const magmaDouble_ptr d_workd, const magmaFloat_ptr d_works, const bool psd, const size_t gpu_id)
{
    cu_errchk(cudaSetDevice(gpu_id));
    magma_int_t err, iter, info;

    magma_dsetmatrix(m, m, a, m, d_a, m, magma_queue); // copy a -> d_a
    magma_dsetmatrix(m, b_n, b, m, d_b, m, magma_queue); // copy b -> d_b

    if (!psd) goto __solve_dgesv;

    if ((err = magma_dshposv_gpu_expert(magma_uplo_t::MagmaLower, m, b_n, d_a, m, d_b, m, d_x, m, d_workd, d_works, &iter, magma_mode_t::MagmaHybrid, 1, 0, 0, 0, &info)) < MAGMA_SUCCESS || info != 0) {
        LOG4_WARN("Call to magma_dshposv_gpu_expert failed with error " << err << ", info " << info << ". Trying magma_dgesv_rbt.");
        if (iter < 0) {
            switch (iter) {
                case -1:
                    LOG4_DEBUG("Iterative magma_dshposv_gpu_expert returned -1 : the routine fell back to full precision for implementation - or machine-specific reasons");
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
    if ((err = magma_dgesv_rbt(magma_bool_t::MagmaTrue, m, b_n, d_a, m, d_b, m, &info)) < MAGMA_SUCCESS)
        LOG4_THROW("Failed calling magma_dgesv_rbt with error code " << err << ", info " << info);
    else if (psd)
        LOG4_DEBUG("Call to magma_dgesv_rbt triunfo.");
    magma_dgetmatrix(m, b_n, d_b, m, output, m, magma_queue); // copy solution d_b -> output
}

// Doesn't work with NVidia CuSolver 12.1
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

    cublas_safe_call(cusolverDnCreate(&cusolverH));
    cublas_safe_call(cublasCreate(&cublasH));

    int *d_devInfo;
    double *d_tau;
    double *d_work;
    double *d_work2;
    cu_errchk(cudaMalloc(&d_tau, sizeof(double) * M));
    cu_errchk(cudaMalloc(&d_devInfo, sizeof(int)));
    int bufSize, bufSize2;
    // in-place A = QR
    cublas_safe_call(
            cusolverDnDgeqrf_bufferSize(
                    cusolverH,
                    M,
                    N,
                    d_Ainput,
                    M,
                    &bufSize
            )
    )
    cu_errchk(cudaMalloc(&d_work, sizeof(double) * bufSize));
    cublas_safe_call(
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
    )

    // Q^T*b
    cublas_safe_call(
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
    )

    cu_errchk(cudaMalloc(&d_work2, sizeof(double) * bufSize2));
    cublas_safe_call(
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
    )

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
    cublas_safe_call(
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
    cublas_safe_call(cublasDestroy(cublasH));
    cublas_safe_call(cusolverDnDestroy(cusolverH));
}


void
call_gpu_overdetermined(
        const size_t Nrows, const size_t Ncols, const size_t Nrhs, const double *cpu_matrix,
        const double *cpu_rhs, double *cpu_output)
{
    svr::common::gpu_context gtx;
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

// Namespaces
}
}

#endif /* USE_CUDA */

