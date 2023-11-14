#ifdef USE_CUDA

#include <cassert>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublasLt.h>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "common/gpu_handler.hpp"
#include "common/cuda_util.cuh"

#include "cuqrsolve.hpp"
#include <thread>

#define SOLVE_TIMEOUT 240 // seconds
#define WATCH_SOLVER

namespace svr::solvers {

void __global__
G_kernel_from_distances(double *__restrict kernel, const double *__restrict dist, const size_t M, const double divisor)
{
    const size_t thr_ix = threadIdx.x;
    const size_t g_thr_ix = thr_ix + blockIdx.x * CUDA_BLOCK_SIZE;
    const size_t grid_size = CUDA_BLOCK_SIZE * gridDim.x;

    for (size_t i = g_thr_ix; i < M; i += grid_size) kernel[i] = 1. - dist[i] / divisor;
}

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

double
score_kernel(
        const double *ref_kernel /* colmaj order */,
        const double norm_ref,
        const double *Z /* colmaj order */,
        const size_t M,
        const double gamma)
{
    double *d_K, *d_Z, *d_ref;
    const size_t elem_ct = M * M;

    common::gpu_context ctx;
    cu_errchk(cudaSetDevice(ctx.phy_id()));
    cu_errchk(cudaMalloc(&d_K, elem_ct * sizeof(double)));
    cu_errchk(cudaMalloc(&d_Z, elem_ct * sizeof(double)));
    cu_errchk(cudaMemcpy(d_Z, Z, elem_ct * sizeof(double), cudaMemcpyHostToDevice));
    G_kernel_from_distances<<<CUDA_THREADS_BLOCKS(elem_ct)>>>(d_K, d_Z, elem_ct, 2. * gamma * gamma);
    cu_errchk(cudaFree(d_Z));
    cu_errchk(cudaDeviceSynchronize());
    cublasHandle_t cublasH;
    cublas_safe_call(cublasCreate(&cublasH));
    double norm_K;
    cublas_safe_call(cublasDnrm2(cublasH, elem_ct, d_K, M, &norm_K));
    cu_errchk(cudaDeviceSynchronize());
    double *d_score;
    cu_errchk(cudaMalloc(&d_score, sizeof(double)));
    cu_errchk(cudaMalloc(&d_ref, elem_ct * sizeof(double)));
    cu_errchk(cudaMemcpy(d_ref, ref_kernel, elem_ct * sizeof(double), cudaMemcpyHostToDevice));
    G_score_kernel<<<CUDA_THREADS_BLOCKS(M)>>>(d_score, d_K, d_ref, elem_ct, norm_K, norm_ref);
    double score;
    cu_errchk(cudaMemcpy(&score, d_score, sizeof(double), cudaMemcpyDeviceToHost));
    cu_errchk(cudaFree(d_score));
    cu_errchk(cudaFree(d_ref));
    cu_errchk(cudaFree(d_K));
    cublas_safe_call(cublasDestroy(cublasH));
    return 2. - score;
}

void kernel_from_distances(double *K, const double *Z, const size_t M, const double gamma)
{
    double *d_K, *d_Z;
    const size_t elem_ct = M * M;
    const size_t mat_size = elem_ct * sizeof(double);

    common::gpu_context ctx;
    cu_errchk(cudaSetDevice(ctx.phy_id()));
    cu_errchk(cudaMalloc(&d_K, mat_size));
    cu_errchk(cudaMalloc(&d_Z, mat_size));
    cu_errchk(cudaMemcpy(d_Z, Z, mat_size, cudaMemcpyHostToDevice));
    G_kernel_from_distances<<<CUDA_THREADS_BLOCKS(elem_ct)>>>(d_K, d_Z, elem_ct, 2. * gamma * gamma);
    cu_errchk(cudaMemcpy(K, d_K, mat_size, cudaMemcpyDeviceToHost));
    cu_errchk(cudaFree(d_K));
    cu_errchk(cudaFree(d_Z));
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


int
dyn_qrsolve(const size_t Nrows, const size_t lda, double *d_Ainput, double *d_rhs, double *d_B, const size_t gpu_phy_dev)
{
    const size_t Nright = 1;
    double *d_work = nullptr;
    int lwork = 0;
    int rc = 0;
    const size_t ldb = lda;
    const size_t Ncols = Nrows;
    const size_t m = Ncols;
    thrust::device_vector<int> d_Ipiv_vector(m);
    int *d_Ipiv = thrust::raw_pointer_cast(d_Ipiv_vector.data());
    thrust::device_vector<int> d_devInfo_vector(1);

    cusolverDnHandle_t cusolverH = nullptr;
    cublasHandle_t cublasH = nullptr;
    cusolverDnCreate(&cusolverH);
    cublas_safe_call(cublasCreate(&cublasH));
    cublas_safe_call(cusolverDnDgetrf_bufferSize(cusolverH, m, m, d_Ainput, lda, &lwork));
    cu_errchk(cudaMalloc((void **) &d_work, sizeof(double) * lwork));

    cu_errchk(cudaMemcpy(d_B, d_rhs, sizeof(double) * Nright * Nrows, cudaMemcpyDeviceToDevice));
    if (const auto errc = cusolverDnDgetrf(cusolverH, m, m, d_Ainput, lda, d_work, d_Ipiv, thrust::raw_pointer_cast(d_devInfo_vector.data()))) {
        LOG4_ERROR("cusolverDnDgetrf call failed with " << int(errc));
        rc = 255;
        goto __bail;
    }
    if (const auto errc = cusolverDnDgetrs(cusolverH, CUBLAS_OP_N, m, 1, d_Ainput, lda, d_Ipiv, d_B, ldb, thrust::raw_pointer_cast(d_devInfo_vector.data()))) {
        LOG4_ERROR("cusolverDnDgetrs call failed with " << int(errc));
        rc = 255;
        goto __bail;
    }
__bail:
    uint flags;
    int active;
    if (cuDevicePrimaryCtxGetState(gpu_phy_dev, &flags, &active) == CUDA_SUCCESS && active) {
        if (d_work) cu_errchk(cudaFree(d_work));
        if (cusolverH) cusolverDnDestroy(cusolverH);
        if (cublasH) cublasDestroy(cublasH);
    }
    return rc;
}


void dyn_gpu_solve(const size_t m, const double *Left, const double *Right, double *output)
{
    svr::common::gpu_context gtx;
    const size_t gpu_phy_id = gtx.phy_id();
    cudaSetDevice(gpu_phy_id);
#ifdef WATCH_SOLVER // A workaround of NVidia's solver looping bug
    bool done = false;
    std::mutex done_mx;
    std::condition_variable cv;
    std::unique_lock<std::mutex> ul(done_mx);
    std::thread cu_watcher([&](){
        cv.wait_for(ul, std::chrono::seconds(SOLVE_TIMEOUT));
        if (!done) {
            cudaSetDevice(gpu_phy_id);
            CUcontext pctx;
            cuCtxGetCurrent(&pctx);
            cuDevicePrimaryCtxReset(gpu_phy_id);
            cuDevicePrimaryCtxRelease(gpu_phy_id);
            cuCtxDestroy(pctx);
            LOG4_ERROR("Failed solving matrix of size " << m*m << " elements in specified time-out " << SOLVE_TIMEOUT << " secs.");
        }
    });
#endif
    thrust::device_vector<double> d_Left(m * m);
    thrust::device_vector<double> d_Right(m);
    thrust::device_vector<double> d_output(m);
    cu_errchk(cudaMemcpy(thrust::raw_pointer_cast(d_Left.data()), Left, sizeof(double) * m * m, cudaMemcpyHostToDevice));
    cu_errchk(cudaMemcpy(thrust::raw_pointer_cast(d_Right.data()), Right, sizeof(double) * m, cudaMemcpyHostToDevice));
    const auto rc = dyn_qrsolve(m, m, thrust::raw_pointer_cast(d_Left.data()), thrust::raw_pointer_cast(d_Right.data()), thrust::raw_pointer_cast(d_output.data()), gpu_phy_id);
#ifdef WATCH_SOLVER
    done = true;
    cv.notify_one();
#endif
    if (!rc) {
        cu_errchk(cudaMemcpy(output, thrust::raw_pointer_cast(d_output.data()), sizeof(double) * m, cudaMemcpyDeviceToHost));
    } else {
        memset(output, 0x80, sizeof(double) * m);
    }
#ifdef WATCH_SOLVER
    cu_watcher.join();
#endif
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


//##############################################################################
// Old solver by Emo
// TODO Port to CUSOLVERMG API
// TODO Warm start for online learn
void qrsolve(const size_t Nrows, const size_t Nright, const double *h_Ainput, const double *h_rhs, double *h_B)
{
    /* Nrows - number of rows in square matrix in h_Ainput
       Nright - number of right hand sides
       d_rhs - ptr to right hand side
       d_B - ptr to result;
    */
    svr::common::gpu_context gtx;
    cudaSetDevice(gtx.phy_id());

    const auto lda = Nrows;
    int lwork = 0;
    int info_gpu = 0;
    const auto ldb = lda;
    const auto Ncols = Nrows;
    const auto m = Ncols;
    thrust::device_vector<double> d_tau_vector(m);
    thrust::device_vector<double> d_A_vector(Nrows * lda);
    thrust::device_vector<double> d_B_vector(Nrows * Nright);

    thrust::device_vector<int> d_Ipiv_vector(m);
    int *d_Ipiv = thrust::raw_pointer_cast(d_Ipiv_vector.data());
    double *d_A = thrust::raw_pointer_cast(d_A_vector.data());
    double *d_B = thrust::raw_pointer_cast(d_B_vector.data());

    cu_errchk(cudaMemcpy(d_A, h_Ainput, sizeof(double) * Nrows * lda, cudaMemcpyHostToDevice));
    double *d_tau = thrust::raw_pointer_cast(d_tau_vector.data());
    thrust::device_vector<int> d_devInfo_vector(1);
    int *devInfo = thrust::raw_pointer_cast(d_devInfo_vector.data());
    cusolverDnHandle_t cusolverH;
    cublasHandle_t cublasH;
    cusolverDnCreate(&cusolverH);
    cublas_safe_call(cublasCreate(&cublasH));
    cublas_safe_call(cusolverDnDgetrf_bufferSize(cusolverH, m, m, d_A, lda, &lwork));
    thrust::device_vector<double> d_work_vector(lwork);
    double *d_work = thrust::raw_pointer_cast(d_work_vector.data());
    cu_errchk(cudaMemcpy(d_B, h_rhs, sizeof(double) * Nright * Nrows, cudaMemcpyHostToDevice));
    cublas_safe_call(cusolverDnDgetrf(cusolverH, m, m, d_A, lda, d_work, d_Ipiv, devInfo));
    //cudaDeviceSynchronize();
    cu_errchk(cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
    //cudaDeviceSynchronize();
    assert(0 == info_gpu);
    cublas_safe_call(cusolverDnDgetrs(cusolverH, CUBLAS_OP_N, m, Nright, /* nrhs */ d_A, lda, d_Ipiv, d_B, ldb, devInfo));
    //cudaDeviceSynchronize();
    cu_errchk(cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
    //cudaDeviceSynchronize();
    if (info_gpu != 0) std::cout << "cusolverDnDgetrs ended with error " << info_gpu << std::endl;
    //assert(0 == info_gpu);
    cu_errchk(cudaMemcpy(h_B, d_B, sizeof(double) * Nright * Nrows, cudaMemcpyDeviceToHost));
    /*if (cublasH) */cublasDestroy(cublasH);
    /*if (cusolverH) */cusolverDnDestroy(cusolverH);
    //cudaDeviceSynchronize();
}

void solve_bi_cg_stab()
{
    /***** BiCGStab Code *****/
/* ASSUMPTIONS:
   1. The cuSPARSE and cuBLAS libraries have been initialized.
   2. The appropriate memory has been allocated and set to zero.
   3. The matrix A (valA, csrRowPtrA, csrColIndA) and the incomplete-
      LU lower L (valL, csrRowPtrL, csrColIndL)  and upper U (valU,
      csrRowPtrU, csrColIndU) triangular factors have been
      computed and are present in the device (GPU) memory. */

//create the info and analyse the lower and upper triangular factors

#if 0 // TODO port to dense cublas
    cusparseCreateSolveAnalysisInfo(&infoL);
    cusparseCreateSolveAnalysisInfo(&infoU);
    cusparseDcsrsv_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            n, descrL, valL, csrRowPtrL, csrColIndL, infoL);
    cusparseDcsrsv_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            n, descrU, valU, csrRowPtrU, csrColIndU, infoU);

//1: compute initial residual r = b - A x0 (using initial guess in x)
    cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, n, 1.0,
                   descrA, valA, csrRowPtrA, csrColIndA, x, 0.0, r);
    cublasDscal(n,-1.0, r, 1);
    cublasDaxpy(n, 1.0, f, 1, r, 1);
//2: Set p=r and \tilde{r}=r
    cublasDcopy(n, r, 1, p, 1);
    cublasDcopy(n, r, 1, rw,1);
    nrmr0 = cublasDnrm2(n, r, 1);

//3: repeat until convergence (based on max. it. and relative residual)
    for (i=0; i<maxit; i++){
        //4: \rho = \tilde{r}^{T} r
        rhop= rho;
        rho = cublasDdot(n, rw, 1, r, 1);
        if (i > 0){
            //12: \beta = (\rho_{i} / \rho_{i-1}) ( \alpha / \omega )
            beta= (rho/rhop)*(alpha/omega);
            //13: p = r + \beta (p - \omega v)
            cublasDaxpy(n,-omega,q, 1, p, 1);
            cublasDscal(n, beta, p, 1);
            cublasDaxpy(n, 1.0,  r, 1, p, 1);
        }
        //15: M \hat{p} = p (sparse lower and upper triangular solves)
        cusparseDcsrsv_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                             n, 1.0, descrL, valL, csrRowPtrL, csrColIndL,
                             infoL, p, t);
        cusparseDcsrsv_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                             n, 1.0, descrU, valU, csrRowPtrU, csrColIndU,
                             infoU, t, ph);

        //16: q = A \hat{p} (sparse matrix-vector multiplication)
        cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, n, 1.0,
                       descrA, valA, csrRowPtrA, csrColIndA, ph, 0.0, q);

        //17: \alpha = \rho_{i} / (\tilde{r}^{T} q)
        temp = cublasDdot(n, rw, 1, q, 1);
        alpha= rho/temp;
        //18: s = r - \alpha q
        cublasDaxpy(n,-alpha, q, 1, r, 1);
        //19: x = x + \alpha \hat{p}
        cublasDaxpy(n, alpha, ph,1, x, 1);

        //20: check for convergence
        nrmr = cublasDnrm2(n, r, 1);
        if (nrmr/nrmr0 < tol){
            break;
        }

        //23: M \hat{s} = r (sparse lower and upper triangular solves)
        cusparseDcsrsv_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                             n, 1.0, descrL, valL, csrRowPtrL, csrColIndL,
                             infoL, r, t);
        cusparseDcsrsv_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                             n, 1.0, descrU, valU, csrRowPtrU, csrColIndU,
                             infoU, t, s);

        //24: t = A \hat{s} (sparse matrix-vector multiplication)
        cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, n, 1.0,
                       descrA, valA, csrRowPtrA, csrColIndA, s, 0.0, t);

        //25: \omega = (t^{T} s) / (t^{T} t)
        temp = cublasDdot(n, t, 1, r, 1);
        temp2= cublasDdot(n, t, 1, t, 1);
        omega= temp/temp2;
        //26: x = x + \omega \hat{s}
        cublasDaxpy(n, omega, s, 1, x, 1);
        //27: r = s - \omega t
        cublasDaxpy(n,-omega, t, 1, r, 1);

        //check for convergence
        nrmr = cublasDnrm2(n, r, 1);
        if (nrmr/nrmr0 < tol){
            break;
        }
    }

//destroy the analysis info (for lower and upper triangular factors)
    cusparseDestroySolveAnalysisInfo(infoL);
    cusparseDestroySolveAnalysisInfo(infoU);
#endif
}

}

#endif /* USE_CUDA */

