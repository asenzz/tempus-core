#include <npp.h>
#include <thrust/sort.h>
#include <cmath>
#include <thread>
#include <cublas_v2.h>
#include <magma_types.h>
#include <magma_v2.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cusolverDn.h>
#include <error.h>
#include "common/compatibility.hpp"
#include "common/gpu_handler.hpp"
#include "cuqrsolve.cuh"
#include "common/constants.hpp"
#include "onlinesvr.hpp"
#include "kernel_base.cuh"
#include "ScalingFactorService.hpp"
#include "common/cuda_util.cuh"

namespace svr {
namespace solvers {


__global__ void G_normalize_distances_I(double *x, const double sf, const double dc, const uint32_t n)
{
    CU_STRIDED_FOR_i(n) common::scale_I(x[i], sf, dc);
}

__global__ void G_div_I(RPTR(double) x, const double a, const uint32_t n)
{
    CU_STRIDED_FOR_i(n) x[i] /= a;
}

__global__ void G_mul_I(RPTR(double) x, const double a, const uint32_t n)
{
    CU_STRIDED_FOR_i(n) x[i] *= a;
}

// error = abs(labels_train - K_train * solved)
__global__ void G_absdif(CRPTRd labels_train, RPTR(double) error_mat, const uint32_t mn)
{
    CU_STRIDED_FOR_i(mn) error_mat[i] = fabs(labels_train[i] - error_mat[i]);
}

// work = abs(j_test_labels - (j_K_test * best_solution - svr_epsilon))
__global__ void G_pred_absdif_I(CRPTRd j_test_labels, RPTR(double) work, const double svr_epsilon, const uint32_t test_len_n)
{
    CU_STRIDED_FOR_i(test_len_n) work[i] = fabs(j_test_labels[i] - work[i] + svr_epsilon);
}

__global__ void G_set_diag(RPTR(double) K, CRPTRd d, const uint32_t m)
{
    CU_STRIDED_FOR_i(m) K[i * m + i] = d[i];
}

__global__ void G_set_diag(RPTR(double) K, const double d, const uint32_t m)
{
    CU_STRIDED_FOR_i(m) K[i * m + i] = d;
}

__global__ void G_augment_K(RPTR(double) K, CRPTRd w, const double d, const uint32_t m)
{
    const auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    const auto j = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= m) return;

    const auto ij = i * m + j;
    K[ij] = i == j ? d : K[ij] * w[ij];
}

__global__ void G_calc_epsco(const double *const K, const double *const L, double *epsco, const uint32_t m, const uint32_t n, const uint32_t ld, const double sum_L)
{
    CU_STRIDED_FOR_i(m) {
        double sum = K[i];
        for (uint32_t j = 1; j < m; ++j) sum += K[j * ld + i];
        double diff = L[i] - sum;
        for (uint32_t j = 1; j < n; ++j) diff += L[j * ld + i] - sum;
        diff /= n;
        epsco[i] = diff;
    }
}

__global__ void G_calc_epsco(CRPTRd K, CRPTRd L, RPTR(double) epsco, const uint32_t m, const uint32_t ld, const double sum_L)
{
    CU_STRIDED_FOR_i(m) {
        double sum_K = K[i];
        UNROLL(common::AppConfig::C_default_kernel_length / 10)
        for (uint32_t j = 1; j < m; ++j) sum_K += K[j * ld + i];
        epsco[i] = L[i] - sum_K - sum_L;
    }
}

double *cu_calc_epscos(CPTRd K, CPTRd L, const uint32_t m, const uint32_t n, const cudaStream_t custream)
{
    double *d_epsco;
    cu_errchk(cudaMallocAsync((void **) &d_epsco, m * sizeof(double), custream));
    if (n == 1) G_calc_epsco<<<CU_BLOCKS_THREADS(m), 0, custream>>>(K, L, d_epsco, m, m, 0.);
    else G_calc_epsco<<<CU_BLOCKS_THREADS(m), 0, custream>>>(K, L, d_epsco, m, n, m, 0);
    return d_epsco;
}

double cu_calc_epsco(const double *const K, const double *const L, const uint32_t m, const uint32_t n, const uint32_t ld, const double sum_L, const cudaStream_t custream)
{
    double *d_epsco;
    cu_errchk(cudaMallocAsync((void **) &d_epsco, m * sizeof(double), custream));
    if (n == 1) G_calc_epsco<<<CU_BLOCKS_THREADS(m), 0, custream>>>(K, L, d_epsco, m, ld, sum_L);
    else
        LOG4_THROW("Not implemented."); // G_calc_epsco<<<CU_BLOCKS_THREADS(m), 0, custream>>>(K, L, d_epsco, m, n, ld);
    const auto mean_epsco = solvers::mean(d_epsco, m, custream);
    cu_errchk(cudaFreeAsync(d_epsco, custream));
    LOG4_TRACE("Mean epsco " << mean_epsco << " m " << m << " n " << n);
    return mean_epsco;
}


const uint16_t score_weights::n_gpus = common::gpu_handler_1::get().get_gpu_devices_count();

score_weights::score_weights(const arma::mat &K, const arma::mat &L, const uint32_t m, const uint32_t n, const uint64_t mn, const uint64_t mm) :
        m(m), n(n), mn(mn), mm(mm), L_size(L.n_elem * sizeof(double))
{
    K_rhs_dev.resize(n_gpus);
    const auto K_size = K.n_elem * sizeof(double);
    OMP_FOR_i(n_gpus) {
        DEV_CUSTREAM(i);
        cu_errchk(cudaMallocAsync((void **) &K_rhs_dev[i].K, K_size, custream));
        cu_errchk(cudaMallocAsync((void **) &K_rhs_dev[i].L, L_size, custream));
        cu_errchk(cudaMemcpyAsync(K_rhs_dev[i].K, K.mem, K_size, cudaMemcpyHostToDevice, custream));
        cu_errchk(cudaMemcpyAsync(K_rhs_dev[i].L, L.mem, L_size, cudaMemcpyHostToDevice, custream));
        cusyndestroy(custream);

        UNROLL(streams_gpu)
        for (DTYPE(streams_gpu) j = 0; j < streams_gpu; ++j) {
            cudaStream_t custream_j;
            cu_errchk(cudaStreamCreateWithFlags(&custream_j, C_cu_default_stream_flags));
            cublasHandle_t cublas_H;
            cb_errchk(cublasCreate(&cublas_H));
            cb_errchk(cublasSetStream(cublas_H, custream_j));
            cb_errchk(cublasSetPointerMode(cublas_H, CUBLAS_POINTER_MODE_HOST));
            double *tmp_L;
            cu_errchk(cudaMallocAsync((void **) &tmp_L, L_size, custream_j));
            K_rhs_dev[i].stream_cublas.emplace_back(dev_ctx::stream_ctx{custream_j, cublas_H, tmp_L});
        }
    }
}

score_weights::~score_weights()
{
    OMP_FOR_i(n_gpus) {
        DEV_CUSTREAM(i);
        UNROLL(streams_gpu)
        for (auto &stream_cubla: K_rhs_dev[i].stream_cublas) {
            cu_errchk(cudaFreeAsync((void *) stream_cubla.tmp_L, stream_cubla.custream));
            cb_errchk(cublasDestroy(stream_cubla.cublas_H));
            cusyndestroy(stream_cubla.custream);
        }
        cu_errchk(cudaFreeAsync((void *) K_rhs_dev[i].K, custream));
        cu_errchk(cudaFreeAsync((void *) K_rhs_dev[i].L, custream));
        cusyndestroy(custream);
    }
}

double score_weights::operator()(CPTRd weights) const
{
    constexpr double one = 1, minus_one = -1;

    const common::gpu_context_<streams_gpu> ctx(false);
    // if (!ctx) return std::numeric_limits<double>::quiet_NaN();
    const auto dev_phy_id = ctx.phy_id();

    const auto &ctx_dev = K_rhs_dev[dev_phy_id];
    const auto &ctx_stream = ctx_dev.stream_cublas[ctx.stream_id()];

    cu_errchk(cudaSetDevice(dev_phy_id));
    const auto d_weights = cumallocopy(weights, ctx_stream.custream, mn);
    cu_errchk(cudaMemcpyAsync(ctx_stream.tmp_L, ctx_dev.L, L_size, cudaMemcpyDeviceToDevice, ctx_stream.custream));
    cb_errchk(cublasDgemm(ctx_stream.cublas_H, CUBLAS_OP_N, CUBLAS_OP_N, m, n, m, &one, ctx_dev.K, m, d_weights, m, &minus_one, ctx_stream.tmp_L, m));
    cu_errchk(cudaFreeAsync(d_weights, ctx_stream.custream));
    // const auto res = sumabs(ctx_stream.tmp_L, mn, ctx_stream.custream);
    double res;
    cb_errchk(cublasDasum(ctx_stream.cublas_H, mn, ctx_stream.tmp_L, 1, &res));
    cu_errchk(cudaStreamSynchronize(ctx_stream.custream));
    LOG4_TRACE("Score " << res << " for weights " << common::to_string(weights, std::min<uint32_t>(4, m * n)) << " on device " << dev_phy_id);
    return res;
}


void __global__
G_score_kernel(
        RPTR(double) score,
        CRPTRd kernel, CRPTRd ref,
        const uint32_t M, const double norm_ker, const double norm_ref)
{
    const auto thr_ix = threadIdx.x;
    const auto g_thr_ix = thr_ix + blockIdx.x * common::C_cu_block_size;
    const auto grid_size = common::C_cu_block_size * gridDim.x;

    double sum = 0;
    for (auto i = g_thr_ix; i < M; i += grid_size) sum += kernel[i] * ref[i];

    __shared__ double _sh_sum[common::C_cu_block_size];
    _sh_sum[thr_ix] = sum;
    __syncthreads();

    for (auto size = common::C_cu_block_size / 2; size > 0; size /= 2) {
        if (thr_ix >= size) continue;
        _sh_sum[thr_ix] += _sh_sum[thr_ix + size];
        __syncthreads();
    }

    if (thr_ix == 0) *score = _sh_sum[0] / (norm_ker * norm_ref);
}


__global__ void G_calc_gamma(CRPTRd Z, CRPTRd L, const uint32_t m, const uint32_t n, const uint32_t mm, const uint32_t mn, const double bias, RPTR(double) gamma)
{
#if 0
    CU_STRIDED_FOR_i(m) {
        double Z_mean = Z[i], Z_min = Z[i], Z_max = Z[i];
        for (uint32_t j = m; j < mm; j += m) {
            const auto Zij = Z[j + i];
            Z_mean += Zij;
            MINAS(Z_min, Zij);
            MAXAS(Z_max, Zij);
        }
        Z_mean /= m;
        double L_mean = L[i], L_min = L[i], L_max = L[i];
        for (uint32_t j = m; j < mn; j += m) {
            const auto Lij = L[j + i];
            L_mean += Lij;
            MINAS(L_min, Lij);
            MAXAS(L_max, Lij);
        }
        L_mean /= n;
        const auto min_qgamma = kernel::path::calc_qgamma(Z_mean, Z_min, L_mean, L_max, m);
        gamma[i] = bias * (kernel::path::calc_qgamma(Z_mean, Z_max, L_mean, L_min, m) - min_qgamma) + min_qgamma;
    }
#endif
}

__global__ void G_calc_gamma(CRPTRd Z, CRPTRd L, const uint32_t m, const uint32_t mm, const double bias, RPTR(double) gamma)
{
#if 0
    CU_STRIDED_FOR_i(m) {
        double Z_mean = Z[i], Z_min = Z[i], Z_max = Z[i];
        for (uint32_t j = m; j < mm; j += m) {
            const auto Zij = Z[j + i];
            Z_mean += Zij;
            MINAS(Z_min, Zij);
            MAXAS(Z_max, Zij);
        }
        Z_mean /= m;
        const auto min_qgamma = kernel::path::calc_qgamma(Z_mean, Z_min, L[i], m);
        gamma[i] = bias * (kernel::path::calc_qgamma(Z_mean, Z_max, L[i], m) - min_qgamma) + min_qgamma;
    }
#endif
}

// Anti-symmetric gamma
__global__ void G_calc_gammas(const double *const Z, const uint32_t lda, const double *const L, const uint32_t m, const uint32_t m_lda, const double sum_L, double *gammas)
{
    CU_STRIDED_FOR_i(m) {
        double Z_row_sum = Z[i];
        UNROLL(common::AppConfig::C_default_kernel_length / 10)
        for (uint32_t j = lda + i; j < m_lda; j += lda) Z_row_sum += Z[j];
        gammas[i] = Z_row_sum / (L[i] - sum_L); // for L = L_col_sum + Z_row_sum / g, therefore g = Z_row_sum / (L - L_col_sum)
    }
}

double *cu_calc_gammas(CPTRd Z, CPTRd L, const uint32_t m, const uint32_t n, const double bias, const cudaStream_t stm)
{
    const auto mm = m * m;
    double *d_gammas;
    cu_errchk(cudaMallocAsync((void **) &d_gammas, m * sizeof(double), stm));
    if (n == 1) G_calc_gamma<<<CU_BLOCKS_THREADS(m), 0, stm>>>(Z, L, m, mm, bias, d_gammas);
    else {
        const auto mn = m * n;
        G_calc_gamma<<<CU_BLOCKS_THREADS(m), 0, stm>>>(Z, L, m, n, mm, mn, bias, d_gammas);
    }
    return d_gammas;
}

double *cu_calc_gammas(const double *const Z, const uint32_t lda, const double *const L, const uint32_t m, const uint32_t n, const double sum_L, const cudaStream_t stm)
{
    double *d_gammas;
    cu_errchk(cudaMallocAsync((void **) &d_gammas, m * sizeof(double), stm));
    if (n == 1) G_calc_gammas<<<CU_BLOCKS_THREADS(m), 0, stm>>>(Z, lda, L, m, m * lda, sum_L, d_gammas);
    else {
        LOG4_THROW("Not implemented.");
    }
    return d_gammas;
}


// TODO Unify calc gamma and mean in a single kernel
double cu_calc_gamma(const double *const Z, const uint32_t lda, const double *const L, const uint32_t m, const uint32_t n, const cudaStream_t stm)
{
    const auto d_gammas = cu_calc_gammas(Z, lda, L, m, n, 0, stm);
    const auto res = mean(d_gammas, m, stm);
    cu_errchk(cudaFreeAsync(d_gammas, stm));
    return res;
}

double cu_calc_gamma(CPTRd Z, CPTRd L, const uint32_t m, const uint32_t n, const double bias, const cudaStream_t stm)
{
    const auto d_gamma = cu_calc_gammas(Z, L, m, n, bias, stm);
    const auto res = mean(d_gamma, m, stm);
    cu_errchk(cudaFree(d_gamma));
    return res;
}

double cu_calc_gamma(CPTRd Z, const double L_mean, const double train_len, const uint32_t n_elem, const cudaStream_t stm)
{
    const auto Z_mm = solvers::mean(Z, n_elem, stm);
    const auto g = 0;//kernel::path::calc_g(train_len, Z_mm, L_mean);
    LOG4_TRACE("Mean Z " << Z_mm << ", mean L " << L_mean << ", n " << train_len << ", gamma " << g);
    return g;
}

std::pair<double, double> cu_calc_minmax_gamma(CPTRd Z, const mmm_t &train_L_m, const double train_len, const uint32_t Z_n_elem, const cudaStream_t stm)
{
    const auto [Z_mean, Z_min, Z_max] = meanminmax(Z, Z_n_elem, stm);
    assert(common::isnormalz(Z_mean));
    assert(common::isnormalz(Z_min));
    assert(common::isnormalz(Z_max));
//    return {kernel::path::calc_qgamma(Z_mean, Z_min, train_L_m.mean, train_L_m.min, train_len),
//            kernel::path::calc_qgamma(Z_mean, Z_max, train_L_m.mean, train_L_m.max, train_len)};
    return {0, 0};
}

double score_kernel(CPTRd ref_kernel /* colmaj order */, const double norm_ref, CPTRd Z /* colmaj order */, const uint32_t m, const double gamma)
{
#if 0
    double *d_K, *d_Z, *d_ref;
    const auto mm = m * m;
    const auto mat_size = mm * sizeof(double);

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
        CRPTRd d_in,
        RPTR(double) d_ou,
        const uint32_t M, const uint32_t N, const uint32_t subM)
{
    const auto i = threadIdx.x + blockIdx.x * blockDim.x;
    const auto j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= subM || j >= N) return;
    d_ou[j * subM + i] = d_in[j * M + i];
}


std::tuple<cusolverDnHandle_t, double *, double *, double *, int32_t *, int32_t *>
init_cusolver(const uint32_t gpu_id, const uint32_t m, const uint32_t n)
{
    cu_errchk(cudaSetDevice(gpu_id));

    cusolverDnHandle_t cusolverH;
    cublasHandle_t cublasH;
    int32_t lwork;
    double *d_Ainput, *d_B, *d_work;
    int32_t *d_Ipiv, *d_devInfo;

    cs_errchk(cusolverDnCreate(&cusolverH));
    cb_errchk(cublasCreate(&cublasH));
    cu_errchk(cudaMalloc((void **) &d_Ainput, m * m * sizeof(double)));
    cu_errchk(cudaMalloc((void **) &d_B, m * n * sizeof(double)));
    cs_errchk(cusolverDnDgetrf_bufferSize(cusolverH, m, m, d_Ainput, m /* lda */, &lwork));
    cu_errchk(cudaMalloc((void **) &d_work, sizeof(double) * lwork));
    cu_errchk(cudaMalloc((void **) &d_Ipiv, m * sizeof(int32_t)));
    cu_errchk(cudaMalloc((void **) &d_devInfo, sizeof(int32_t)));

    return {cusolverH, d_Ainput, d_B, d_work, d_Ipiv, d_devInfo};
}


void uninit_cusolver(const uint32_t gpu_id, const cusolverDnHandle_t cusolverH, double *d_Ainput, double *d_B, double *d_work, int32_t *d_Ipiv, int32_t *d_devInfo)
{
    cu_errchk(cudaSetDevice(gpu_id));

    if (d_Ainput) cu_errchk(cudaFree(d_Ainput));
    if (d_B) cu_errchk(cudaFree(d_B));
    if (d_work) cu_errchk(cudaFree(d_work));
    if (d_Ipiv) cu_errchk(cudaFree(d_Ipiv));
    if (d_devInfo) cu_errchk(cudaFree(d_devInfo));

    if (cusolverH) cs_errchk(cusolverDnDestroy(cusolverH));
}

void dyn_gpu_solve(const cusolverDnHandle_t cusolver_H, const uint32_t m, const uint32_t n, CPTRd d_a, double *d_b, double *d_work, int32_t *d_piv, int32_t *d_info)
{
    cs_errchk(cusolverDnDgetrf(cusolver_H, m, m, (double *) d_a, m, d_work, d_piv, d_info));
    cs_errchk(cusolverDnDgetrs(cusolver_H, CUBLAS_OP_N, m, n, d_a, m, d_piv, d_b, m, d_info));
}

void h_dyn_gpu_solve(
        const uint32_t gpu_id, const uint32_t m, const uint32_t n, CPTRd h_K, CPTRd h_L, double *h_weights, cusolverDnHandle_t cusolver_H,
        double *d_a, double *d_b, double *d_work, int32_t *d_piv, int32_t *d_info)
{
    cu_errchk(cudaSetDevice(gpu_id));
    cu_errchk(cudaMemcpy(d_a, h_K, sizeof(double) * m * m, cudaMemcpyHostToDevice));
    cu_errchk(cudaMemcpy(d_b, h_L, sizeof(double) * m * n, cudaMemcpyHostToDevice));
    dyn_gpu_solve(cusolver_H, m, n, d_a, d_b, d_work, d_piv, d_info);
    cu_errchk(cudaMemcpy(h_weights, d_b, sizeof(double) * m * n, cudaMemcpyDeviceToHost));
}


std::tuple<magma_queue_t, magmaDouble_ptr, magmaDouble_ptr, magmaDouble_ptr, magmaDouble_ptr, magmaFloat_ptr, magmaInt_ptr>
init_magma_solver(const uint32_t m, const uint32_t b_n, const bool psd, const uint32_t gpu_id)
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
init_magma_batch_solver(const uint32_t batch_size, const uint32_t m, const uint32_t n)
{
    std::vector<magmaDouble_ptr> d_a(batch_size, nullptr), d_b(batch_size, nullptr);
    UNROLL()
    for (uint32_t i = 0; i < batch_size; ++i) {
        ma_errchk(magma_dmalloc(&d_a[i], m * m));
        ma_errchk(magma_dmalloc(&d_b[i], m * n));
    }
    return {d_a, d_b};
}


void uninit_magma_solver(
        const magma_queue_t &magma_queue,
        const magmaDouble_ptr d_a, const magmaDouble_ptr d_b, const magmaDouble_ptr d_x, const magmaDouble_ptr d_wd, const magmaFloat_ptr d_ws, const magmaInt_ptr piv,
        const uint32_t gpu_id)
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
    UNROLL()
    for (uint32_t i = 0; i < d_a.size(); ++i) {
        if (d_a[i]) ma_errchk(magma_free(d_a[i]));
        if (d_b[i]) ma_errchk(magma_free(d_b[i]));
    }
}


void iter_magma_solve(
        const int32_t m, const int32_t b_n, CPTRd a, CPTRd b, double *output, const magma_queue_t magma_queue,
        const magmaDouble_ptr d_a, const magmaDouble_ptr d_b, const magmaDouble_ptr d_x, const magmaDouble_ptr d_workd,
        const magmaFloat_ptr d_works, const bool psd, const uint32_t gpu_id)
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
    ma_errchk(magma_dgesv_rbt(magma_bool_t::MagmaTrue, m, b_n, d_a, m, d_b, m, &info));
    if (psd) LOG4_DEBUG("Call to magma_dgesv_rbt triunfo.");
    magma_dgetmatrix(m, b_n, d_b, m, output, m, magma_queue); // copy solution d_b -> output
}

void iter_magma_solve(
        const int32_t m, const int32_t n, CPTRd a, CPTRd b, double *output, const magma_queue_t &magma_queue,
        const magmaDouble_ptr d_a, const magmaDouble_ptr d_b)
{
    magma_int_t info;
    magma_dsetmatrix(m, m, a, m, d_a, m, magma_queue);
    magma_dsetmatrix(m, n, b, m, d_b, m, magma_queue);
    cu_errchk(cudaDeviceSynchronize());
    ma_errchk(magma_dgesv_rbt(magma_bool_t::MagmaTrue, m, n, d_a, m, d_b, m, &info));
    cu_errchk(cudaDeviceSynchronize());
    magma_dgetmatrix(m, n, d_b, m, output, m, magma_queue);
    cu_errchk(cudaDeviceSynchronize());
}

void iter_magma_batch_solve(
        const int32_t m, const int32_t n, const std::deque<arma::mat> &a, const std::deque<arma::mat> &b, std::deque<arma::mat> &output,
        const magma_queue_t magma_queue, std::vector<magmaDouble_ptr> &d_a, std::vector<magmaDouble_ptr> &d_b, const uint32_t gpu_id)
{
    const auto batch_size = a.size();
    LOG4_DEBUG("m " << magma_int_t(m) << ", n " << magma_int_t(n) << ", batch size " << batch_size);
// #pragma omp parallel for schedule(static, 1) num_threads(adj_threads(batch_size))
    cu_errchk(cudaSetDevice(gpu_id));
    UNROLL()
    for (uint32_t i = 0; i < batch_size; ++i) {
        magma_dsetmatrix(m, m, a[i].mem, m, d_a[i], m, magma_queue);
        magma_dsetmatrix(m, n, b[i].mem, m, d_b[i], m, magma_queue);
    }
    std::vector<magma_int_t> info(batch_size);
    cu_errchk(cudaSetDevice(gpu_id));
    auto da_data = &d_a[0];
    auto db_data = &d_b[0];
    ma_errchk(magma_dgesv_rbt_batched(magma_int_t(m), magma_int_t(n), da_data, m, db_data, m, info.data(), magma_int_t(batch_size), magma_queue));

#pragma omp parallel for schedule(static, 1) num_threads(adj_threads(batch_size))
    for (uint32_t i = 0; i < batch_size; ++i) {
        cu_errchk(cudaSetDevice(gpu_id));
        magma_dgetmatrix(m, n, d_b[i], m, output[i].memptr(), m, magma_queue); // copy solution d_b -> output
    }
}

// Doesn't work with NVidia CuSolver 12.1, leaks memory
void dyn_magma_solve(const int32_t m, const int32_t b_n, CPTRd a, CPTRd b, double *output, magma_queue_t magma_queue,
                     const magmaInt_ptr piv, const magmaDouble_ptr d_a, const magmaDouble_ptr d_b, const uint32_t gpu_id)
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
qrsolve_over(const uint32_t Nrows, const uint32_t Ncols, const uint32_t Nrhs, double *d_Ainput, double *d_b, double *d_output)
{
    // define handles
    cusolverDnHandle_t cusolverH = nullptr;
    cublasHandle_t cublasH = nullptr;
    const auto M = Nrows;
    const auto N = Ncols;
    const auto K = Nrhs;

    cs_errchk(cusolverDnCreate(&cusolverH));
    cb_errchk(cublasCreate(&cublasH));

    int32_t *d_devInfo;
    double *d_tau;
    double *d_work;
    double *d_work2;
    cu_errchk(cudaMalloc(&d_tau, sizeof(double) * M));
    cu_errchk(cudaMalloc(&d_devInfo, sizeof(int32_t)));
    int32_t bufSize, bufSize2;

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
                    std::min<uint32_t>(M, N),
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
    dim3 blk_size((N + C_cu_tile_dim.x - 1) / C_cu_tile_dim.x, (N + C_cu_tile_dim.y - 1) / C_cu_tile_dim.y);
    gpu_copy_upper_submatrix<<<blk_size, C_cu_tile_dim>>>(d_Ainput, d_R, M, N, N);
    blk_size = dim3((N + C_cu_tile_dim.x - 1) / C_cu_tile_dim.x, (K + C_cu_tile_dim.y - 1) / C_cu_tile_dim.y);
    gpu_copy_upper_submatrix<<<blk_size, C_cu_tile_dim>>>(d_b, d_output, M, K, N);

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
        const uint32_t Nrows, const uint32_t Ncols, const uint32_t Nrhs, CPTRd cpu_matrix,
        CPTRd cpu_rhs, double *cpu_output)
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
/*
__global__ void G_irwls_op2(
        CRPTRd err,
        CRPTRd K,
        CRPTRd labels,
        RPTR(double) out_K,
        RPTR(double) solved,
        const double additive,
        const uint32_t m,
        const uint32_t mn,
        const uint32_t mm)
{
    double sum_err_i;
    CU_STRIDED_FOR_i(mm) {
        if (i < mn) solved[i] = (err[i] + additive) * labels[i];
        sum_err_i = 0;
        out_K[i] = K[i];
UNROLL()
        for (uint32_t j = i % m; j < mn; j += m) sum_err_i += err[j] + additive;
        out_K[i] *= sum_err_i;
    }
}
*/
/* LDA version */
__global__ void G_irwls_op2(
        CRPTRd err,
        CRPTRd K,
        const uint32_t ldK,
        CRPTRd labels,
        RPTR(double) out_K,
        RPTR(double) solved,
        const double additive,
        const uint32_t m,
        const uint32_t mn,
        const uint32_t mm)
{
    CU_STRIDED_FOR_i(mm) {
        const auto row = i % m;
        const auto in_i = (i / m) * ldK + row;
        if (i < mn) solved[i] = (err[i] + additive) * labels[in_i];
        auto sum_err_i = additive;
        for (auto j = row; j < mn; j += m) sum_err_i += err[j];
        out_K[i] = K[in_i] * sum_err_i;
    }
}


__global__ void G_abs(RPTR(double) inout, const uint32_t N)
{
    CU_STRIDED_FOR_i(N)inout[i] = _ABS(inout[i]);
}


__global__ void G_sumabsdif(CRPTRd d_in1, CRPTRd d_in2, RPTR(double) d_result_sum, const size_t n)
{
    double sum = 0;
    CU_STRIDED_FOR_i(n) sum += fabs(d_in1[i] - d_in2[i]);
    atomicAdd(d_result_sum, sum);
}

double cu_mae(CRPTRd d_in1, CRPTRd d_in2, const size_t n, const cudaStream_t custream)
{
    auto const d_result_sum = cucalloc<double>(custream);
    G_sumabsdif<<<CU_BLOCKS_THREADS(n), 0, custream>>>(d_in1, d_in2, d_result_sum, n);
    double r;
    cufreecopy(&r, d_result_sum, custream);
    return r / n;
}

#if 1

template<const uint32_t block_size> __global__ void
G_sumabs(CRPTRd d_input, RPTR(double) d_result_sum, const uint32_t n)
{
    __shared__ double sumdata[common::C_cu_block_size];
    auto i = blockIdx.x * block_size + tid;
    if (i < n) {
        sumdata[tid] = fabs(d_input[i]);
        const auto stride1 = blockDim.x * gridDim.x;
        UNROLL()
        for (i += stride1; i < n; i += stride1) sumdata[tid] += fabs(d_input[i]);
    } else sumdata[tid] = 0;

    __syncthreads();
    const auto sh_limit = _MIN(n, block_size);
#define stride_reduce_sum(block_low_)                        \
        if (block_size >= block_low_) {                      \
            constexpr uint32_t stride2 = block_low_ / 2;     \
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
    warp_reduce_sum<block_size>(sumdata, tid, sh_limit);

    if (tid) return;
    atomicAdd(d_result_sum, *sumdata);
}

double sumabs(CPTRd d_in, const size_t n, const cudaStream_t stm)
{
    double sum, *d_sum = cucalloc<double>(stm);
    G_sumabs<common::C_cu_block_size><<<CU_BLOCKS_THREADS(n), 0, stm>>>(d_in, d_sum, n);
    cufreecopy(&sum, d_sum, stm);
    cu_errchk(cudaStreamSynchronize(stm));
    return sum;
}

#else

#define CU_SHLEN_CHECK(N_)                              \
    const auto block_start_ = blockIdx.x * blockDim.x;  \
    if (block_start_ >= (N_)) return;                   \
    auto sh_len = (N_) - block_start_;                  \
    if (sh_len > blockDim.x) sh_len = blockDim.x;       \
    if (tid >= sh_len) return;


__global__ void
G_sumabs(CRPTRd d_input, RPTR(double) d_result_sum, const uint32_t n)
{
    extern __shared__ double sumdata[];

    CU_SHLEN_CHECK(n);

    sumdata[tid] = 0;
    CU_STRIDED_FOR_i(n) sumdata[tid] += fabs(d_input[i]);

    __syncthreads();

#define stride_reduce_sum(reduce_block_)                        \
        if (sh_len >= reduce_block_) {                          \
            constexpr uint32_t stride_2 = reduce_block_ / 2;    \
            const auto tid_stride_2 = tid + stride_2;        \
            if (tid < stride_2 && tid_stride_2 < sh_len)     \
                sumdata[tid] += sumdata[tid_stride_2];       \
            __syncthreads();                                 \
        }

    stride_reduce_sum(1024);
    stride_reduce_sum(512);
    stride_reduce_sum(256);
    stride_reduce_sum(128);

    if (tid >= 32) return;
    warp_reduce_sum(sumdata, tid, sh_len);

    if (tid) return;
    atomicAdd(d_result_sum, *sumdata);
}

double sumabs(CPTRd d_in, const uint32_t n, const cudaStream_t stm)
{
    double sum, *d_sum = cucalloc<double>(stm);
    const auto [blocks, threads] = CU_BLOCKS_THREADS_t(n);
    G_sumabs<<<blocks, threads, threads * sizeof(double), stm>>>(d_in, d_sum, n);
    cufreecopy(&sum, d_sum, stm);
    cu_errchk(cudaStreamSynchronize(stm));
    return sum;
}

#endif

double median_(const size_t n, cudaStream_t const stm, double *d_tmp)
{
    thrust::sort(thrust::cuda::par.on(stm), d_tmp, d_tmp + n);
    const auto n_size = n * sizeof(double);
    double *tmp = static_cast<double *>(malloc(n_size));
    cu_errchk(cudaMemcpyAsync(tmp, d_tmp, n_size, cudaMemcpyDeviceToHost, stm));
    cu_errchk(cudaFreeAsync(d_tmp, stm));
    cu_errchk(cudaStreamSynchronize(stm));
    return n % 2 ? tmp[n / 2] : (tmp[n / 2 - 1] + tmp[n / 2]) / 2;
}

double median(CPTRd d_in, const size_t n, const cudaStream_t stm)
{
    auto d_tmp = cumallocopy(d_in, stm, n);
    return median_(n, stm, d_tmp);
}

double medianabs(CPTRd d_in, const size_t n, const cudaStream_t stm)
{
    auto d_tmp = cumallocopy(d_in, stm, n);
    G_abs<<<CU_BLOCKS_THREADS(n), 0, stm>>>(d_tmp, n);
    return median_(n, stm, d_tmp);
}

double meanabs(CPTRd d_in, const size_t n, const cudaStream_t stm)
{
    return sumabs(d_in, n, stm) / n;
}


__global__ void G_score_diff_dir(RPTR(double) score, CPTRd dif, CPTRd ref, const uint32_t ld_in, const uint32_t ld_ref, const uint32_t m, const uint32_t n)
{
    const auto i = blockIdx.x * blockDim.x + threadIdx.x;
    const auto j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= m || j >= n) return;
    const auto difv = dif[j * ld_in + i];
    const auto refv = ref[j * ld_ref + i];
    atomicAdd(score, fabs(difv) * (signbit(refv) == signbit(refv + difv) ? 1 : 10));
}

/* Sign-weighted score */
double swscore(CPTRd d_in, const uint32_t ld_in, CPTRd d_ref, const uint32_t ld_ref, const uint32_t m, const uint32_t n, const cudaStream_t custream)
{
    double score, *d_score = cucalloc<double>(custream);
    G_score_diff_dir<<<CU_BLOCKS_THREADS_2D2(m, n), 0, custream>>>(d_score, d_in, d_ref, ld_in, ld_ref, m, n);
    cu_errchk(cudaMemcpyAsync(&score, d_score, sizeof(double), cudaMemcpyDeviceToHost, custream));
    cu_errchk(cudaFreeAsync(d_score, custream));
    cu_errchk(cudaStreamSynchronize(custream));
    return score;
}

#if 1

template<const uint32_t block_size, typename T> __device__ __forceinline__ void warp_reduce_minsum(volatile T *sumabs, volatile T *minabs, const uint32_t ix, const uint32_t n)
{
    assert(ix < 32);

#define _DO_WARP_REDUCE_SUMMIN(N)                           \
    if (block_size >= (N)) {                                \
        const uint32_t ix_N_2 = ix + (N) / 2;               \
        if (ix_N_2 < n) {                                   \
            sumabs[ix] += sumabs[ix_N_2];                   \
            MINAS(minabs[ix], minabs[ix_N_2]);              \
        }                                                   \
    }

    _DO_WARP_REDUCE_SUMMIN(64);
    _DO_WARP_REDUCE_SUMMIN(32);
    _DO_WARP_REDUCE_SUMMIN(16);
    _DO_WARP_REDUCE_SUMMIN(8);
    _DO_WARP_REDUCE_SUMMIN(4);
    _DO_WARP_REDUCE_SUMMIN(2);
}

// Inplace abs and returns sumabs
template<const uint32_t block_size> __global__ void
G_irwls_op1(RPTR(double) d_input, RPTR(double) d_result_sumabs, RPTR(double) d_result_minabs, const uint32_t n)
{
    __shared__ double sumabs[block_size];
    __shared__ double minabs[block_size];
    auto i = blockIdx.x * block_size + tid;
    if (i < n) {
        d_input[i] = fabs(d_input[i]);
        sumabs[tid] = d_input[i];
        minabs[tid] = d_input[i];
        const auto stride1 = blockDim.x * gridDim.x;
        UNROLL()
        for (i += stride1; i < n; i += stride1) {
            d_input[i] = fabs(d_input[i]);
            sumabs[tid] += d_input[i];
            MINAS(minabs[tid], d_input[i]);
        }
    } else {
        sumabs[tid] = 0;
        minabs[tid] = common::C_bad_validation;
    }

    __syncthreads();
    const auto sh_limit = _MIN(n, block_size);
#define stride_reduce_minsum(block_low_)                        \
        if (block_size >= block_low_) {                      \
            constexpr uint32_t stride2 = block_low_ / 2;     \
            const auto tid_stride2 = tid + stride2;          \
            if (tid < stride2 && tid_stride2 < sh_limit) {   \
                sumabs[tid] += sumabs[tid_stride2];          \
                MINAS(minabs[tid], minabs[tid_stride2]);     \
            }                                                \
            __syncthreads();                                 \
        }

    stride_reduce_minsum(1024);
    stride_reduce_minsum(512);
    stride_reduce_minsum(256);
    stride_reduce_minsum(128);
    if (tid >= 32) return;
    warp_reduce_minsum<block_size>(sumabs, minabs, tid, sh_limit);
    if (tid) return;
    atomicAdd(d_result_sumabs, *sumabs);
    atomicMin(d_result_minabs, *minabs);
}

template<const uint32_t block_size> __global__ void
G_irwls_op1(RPTR(double) d_input, RPTR(double) d_result_sum, const uint32_t n)
{
//    constexpr double C_error_threshold = 1e-8;
    __shared__ double sumdata[block_size];
    auto i = blockIdx.x * block_size + tid;
    if (i < n) {
        d_input[i] = fabs(d_input[i]);
        sumdata[tid] = d_input[i];
        // d_input[i] = 1. / _MAX(d_input[i], C_error_threshold);
        const auto stride1 = blockDim.x * gridDim.x;
        UNROLL()
        for (i += stride1; i < n; i += stride1) {
            d_input[i] = fabs(d_input[i]);
            sumdata[tid] += d_input[i];
            // d_input[i] = 1. / _MAX(d_input[i], C_error_threshold);
        }
    } else
        sumdata[tid] = 0;

    __syncthreads();
    const auto sh_limit = _MIN(n, block_size);
#define stride_reduce_sum(block_low_)                        \
        if (block_size >= block_low_) {                      \
            constexpr uint32_t stride2 = block_low_ / 2;     \
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
    warp_reduce_sum<block_size>(sumdata, tid, sh_limit);
    if (tid) return;
    atomicAdd(d_result_sum, *sumdata);
}

// Inplace meanabs, and returns meanabs and minabs
std::pair<double, double> irwls_op1w(double *d_in, const uint32_t n, const cudaStream_t stm)
{
    double sumabs, *d_sumabs = cucalloc<double>(stm), minabs = common::C_bad_validation, *d_minabs;
    cu_errchk(cudaMallocAsync(&d_minabs, sizeof(double), stm));
    cu_errchk(cudaMemcpyAsync(d_minabs, &minabs, sizeof(double), cudaMemcpyHostToDevice, stm));
    G_irwls_op1 < common::C_cu_block_size ><<<CU_BLOCKS_THREADS(n), 0, stm>>>(d_in, d_sumabs, d_minabs, n);
    cufreecopy(&sumabs, d_sumabs, stm);
    cufreecopy(&minabs, d_minabs, stm);
    cu_errchk(cudaStreamSynchronize(stm));
    return {sumabs / n, minabs}; // Return mean
}


// Inplace meanabs, and returns meanabs
double irwls_op1(double *const d_in, const uint32_t n, const cudaStream_t stm)
{
    double sumabs, *d_sumabs = cucalloc<double>(stm);
    G_irwls_op1 < common::C_cu_block_size ><<<CU_BLOCKS_THREADS(n), 0, stm>>>(d_in, d_sumabs, n);
    cufreecopy(&sumabs, d_sumabs, stm);
    cu_errchk(cudaStreamSynchronize(stm));
    return sumabs / n; // Return mean
}

#else

// Inplace abs and returns sumabs
__global__ void
G_irwls_op1(RPTR(double) d_input, RPTR(double) d_result_sum, const uint32_t n)
{
    extern __shared__ double sumdata[];

    CU_SHLEN_CHECK(n);

    auto i = blockIdx.x * blockDim.x + tid;
    d_input[i] = fabs(d_input[i]);
    sumdata[tid] = d_input[i];
    const auto stride1 = blockDim.x * gridDim.x;
UNROLL()
    for (i += stride1; i < n; i += stride1) {
        d_input[i] = fabs(d_input[i]);
        sumdata[tid] += d_input[i];
    }

    __syncthreads();

    stride_reduce_sum(1024);
    stride_reduce_sum(512);
    stride_reduce_sum(256);
    stride_reduce_sum(128);

    if (tid >= 32) return;
    warp_reduce_sum(sumdata, tid, sh_len);

    if (tid) return;
    atomicAdd(d_result_sum, *sumdata);
}

// Inplace meanabs, and returns meanabs
double irwls_op1(double *d_in, const uint32_t n, const cudaStream_t stm)
{
    double sum, *d_sum = cucalloc<double>(stm);
    const auto [blocks, threads] = CU_BLOCKS_THREADS_t(n);
    G_irwls_op1<<<blocks, threads, threads * sizeof(double), stm>>>(d_in, d_sum, n);
    cufreecopy(&sum, d_sum, stm);
    cu_errchk(cudaStreamSynchronize(stm));
    return sum / n; // Return mean
}

#endif

#define _SMM_OP(X1, X2, X3, Y1, Y2, Y3) {       \
        (X1) += (Y1);                           \
        MINAS((X2), (Y2));                     \
        MAXAS((X3), (Y3));                     \
    }

#if 1

template<const uint32_t block_size> __device__ inline void
warp_reduce_suminmax(volatile double *sumdata, volatile double *mindata, volatile double *maxdata, const uint32_t ix, const uint32_t n)
{
#define _DO_WARP_REDUCE(N)                      \
    if (block_size >= (N)) {                    \
        const uint32_t ix_N_2 = ix + (N) / 2;   \
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

template<const uint32_t block_size> __global__ void
G_suminmax(CRPTRd d_input, RPTR(double) d_result_sum, RPTR(double) d_result_min, RPTR(double) d_result_max, const uint32_t n)
{
    static __shared__ double sumdata[block_size], mindata[block_size], maxdata[block_size];
    auto i = blockIdx.x * block_size + tid;
    if (i < n) {
        sumdata[tid] = mindata[tid] = maxdata[tid] = d_input[i];
        const auto stride1 = blockDim.x * gridDim.x;
        UNROLL()
        for (i += stride1; i < n; i += stride1) _SMM_OP(sumdata[tid], mindata[tid], maxdata[tid], d_input[i], d_input[i], d_input[i]);
    } else {
        sumdata[tid] = 0;
        mindata[tid] = std::numeric_limits<double>::max();
        maxdata[tid] = std::numeric_limits<double>::min();
    }
    __syncthreads();
    const auto sh_limit = _MIN(n, block_size);
#define stride_reduce_suminmax(block_low_)                          \
        if (block_size >= block_low_) {                             \
            constexpr uint32_t stride2 = block_low_ / 2;            \
            const auto tid_stride2 = tid + stride2;                 \
            if (tid < stride2 && tid_stride2 < sh_limit)            \
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
    atomicAdd(d_result_sum, *sumdata);
    atomicMin(d_result_min, *mindata);
    atomicMax(d_result_max, *maxdata);
}

std::tuple<double, double, double> suminmax(CPTRd d_in, const uint32_t n, const cudaStream_t stm)
{
    double sum, min = std::numeric_limits<double>::max(), max = std::numeric_limits<double>::min();
    double *d_min, *d_max, *d_sum = cucalloc<double>(stm);
    cu_errchk(cudaMallocAsync((void **) &d_min, sizeof(double), stm));
    cu_errchk(cudaMemcpyAsync(d_min, &min, sizeof(double), cudaMemcpyHostToDevice, stm));
    cu_errchk(cudaMallocAsync((void **) &d_max, sizeof(double), stm));
    cu_errchk(cudaMemcpyAsync(d_max, &max, sizeof(double), cudaMemcpyHostToDevice, stm));
    G_suminmax < common::C_cu_block_size ><<<CU_BLOCKS_THREADS(n), 0, stm>>>(d_in, d_sum, d_min, d_max, n);
    cufreecopy(&sum, d_sum, stm);
    cufreecopy(&min, d_min, stm);
    cufreecopy(&max, d_max, stm);
    cu_errchk(cudaStreamSynchronize(stm));
    return {sum, min, max};
}

#else

__device__ inline void
warp_reduce_suminmax(volatile double *sumdata, volatile double *mindata, volatile double *maxdata, const uint32_t ix, const uint32_t n)
{
#define _DO_WARP_REDUCE(N_)                      \
    if (n >= (N_)) {                             \
        const uint32_t ix_N_2 = ix + (N_) / 2;   \
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

__global__ void
G_suminmax(CRPTRd d_input, RPTR(double) d_result_sum, RPTR(double) d_result_min, RPTR(double) d_result_max,
           const uint32_t n)
{
    extern __shared__ double sumdata[], mindata[], maxdata[];

    CU_SHLEN_CHECK(n);

    auto i = blockIdx.x * blockDim.x + tid;
    sumdata[tid] = mindata[tid] = maxdata[tid] = d_input[i];
    const auto stride1 = blockDim.x * gridDim.x;
UNROLL()
    for (i += stride1; i < n; i += stride1) _SMM_OP(sumdata[tid], mindata[tid], maxdata[tid], d_input[i], d_input[i], d_input[i]);

    __syncthreads();

#define stride_reduce_suminmax(reduce_block_)                           \
        if (sh_len >= reduce_block_) {                                  \
            constexpr uint32_t stride_2 = reduce_block_ / 2;            \
            const auto tid_stride_2 = tid + stride_2;                   \
            if (tid < stride_2 && tid_stride_2 < sh_len)                \
                _SMM_OP(sumdata[tid], mindata[tid], maxdata[tid], sumdata[tid_stride_2], mindata[tid_stride_2], maxdata[tid_stride_2]); \
            __syncthreads();                                            \
        }

    stride_reduce_suminmax(1024);
    stride_reduce_suminmax(512);
    stride_reduce_suminmax(256);
    stride_reduce_suminmax(128);

    if (tid >= 32) return;
    warp_reduce_suminmax(sumdata, mindata, maxdata, tid, sh_len);

    if (tid) return;
    atomicAdd(d_result_sum, *sumdata);
    atomicMin(d_result_min, *mindata);
    atomicMax(d_result_max, *maxdata);
}

std::tuple<double, double, double> suminmax(CPTRd d_in, const uint32_t n, const cudaStream_t stm)
{
    double sum, min = std::numeric_limits<double>::max(), max = std::numeric_limits<double>::min();
    auto d_sum = cucalloc<double>(stm);
    auto d_min = cumallocopy(&min, stm);
    auto d_max = cumallocopy(&max, stm);
    const auto [blocks, threads] = CU_BLOCKS_THREADS_t(n);
    LOG4_DEBUG("Blocks " << blocks << ", threads " << threads);
    G_suminmax<<<blocks, threads, 3 * threads * sizeof(double), stm>>>(d_in, d_sum, d_min, d_max, n);
    cufreecopy(&sum, d_sum, stm);
    cufreecopy(&min, d_min, stm);
    cufreecopy(&max, d_max, stm);
    cu_errchk(cudaStreamSynchronize(stm));
    return {sum, min, max};
}

#endif

std::tuple<double, double, double> meanminmax(CPTRd d_in, const uint32_t n, const cudaStream_t stm)
{
    const auto [sum, min, max] = suminmax(d_in, n, stm);
    return {sum / n, min, max};
}

#if 1

template<const uint32_t block_size> __global__ void G_dist_unscaled(
        RPTR(double) d_sum, CRPTRd d_labels, CRPTRd d_predictions,
        const uint32_t m, const uint32_t mn, const uint32_t ldl)
{
    static __shared__ double sh_dist[block_size];
    sh_dist[tid] = 0;
    CU_STRIDED_FOR_i(mn) sh_dist[tid] += fabs(d_labels[LDi(i, m, ldl)] - d_predictions[i]);

    __syncthreads();
    const auto n_min = _MIN(mn, block_size);

#define stride_reduce_dist(block_low_)                                                  \
        if (block_size >= block_low_) {                                                 \
            constexpr uint32_t stride2 = block_low_ / 2;                                \
            const auto tid_stride2 = tid + stride2;                                     \
            if (tid < stride2 && tid_stride2 < n_min)                                      \
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
    atomicAdd(d_sum, sh_dist[0]);
}

template<const uint32_t block_size> __global__ void G_dist_unscaled(
        RPTR(double) d_sum, CRPTRd d_labels, CRPTRd d_predictions, const uint32_t mn)
{
    static __shared__ double sh_dist[block_size];
    sh_dist[tid] = 0;
    CU_STRIDED_FOR_i(mn) sh_dist[tid] += fabs(d_labels[i] - d_predictions[i]);

    __syncthreads();
    const auto n_min = _MIN(mn, block_size);

#define stride_reduce_dist(block_low_)                                                  \
        if (block_size >= block_low_) {                                                 \
            constexpr uint32_t stride2 = block_low_ / 2;                                \
            const auto tid_stride2 = tid + stride2;                                     \
            if (tid < stride2 && tid_stride2 < n_min)                                      \
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

double unscaled_distance(CPTRd d_labels, CPTRd d_predictions, const double scale, const uint32_t m, const uint32_t n, const uint32_t ldl, const cudaStream_t stm)
{
    const auto mn = m * n;
    double sum, *d_sum = cucalloc<double>(stm);
    if (ldl == m)
        G_dist_unscaled < common::C_cu_block_size ><<<CU_BLOCKS_THREADS(mn), 0, stm>>>(d_sum, d_labels, d_predictions, mn);
    else
    G_dist_unscaled < common::C_cu_block_size ><<<CU_BLOCKS_THREADS(mn), 0, stm>>>(d_sum, d_labels, d_predictions, m, mn, ldl);
    cufreecopy(&sum, d_sum, stm);
    cu_errchk(cudaStreamSynchronize(stm));
    return scale * sum / mn;
}

#else

__global__ void G_dist_unscaled(
        RPTR(double) d_sum, CRPTRd d_labels, CRPTRd d_predictions,
        const uint32_t m, const uint32_t mn, const uint32_t ldl)
{
    extern __shared__ double sumdata[];

    CU_SHLEN_CHECK(mn);

    sumdata[tid] = 0;
    CU_STRIDED_FOR_i(mn) sumdata[tid] += fabs(d_labels[LDi(i, m, ldl)] - d_predictions[i]);

    __syncthreads();

    stride_reduce_sum(1024);
    stride_reduce_sum(512);
    stride_reduce_sum(256);
    stride_reduce_sum(128);

    if (tid >= 32) return;
    warp_reduce_sum(sumdata, tid, sh_len);

    if (tid) return;
    atomicAdd(d_sum, *sumdata);
}

__global__ void G_dist_unscaled(RPTR(double) d_sum, CRPTRd d_labels, CRPTRd d_predictions, const uint32_t mn)
{
    extern __shared__ double sumdata[];

    CU_SHLEN_CHECK(mn);

    sumdata[tid] = 0;
    CU_STRIDED_FOR_i(mn) sumdata[tid] += fabs(d_labels[i] - d_predictions[i]);

    __syncthreads();

    stride_reduce_sum(1024);
    stride_reduce_sum(512);
    stride_reduce_sum(256);
    stride_reduce_sum(128);

    if (tid >= 32) return;
    warp_reduce_sum(sumdata, tid, sh_len);

    if (tid) return;
    atomicAdd(d_sum, *sumdata);
}

double
unscaled_distance(CPTRd d_labels, CPTRd d_predictions, const double scale, const uint32_t m, const uint32_t n, const uint32_t ldl, const cudaStream_t stm)
{
    const auto mn = m * n;
    double sum, *d_sum = cucalloc<double>(stm);
    const auto [blocks, threads] = CU_BLOCKS_THREADS_t(mn);
    if (ldl == m)
        G_dist_unscaled<<<blocks, threads, threads * sizeof(double), stm>>>(d_sum, d_labels, d_predictions, mn);
    else
        G_dist_unscaled<<<blocks, threads, threads * sizeof(double), stm>>>(d_sum, d_labels, d_predictions, m, mn, ldl);
    cufreecopy(&sum, d_sum, stm);
    cu_errchk(cudaStreamSynchronize(stm));
    return scale * sum / mn;
}

#endif

double max(CPTRd d_in, const size_t n, const cudaStream_t stm)
{
    const auto r = thrust::reduce(thrust::cuda::par.on(stm), d_in, d_in + n, std::numeric_limits<double>::min(), thrust::maximum<double>());
    cu_errchk(cudaStreamSynchronize(stm));
    return r;
}

double min(CPTRd d_in, const size_t n, const cudaStream_t stm)
{
    const auto r = thrust::reduce(thrust::cuda::par.on(stm), d_in, d_in + n, std::numeric_limits<double>::max(), thrust::minimum<double>());
    cu_errchk(cudaStreamSynchronize(stm));
    return r;
}

double mean(CPTRd d_in, const size_t n, const cudaStream_t stm)
{
    return sum(d_in, n, stm) / n;
}

double sum(CPTRd d_in, const size_t n, const cudaStream_t stm)
{
    const auto r = thrust::reduce(thrust::cuda::par.on(stm), d_in, d_in + n);
    cu_errchk(cudaStreamSynchronize(stm));
    return r;
}

double sum(CPTRd d_in, const size_t n, const NppStreamContext &npp_ctx)
{
    size_t npp_buffer_size;
    np_errchk(nppsSumGetBufferSize_64f_Ctx(n, &npp_buffer_size, npp_ctx));

    Npp8u *npp_sum_buf;
    Npp64f *dres;
    cu_errchk(cudaMallocAsync((void **) &npp_sum_buf, npp_buffer_size, npp_ctx.hStream));
    cu_errchk(cudaMallocAsync((void **) &dres, sizeof(*dres), npp_ctx.hStream));
    np_errchk(nppsMean_64f_Ctx(d_in, n, dres, npp_sum_buf, npp_ctx));

    double res;
    cu_errchk(cudaMemcpyAsync(&res, dres, sizeof(*dres), cudaMemcpyDeviceToHost, npp_ctx.hStream));
    cu_errchk(cudaFreeAsync(npp_sum_buf, npp_ctx.hStream));
    cu_errchk(cudaFreeAsync(dres, npp_ctx.hStream));
    cu_errchk(cudaStreamSynchronize(npp_ctx.hStream));

    return res;
}


__global__ void G_sqrt_add(RPTR(double) input, const double a, const uint32_t N)
{
    CU_STRIDED_FOR_i(N)input[i] = sqrt(input[i] + a);
}

__global__ void G_matmul_I(CRPTRd input, RPTR(double) output, const uint32_t N)
{
    CU_STRIDED_FOR_i(N)output[i] *= input[i];
}

__global__ void G_eq_matmul(CRPTRd input1, CRPTRd input2, RPTR(double) output, const uint32_t N)
{
    CU_STRIDED_FOR_i(N)output[i] = input1[i] * input2[i];
}

__global__ void G_abs_subtract(CRPTRd input1, RPTR(double) input2, const uint32_t N)
{
    CU_STRIDED_FOR_i(N)input2[i] = fabs(input1[i] - input2[i]);
}


#if 0 // Solve Ax=B using Bandicoot
coot::mat OnlineMIMOSVR::solve_irwls(const coot::mat &K_epsco, const coot::mat &K, const coot::mat &rhs, const size_t iters, const size_t gpu_id)
{
    magma_queue_t magma_queue = nullptr;
    magma_queue_create(gpu_id, &magma_queue);
    if (!magma_queue) LOG4_THROW("Failed creating MAGMA queue.");
    const auto m = K.n_rows;
    const auto b_n = rhs.n_cols;
    coot::mat solved = rhs;
    coot::get_rt().synchronise();
    magma_int_t err, info;
    if ((err = magma_dgesv_rbt(magma_bool_t::MagmaTrue, m, b_n, K.get_dev_mem(false).cuda_mem_ptr, m, solved.get_dev_mem(false).cuda_mem_ptr, m, &info)) < MAGMA_SUCCESS)
        LOG4_THROW("Failed calling magma_dgesv_rbt with error code " << err << ", info " << info);

    auto best_sae = std::numeric_limits<double>::infinity();
    coot::mat best_solution = solved;
    size_t best_iter = 0;
UNROLL()
    for (size_t i = 1; i < iters; ++i) {
        const coot::mat error_mat = coot::abs(K * solved - rhs);
        const double this_sae = coot::accu(error_mat);
        if (this_sae < best_sae) {
            LOG4_TRACE("IRWLS iteration " << i << ", SAE " << this_sae << ", kernel dimensions " << coot::size(K) << ", best SAE " << best_sae);
            best_sae = this_sae;
            best_solution = solved;
            best_iter = i;
        }
        const coot::mat mult = coot::sqrt(error_mat + common::C_itersolve_delta / (double(i) * common::C_itersolve_range / double(iters)));
        const coot::mat left = (mult * coot::ones<coot::mat>(mult.n_cols, K.n_cols)) % K_epsco;
        solved = rhs % mult;
        coot::coot_synchronise();
        if ((err = magma_dgesv_rbt(magma_bool_t::MagmaTrue, m, b_n, left.get_dev_mem(false).cuda_mem_ptr, m, solved.get_dev_mem(false).cuda_mem_ptr, m, &info)) < MAGMA_SUCCESS)
            LOG4_THROW("Failed calling magma_dgesv_rbt with error code " << err << ", info " << info);
        //magma_queue_sync(queue);
    }

    magma_queue_destroy(magma_queue);

    if ((err = magma_finalize()) < MAGMA_SUCCESS)
        LOG4_THROW("Failed calling magma_finalize with error code " << err);
    LOG4_DEBUG("IRWLS best iteration " << best_iter << ", MAE " << best_sae / double(solved.n_elem) << ", kernel dimensions " << coot::size(K) <<
                                       ", delta " << common::C_itersolve_delta << ", range " << common::C_itersolve_range << ", solution " << coot::size(solved));
    return best_solution;
}
#endif


__global__ void G_prepare_labels(RPTR(double) d_labels, const double L_sum, const uint32_t n)
{
    CU_STRIDED_FOR_i(n)d_labels[i] = n * d_labels[i] - L_sum;
}

void solve_irwls(const arma::mat &K_epsco, const arma::mat &K, const arma::mat &rhs, arma::mat &solved, const uint16_t iters)
{
    LOG4_BEGIN();

    assert(K.n_cols == rhs.n_rows && K.n_rows == K_epsco.n_rows && K.n_cols == K_epsco.n_cols);

    double *d_solved, *d_best_solution, *d_rwork, *d_Kwork;
    const auto K_size = K.n_elem * sizeof(double);
    const auto mn = rhs.n_elem;
    const auto rhs_size = mn * sizeof(double);
    common::gpu_context_4 ctx;
    cu_errchk(cudaSetDevice(ctx.phy_id()));
    magma_queue_t maqueue;
    magma_queue_create(ctx.phy_id(), &maqueue);
    const cudaStream_t custream = magma_queue_get_cuda_stream(maqueue);
    const cublasHandle_t cublas_H = magma_queue_get_cublas_handle(maqueue);
    cu_errchk(cudaMallocAsync(&d_solved, rhs_size, custream));
    cu_errchk(cudaMallocAsync(&d_best_solution, rhs_size, custream));
    if (arma::size(rhs) == arma::size(solved)) {
//        cu_errchk(cudaHostRegister((void *) solved.memptr(), rhs_size, cudaHostRegisterDefault));
        cu_errchk(cudaMemcpyAsync(d_solved, solved.memptr(), rhs_size, cudaMemcpyHostToDevice, custream));
    } else {
        solved.set_size(arma::size(rhs));
//        cu_errchk(cudaHostRegister((void *) solved.memptr(), rhs_size, cudaHostRegisterDefault));
    }

    /* CUDA bug disabled, hostRegister() introduces contention among threads working on different GPUs (why?)
    cu_errchk(cudaHostRegister((void *) K_epsco.memptr(), K_size, cudaHostRegisterReadOnly));
    cu_errchk(cudaHostRegister((void *) K.memptr(), K_size, cudaHostRegisterReadOnly));
    cu_errchk(cudaHostRegister((void *) rhs.memptr(), rhs_size, cudaHostRegisterReadOnly));
    */

    const auto d_K_epsco = cumallocopy(K_epsco, custream);
    const auto d_K = cumallocopy(K, custream);
    auto d_labels = cumallocopy(rhs, custream);
    cu_errchk(cudaMallocAsync(&d_rwork, rhs_size, custream));
    cu_errchk(cudaMallocAsync(&d_Kwork, K_size, custream));

    const auto iters_mul = common::C_itersolve_range / iters;

    (void) solvers::solve_hybrid(d_K_epsco, rhs.n_cols, K.n_rows, d_solved, maqueue, iters, d_labels, rhs_size, d_rwork, custream, cublas_H, d_K, 1, mn, d_best_solution,
                                 K.n_elem, d_Kwork, iters_mul);
    cu_errchk(cudaFreeAsync(d_K_epsco, custream));
    cu_errchk(cudaFreeAsync(d_K, custream));
    cu_errchk(cudaFreeAsync(d_labels, custream));
    cu_errchk(cudaFreeAsync(d_rwork, custream));
    cu_errchk(cudaFreeAsync(d_Kwork, custream));
    cu_errchk(cudaFreeAsync(d_solved, custream));
    cu_errchk(cudaMemcpyAsync(solved.memptr(), d_best_solution, rhs_size, cudaMemcpyDeviceToHost, custream));
    cu_errchk(cudaFreeAsync(d_best_solution, custream));
    /*
    cu_errchk(cudaHostUnregister((void *) K_epsco.memptr()));
    cu_errchk(cudaHostUnregister((void *) K.memptr()));
    cu_errchk(cudaHostUnregister((void *) rhs.memptr()));
    cu_errchk(cudaHostUnregister((void *) solved.memptr()));
    */
    cu_errchk(cudaStreamSynchronize(custream));
    magma_queue_destroy(maqueue);

    LOG4_END();
}


#if 0 // Not used
void solve_irwls(const arma::mat &K_epsco, const arma::mat &K, const arma::mat &rhs, arma::mat &solved, const uint32_t iters, const uint32_t gpu_phy_id)
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
    uint32_t best_iter = 0;
UNROLL()
    for (uint32_t i = 1; i < iters; ++i) {
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

double solve_hybrid(
        const double *const j_K_epsco, const uint32_t n, const uint32_t train_len, double *const j_solved, const magma_queue_t ma_queue,
        const uint16_t irwls_iters, const double *const j_train_labels, const size_t train_n_size, double *const j_work, const cudaStream_t custream,
        const cublasHandle_t cublas_H, const double *const j_K_tune, const double labels_factor, const uint32_t train_len_n, double *const d_best_weights,
        const uint32_t K_train_len, double *j_K_work, const double iters_mul)
{
    LOG4_BEGIN();
    magma_int_t info;
    constexpr double one = 1, oneneg = -1;
    const auto delta_iter_mul = common::C_itersolve_delta / iters_mul;
    auto best_score = std::numeric_limits<double>::max();
    copy_submat(j_train_labels, j_solved, train_len, 0, 0, train_len, n, train_len, cudaMemcpyDeviceToDevice, custream);
    cu_errchk(cudaStreamSynchronize(custream));
    ma_errchki(magma_dgesv_rbt(MagmaTrue, train_len, n, (double *) j_K_epsco, train_len, j_solved, train_len, &info), info);
    UNROLL()
    for (uint32_t i = 1; i < irwls_iters + 1; ++i) {
        cu_errchk(cudaStreamSynchronize(custream));
        copy_submat(j_train_labels, j_work, train_len, 0, 0, train_len, n, train_len, cudaMemcpyDeviceToDevice, custream);
        cu_errchk(cudaStreamSynchronize(custream));
        cb_errchk(cublasDgemm(cublas_H, CUBLAS_OP_N, CUBLAS_OP_N, train_len, n, train_len, &one, (double *) j_K_tune, train_len, j_solved, train_len, &oneneg, j_work, train_len));
        cu_errchk(cudaStreamSynchronize(custream));
#ifdef INSTANCE_WEIGHTS
        auto [score, minabs] = irwls_op1w(j_work, train_len_n, custream); // TODO Test
        score *= labels_factor;
#else
        const auto score = irwls_op1(j_work, train_len_n, custream) * labels_factor;
#endif
        if (!std::isnormal(score)) {
            LOG4_WARN("Bailing, score not normal " << score << ", iteration " << i << ", train len " << train_len << ", best score " << best_score);
	        if (i == 1) cu_errchk(cudaMemsetAsync(d_best_weights, 0, train_len * sizeof(double), custream)); // thrust::fill(thrust::cuda::par.on(custream), d_best_weights, d_best_weights + train_len_n, 0.);
            goto __bail;
        } else if (score < best_score) {
            LOG4_TRACE("IRWLS iteration " << i << ", kernel dimensions " << train_len << "x" << train_len << ", former best score " << best_score <<
                                          ", new best score " << score << ", improvement " << 100. * (1. - score / best_score) << " pct.");
            best_score = score;
            cu_errchk(cudaMemcpyAsync(d_best_weights, j_solved, train_n_size, cudaMemcpyDeviceToDevice, custream));
            cu_errchk(cudaStreamSynchronize(custream));
        }
        if (i == irwls_iters) break;
        G_irwls_op2<<<CU_BLOCKS_THREADS(K_train_len), 0, custream>>>(
                j_work, j_K_epsco, train_len, j_train_labels, j_K_work, j_solved, delta_iter_mul / i, train_len, train_len_n, K_train_len);
        cu_errchk(cudaStreamSynchronize(custream));
        ma_errchki(magma_dgesv_rbt(MagmaTrue, train_len, n, j_K_work, train_len, j_solved, train_len, &info), info);
    }
    __bail:
    cu_errchk(cudaStreamSynchronize(custream));
    LOG4_END();
    return best_score;
}

constexpr auto C_lower_precision_type = CUSOLVER_R_64F;
constexpr auto C_ge_tol = 1e-7;
constexpr cusolver_int_t C_inner_iter = 50;
constexpr auto C_refine_type = CUSOLVER_IRS_REFINE_GMRES_GMRES;

void cs_gels_iter(CPTR(double) A, double *const x, CPTR(double) b, const uint32_t m, const uint32_t n, const uint32_t iter)
{
    assert(n == 1);
    CTX4_CUSTREAM;
    cusolverDnHandle_t cusolverH;
    cs_errchk(cusolverDnCreate(&cusolverH));
    cs_errchk(cusolverDnSetStream(cusolverH, custream));
    const auto d_A = cumallocopy(A, custream, m * m);
    const auto d_b = cumallocopy(b, custream, m * n);
    double *d_x, *d_W;
    cu_errchk(cudaMallocAsync(&d_x, m * n * sizeof(double), custream));
    cusolverDnIRSParams_t irs_params;
    cs_errchk(cusolverDnIRSParamsCreate(&irs_params));
    cs_errchk(cusolverDnIRSParamsSetSolverPrecisions(irs_params, CUSOLVER_R_64F, C_lower_precision_type));
    cs_errchk(cusolverDnIRSParamsSetRefinementSolver(irs_params, C_refine_type));
    cs_errchk(cusolverDnIRSParamsSetTol(irs_params, C_ge_tol));
    cs_errchk(cusolverDnIRSParamsSetTolInner(irs_params, C_ge_tol));
    cs_errchk(cusolverDnIRSParamsSetMaxIters(irs_params, iter));
    cs_errchk(cusolverDnIRSParamsSetMaxItersInner(irs_params, C_inner_iter));
    cusolverDnIRSInfos_t irs_info;
    cs_errchk(cusolverDnIRSInfosCreate(&irs_info));
    cusolver_int_t niters;
    cusolver_int_t *d_info;
    cu_errchk(cudaMallocAsync(&d_info, sizeof(cusolver_int_t), custream));
    size_t worksize;
    cs_errchk(cusolverDnIRSXgels_bufferSize(cusolverH, irs_params, m, m, n, &worksize));
    cu_errchk(cudaMallocAsync(&d_W, worksize, custream));
    cu_errchk(cudaStreamSynchronize(custream));
    cs_errchk(cusolverDnIRSXgels(cusolverH, irs_params, irs_info, m, m, n, d_A, m, d_b, m, d_x, m, d_W, worksize, &niters, d_info));
    cu_errchk(cudaStreamSynchronize(custream));
    cu_errchk(cudaFreeAsync(d_W, custream));
    cu_errchk(cudaStreamSynchronize(custream));
    cu_errchk(cudaFreeAsync(d_A, custream));
    cu_errchk(cudaStreamSynchronize(custream));
    cu_errchk(cudaFreeAsync(d_b, custream));
    cu_errchk(cudaStreamSynchronize(custream));
    cu_errchk(cudaFreeAsync(d_info, custream));
    cu_errchk(cudaStreamSynchronize(custream));
    cu_errchk(cudaMemcpyAsync(x, d_x, m * n * sizeof(double), cudaMemcpyDeviceToHost, custream));
    cu_errchk(cudaStreamSynchronize(custream));
    cu_errchk(cudaFreeAsync(d_x, custream));
    cu_errchk(cudaStreamSynchronize(custream));
    cs_errchk(cusolverDnIRSParamsDestroy(irs_params));
    cs_errchk(cusolverDnIRSInfosDestroy(irs_info));
    cs_errchk(cusolverDnDestroy(cusolverH));
    cusyndestroy(custream);
    LOG4_DEBUG("Solved " << m << "x" << n << ", iterations " << niters << ", worksize " << worksize);
}

void cs_gesv_iter(CPTR(double) A, double *const x, CPTR(double) b, const uint32_t m, const uint32_t n, const uint32_t iter)
{
    assert(n == 1);
    CTX4_CUSTREAM;
    cusolverDnHandle_t cusolverH;
    cs_errchk(cusolverDnCreate(&cusolverH));
    cs_errchk(cusolverDnSetStream(cusolverH, custream));
    const auto d_A = cumallocopy(A, custream, m * m);
    const auto d_b = cumallocopy(b, custream, m * n);
    double *d_x, *d_W;
    cu_errchk(cudaMallocAsync(&d_x, m * n * sizeof(double), custream));
    cusolverDnIRSParams_t irs_params;
    cs_errchk(cusolverDnIRSParamsCreate(&irs_params));
    cs_errchk(cusolverDnIRSParamsSetSolverPrecisions(irs_params, CUSOLVER_R_64F, C_lower_precision_type));
    cs_errchk(cusolverDnIRSParamsSetRefinementSolver(irs_params, C_refine_type));
    cs_errchk(cusolverDnIRSParamsSetTol(irs_params, C_ge_tol));
    cs_errchk(cusolverDnIRSParamsSetTolInner(irs_params, C_ge_tol));
    cs_errchk(cusolverDnIRSParamsSetMaxIters(irs_params, iter));
    cs_errchk(cusolverDnIRSParamsSetMaxItersInner(irs_params, C_inner_iter));
    cusolverDnIRSInfos_t irs_info;
    cs_errchk(cusolverDnIRSInfosCreate(&irs_info));
    cusolver_int_t niters;
    cusolver_int_t *d_info;
    cu_errchk(cudaMallocAsync(&d_info, sizeof(cusolver_int_t), custream));
    size_t worksize;
    cs_errchk(cusolverDnIRSXgesv_bufferSize(cusolverH, irs_params, m, n, &worksize));
    cu_errchk(cudaMallocAsync(&d_W, worksize, custream));
    cs_errchk(cusolverDnIRSXgesv(cusolverH, irs_params, irs_info, m, n, d_A, m, d_b, m, d_x, m, d_W, worksize, &niters, d_info));
    cu_errchk(cudaFreeAsync(d_W, custream));
    cu_errchk(cudaFreeAsync(d_A, custream));
    cu_errchk(cudaFreeAsync(d_b, custream));
    cu_errchk(cudaFreeAsync(d_info, custream));
    cu_errchk(cudaMemcpyAsync(x, d_x, m * n * sizeof(double), cudaMemcpyDeviceToHost, custream));
    cu_errchk(cudaFreeAsync(d_x, custream));
    cs_errchk(cusolverDnIRSParamsDestroy(irs_params));
    cs_errchk(cusolverDnIRSInfosDestroy(irs_info));
    cs_errchk(cusolverDnDestroy(cusolverH));
    cusyndestroy(custream);
    LOG4_DEBUG("Solved " << m << "x" << n << ", iterations " << niters << ", worksize " << worksize);
}


// Namespaces
}
}
