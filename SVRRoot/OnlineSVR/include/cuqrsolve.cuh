#pragma once

#include <npp.h>
#include <cublas_v2.h>
#include <magma_types.h>
#include <cusolverDn.h>


namespace svr::solvers {

struct mmm_t { double mean = 0, max = 0, min = 0; };

constexpr double C_gamma_variance = 6e4; // Inverse tuning gamma variance

__global__ void G_normalize_distances_I(RPTR(double) x, const double dc, const double a, const unsigned n);

__global__ void G_div_I(RPTR(double) x, const double a, const unsigned n);

__global__ void G_mul_I(RPTR(double) x, const double a, const unsigned n);

__global__ void G_absdif(CRPTRd labels_train, RPTR(double) error_mat, const unsigned mn);

__global__ void G_pred_absdif_I(CRPTRd j_test_labels, RPTR(double) work, const double svr_epsilon, const unsigned test_len_n);

__global__ void G_set_diag(RPTR(double) K, CRPTRd d, const unsigned m);

__global__ void G_set_diag(RPTR(double) K, const double d, const unsigned m);

__global__ void G_augment_K(RPTR(double) K, CRPTRd w, const double d, const unsigned m);

__global__ void G_calc_epsco(CRPTRd K, CRPTRd L, RPTR(double) epsco, const unsigned m, const unsigned n, const uint32_t ld);

double *cu_calc_epscos(CPTRd K, CPTRd L, const unsigned m, const unsigned n, const cudaStream_t custream);

double cu_calc_epsco(CPTRd K, CPTRd L, const unsigned m, const unsigned n, const uint32_t ld, const cudaStream_t custream);


void kernel_from_distances(double *K, CPTRd Z, const uint32_t m, const uint32_t n, const double gamma);

void kernel_from_distances_I(arma::mat &Kz, const double gamma);

void kernel_from_distances(double *const K, CPTRd Z, const uint32_t m, const uint32_t n, CPTRd gammas);

void kernel_from_distances_I(arma::mat &Kz, const arma::vec &gamma);

void kernel_from_distances_symm(double *const K, CPTRd Z, const uint32_t m, const double gamma); // TODO Buggy fix

__global__ void G_kernel_from_distances(RPTR(double) K, CRPTRd Z, const uint32_t mn, const double gamma);

__global__ void G_kernel_from_distances(RPTR(double) K, CRPTR(double) Z, const uint32_t mn, const uint32_t m, CRPTR(double) gamma);

__global__ void G_kernel_from_distances_I(RPTR(double) Kz, const uint32_t mn, const double gamma);

__global__ void G_kernel_from_distances_I(RPTR(double) Kz, const uint32_t mn, const uint32_t m, CRPTR(double) gamma);

// double score_kernel(CPTRd ref_kernel /* colmaj order */, const double norm_ref, CPTRd Z /* colmaj order */, const uint32_t M, const double gamma); // TODO  Test

std::pair<double, double> cu_calc_minmax_gamma(CPTRd Z, const mmm_t &train_L_m, const double train_len, const uint32_t Z_n_elem, const cudaStream_t stm);

double cu_calc_gamma(CPTRd Z, const double L_mean, const double train_len, const uint32_t n_elem, const cudaStream_t stm);

double cu_calc_gamma(CPTRd Z, CPTRd L, const uint32_t m, const uint32_t n, const double bias, const cudaStream_t stm);

double cu_calc_gamma(const double *const Z, const uint32_t lda, const double *const L, const uint32_t m, const uint32_t n, const double bias, const cudaStream_t stm);

double *cu_calc_gammas(CPTRd Z, CPTRd L, const uint32_t m, const uint32_t n, const double bias, const cudaStream_t stm);

__global__ void G_matmul_I(CRPTRd input, RPTR(double) output, const uint32_t N);

__global__ void G_sqrt_add(RPTR(double) input, const double a, const uint32_t N);

__global__ void G_eq_matmul(CRPTRd input1, CRPTRd input2, RPTR(double) output, const uint32_t N);

__global__ void G_abs_subtract(CRPTRd input1, RPTR(double) input2, const uint32_t N);

__global__ void G_abs(RPTR(double) inout, const uint32_t N);

std::tuple<double, double, double> suminmax(CPTRd d_in, const uint32_t n, const cudaStream_t stm);

std::tuple<double, double, double> meanminmax(CPTRd d_in, const uint32_t n, const cudaStream_t stm);

std::pair<double, double> irwls_op1w(double *const d_in, const uint32_t n, const cudaStream_t stm);

double irwls_op1(double *const d_in, const uint32_t n, const cudaStream_t stm);

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
        const uint32_t mm);

double meanabs(CPTRd d_in, const uint32_t n, const cudaStream_t stm);

double sumabs(CPTRd d_in, const uint32_t n, const cudaStream_t stm);

double sum(CPTRd d_data, const uint32_t n, const cudaStream_t strm);

double sum(CPTRd d_in, const uint32_t n, const NppStreamContext &npp_ctx);

double mean(CPTRd d_in, const uint32_t n, const cudaStream_t strm);

double max(CPTRd d_in, const uint32_t n, const cudaStream_t strm);

double min(CPTRd d_in, const uint32_t n, const cudaStream_t strm);

double unscaled_distance(
        CPTRd d_labels, CPTRd d_predictions, const double scale, const uint32_t m, const uint32_t n, const uint32_t ldl, const cudaStream_t stm);

double autocorrelation(CPTRd d_x, CPTRd d_y, const uint32_t n, const cudaStream_t stm);

double autocorrelation_n(CPTRd d_in, const uint32_t n, const std::vector<uint32_t> &offsets, const cudaStream_t stm);

double score_weights(CPTRd K, CPTRd weights, CPTRd labels, const uint32_t m, const uint32_t n, const uint32_t mn, const uint32_t mm);


// Solvers

double solve_hybrid(const double *const j_K_epsco, const uint32_t n, const uint32_t train_len, double *j_solved, const uint32_t magma_iters, const double magma_threshold,
                    const magma_queue_t ma_queue, const uint16_t irwls_iters, const double *const j_train_labels, const size_t train_n_size, double *j_work,
                    const cudaStream_t custream, const cublasHandle_t cublas_H, const double *const j_K_tune, const double labels_factor, const uint32_t train_len_n,
                    double *d_best_weights, const uint32_t K_train_len, double *j_K_epsco_reweighted, magma_int_t &info, const double iters_mul, const uint32_t m);

/* CuSolver */

std::tuple<cusolverDnHandle_t, double *, double *, double *, int32_t *, int32_t *>
init_cusolver(const uint32_t gpu_phy_id, const uint32_t m, const uint32_t n);

void uninit_cusolver(const uint32_t gpu_phy_id, const cusolverDnHandle_t cusolverH, double *d_Ainput, double *d_B, double *d_work, int32_t *d_Ipiv, int32_t *d_devInfo);

void dyn_gpu_solve(const cusolverDnHandle_t cusolver_H, const uint32_t m, const uint32_t n, CPTRd d_a, double *d_b, double *d_work, int32_t *d_piv, int32_t *d_info);

void h_dyn_gpu_solve(
        const uint32_t gpu_phy_id, const uint32_t m, const uint32_t n, CPTRd h_K, CPTRd h_L, double *h_weights,
        cusolverDnHandle_t cusolverH,
        double *d_a, double *d_b, double *d_work, int32_t *d_piv, int32_t *d_info);

void strum_solve(const uint32_t Nrows, const uint32_t Nright, double *h_Ainput, double *h_rhs, double *h_B, const double epsco);

/* MAGMA */
std::tuple<magma_queue_t, magmaDouble_ptr, magmaDouble_ptr, magmaDouble_ptr, magmaDouble_ptr, magmaFloat_ptr, magmaInt_ptr>
init_magma_solver(const uint32_t m, const uint32_t b_n, const bool psd, const uint32_t gpu_id = 0);

std::tuple<std::vector<magmaDouble_ptr>, std::vector<magmaDouble_ptr>>
init_magma_batch_solver(const uint32_t batch_size, const uint32_t m, const uint32_t n);

void uninit_magma_solver(
        const magma_queue_t &magma_queue,
        const magmaDouble_ptr d_a, const magmaDouble_ptr d_b, const magmaDouble_ptr d_x, const magmaDouble_ptr d_wd, const magmaFloat_ptr d_ws, const magmaInt_ptr piv,
        const uint32_t gpu_id = 0);

void uninit_magma_batch_solver(std::vector<magmaDouble_ptr> &d_a, std::vector<magmaDouble_ptr> &d_b);

void dyn_magma_solve(
        const int32_t m, const int32_t b_n, CPTRd a, CPTRd b, double *output, magma_queue_t magma_queue = nullptr, const magmaInt_ptr piv = nullptr,
        const magmaDouble_ptr d_a = nullptr, const magmaDouble_ptr d_b = nullptr, const uint32_t gpu_id = 0);

void iter_magma_solve(
        const int32_t m, const int32_t b_n, CPTRd a, CPTRd b, double *output, magma_queue_t magma_queue,
        const magmaDouble_ptr d_a, const magmaDouble_ptr d_b, const magmaDouble_ptr d_x, const magmaDouble_ptr d_wd,
        const magmaFloat_ptr d_ws, const bool psd = false, const uint32_t gpu_id = 0);

void iter_magma_solve(
        const int32_t m, const int32_t n, CPTRd a, CPTRd b, double *output, const magma_queue_t &magma_queue,
        const magmaDouble_ptr d_a, const magmaDouble_ptr d_b);

void iter_magma_batch_solve(
        const int32_t m, const int32_t n, const std::deque<arma::mat> &a, const std::deque<arma::mat> &b, std::deque<arma::mat> &output,
        const magma_queue_t magma_queue, std::vector<magmaDouble_ptr> &d_a, std::vector<magmaDouble_ptr> &d_b, const uint32_t gpu_id);

double solve_validate_host(
        CPTRd K_epsco, CPTRd K, CPTRd rhs, CPTRd K_test, CPTRd labels, const double meanabs_labels, const double neg_epsilon,
        const uint32_t m, const uint32_t n, const uint32_t test_m, const uint32_t iters, const magma_queue_t magma_queue, const cublasHandle_t cublas_h);

}
