#pragma once

#include <npp.h>
#include <cublas_v2.h>
#include <magma_types.h>
#include <cusolverDn.h>


namespace svr::solvers {

struct mmm_t { double mean = 0, max = 0, min = 0; };

constexpr double C_gamma_variance = 6e4;

void kernel_from_distances(double *K, CPTRd Z, const unsigned m, const unsigned n, const double gamma);

void kernel_from_distances_I(arma::mat &Kz, const double gamma);

void kernel_from_distances(double *const K, CPTRd Z, const unsigned m, const unsigned n, CPTRd gammas);

void kernel_from_distances_I(arma::mat &Kz, const arma::vec &gamma);

void kernel_from_distances_symm(double *const K, CPTRd Z, const unsigned m, const double gamma); // TODO Buggy fix

__global__ void G_kernel_from_distances(RPTR(double) K, CRPTRd Z, const unsigned mn, const double gamma);

__global__ void G_kernel_from_distances(RPTR(double) K, CRPTR(double) Z, const unsigned mn, const unsigned m, CRPTR(double) gamma);

__global__ void G_kernel_from_distances_I(RPTR(double) Kz, const unsigned mn, const double gamma);

__global__ void G_kernel_from_distances_I(RPTR(double) Kz, const unsigned mn, const unsigned m, CRPTR(double) gamma);

// double score_kernel(CPTRd ref_kernel /* colmaj order */, const double norm_ref, CPTRd Z /* colmaj order */, const unsigned M, const double gamma); // TODO  Test

std::pair<double, double> cu_calc_minmax_gamma(CPTRd Z, const mmm_t &train_L_m, const double train_len, const unsigned Z_n_elem, const cudaStream_t stm);

double cu_calc_gamma(CPTRd Z, const double L_mean, const double train_len, const unsigned n_elem, const cudaStream_t stm);

double cu_calc_gamma(CPTRd Z, CPTRd L, const unsigned m, const unsigned n, const double bias, const cudaStream_t stm);

double cu_calc_gamma(const double *const Z, const unsigned lda, const double *const L, const unsigned m, const unsigned n, const double bias, const cudaStream_t stm);

double *cu_calc_gammas(CPTRd Z, CPTRd L, const unsigned m, const unsigned n, const double bias, const cudaStream_t stm);

__global__ void G_matmul_I(CRPTRd input, RPTR(double) output, const unsigned N);

__global__ void G_sqrt_add(RPTR(double) input, const double a, const unsigned N);

__global__ void G_eq_matmul(CRPTRd input1, CRPTRd input2, RPTR(double) output, const unsigned N);

__global__ void G_abs_subtract(CRPTRd input1, RPTR(double) input2, const unsigned N);

__global__ void G_abs(RPTR(double) inout, const unsigned N);

std::tuple<double, double, double> suminmax(CPTRd d_in, const unsigned n, const cudaStream_t stm);

std::tuple<double, double, double> meanminmax(CPTRd d_in, const unsigned n, const cudaStream_t stm);

std::pair<double, double> irwls_op1w(double *const d_in, const unsigned n, const cudaStream_t stm);

double irwls_op1(double *const d_in, const unsigned n, const cudaStream_t stm);

__global__ void G_irwls_op2(
        CRPTRd err,
        CRPTRd K,
        const unsigned ldK,
        CRPTRd labels,
        RPTR(double) out_K,
        RPTR(double) solved,
        const double additive,
        const unsigned m,
        const unsigned mn,
        const unsigned mm);

double meanabs(CPTRd d_in, const unsigned n, const cudaStream_t stm);

double sumabs(CPTRd d_in, const unsigned n, const cudaStream_t stm);

double sum(CPTRd d_data, const unsigned n, const cudaStream_t strm);

double sum(CPTRd d_in, const unsigned n, const NppStreamContext &npp_ctx);

double mean(CPTRd d_in, const unsigned n, const cudaStream_t strm);

double max(CPTRd d_in, const unsigned n, const cudaStream_t strm);

double min(CPTRd d_in, const unsigned n, const cudaStream_t strm);

double unscaled_distance(
        CPTRd d_labels, CPTRd d_predictions, const double scale, const unsigned m, const unsigned n, const unsigned ldl, const cudaStream_t stm);

double autocorrelation(CPTRd d_x, CPTRd d_y, const unsigned n, const cudaStream_t stm);

double autocorrelation_n(CPTRd d_in, const unsigned n, const std::vector<unsigned> &offsets, const cudaStream_t stm);

double score_weights(CPTRd K, CPTRd weights, CPTRd labels, const unsigned m, const unsigned n, const unsigned mn, const unsigned mm);

// Solvers

double solve_hybrid(CPTRd j_K_epsco, const unsigned n, const unsigned train_len, double *const j_solved, const unsigned magma_iters, const double magma_threshold,
             const magma_queue_t ma_queue, const unsigned irwls_iters, CPTRd j_train_labels, const size_t train_n_size, double *const j_work,
             const cudaStream_t custream, const cublasHandle_t cublas_H, CPTRd j_K_tune, const double labels_factor, const unsigned train_len_n,
             unsigned &best_iter, double *const d_best_weights, const unsigned K_train_len, double *const j_K_work, magma_int_t &info,
             const double iters_mul, const unsigned m);

/* CuSolver */

std::tuple<cusolverDnHandle_t, double *, double *, double *, int *, int *>
init_cusolver(const unsigned gpu_phy_id, const unsigned m, const unsigned n);

void uninit_cusolver(const unsigned gpu_phy_id, const cusolverDnHandle_t cusolverH, double *d_Ainput, double *d_B, double *d_work, int *d_Ipiv, int *d_devInfo);

void dyn_gpu_solve(const cusolverDnHandle_t cusolver_H, const unsigned m, const unsigned n, CPTRd d_a, double *d_b, double *d_work, int *d_piv, int *d_info);

void h_dyn_gpu_solve(
        const unsigned gpu_phy_id, const unsigned m, const unsigned n, CPTRd h_K, CPTRd h_L, double *h_weights,
        cusolverDnHandle_t cusolverH,
        double *d_a, double *d_b, double *d_work, int *d_piv, int *d_info);

void strum_solve(const unsigned Nrows, const unsigned Nright, double *h_Ainput, double *h_rhs, double *h_B, const double epsco);

/* MAGMA */
std::tuple<magma_queue_t, magmaDouble_ptr, magmaDouble_ptr, magmaDouble_ptr, magmaDouble_ptr, magmaFloat_ptr, magmaInt_ptr>
init_magma_solver(const unsigned m, const unsigned b_n, const bool psd, const unsigned gpu_id = 0);

std::tuple<std::vector<magmaDouble_ptr>, std::vector<magmaDouble_ptr>>
init_magma_batch_solver(const unsigned batch_size, const unsigned m, const unsigned n);

void uninit_magma_solver(
        const magma_queue_t &magma_queue,
        const magmaDouble_ptr d_a, const magmaDouble_ptr d_b, const magmaDouble_ptr d_x, const magmaDouble_ptr d_wd, const magmaFloat_ptr d_ws, const magmaInt_ptr piv,
        const unsigned gpu_id = 0);

void uninit_magma_batch_solver(std::vector<magmaDouble_ptr> &d_a, std::vector<magmaDouble_ptr> &d_b);

void dyn_magma_solve(
        const int m, const int b_n, CPTRd a, CPTRd b, double *output, magma_queue_t magma_queue = nullptr, const magmaInt_ptr piv = nullptr,
        const magmaDouble_ptr d_a = nullptr, const magmaDouble_ptr d_b = nullptr, const unsigned gpu_id = 0);

void iter_magma_solve(
        const int m, const int b_n, CPTRd a, CPTRd b, double *output, magma_queue_t magma_queue,
        const magmaDouble_ptr d_a, const magmaDouble_ptr d_b, const magmaDouble_ptr d_x, const magmaDouble_ptr d_wd,
        const magmaFloat_ptr d_ws, const bool psd = false, const unsigned gpu_id = 0);

void iter_magma_solve(
        const int m, const int n, CPTRd a, CPTRd b, double *output, const magma_queue_t &magma_queue,
        const magmaDouble_ptr d_a, const magmaDouble_ptr d_b);

void iter_magma_batch_solve(
        const int m, const int n, const std::deque<arma::mat> &a, const std::deque<arma::mat> &b, std::deque<arma::mat> &output,
        const magma_queue_t magma_queue, std::vector<magmaDouble_ptr> &d_a, std::vector<magmaDouble_ptr> &d_b, const unsigned gpu_id);

double solve_validate_host(
        CPTRd K_epsco, CPTRd K, CPTRd rhs, CPTRd K_test, CPTRd labels, const double meanabs_labels, const double neg_epsilon,
        const unsigned m, const unsigned n, const unsigned test_m, const unsigned iters, const magma_queue_t magma_queue, const cublasHandle_t cublas_h);

double solve_hybrid(CPTRd j_K_epsco, const unsigned n, const unsigned train_len, double *const j_solved, const unsigned magma_iters, const double magma_threshold,
             const magma_queue_t ma_queue, const unsigned irwls_iters, CPTRd j_train_labels, const size_t train_n_size, double *const j_work,
             const cudaStream_t custream, const cublasHandle_t cublas_H, CPTRd j_K_tune, const double labels_factor, const unsigned train_len_n,
             unsigned &best_iter, double *const d_best_weights, const unsigned K_train_len, double *const j_K_work, magma_int_t &info,
             const double iters_mul, const unsigned m);

}
