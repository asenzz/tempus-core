#pragma once

#include <cublas_v2.h>
#include <magma_types.h>
#include <cusolverDn.h>



namespace svr::solvers {

void kernel_from_distances(double *K, const double *Z, const size_t m, const size_t n, const double gamma);

void kernel_from_distances(double *K, const double *Z, const size_t m, const size_t n, const double gamma, const size_t gpu_id);

void kernel_from_distances_inplace(double *Kz, const size_t m, const size_t n, const double gamma);

void kernel_from_distances_inplace(double *Kz, const size_t m, const size_t n, const double gamma, const size_t gpu_id);

void kernel_from_distances_symm(double *K, const double *Z, const size_t m, const double gamma); // Buggy

__global__ void G_kernel_from_distances(double *__restrict K, const double *__restrict Z, const size_t mn, const double divisor);

__global__ void G_kernel_from_distances_inplace(double *__restrict Kz, const size_t mn, const double divisor);

//double score_kernel(const double *ref_kernel /* colmaj order */, const double norm_ref, const double *Z /* colmaj order */, const size_t M, const double gamma); // TODO  Test

__global__ void G_matmul_inplace(const double *__restrict__ input, double *__restrict__ output, const size_t N);

__global__ void G_sqrt_add(double *__restrict__ input, const double a, const size_t N);

__global__ void G_eq_matmul(const double *__restrict__ input1, const double *__restrict__ input2, double *__restrict__ output, const size_t N);

__global__ void G_abs_subtract(const double *__restrict__ input1, double *__restrict__ input2, const size_t N);

__global__ void G_abs(double *__restrict__ input, const size_t N);

double sum(const double *d_data, const size_t n, const cudaStream_t &strm);

double sumabs(const double *d_data, const size_t n, const cudaStream_t &strm);


// Solvers

/* CuSolver */

std::tuple<cusolverDnHandle_t, double *, double *, double *, int *, int *>
init_cusolver(const size_t gpu_phy_id, const size_t m, const size_t n);

void uninit_cusolver(const size_t gpu_phy_id, const cusolverDnHandle_t cusolverH, double *d_Ainput, double *d_B, double *d_work, int *d_Ipiv, int *d_devInfo);


void dyn_gpu_solve(
        const size_t gpu_phy_id, const size_t m, const size_t n, const double *Left, const double *Right, double *output,
        cusolverDnHandle_t cusolverH,
        double *d_Ainput, double *d_B, double *d_work, int *d_Ipiv, int *d_devInfo);

void strum_solve(const size_t Nrows, const size_t Nright, double *h_Ainput, double *h_rhs, double *h_B, const double epsco);

/* MAGMA */
std::tuple<magma_queue_t, magmaDouble_ptr, magmaDouble_ptr, magmaDouble_ptr, magmaDouble_ptr, magmaFloat_ptr, magmaInt_ptr>
init_magma_solver(const size_t m, const size_t b_n, const bool psd, const size_t gpu_id = 0);

std::tuple<std::vector<magmaDouble_ptr>, std::vector<magmaDouble_ptr>>
init_magma_batch_solver(const size_t batch_size, const size_t m, const size_t n);

void uninit_magma_solver(
        const magma_queue_t &magma_queue,
        const magmaDouble_ptr d_a, const magmaDouble_ptr d_b, const magmaDouble_ptr d_x, const magmaDouble_ptr d_wd, const magmaFloat_ptr d_ws, const magmaInt_ptr piv, const size_t gpu_id = 0);

void uninit_magma_batch_solver(std::vector<magmaDouble_ptr> &d_a, std::vector<magmaDouble_ptr> &d_b);

void dyn_magma_solve(
        const int m, const int b_n, const double *a, const double *b, double *output, magma_queue_t magma_queue = nullptr, const magmaInt_ptr piv = nullptr,
        const magmaDouble_ptr d_a = nullptr, const magmaDouble_ptr d_b = nullptr, const size_t gpu_id = 0);

void iter_magma_solve(
        const int m, const int b_n, const double *a, const double *b, double *output, magma_queue_t magma_queue,
        const magmaDouble_ptr d_a, const magmaDouble_ptr d_b, const magmaDouble_ptr d_x, const magmaDouble_ptr d_wd,
        const magmaFloat_ptr d_ws, const bool psd = false, const size_t gpu_id = 0);

void iter_magma_solve(
        const int m, const int n, const double *a, const double *b, double *output, const magma_queue_t &magma_queue,
        const magmaDouble_ptr d_a, const magmaDouble_ptr d_b);

void iter_magma_batch_solve(
        const int m, const int n, const std::deque<arma::mat> &a, const std::deque<arma::mat> &b, std::deque<arma::mat> &output,
        const magma_queue_t magma_queue, std::vector<magmaDouble_ptr> &d_a, std::vector<magmaDouble_ptr> &d_b, const size_t gpu_id);

double solve_validate(
        const double *K_epsco, const double *K, const double *rhs, const double *K_test, const double *labels, const double meanabs_labels, const double neg_epsilon,
        const double *d_one, const double *d_neg_one, double *error_mat, double *left, double *mult, double *work, double *best_solution, double *solved, double *d_neg_epsilon,
        const size_t m, const size_t n, const size_t test_m, const size_t iters, const magma_queue_t &magma_queue, const cublasHandle_t &cublas_h, const cudaStream_t &strm);

double solve_validate_host(
        const double *K_epsco, const double *K, const double *rhs, const double *K_test, const double *labels, const double meanabs_labels, const double neg_epsilon,
        const size_t m, const size_t n, const size_t test_m, const size_t iters, const magma_queue_t magma_queue, const cublasHandle_t cublas_h);

}
