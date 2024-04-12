#pragma once

#include <magma_types.h>
#include <cusolverDn.h>



namespace svr::solvers {

void kernel_from_distances(double *K, const double *Z, const size_t m, const size_t n, const double gamma);
void kernel_from_distances_symm(double *K, const double *Z, const size_t m, const double gamma);

//double score_kernel(const double *ref_kernel /* colmaj order */, const double norm_ref, const double *Z /* colmaj order */, const size_t M, const double gamma); // TODO  Test

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

void uninit_magma_solver(
        const magma_queue_t &magma_queue,
        const magmaDouble_ptr d_a, const magmaDouble_ptr d_b, const magmaDouble_ptr d_x, const magmaDouble_ptr d_wd, const magmaFloat_ptr d_ws, const magmaInt_ptr piv, const size_t gpu_id = 0);

void dyn_magma_solve(
        const int m, const int b_n, const double *a, const double *b, double *output, magma_queue_t magma_queue = nullptr, const magmaInt_ptr piv = nullptr,
        const magmaDouble_ptr d_a = nullptr, const magmaDouble_ptr d_b = nullptr, const size_t gpu_id = 0);

void iter_magma_solve(
        const int m, const int b_n, const double *a, const double *b, double *output, magma_queue_t magma_queue = nullptr,
        const magmaDouble_ptr d_a = nullptr, const magmaDouble_ptr d_b = nullptr, const magmaDouble_ptr d_x = nullptr, const magmaDouble_ptr d_wd = nullptr,
        const magmaFloat_ptr d_ws = nullptr, const bool psd = false, const size_t gpu_id = 0);


}
