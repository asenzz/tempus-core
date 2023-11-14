#pragma once

namespace svr::solvers {

//void kernel_from_distances(double *K, const double *Z, const size_t M, const double gamma);

//double score_kernel(const double *ref_kernel /* colmaj order */, const double norm_ref, const double *Z /* colmaj order */, const size_t M, const double gamma);

void call_gpu_overdetermined(const size_t Nrows, const size_t Ncols, const size_t Nrhs, const double *cpu_matrix, const double *cpu_rhs, double *cpu_output);

void dyn_gpu_solve(const size_t m, const double *left, const double *right, double *output);

void qrsolve(const size_t Nrows, const size_t Nright, const double *d_Ainput, const double *d_rhs, double *d_B);

}