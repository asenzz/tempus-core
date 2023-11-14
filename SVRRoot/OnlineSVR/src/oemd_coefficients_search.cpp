//
// Created by zarko on 10/3/22.
//
#include "oemd_coefficients_search.hpp"

#include <algorithm>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <unistd.h>
#include <unistd.h>
#include <string>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <mutex>
#include <thread>
#include <omp.h>
//#include <osqp.h>
#include <complex>
#include <semaphore.h>
#include <iomanip>

#include "common/compatibility.hpp"
#include "common/gpu_handler.hpp"
#include "common/parallelism.hpp"


namespace svr {
namespace oemd_search {


double
do_quality(const cufftDoubleComplex *const h_mask_fft, const size_t fft_size, const size_t siftings)
{
    const double coeff = double(fft_size) / 250.;
    double result = 0;
    const auto cplx_one = std::complex<double>(1.);
    const size_t end_i = fft_size * 2. * C_lambda1 / coeff;
#pragma omp parallel
    {
#pragma omp for simd reduction(+:result) schedule(dynamic, 1024)
        for (size_t i = 0; i < end_i; ++i) {
            std::complex<double> zz(1 - h_mask_fft[i].x, -h_mask_fft[i].y);
            std::complex<double> p(1, 0);
            for (size_t k = 0; k < siftings; ++k) p *= zz;
            result += std::norm(p) + fabs(1. - std::norm(cplx_one - p));
        }
#pragma omp for simd reduction(+:result) schedule(dynamic, 1024)
        for (size_t i = end_i; i < fft_size; ++i) {
            std::complex<double> zz(h_mask_fft[i].x, h_mask_fft[i].y);
            std::complex<double> p(1, 0);
            for (size_t k = 0; k < siftings; ++k) p *= zz;
            result += i < fft_size * 2. * C_lambda2 / coeff ? std::norm(p) : C_smooth_factor * std::norm(p);
        }
#pragma omp for reduction(+:result)
        for (size_t i = 0; i < fft_size; ++i) {
            const double zz_norm = std::norm(std::complex<double>{h_mask_fft[i].x, h_mask_fft[i].y});
            result += zz_norm > 1 ? zz_norm : 0;
        }
    }
    return result / double(fft_size);
}

void do_mult(const size_t m, const size_t n, const std::vector<double> &A_x, std::vector<double> &H)
{
    H.resize(m * m);
    __omp_pfor_i(0, m,
        for (size_t j = 0; j < m; ++j) {
            const auto ptr_ix = H.data() + i * m + j;
            *ptr_ix = 0;
            for (size_t k = 0; k < n; ++k)
                *ptr_ix += A_x[i * n + k] * A_x[j * n + k];
        }
    )
}

void do_diff_mult(const size_t M, const size_t N, const std::vector<double> &diff, std::vector<double> &Bmatrix)
{
    Bmatrix.resize(M * M, 0.);
    __omp_pfor_i (0, M,
        for (size_t j = 0; j <= i; ++j) {
            auto ptr_ix = Bmatrix.data() + i * M + j;
            *ptr_ix = 0;
            for (size_t k = 0; k < N - M - 1; ++k)
              *ptr_ix += diff[i + k] * diff[j + k];
        }
    )

    for (size_t i = 0; i < M; ++i) {
        for (size_t j = i + 1; j < M; ++j)
            Bmatrix[i * M + j] = Bmatrix[j * M + i];
    }
}

void prep_x_matrix(const size_t M, const size_t N, const double *x, std::vector<double> &Bmatrix, std::vector<double> &Fmatrix)
{
    std::vector<double> diff(N - 1, 0.);
    for (size_t i = 0; i < N - 1; ++i)
        diff[i] = x[i + 1] - x[i];
    do_diff_mult(M, N, diff, Bmatrix);
    Fmatrix.resize(M);
#pragma omp simd
    for (size_t i = 0; i < M; ++i)
        Fmatrix[i] = -2. * Bmatrix[i];
}

#if 0

int make_P_from_dense(const double *H, size_t m, c_int &P_nnz, std::vector<c_float> &P_x_vector, std::vector<c_int> &P_i_vector, std::vector<c_int> &P_p_vector)
{
    P_nnz = 0;
    P_x_vector.clear();
    P_i_vector.clear();
    P_p_vector.clear();
    for (size_t i = 0; i < m; i++) {
        P_p_vector.push_back(P_nnz);
        for (size_t j = 0; j <= i; j++) {
            P_i_vector.push_back(j);
            P_x_vector.push_back(H[i * m + j]);//assume H is symmetric!
            P_nnz++;
        }
    }
    P_p_vector.push_back(P_nnz);
    return 0;
}


int do_osqp(const size_t mask_size, std::vector<double> &good_mask, const size_t input_size, const double *x, const int gpu_id)
{
    size_t fft_size = C_mask_expander * mask_size;
    size_t M = mask_size;
    size_t N = input_size;
    std::vector<double> Bmatrix;
    std::vector<double> Fmatrix;
    prep_x_matrix(M, N, x, Bmatrix, Fmatrix);

    std::vector<double> lo_vector(1 + mask_size, 0.), up_vector(1 + mask_size, 1.);

    c_float *lo = lo_vector.data();
    c_float *up = up_vector.data();

    std::vector<c_int> A_i_vector;
    std::vector<c_int> A_p_vector;
    std::vector<c_float> A_x_vector;
    const double eps = 0.00000001;
    lo_vector[0] = 1 - eps;
    up_vector[0] = 1 + eps;

    size_t A_nnz = C_mask_expander * mask_size;
    for (size_t i = 0; i < mask_size; i++) {
        A_p_vector.push_back(2 * i);
        A_i_vector.push_back(0);
        A_x_vector.push_back(1.);
        A_i_vector.push_back(1 + i);
        A_x_vector.push_back(1.);
    }
    A_p_vector.push_back(A_nnz);


    std::vector<csc> A_values(1);

    csc *A = &A_values[0];

    csc_set_data(A, 1 + mask_size, mask_size, A_nnz, A_x_vector.data(), A_i_vector.data(), A_p_vector.data());


    std::vector<double> A_x;
    std::vector<double> F_x;

    first_matrix(mask_size, fft_size, C_lambda1, C_lambda2, smooth_factor, A_x, F_x);

    std::vector<double> H(mask_size * mask_size, 0.);

    do_mult(mask_size, 2 * (fft_size / 2 + 1), A_x, H);//mult by 2 because both real and imaginary

    for (size_t i = 0; i < H.size(); i++) {
        H[i] += Bmatrix[i] * C_b_factor;
    }
    for (size_t i = 0; i < F_x.size(); i++) {
        F_x[i] += Fmatrix[i] * C_b_factor;
    }

    std::vector<csc> P_values(1);

    csc *P = &P_values[0];

    std::vector<c_float> P_x_vector;
    std::vector<c_int> P_i_vector;
    std::vector<c_int> P_p_vector;

    c_int P_nnz = 0;

    make_P_from_dense(H.data(), mask_size, P_nnz, P_x_vector, P_i_vector, P_p_vector);


    csc_set_data(P, mask_size, mask_size, P_nnz, P_x_vector.data(), P_i_vector.data(), P_p_vector.data());

    c_float *q = F_x.data();
    for (size_t i = 0; i < mask_size; ++i) {
        q[i] = q[i] / 2;
    }


    c_int exitflag;

    /* Workspace, settings, matrices */
    OSQPSolver *solver;
    std::vector<OSQPSettings> vec_settings(1);
    OSQPSettings *settings = &vec_settings[0];
    osqp_set_default_settings(settings);


    settings->polish = 0;
    settings->max_iter = 4000;
    settings->eps_abs = 1.0e-05;
    settings->eps_rel = 1.0e-05;
    settings->alpha = 0.6;
    settings->eps_prim_inf = 1.0e-06;
    settings->eps_dual_inf = 1.0e-06;
    settings->deviceId = gpu_id;

    c_int m = mask_size;
    c_int n = mask_size + 1;//for the inequalities

    exitflag = osqp_setup(&solver, P, q, A, lo, up, n, m, settings);

    if (exitflag != 0) return -1;
    LOG4_DEBUG("Starting OSQP");
    osqp_solve(solver);
    LOG4_DEBUG("OSQP Done.");
    OSQPSolution *sol = solver->solution;
    c_float *real_solution = sol->x;

    good_mask.resize(mask_size, 0.);
    for (size_t i = 0; i < mask_size; i++) good_mask[i] = real_solution[i];
    fix_mask(mask_size, good_mask.data());
    osqp_cleanup(solver);
    return 0;
}
#endif

void
make_oemd_levels(
        std::vector<std::vector<double>> &masks,
        std::vector<size_t> &siftings,
        const size_t start_size,
        const size_t end_size,
        const size_t levels)
{
    if (masks.size() != levels - 1) masks.resize(levels - 1);
    if (masks.front().size() != start_size) {
        LOG4_DEBUG("Resizing first mask " << 0 << " to " << start_size);
        masks.front().resize(start_size);
    }
    if (masks.back().size() != end_size) {
        LOG4_DEBUG("Resizing last mask " << masks.size() - 1 << " to " << end_size);
        masks.back().resize(end_size);
    }
    __omp_pfor_i(1, masks.size() - 1,
        const auto new_size = (size_t) round(std::pow<double>(start_size, double(masks.size() - i) / double(masks.size())) * std::pow<double>(end_size, double(i) / double(masks.size())));
        if (masks[i].size() != new_size) {
            LOG4_DEBUG("Resizing level " << i << " out of " << levels << " masks to " << new_size);
            masks[i].resize(new_size);
        }
    )
    if (siftings.size() != masks.size()) {
        LOG4_DEBUG("Resizing siftings to " << masks.size());
        siftings.resize(masks.size());
    }
    for (size_t i = 0; i < siftings.size(); ++i) siftings[i] = GENERATE_SIFTINGS;
}

void
oemd_coefs_search(
        const std::vector<double> &val,
        const size_t mask_start_size,
        const size_t mask_end_size,
        const size_t window_start,
        const size_t window_end,
        const size_t levels,
        std::vector<std::vector<double>> &masks,
        std::vector<size_t> &siftings)
{
    for (auto &m: masks) std::reverse(m.begin(), m.end());
    make_oemd_levels(masks, siftings, mask_start_size, mask_end_size, levels);
    optimize_levels(val, masks, siftings, window_start, window_end);
    for (auto &m: masks) std::reverse(m.begin(), m.end());
}

}
}