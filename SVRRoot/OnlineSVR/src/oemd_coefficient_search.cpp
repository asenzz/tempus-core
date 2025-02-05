//
// Created by zarko on 10/3/22.
//
#include <vector>
#include <cmath>
#include <cstdio>
//#include <osqp.h>
#include <complex>
#include <iomanip>
#include "appcontext.hpp"
#include "util/math_utils.hpp"
#include "oemd_coefficient_search.hpp"
#include "common/gpu_handler.hpp"
#include "oemd_coefficients.hpp"
#include "ModelService.hpp"
#include "align_features.cuh"

namespace svr {
namespace oemd {


oemd_coefficients_search::oemd_coefficients_search(const uint16_t levels, const bpt::time_duration &resolution, const uint32_t label_len) :
        resolution(resolution),
        sample_rate(onesec / resolution),
        levels(levels),
        max_row_len(business::ModelService::get_max_row_len()), // Not a constexpr because business::ModelService::t_quantisations::get_max_quantisation() is initialized after this class
        label_len(label_len),
        fir_validation_window(levels * .5 * C_sifted_fir_max_len + C_align_window_oemd * label_len + max_row_len)
{
}


double
oemd_coefficients_search::do_quality(const std::vector<cufftDoubleComplex> &h_mask_fft, const uint16_t siftings)
{
    const double coeff = double(h_mask_fft.size()) / 250.;
    double result = 0;
    constexpr auto cplx_one = std::complex<double>(1.);
    const uint32_t end_i = h_mask_fft.size() * 2. * lambda1 / coeff;
    OMP_FOR_(end_i, simd reduction(+:result))
    for (DTYPE(end_i) i = 0; i < end_i; ++i) {
        std::complex<double> zz(1 - h_mask_fft[i].x, -h_mask_fft[i].y);
        std::complex<double> p(1, 0);
        // UNROLL()
        for (DTYPE(siftings) k = 0; k < siftings; ++k) p *= zz;
        result += std::norm(p) + fabs(1. - std::norm(cplx_one - p));
    }
    OMP_FOR_(h_mask_fft.size(), simd reduction(+:result))
    for (auto i = end_i; i < h_mask_fft.size(); ++i) {
        std::complex<double> zz(h_mask_fft[i].x, h_mask_fft[i].y);
        std::complex<double> p(1, 0);
        // UNROLL()
        for (DTYPE(siftings) k = 0; k < siftings; ++k) p *= zz;
        result += i < h_mask_fft.size() * 2. * lambda2 / coeff ? std::norm(p) : C_smooth_factor * std::norm(p);
    }
    OMP_FOR_(h_mask_fft.size(), SSIMD reduction(+:result))
    for (const auto &i: h_mask_fft) {
        const double zz_norm = std::norm(std::complex<double>{i.x, i.y});
        result += zz_norm > 1 ? zz_norm : 0;
    }
    return result / double(h_mask_fft.size());
}


void oemd_coefficients_search::do_diff_mult(const uint32_t M, const uint32_t N, const std::vector<double> &diff, std::vector<double> &Bmatrix)
{
    const auto MM = M * M;
    const auto MM2 = MM / 2;
    Bmatrix.resize(MM, 0.);
#ifdef __GNUC__
    OMP_FOR_(MM2, simd)
#else
    OMP_FOR_(MM2, simd collapse(2))
#endif
    for (DTYPE(M) i = 0; i < M; ++i) {
        for (DTYPE(i) j = 0; j <= i; ++j) {
            auto ptr_ix = Bmatrix.data() + i * M + j;
            *ptr_ix = 0;
            for (DTYPE(N) k = 0; k < N - M - 1; ++k)
                *ptr_ix += diff[i + k] * diff[j + k];
        }
    }
#ifdef __GNUC__
    OMP_FOR_(MM2, simd)
#else
    OMP_FOR_(MM2, simd collapse(2))
#endif
    for (DTYPE(M) i = 0; i < M; ++i)
        for (auto j = i + 1; j < M; ++j)
            Bmatrix[i * M + j] = Bmatrix[j * M + i];
}

void oemd_coefficients_search::prep_x_matrix(const uint32_t M, const uint32_t N, CPTRd x, std::vector<double> &Bmatrix, std::vector<double> &Fmatrix)
{
    std::vector<double> diff(N - 1, 0.);
    for (DTYPE(N) i = 0; i < N - 1; ++i) diff[i] = x[i + 1] - x[i];
    do_diff_mult(M, N, diff, Bmatrix);
    Fmatrix.resize(M);
    OMP_FOR(M)
    for (DTYPE(M) i = 0; i < M; ++i)
        Fmatrix[i] = -2. * Bmatrix[i];
}

#if 0
int make_P_from_dense(CPTRd H, uint32_t m, c_int &P_nnz, std::vector<c_float> &P_x_vector, std::vector<c_int> &P_i_vector, std::vector<c_int> &P_p_vector)
{
    P_nnz = 0;
    P_x_vector.clear();
    P_i_vector.clear();
    P_p_vector.clear();
    for (DTYPE(m) i = 0; i < m; i++) {
        P_p_vector.push_back(P_nnz);
        for (DTYPE(i) j = 0; j <= i; j++) {
            P_i_vector.push_back(j);
            P_x_vector.push_back(H[i * m + j]);//assume H is symmetric!
            P_nnz++;
        }
    }
    P_p_vector.push_back(P_nnz);
    return 0;
}


int do_osqp(const uint32_t mask_size, std::vector<double> &good_mask, const uint32_t input_size, CPTRd x, const int gpu_id)
{
    uint32_t fft_size = C_mask_expander * mask_size;
    uint32_t M = mask_size;
    uint32_t N = input_size;
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

    uint32_t A_nnz = C_mask_expander * mask_size;
    for (uint32_t i = 0; i < mask_size; i++) {
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

    for (uint32_t i = 0; i < H.size(); i++) {
        H[i] += Bmatrix[i] * C_b_factor;
    }
    for (uint32_t i = 0; i < F_x.size(); i++) {
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
    for (uint32_t i = 0; i < mask_size; ++i) {
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
    for (uint32_t i = 0; i < mask_size; i++) good_mask[i] = real_solution[i];
    fix_mask(mask_size, good_mask.data());
    osqp_cleanup(solver);
    return 0;
}
#endif

void
oemd_coefficients_search::prepare_masks(
        std::deque<std::vector<double>> &masks,
        std::deque<uint16_t> &siftings,
        const uint16_t levels)
{
    assert(levels > 0);
#if 0
    if (masks.size() != levels - 1) masks.resize(levels - 1);
    if (masks.front().size() != C_fir_mask_start_len) {
        LOG4_DEBUG("Resizing first mask " << 0 << " to " << C_fir_mask_start_len);
        masks.front().resize(C_fir_mask_start_len);
    }
    if (masks.back().size() != C_fir_mask_end_len) {
        LOG4_DEBUG("Resizing last mask " << masks.size() - 1 << " to " << C_fir_mask_end_len);
        masks.back().resize(C_fir_mask_end_len);
    }
    // OMP_FOR(masks.size() - 1)
    for (uint16_t i = 1; i < masks.size() - 1; ++i) { // TODO Masks are in inverted order, fix!
        const auto new_size = (uint32_t) round(
                std::pow<double>(C_fir_mask_start_len, double(masks.size() - i) / masks.size()) *
                std::pow<double>(C_fir_mask_end_len, double(i) / double(masks.size())));
        if (masks[i].size() != new_size) {
            LOG4_DEBUG("Resizing level " << i << " out of " << levels << " masks to " << new_size);
            masks[i].resize(new_size);
        }
    }
#else
    if (masks.size() != DTYPE(levels)(levels - 1)) masks.resize(levels - 1);
#endif
    if (siftings.size() != masks.size()) {
        LOG4_DEBUG("Resizing siftings to " << masks.size());
        siftings.resize(masks.size());
    }
    std::fill(C_default_exec_policy, siftings.begin(), siftings.end(), oemd_coefficients::C_default_siftings);
}


void
oemd_coefficients_search::smoothen_mask(std::vector<double> &mask, common::t_drand48_data_ptr buffer)
{
    const uint32_t window_size = 3 + 2. * (mask.size() * common::drander(buffer) / 10.);
    const uint32_t mask_size = mask.size();

    std::vector<double> weights(window_size);
    double wsum = 0;
#pragma omp simd reduction(+:wsum)
    for (DTYPE(window_size) i = 0; i < window_size; ++i) {
        weights[i] = exp(-pow((3. * ((double) i - (window_size / 2))) / (double) (window_size / 2), 2) / 2.);
        wsum += weights[i];
    }

    std::vector<double> nmask(mask_size);
#pragma omp simd
    for (DTYPE(mask_size) i = 0; i < mask_size; ++i) {
        double sum = 0;
        for (auto j = std::max<DTYPE(window_size) >(0, i - window_size / 2); j <= std::min<DTYPE(window_size) >(i + window_size / 2, mask_size - 1); ++j)
            sum += weights[window_size / 2 + i - j] * mask[j];
        nmask[i] = sum / wsum;
    }
    mask = nmask;
}


void
oemd_coefficients_search::save_mask(
        const std::vector<double> &mask,
        const std::string &queue_name,
        const uint16_t level,
        const uint16_t levels)
{
    uint16_t ctr = 0;
    LOG4_TRACE("Saving mask for level " << level << " of " << levels << ", queue " << queue_name << ", mask " <<
                                        common::present(arma::vec((double *) mask.data(), mask.size(), false, true)));
    while (common::file_exists(oemd_coefficients::get_mask_file_name(ctr, level, levels, queue_name))) { ++ctr; }
    const auto mask_file_name = oemd_coefficients::get_mask_file_name(ctr, level, levels, queue_name);
    std::ofstream mask_file(svr::common::formatter() << mask_file_name);
    if (mask_file.is_open()) {
        LOG4_DEBUG("Saving mask for " << level << " to " << mask_file_name);
        mask_file << std::setprecision(std::numeric_limits<double>::max_digits10);
        for (auto it = mask.cbegin(); it != mask.cend(); ++it)
            if (it != std::prev(mask.cend()))
                mask_file << *it << ",";
            else
                mask_file << *it;
        mask_file.close();
    } else
        LOG4_ERROR("Aborting saving! Unable to open file " << mask_file_name << " for writing.");
}

#if 0

double get_std(double *x, const uint32_t input_size)
{
    double sum = 0.;
    for (uint32_t i = 0; i < input_size - 1; ++i) {
        sum += pow(x[i] - x[i + 1], 2);
    }
    return sqrt(sum / (double) input_size);
}

std::vector<double>
oemd_coefficients_search::fill_auto_matrix(const uint32_t M, const uint16_t siftings, const uint32_t N, CPTRd x)
{
    std::vector<double> diff(N - 1);
#pragma omp parallel for num_threads(adj_threads(N - 1))
    for (DTYPE(N) i = 0; i < N - 1; ++i)
        diff[i] = x[i + 1] - x[i];

    const DTYPE(M) Msift = M * siftings;
    std::vector<double> global_sift_matrix(Msift);
#pragma omp parallel for num_threads(adj_threads(Msift))
    for (DTYPE(Msift) i = 0; i < Msift; ++i) {
        double sum = 0;
        for (auto j = N / 2; j < N - 1 - i; ++j)
            sum += diff[j] * diff[j + i];
        global_sift_matrix[i] = sum / double(N - 1. - i - N / 2.);
    }
    return global_sift_matrix;
}

#endif


int oemd_coefficients_search::do_filter(const std::vector<cufftDoubleComplex> &h_mask_fft)
{
    UNROLL()
    for (auto i: h_mask_fft)
        if (std::norm<double>({i.x, i.y}) > norm_thresh)
            return 1;
    return 0;
}


}
}
