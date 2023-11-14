//
// Created by zarko on 7/5/22.
//

#include "common/defines.h"
#include "fast_cvmd.hpp"
#include "util/math_utils.hpp"
#include "common/compatibility.hpp"
#include "common/Logging.hpp"
#include "common/parallelism.hpp"
#include <armadillo>
#include <vector>


namespace svr {


static void compute_cos_sin(
        const std::vector<double> &omega,
        std::vector<double> &phase_cos,
        std::vector<double> &phase_sin,
        const double step = DEFAULT_PHASE_STEP)
{
    if (phase_cos.size() != omega.size()) phase_cos.resize(omega.size(), 0.);
    if (phase_sin.size() != omega.size()) phase_sin.resize(omega.size(), 0.);

    __omp_pfor_i(0, omega.size(), phase_cos[i] = cos(2 * M_PI * step * omega[i]); phase_sin[i] = sin(2 * M_PI * step * omega[i]) )
    LOG4_DEBUG("Omega " << common::deep_to_string(omega) << ", phase cos " << common::deep_to_string(phase_cos) << ", phase sin " << common::deep_to_string(phase_sin) << ", step " << step << ", alpha bins " << ALPHA_BINS << ", tau fidelity " << TAU_FIDELITY << ", max VMD iterations " << MAX_VMD_ITERATIONS << ", epsilon " << EPS << ", tolerance " << CVMD_TOL << ", omega divisor " << OMEGA_DIVISOR);
}


#if defined(ORTHO_VMD) // Orthogonally parallelized VMD for CUDA, TODO port and replace thread primitives with CUDA kernel

#include "common/barrier.hpp"

void
fast_cvmd::step_decompose_matrix(
        const std::vector<double> &phase_cos,
        const std::vector<double> &phase_sin,
        const std::vector<double> &values,
        const std::vector<double> &previous,
        std::vector<std::vector<double>> &decomposition,
        arma::mat &H) const
{
    if (values.empty()) LOG4_THROW("Input empty!");
    const size_t K = phase_cos.size();
    if (K * 2 != levels || K != phase_sin.size()) LOG4_THROW("Invalid phase cos size " << phase_cos.size());
    if (decomposition.size() != values.size()) decomposition.resize(values.size(), std::vector<double>(levels));

    arma::mat f(levels, 1), A = arma::zeros(1, levels);
    if (H.empty()) {
        H = arma::zeros(levels, levels);
#pragma omp parallel for
        for (size_t j = 0; j < K; ++j) {
            H(2 * j, 2 * j) = 1;
            H(2 * j + 1, 2 * j + 1) = 1;
            A(0, 2 * j) = 1;
        }
    }
    arma::mat solution(levels, 1);
    svr::common::barrier bar(K);
    __pxt_tpfor(size_t, j, 0, K,
        const auto l = 2 * j;
        const auto l1 = l + 1;
        for (size_t t = 0; t < values.size(); ++t) {
            if (!j) f.zeros();
            else bar.wait();

            const double prev_ui = t ? decomposition[t - 1][l] : previous[l];
            const double prev_vi = t ? decomposition[t - 1][l1] : previous[l1];
            f(l, 0) += -phase_cos[j] * prev_ui + phase_sin[j] * prev_vi;
            f(l1, 0) += -phase_sin[j] * prev_ui + -phase_cos[j] * prev_vi;

            bar.wait();

            if (!j) {
                if (decomposition[t].size() != levels) decomposition[t].resize(levels);
                solution = arma::solve(H, -A.t() * (-arma::solve(A * arma::solve(H, A.t()), A * arma::solve(H, f) + values[t])) - f);
            } else  bar.wait();

            decomposition[t][l] = solution(l, 0);
            decomposition[t][l1] = solution(l1, 0);
        }
    )
}

#elif defined(FASTER_CVMD) // My faster but buggy TODO test and fix

void fast_cvmd::step_decompose_matrix(
        const std::vector<double> &phase_cos,
        const std::vector<double> &phase_sin,
        const std::vector<double> &values,
        const std::vector<double> &previous,
        std::vector<std::vector<double>> &decomposition,
        arma::mat &H) const
{
    if (values.empty()) {
        LOG4_ERROR("Input empty!");
        return;
    }
    const size_t K = phase_cos.size();
    if (K * 2 != levels || K != phase_sin.size() || previous.size() != levels) LOG4_THROW("Invalid phase cos size " << phase_cos.size());
    if (decomposition.size() != values.size()) decomposition.resize(values.size());

    arma::mat f = arma::zeros(levels, 1);
    arma::mat A = arma::zeros(1, levels);
    if (H.empty()) {
        H = arma::zeros(levels, levels);
#pragma omp simd
        for (size_t j = 0; j < K; ++j) {
            H(2 * j, 2 * j) += 1;
            H(2 * j + 1, 2 * j + 1) += 1;
        }
    }
#pragma omp simd
    for (size_t j = 0; j < levels; j += 2) {
        A(0, j) = 1;
        f(j, 0) = -phase_cos[j / 2] * previous[j] + phase_sin[j / 2] * previous[j + 1];
        f(j + 1, 0) = -phase_sin[j / 2] * previous[j] - phase_cos[j / 2] * previous[j + 1];
    }
    {
        const arma::mat solution = arma::solve(H, -A.t() * (-arma::solve(A * arma::solve(H, A.t()), A * arma::solve(H, f) + values.front())) - f);
        if (decomposition.front().size() != levels) decomposition.front().resize(levels);
        std::memcpy(decomposition.front().data(), solution.memptr(), levels * sizeof(double));
    }
    for (size_t t = 1; t < values.size(); ++t) {
        const auto &prev_t = decomposition[t - 1];
#pragma omp simd
        for (size_t j = 0; j < K; ++j) {
            f(2 * j, 0) = -phase_cos[j] * prev_t[2 * j] + phase_sin[j] * prev_t[2 * j + 1];
            f(2 * j + 1, 0) = -phase_sin[j] * prev_t[2 * j] - phase_cos[j] * prev_t[2 * j + 1];
        }
        const arma::mat solution = arma::solve(H, -A.t() * (-arma::solve(A * arma::solve(H, A.t()), A * arma::solve(H, f) + values[t])) - f);
        if (decomposition[t].size() != levels) decomposition[t].resize(levels);
        std::memcpy(decomposition[t].data(), solution.memptr(), levels * sizeof(double));
    }
}

#else // Emo's implementation

static void
create_solve_problem(
        const std::vector<double> &previous,
        const std::vector<double> &values,
        const std::vector<double> &phase_cos,
        const std::vector<double> &phase_sin,
        arma::mat &H,
        arma::mat &f,
        arma::mat &A,
        bool &first_call)
{
    const size_t n = values.size();
    const size_t K = phase_cos.size();
    A.zeros(); // = arma::zeros(n, n * 2 * K);
    /*
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < K; j++) {
            A(i, 2 * j + i * 2 * K) = 1.;
        }
    }
    */
    f.zeros();

    for (size_t i = 0; i < n; ++i) {
        if (i == 0) {
//#pragma omp parallel for default(none) shared(f, A, i, H, phase_cos, phase_sin, K, first_call, previous)
            for (size_t j = 0; j < K; ++j) {
                A(i, 2 * j + i * 2 * K) = 1;

                const double prev_ui = previous[2 * j];
                const double prev_vi = previous[2 * j + 1];
                const double CC = phase_cos[j];
                const double SS = phase_sin[j];

                //%u_i+1, u_i+1
                if (first_call) {
                    H(2 * j + i * 2 * K, 2 * j + i * 2 * K) += 1.;
                    H(2 * j + 1 + i * 2 * K, 2 * j + 1 + i * 2 * K) += 1.;
                }
                //%f for u_i+1 u_i
                f(2 * j + i * 2 * K, 0) += -CC * prev_ui;
                //%f for v_i+1 u_i
                f(2 * j + 1 + i * 2 * K, 0) += -SS * prev_ui;

                //%f for v_i+1 v_i
                f(2 * j + 1 + i * 2 * K, 0) += -CC * prev_vi;
                //%f for u_i+1 v_i
                f(2 * j + i * 2 * K) += SS * prev_vi;
            }
            first_call = false;
        } else {
#pragma omp parallel for default(none) shared(A, i, H, phase_cos, phase_sin, K) schedule(dynamic)
            for (size_t j = 0; j < K; ++j) {
                A(i, 2 * j + i * 2 * K) = 1;

                const double CC = phase_cos[j];
                const double SS = phase_sin[j];

                H(2 * j + (i - 1) * 2 * K, 2 * j + (i - 1) * 2 * K) += 1.;
                H(2 * j + 1 + (i - 1) * 2 * K, 2 * j + 1 + (i - 1) * 2 * K) += 1.;
                H(2 * j + i * 2 * K, 2 * j + i * 2 * K) += 1.;
                H(2 * j + 1 + i * 2 * K, 2 * j + 1 + i * 2 * K) += 1.;

                H(2 * j + (i - 1) * 2 * K, 2 * j + i * 2 * K) += (-CC);
                H(2 * j + i * 2 * K, 2 * j + (i - 1) * 2 * K) += (-CC);

                H(2 * j + (i - 1) * 2 * K, 2 * j + 1 + i * 2 * K) += (-SS);
                H(2 * j + 1 + i * 2 * K, 2 * j + (i - 1) * 2 * K) += (-SS);

                H(2 * j + 1 + (i - 1) * 2 * K, 2 * j + 1 + i * 2 * K) += (-CC);
                H(2 * j + 1 + i * 2 * K, 2 * j + 1 + (i - 1) * 2 * K) += (-CC);

                H(2 * j + 1 + (i - 1) * 2 * K, 2 * j + i * 2 * K) += SS;
                H(2 * j + i * 2 * K, 2 * j + 1 + (i - 1) * 2 * K) += SS;
            }
        }
    }
}


void fast_cvmd::step_decompose_matrix(
        const std::vector<double> &phase_cos,
        const std::vector<double> &phase_sin,
        const std::vector<double> &values,
        const std::vector<double> &previous,
        std::vector<std::vector<double>> &decomposition,
        arma::mat &H) const
{
    const size_t len = values.size();
    const size_t K = phase_cos.size();
    if (K * 2 != levels) LOG4_THROW("Invalid phase cos size " << phase_cos.size());
    if (decomposition.size() != len) decomposition.resize(len);
    arma::mat f = arma::zeros(levels, 1), A = arma::zeros(1, levels);
    bool first_call;
    if (H.empty()) {
        H = arma::zeros(levels, levels);
        first_call = true;
    } else {
        first_call = false;
    }

    for (size_t t = 0; t < values.size(); ++t) {
        create_solve_problem(t > 0 ? decomposition[t - 1] : previous, {values[t]}, phase_cos, phase_sin, H, f, A, first_call);
        const arma::mat lambda_rhs = A * arma::solve(H,f) + values[t];
        const arma::mat lambda = -arma::solve((A * arma::solve(H, A.t())), lambda_rhs);
        const arma::mat solution = arma::solve(H, -A.t() * lambda - f);

        if (decomposition[t].size() != levels) decomposition[t].resize(levels);
        // LOG4_DEBUG("Solution size " << arma::size(solution));
        std::memcpy(decomposition[t].data(), solution.memptr(), levels * sizeof(double));
    }
}

#endif

fast_cvmd::fast_cvmd(const size_t _levels) : spectral_transform(std::string("cvmd"), _levels), levels(_levels)
{
    if (levels < 2 or levels % 2) LOG4_THROW("Invalid number of levels " << levels);
}


static void
do_vmd_freqs(
        const arma::mat &signal, const double alpha, const double tau, const size_t K, const size_t DC, const size_t init,
        const double tol, /* outputs */ arma::mat &u, arma::cx_mat &u_hat, std::vector<double> &omega_plus)
{
#ifdef EMOS_OMEGAS
    return;
#endif
    if ((signal.n_elem & 1) != 0) {
        LOG4_ERROR("Odd series length " << arma::size(signal) << " not supported.");
        return;
    }
    const size_t save_T = signal.n_elem;
    const size_t T = 2 * save_T;

    //create the mirrored signal (TODO Why? Maybe not needed!)
    arma::mat f_mirror(1, T);
#pragma omp parallel
    {
#pragma omp for simd nowait schedule(dynamic, 32)
        for (size_t i = 0; i < save_T / 2; ++i)
            f_mirror(0, i) = signal(0, save_T / 2 - 1 - i);

#pragma omp for simd nowait schedule(dynamic, 32)
        for (size_t i = 0; i < save_T; ++i)
            f_mirror(0, save_T / 2 + i) = signal(0, i);

#pragma omp for simd nowait schedule(dynamic, 32)
        for (size_t i = 0; i < save_T / 2; ++i)
            f_mirror(0, 3 * save_T / 2 + i) = signal(0, save_T - 1 - i);
    }

    // arma::mat Alpha(1, K, arma::fill::value(alpha)); // provision to use different alpha for the different frequences, at a later stage
    const double fs = 1. / double(save_T); // step
    u = arma::zeros(T, K);

    //this may or may not work instead. double EPS=std::numeric_limits<T>::epsilon();

    //% Time Domain 0 to T (of mirrored signal)
    arma::mat t(1, T);
    __tbb_pfor_i(0, T, t(0, i) = (double(i) + 1.) / double(T));

    //% Spectral Domain discretization
    arma::mat freqs = t - 0.5 - 1. / (double) T;

    omega_plus.resize(K, 0.);

    arma::cx_mat f_hat_plus = common::fftshift(common::matlab_fft(f_mirror));
    f_hat_plus.cols(arma::span(0, T / 2 - 1)).fill(std::complex<double>(0));

    switch (init) {
        case 1: {
            __tbb_pfor_i(0, K, omega_plus[i] = .5 / K * i)
            break;
        }
        case 2: {
            arma::mat real_ran(K, 1);
            __tbb_pfor_i(0, K, real_ran(i, 0) = exp(log(fs) + (log(.5) - log(fs)) * common::randouble()))
            real_ran = arma::sort(real_ran);
            memcpy(omega_plus.data(), real_ran.memptr(), K * sizeof(double));
            break;
        }
        case 0: {
            memset(omega_plus.data(), 0, omega_plus.size() * sizeof(omega_plus[0]));
            break;
        }
        default:
            LOG4_THROW("Init should be 0, 1 or 2!");

    }
    if (DC == 1) omega_plus[0] = 0; // if DC component, then first frequency is 0!
    /* completed initialization of frequencies*/

    arma::cx_mat lambda_hat(arma::zeros(1, T), arma::zeros(1, T)); // Keeping track of the evolution of lambda_hat can be done with the iterations
    arma::cx_mat u_hat_plus(arma::zeros(T, K), arma::zeros(T, K));
    arma::cx_mat u_hat_plus_old = u_hat_plus;

    double uDiff = tol + EPS; //tol+EPS must be different from just tol, that is why EPS should not be too small

    arma::cx_mat sum_uk(arma::zeros(1, T), arma::zeros(1, T));
    size_t n_iter = 0;
    while (n_iter < MAX_VMD_ITERATIONS && uDiff > tol) {
        //% update first mode accumulator
        size_t k = 0;

        __tbb_pfor_i(0, T, sum_uk(0, i) += u_hat_plus_old(i, K - 1) - u_hat_plus_old(i, 0))

        //% update spectrum of first mode through Wiener filter of residuals
        __tbb_pfor_i(0, T, u_hat_plus(i, k) = (f_hat_plus(0, i) - sum_uk(0, i) - lambda_hat(0, i) / 2.) / (1. + alpha * std::pow((freqs(0, i) - omega_plus[k]), 2)))

        if (!DC) {
            //when DC==1, the first frequency can not be changed. that is why this cycle only for DC==0.
            double sum_up = 0;
            double sum_down = 0;
#pragma omp parallel for reduction(+:sum_up, sum_down)
            for (size_t i = 0; i < T / 2; i++) {
                sum_up += freqs(0, T / 2 + i) * std::norm(u_hat_plus(T / 2 + i, k));
                sum_down += std::norm(u_hat_plus(T / 2 + i, k));
            }
            omega_plus[0] = sum_up / sum_down;
        }

        //update of any other mode (k from 1 to K-1), after the first
        for (k = 1; k < K; ++k) {
            // accumulator
            __tbb_pfor_i(0, T, sum_uk(0, i) += u_hat_plus(i, k - 1) - u_hat_plus_old(i, k))

            // mode spectrum
            __tbb_pfor_i(0, T, u_hat_plus(i, k) = (f_hat_plus(0, i) - sum_uk(0, i) - lambda_hat(0, i) / 2.) / (1. + alpha * std::pow(freqs(0, i) - omega_plus[k], 2)))

            //re-compute center frequencies
            double sum_up = 0;
            double sum_down = 0;
#pragma omp parallel for reduction(+:sum_up, sum_down)
            for (size_t i = 0; i < T / 2; ++i) {
                sum_up += freqs(0, T / 2 + i) * std::norm(u_hat_plus(T / 2 + i, k));
                sum_down += std::norm(u_hat_plus(T / 2 + i, k));
            }
            omega_plus[k] = sum_up / sum_down;
        }
        // Dual ascent
        arma::cx_mat sum_u_hat_plus = arma::sum(u_hat_plus.t());

        lambda_hat += tau * (sum_u_hat_plus - f_hat_plus.cols(0, T - 1)); // xxx

        // loop counter
        ++n_iter;

        //compute uDiff
        uDiff = 0;
#pragma omp parallel for simd reduction(+:uDiff) num_threads(K) schedule(dynamic, 1) default(none) shared(u_hat_plus, u_hat_plus_old, K, T)
        for (size_t i = 0; i < K; ++i) {
            double s = 0;
            double s_norm_old = 0;
            for (size_t j = 0; j < T; ++j) {
                s += std::norm(u_hat_plus(j, i) - u_hat_plus_old(j, i));
                s_norm_old += std::norm(u_hat_plus_old(j, i));
            }
            uDiff += s / s_norm_old;
        }
        u_hat_plus_old = u_hat_plus;
    }
    //%------ Postprocessing and cleanup
    //% discard empty space if converged early - this step is not used here
    //N = min(N,n);
    //omega = omega_plus(1:N,:);
    // - this not needed, since we do not keep all omega, only the latest!

    // Signal reconstruction
    u_hat = arma::cx_mat(arma::zeros(T, K), arma::zeros(T, K));
#pragma omp parallel for collapse(2)
    for (size_t i = 0; i < T / 2; ++i) {
        for (size_t k = 0; k < K; ++k) {
            u_hat(T / 2 + i, k) = u_hat_plus(T / 2 + i, k);
            // here it is not clear why, but T/2 is set both above and below. Instead, the 0 is set to conj of T-1
            u_hat(T / 2 - i, k) = std::conj(u_hat_plus(T / 2 + i, k));
        }
    }

    __tbb_pfor(k, 0, K, u_hat(0, k) = std::conj(u_hat(T - 1, k)))

    arma::mat u_big = arma::zeros(K, t.n_elem);
    __tbb_pfor(k, 0, K, u_big.row(k) = arma::trans(arma::real(common::matlab_ifft(common::ifftshift(u_hat.col(k))))))

    u = arma::zeros(K, T / 2);
#pragma omp parallel for collapse(2)
    for (size_t i = 0; i < T / 2; ++i)
        for (size_t k = 0; k < K; ++k)
            u(k, i) = u_big(k, T / 4 + i);

    //recompute spectrum
    //clear u_hat;
    u_hat = arma::cx_mat(arma::zeros(K, T / 2), arma::zeros(K, T / 2));
    __tbb_pfor(k, 0, K,
        u_hat.row(k) = common::fftshift(common::matlab_fft(u.row(k)));  // should work without conj. not clear why conj needed.
    )
    omega_plus /= OMEGA_DIVISOR;
    //while (omega_plus[0] < 1e-12) omega_plus[0] *= 10.;
    //Emo // omega_plus = {1.000000000000000e-12, 4.830917874396136e-08, 9.661835748792271e-08, 2.415458937198068e-07, 6.038647342995169e-07, 1.207729468599034e-06, 2.415458937198068e-06, 1.207729468599034e-05, 3.472222222222222e-05, 6.944444444444444e-05, 1.388888888888889e-04, 2.777777777777778e-04, 5.555555555555556e-04, 1.111111111111111e-03, 3.333333333333333e-03, 1.666666666666667e-02};
};

#if 0 // My parallelized version
static void
do_vmd_freqs(
    const arma::mat &signal, const double alpha, const double tau, const int K, const int DC, const int init,
    const double tol, /* outputs */ arma::mat &u, arma::cx_mat &u_hat, std::vector<double> &omega_plus)
{
    //Initialize u, w, lambda, n
    //
    const int save_T = signal.n_elem;

    int T = save_T;
    if (T & 1) LOG4_THROW("Input data size " << T << " must be even!");

    //create the mirrored signal
    arma::mat f_mirror = arma::zeros(1,2 * T);

    __omp_pfor_i(0, T / 2, f_mirror(0, i) = signal(0, T / 2 - 1 - i))
    __omp_pfor_i(0, T, f_mirror(0, T / 2 + i) = signal(0, i))
    __omp_pfor_i(0, T / 2, f_mirror(0, 3 * T / 2 + i) = signal(0, T - 1 - i))

    arma::mat f = f_mirror;
    T = 2 * T;

    arma::mat Alpha = alpha * arma::ones(1, K); //provision to use different alpha for the different frequences, at a later stage

    const double fs = 1. / double(save_T); //step

    const int N_max = MAX_VMD_ITERATIONS; //max number of iterations

    u = arma::zeros(T,K);
    const double eps = 2.220446049250313e-16;
    //this may or may not work instead. double eps=std::numeric_limits<T>::epsilon();

    // Time Domain 0 to T (of mirrored signal)
    arma::mat t(1, T);
    __omp_pfor_i(0, T, t(0,i) = (double(i) + 1.) / double(T))
    // Spectral Domain discretization
    arma::mat freqs = t - 0.5 - 1. / double(T);
    omega_plus.resize(K);
    memset(omega_plus.data(), 0, omega_plus.size() * sizeof(double));
    arma::cx_mat f_hat = common::fftshift(common::matlab_fft(f));
    arma::cx_mat f_hat_plus = f_hat;

    __omp_pfor_i(0, T / 2, f_hat_plus(0, i) = 0)

    switch (init) {
        case 1:{
            __omp_pfor_i(0, K, omega_plus[i] = 0.5 / K * i);
            break;
        }
        case 2:{
            arma::mat real_ran (K,1);
            __omp_pfor_i(0, K, real_ran(i, 0) = exp(log(fs) + (log(0.5) - log(fs)) * common::randouble()));
            real_ran = arma::sort(real_ran);
            __omp_pfor_i(0, K, omega_plus[i] = real_ran(i, 0));
            break;
        }
        case 0:{
            //__omp_pfor_i(0, K, omega_plus[i] = 0);
            break;
        }
        default:{
            LOG4_THROW("Init should be 0,1 or 2!");
        }
    }
    if (DC == 1) { //if DC component, then first frequency is 0!
        omega_plus[0] = 0;
    }
    /* completed initialization of frequencies */
    arma::cx_mat lambda_hat( arma::zeros(1, freqs.n_elem), arma::zeros(1, freqs.n_elem)); // keeping track of the evolution of lambda_hat with the iterations
    arma::cx_mat u_hat_plus(arma::zeros(T,K),arma::zeros(T,K));
    arma::cx_mat u_hat_plus_old = u_hat_plus;

#ifdef DEBUG_CVMD
    double uDiff{tol+eps};
#else
    double uDiff = tol + eps; //tol+eps must be different from just tol, that is why eps should not be too small
#endif

    arma::cx_mat sum_uk(arma::zeros(1, T),arma::zeros(1, T));
    int n_iter = 0;
#ifdef DEBUG_CVMD
    while  ((n_iter < MAX_VMD_ITERATIONS) && (uDiff > tol)) {
#else
    while (n_iter < MAX_VMD_ITERATIONS && uDiff > tol) {
#endif
        //% update first mode accumulator
        __omp_pfor_i(0, T, sum_uk(0, i) = u_hat_plus_old(i, K-1) + sum_uk(0, i) - u_hat_plus_old(i, 0));

        //% update spectrum of first mode through Wiener filter of residuals
        int k = 0;
        __omp_pfor_i(0, T,
              u_hat_plus(i, k) = (f_hat_plus(0, i) - sum_uk(0, i) - lambda_hat(0, i) / 2.) / (1. + Alpha(0, k) * std::pow((freqs(0, i) - omega_plus[k]), 2)) )

        if (DC == 0) {
            //when DC==1, the first frequency can not be changed. that is why this cycle only for DC==0.
#ifdef DEBUG_CVMD
            double sum_up = 0.;
            double sum_down = 0.;
            for (int i=0; i < T/2; ++i) {
                sum_up += freqs(0,T/2+i)*std::norm(u_hat_plus(T/2+i,k));
                sum_down += std::norm(u_hat_plus(T/2+i,k));
            }
            omega_plus(0,0) = sum_up / sum_down;
#else
            double sum_up = 0;
            double sum_down = 0;
            for (int i = 0; i < T / 2; ++i) {
                const auto norm_uhat_plus = std::norm(u_hat_plus(T / 2 + i, k));
                sum_up += freqs(0,T / 2 + i) * norm_uhat_plus;
                sum_down += norm_uhat_plus;
            }
            omega_plus[0] = sum_up / sum_down;
#endif
        }

        // update of any other mode (k from 1 to K-1), after the first
        for (k = 1; k < K; ++k) {
            // accumulator
            __omp_pfor_i(0, T, sum_uk(0, i) = u_hat_plus(i, k - 1) + sum_uk(0, i) - u_hat_plus_old(i, k))
            // mode spectrum
            __omp_pfor_i(0, T, u_hat_plus(i, k) = (f_hat_plus(0, i) - sum_uk(0, i) - lambda_hat(0, i) / 2.) / (1. + Alpha(0, k) * std::pow(freqs(0, i) - omega_plus[k], 2)))

            //re-compute center frequencies
#ifdef DEBUG_CVMD
            double sum_up = 0.;
            double sum_down = 0.;
            for (int i = 0; i < T / 2; ++i) {
                sum_up += freqs(0, T / 2 + i) * std::norm(u_hat_plus(T / 2 + i, k));
                sum_down += std::norm(u_hat_plus(T / 2 + i, k));
            }
            omega_plus(k, 0) = sum_up / sum_down;
#else
            double clk_sum_up = 0;
            double clk_sum_down = 0;
            for (int i = 0; i < T / 2; ++i) {
                const double u_hat_plus_norm = std::norm(u_hat_plus(T / 2 + i, k));
                clk_sum_up += freqs(0, T / 2 + i) * u_hat_plus_norm;
                clk_sum_down += u_hat_plus_norm;
            }
            omega_plus[k] = clk_sum_up / clk_sum_down;
#endif
        }
        // Dual ascent
        arma::cx_mat sum_u_hat_plus = arma::sum(u_hat_plus.t());

        __omp_pfor_i(0, T, lambda_hat(0, i) += tau * (sum_u_hat_plus(0, i) - f_hat_plus(0, i)) )
        // loop counter
        ++n_iter;

        //compute uDiff
#ifdef DEBUG_CVMD
        uDiff = 0.;
#else
        uDiff = 0;
#endif
        for (int i = 0; i < K; ++i) {
            double s = 0;
            double s_norm_old = 0;
            for (int j = 0; j < T; ++j) {
                s += std::norm(u_hat_plus(j,i) - u_hat_plus_old(j,i));
                s_norm_old += std::norm(u_hat_plus_old(j,i));
            }
            uDiff += (s / s_norm_old);
        }
        u_hat_plus_old = u_hat_plus;
    }
    //%------ Postprocessing and cleanup
    //% discard empty space if converged early - this step is not used here
    //N = min(N,n);
    //omega = omega_plus(1:N,:);
    // - this not needed, since we do not keep all omega, only the latest!

    // Signal reconstruction
    arma::cx_mat zmat(arma::zeros(T, K), arma::zeros(T, K));
    u_hat = zmat;
    __omp_pfor_i(0, T / 2,
          for (int k = 0; k < K; ++k) u_hat(T / 2 + i, k) = u_hat_plus(T / 2 + i, k) )
    // here it is not clear why, but T/2 is set both above and below. Instead, the 0 is set to conj of T-1
    __omp_pfor_i(0, T / 2, for (int k=0; k < K; ++k) u_hat(T / 2 - i, k) = std::conj(u_hat_plus(T / 2 + i, k)) )
    for (int k = 0; k < K; ++k) u_hat(0, k) = std::conj(u_hat(T - 1, k));

    arma::mat u_big = arma::zeros(K, t.n_elem);
    __omp_pfor(k, 0, K, u_big.rows(k, k) = arma::trans( arma::real(common::matlab_ifft(common::ifftshift(u_hat.cols(k, k))))) )
    u = arma::zeros(K,T / 2);
    __omp_pfor_i(0, T / 2, for (int k = 0; k < K; ++k) u(k, i) = u_big(k, T / 4 + i) )
    //recompute spectrum
    //clear u_hat;
    u_hat = arma::cx_mat(arma::zeros(K,T / 2),arma::zeros(K,T / 2));
    __omp_pfor(k, 0, K,
         u_hat.rows(k,k) = common::fftshift(common::matlab_fft(u.rows(k,k)));
         omega_plus[k] /= OMEGA_DIVISOR )

    LOG4_DEBUG("Omega is " << common::deep_to_string(omega_plus));
}
#endif


static fcvmd_frequency_outputs
calculate_vmd_frequencies(const std::vector<double> &input, const size_t levels_half)
{
    LOG4_DEBUG("Calculating VMD frequencies for " << levels_half << " half levels.");

    arma::rowvec x_signal;
    if (input.size() % 2) LOG4_WARN("Odd signal length " << input.size() << ", trimming first value.");
    x_signal = input.size() % 2 ? arma::rowvec(&*std::next(input.begin()), input.size() - 1) : input;

    /* constants */
    const double alpha = ALPHA_BINS; // bands
    const double tau = TAU_FIDELITY; // fidelity - 0 means no strict enforcement of decomposition. In reality, we should use something like 0.1 or higher.
    const size_t DC = 0; // has a DC component or not.
    const size_t K = levels_half; // number of modes/frequencies. because of DC=1 we use 1 more than the natural number of frequencies (3 in the signal above).
    const size_t init = 1; // kind of initialization, let's use 1 for now.
    const double tol = CVMD_TOL; // some tolerance

    /* outputs */
    arma::mat u;
    arma::cx_mat u_hat;
#ifdef EMOS_OMEGAS
    std::vector<double> omega = {
            1.000000000000000e-12, 4.830917874396136e-08, 9.661835748792271e-08, 2.415458937198068e-07, 6.038647342995169e-07,
            1.207729468599034e-06, 2.415458937198068e-06, 1.207729468599034e-05, 3.472222222222222e-05, 6.944444444444444e-05,
            1.388888888888889e-04, 2.777777777777778e-04, 5.555555555555556e-04, 1.111111111111111e-03, 3.333333333333333e-03,
            1.666666666666667e-02};
#else
    std::vector<double> omega;
#endif
    do_vmd_freqs(x_signal, alpha, tau, K, DC, init, tol, u, u_hat, omega);

    std::vector<double> phase_cos, phase_sin;
    compute_cos_sin(omega, phase_cos, phase_sin, DEFAULT_PHASE_STEP);
    fcvmd_frequency_outputs res;
    res.phase_cos = phase_cos;
    res.phase_sin = phase_sin;
    res.H.clear();
    return res;
}


bool
fast_cvmd::initialized(const std::string &decon_queue_table_name)
{
    return not vmd_frequencies.empty() and vmd_frequencies.find({decon_queue_table_name, levels}) != vmd_frequencies.end();
}


void
fast_cvmd::initialize(const std::vector<double> &input, const std::string &decon_queue_table_name)
{
    LOG4_DEBUG("Initializing omega on " << input.size() << " rows, for " << levels << " levels, " << decon_queue_table_name << " table.");
    const auto freq_key = freq_key_t(decon_queue_table_name, levels);
    if (vmd_frequencies.empty() or vmd_frequencies.find(freq_key) == vmd_frequencies.end())
        vmd_frequencies[freq_key] = calculate_vmd_frequencies(input, levels / 2);
}


void
fast_cvmd::transform(
        const std::vector<double> &input,
        std::vector<std::vector<double>> &decon,
        const size_t padding = 0)
{
    THROW_EX_FS(std::logic_error, "Not implemented!");
}


// Online VMD transform
void
fast_cvmd::transform(
        const std::vector<double> &input,
        std::vector<std::vector<double>> &decon,
        const std::string &decon_queue_table_name,
        const std::vector<double> &prev_decon)
{
    const auto found_freqs = vmd_frequencies.find({decon_queue_table_name, levels});
    if (vmd_frequencies.empty() or found_freqs == vmd_frequencies.end()) LOG4_THROW("VMD frequencies not found!");

    step_decompose_matrix(found_freqs->second.phase_cos, found_freqs->second.phase_sin, input, prev_decon, decon, found_freqs->second.H);
}


// Batch transform
void
fast_cvmd::transform(
        const std::vector<double> &input,
        std::vector<std::vector<double>> &decon,
        const std::string &table_name)
{
    const auto found_freqs = vmd_frequencies.find({table_name, levels});
    if (vmd_frequencies.empty() or found_freqs == vmd_frequencies.end())
        LOG4_THROW("VMD frequencies for " << table_name << " " << levels << " not found!");

    auto prev = std::vector<double>(levels, 0.);
    prev[0] = input.front();
    step_decompose_matrix(found_freqs->second.phase_cos, found_freqs->second.phase_sin, input, prev, decon, found_freqs->second.H);
}


void
fast_cvmd::inverse_transform(
        const std::vector<double> &decon,
        std::vector<double> &recon,
        const size_t padding) const
{
    const size_t input_size = decon.size() / levels;
    if (recon.size() != input_size) recon.resize(input_size, 0);
    __omp_pfor(t, 0, input_size,
        recon[t] = 0;
        for (size_t l = 0; l < levels; l += 2) recon[t] += decon[t + l * input_size]
    )
}


size_t
fast_cvmd::get_residuals_length(const std::string &decon_queue_table_name)
{
    LOG4_DEBUG("Getting residuals length for " << decon_queue_table_name << " " << levels << " levels.");
    const auto vmd_freq_iter = vmd_frequencies.find({decon_queue_table_name, levels});
    return vmd_freq_iter != vmd_frequencies.end() && !vmd_freq_iter->second.H.empty() ? 0 : std::pow<size_t>(levels / 2, 4);
}


}
