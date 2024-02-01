//
// Created by zarko on 7/5/22.
//

#include "common/defines.h"
#include "fast_cvmd.hpp"
#include "util/math_utils.hpp"
#include "common/compatibility.hpp"
#include "common/Logging.hpp"
#include "common/parallelism.hpp"
#include "model/DeconQueue.hpp"
#include "model/DataRow.hpp"
#include <armadillo>
#include <vector>


namespace svr {

inline namespace {

void compute_cos_sin(
        const arma::colvec &omega,
        arma::vec &phase_cos,
        arma::vec &phase_sin,
        const double step = DEFAULT_PHASE_STEP)
{
    if (phase_cos.size() != omega.size()) phase_cos.set_size(omega.size());
    if (phase_sin.size() != omega.size()) phase_sin.set_size(omega.size());

    __omp_pfor_i(0, omega.size(), phase_cos[i] = cos(2 * M_PI * step * omega[i]); phase_sin[i] = sin(2 * M_PI * step * omega[i]))
    LOG4_DEBUG("Omega " << omega << ", phase cos " << phase_cos << ", phase sin " << phase_sin << ", step " << step <<
                        ", alpha bins " << ALPHA_BINS << ", tau fidelity " << TAU_FIDELITY << ", max VMD iterations " << MAX_VMD_ITERATIONS << ", epsilon " << EPS <<
                        ", tolerance " << CVMD_TOL << ", omega divisor " << OMEGA_DIVISOR);
}

}


fast_cvmd::fast_cvmd(const size_t _levels) : spectral_transform(std::string("cvmd"), _levels), levels(_levels)
{
    if (levels < 2 or levels % 2) LOG4_THROW("Invalid number of levels " << levels);

    K = levels / 2;
    f = arma::vec(levels, arma::fill::zeros);
    row_values = arma::vec(levels * 2, arma::fill::zeros);
    soln = arma::mat(row_values.memptr() + levels, levels, 1, false, true);
    even_ixs = arma::regspace<arma::uvec>(0, 2, levels - 2);
    odd_ixs = arma::regspace<arma::uvec>(1, 2, levels - 1);
    timenow = boost::posix_time::second_clock::local_time();
    K_ixs = arma::regspace<arma::uvec>(0, K - 1);
}


bool fast_cvmd::initialized(const std::string &decon_queue_table_name)
{
    return not vmd_frequencies.empty() and vmd_frequencies.find({decon_queue_table_name, levels}) != vmd_frequencies.end();
}


void
fast_cvmd::initialize(const datamodel::datarow_crange &input, const size_t input_column_index, const std::string &decon_queue_table_name)
{
    if (input.distance() < 1) LOG4_THROW("Illegal input size " << input.distance());
    LOG4_DEBUG("Initializing omega on " << input.distance() << " rows, for " << levels << " levels, " << decon_queue_table_name << " table.");
    const auto freq_key = freq_key_t(decon_queue_table_name, levels);
    if (vmd_frequencies.find(freq_key) != vmd_frequencies.end()) return;

    /* constants */
    constexpr double alpha = ALPHA_BINS; // bands
    constexpr double tau = TAU_FIDELITY; // fidelity - 0 means no strict enforcement of decomposition. In reality, we should use something like 0.1 or higher.
    constexpr size_t DC = HAS_DC; // has a DC component or not.
    constexpr size_t init = 1; // Type of initialization, let's use 1 for now.
    constexpr double tol = CVMD_TOL; // some tolerance

    /* outputs */
    arma::mat u;
    arma::cx_mat u_hat;
#ifdef EMOS_OMEGAS
    {
        const arma::colvec omega = {
                1.000000000000000e-12, 4.830917874396136e-08, 9.661835748792271e-08, 2.415458937198068e-07, 6.038647342995169e-07,
                1.207729468599034e-06, 2.415458937198068e-06, 1.207729468599034e-05, 3.472222222222222e-05, 6.944444444444444e-05,
                1.388888888888889e-04, 2.777777777777778e-04, 5.555555555555556e-04, 1.111111111111111e-03, 3.333333333333333e-03,
                1.666666666666667e-02};
        arma::vec phase_cos, phase_sin;
        compute_cos_sin(omega, phase_cos, phase_sin, DEFAULT_PHASE_STEP);
        vmd_frequencies.emplace(freq_key, fcvmd_frequency_outputs{phase_cos, phase_sin, {}});
        return;
    }
#endif

    //Initialize u, w, lambda, n
    const size_t save_T = input.distance();
    size_t T = save_T;
    if (T & 1) LOG4_THROW("Input data size " << T << " must be even!");

    //create the mirrored signal
    arma::mat f_init = arma::zeros(1, 2 * T);

#pragma omp parallel for
    for (size_t i = 0; i < T / 2; ++i)
        f_init[i] = (input.begin() + T / 2 - 1 - i)->get()->get_value(input_column_index);
#pragma omp parallel for
    for (size_t i = 0; i < T; ++i)
        f_init[T / 2 + i] = (input.begin() + i)->get()->get_value(input_column_index);
#pragma omp parallel for
    for (size_t i = 0; i < T / 2; ++i)
        f_init[3 * T / 2 + i] = (input.begin() + T - 1 - i)->get()->get_value(input_column_index);

    T = 2 * T;

    const arma::rowvec Alpha = alpha * arma::ones(1, K);//provision to use different alpha for the different frequences, at a later stage

    const double fs = 1. / double(save_T); //step

    constexpr size_t N_max = MAX_VMD_ITERATIONS; // max number of iterations

    u = arma::zeros(T, K);
    constexpr double eps = 2.220446049250313e-16;
    //this may or may not work instead. double eps=std::numeric_limits<T>::epsilon();

    // Time Domain 0 to T (of mirrored signal)
    arma::rowvec t(T);
    __omp_pfor_i(0, T, t[i] = (double(i) + 1.) / double(T))
    // Spectral Domain discretization
    const arma::rowvec freqs = t - .5 - 1. / double(T);
    arma::colvec omega = arma::zeros(K, 1);
    arma::cx_rowvec f_hat_plus = common::fftshift(common::matlab_fft(f_init));

    __omp_pfor_i(0, T / 2, f_hat_plus[i] = 0)

    switch (init) {
        case 1:
            __omp_pfor_i(0, K, omega[i] = .5 / K * i);
            break;

        case 2:{
            arma::colvec real_ran(K);
            __omp_pfor_i(0, K, real_ran[i] = exp(log(fs) + (log(0.5) - log(fs)) * common::randouble()));
            real_ran = arma::sort(real_ran);
            __omp_pfor_i(0, K, omega[i] = real_ran[i]);
            break;
        }
        case 0:
            __omp_pfor_i(0, K, omega[i] = 0); // Was commented out originally!
            break;

        default:
            LOG4_THROW("Init should be 0,1 or 2!");
    }
    if (DC == 1) //if DC component, then first frequency is 0!
        omega.front() = 0;

    /* completed initialization of frequencies */
    arma::cx_mat lambda_hat( arma::zeros(1, freqs.n_elem), arma::zeros(1, freqs.n_elem)); // keeping track of the evolution of lambda_hat with the iterations
    arma::cx_mat u_hat_plus(arma::zeros(T,K), arma::zeros(T,K));
    arma::cx_mat u_hat_plus_old = u_hat_plus;

    double uDiff = tol + eps; //tol+eps must be different from just tol, that is why eps should not be too small

    arma::cx_mat sum_uk(arma::zeros(1,T), arma::zeros(1, T));
    size_t n_iter = 0;
    while (n_iter < N_max && uDiff > tol) {
        //% update first mode accumulator
        __omp_pfor_i(0, T, sum_uk(0, i) = u_hat_plus_old(i, K - 1) + sum_uk(0, i) - u_hat_plus_old(i, 0));

        //% update spectrum of first mode through Wiener filter of residuals
        __omp_pfor_i(0, T, u_hat_plus(i, 0) = (f_hat_plus[i] - sum_uk(0, i) - lambda_hat(0, i) / 2.) / (1. + Alpha(0, 0) * std::pow((freqs(0, i) - omega[0]), 2)) )

        if (DC == 0) {
            //when DC==1, the first frequency can not be changed. that is why this cycle only for DC==0.
            double sumup = 0;
            double sumdown = 0;
#pragma omp parallel for reduction(+:sumup, sumdown)
            for (size_t i = 0; i < T / 2; ++i) {
                const auto norm_uhat_plus = std::norm(u_hat_plus(T / 2 + i, 0));
                sumup += freqs[T / 2 + i] * norm_uhat_plus;
                sumdown += norm_uhat_plus;
            }
            omega.front() = sumup / sumdown;
        }

        // update of any other mode (k from 1 to K-1), after the first
#pragma omp parallel for ordered schedule(static, 1)
        for (size_t k = 1; k < K; ++k) {
#pragma omp ordered
            {
                // accumulator
                __omp_pfor_i(0, T, sum_uk(0, i) = u_hat_plus(i, k - 1) + sum_uk(0, i) - u_hat_plus_old(i, k))
                // mode spectrum
                __omp_pfor_i(0, T,
                             u_hat_plus(i, k) = (f_hat_plus[i] - sum_uk(0, i) - lambda_hat(0, i) / 2.) / (1. + Alpha(0, k) * std::pow(freqs(0, i) - omega[k], 2)))
            }
            //re-compute center frequencies
            double sumup = 0;
            double sumdown = 0;
#pragma omp parallel for reduction(+:sumup, sumdown)
            for (size_t i = 0; i < T / 2; ++i) {
                const double u_hat_plus_norm = std::norm(u_hat_plus(T / 2 + i, k));
                sumup += freqs(0, T / 2 + i) * u_hat_plus_norm;
                sumdown += u_hat_plus_norm;
            }
            omega[k] = sumup / sumdown;
        }
        // Dual ascent
        const arma::cx_mat sum_u_hat_plus = arma::sum(u_hat_plus.t());

        __omp_pfor_i(0, T, lambda_hat(0, i) += tau * (sum_u_hat_plus(0, i) - f_hat_plus[i]) )
        // loop counter
        ++n_iter;

        //compute uDiff
        uDiff = 0;
#pragma omp parallel for reduction(+:uDiff)
        for (size_t i = 0; i < K; ++i) {
            double s = 0;
            double s_norm_old = 0;
            for (size_t j = 0; j < T; ++j) {
                s += std::norm(u_hat_plus(j, i) - u_hat_plus_old(j, i));
                s_norm_old += std::norm(u_hat_plus_old(j, i));
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
    arma::cx_mat zmat (arma::zeros(T, K), arma::zeros(T, K));
    u_hat = zmat;
    __omp_pfor_i(0, T / 2,
                 for (size_t k = 0; k < K; ++k)
                     u_hat(T / 2 + i, k) = u_hat_plus(T / 2 + i, k) )
    // here it is not clear why, but T/2 is set both above and below. Instead, the 0 is set to conj of T-1
    __omp_pfor_i(0, T / 2, for (size_t k = 0; k < K; ++k) u_hat(T / 2 - i, k) = std::conj(u_hat_plus(T / 2 + i, k)) )
#pragma omp parallel for
    for (size_t k = 0; k < K; ++k) u_hat(0, k) = std::conj(u_hat(T - 1, k));

    arma::mat u_big = arma::zeros(K, t.n_elem);
    __omp_pfor(k, 0, K, u_big.rows(k, k) = arma::trans(arma::real(common::matlab_ifft(common::ifftshift(u_hat.cols(k, k))))); )
    u = arma::zeros(K,T/2);
    __omp_pfor_i(0, T / 2, for (size_t k = 0; k < K; ++k) u(k, i) = u_big(k, T / 4 + i); )
    //recompute spectrum
    //clear u_hat;
    u_hat = arma::cx_mat(arma::zeros(K, T / 2), arma::zeros(K, T / 2));
    __omp_pfor(k, 0, K, u_hat.rows(k, k) = common::fftshift(common::matlab_fft(u.rows(k, k))) )
    omega /= OMEGA_DIVISOR;
    LOG4_DEBUG("Calculated Omega is " << omega);

    arma::vec phase_cos, phase_sin;
    compute_cos_sin(omega, phase_cos, phase_sin, DEFAULT_PHASE_STEP);
    vmd_frequencies.emplace(freq_key, fcvmd_frequency_outputs{phase_cos, phase_sin});
}


void
fast_cvmd::transform(
        const data_row_container &input,
        datamodel::DeconQueue_ptr &p_decon_queue,
        const size_t in_colix,
        const size_t test_offset)
{
    if (input.size() == 0) {
        LOG4_ERROR("Input empty!");
        return;
    }

    // VMD
    if (!initialized(p_decon_queue->get_table_name())) {
        auto start_cvmd_init = input.begin() + std::max<ssize_t>(0, input.size() - CVMD_INIT_LEN - test_offset);
        auto end_cvmd_init = start_cvmd_init + std::min<ssize_t>(std::distance(start_cvmd_init, input.end() - test_offset), CVMD_INIT_LEN);
        if (std::distance(start_cvmd_init, end_cvmd_init) % 2) ++start_cvmd_init; // Ensure even distance
        PROFILE_EXEC_TIME(initialize(
                datamodel::datarow_crange{start_cvmd_init, end_cvmd_init, input}, in_colix, p_decon_queue->get_table_name()), "CVMD init");
    }

    const auto found_freqs = vmd_frequencies.find({p_decon_queue->get_table_name(), levels});
    if (vmd_frequencies.empty() or found_freqs == vmd_frequencies.end())
        LOG4_THROW("VMD frequencies for " << p_decon_queue->get_table_name() << " " << levels << " not found!");

    const auto &phase_cos = found_freqs->second.phase_cos;
    const auto &phase_sin = found_freqs->second.phase_sin;
    auto &decon = p_decon_queue->get_data();

    if (K != phase_sin.size() || K != phase_cos.size()) LOG4_THROW("Invalid phase cos size " << phase_cos.size() << " or phase sin size " << phase_sin.size());

    data_row_container::const_iterator iterin;
    if (decon.empty()) {
        iterin = input.begin();
    } else {
        iterin = lower_bound_back(input, decon.back()->get_value_time());
        if (iterin->get()->get_value_time() != decon.back()->get_value_time()) {
            LOG4_ERROR("Input time " << iterin->get()->get_value_time() << " does not match last decon time " << decon.back()->get_value_time());
            --iterin;
        }
    }

    size_t n = std::distance(iterin, input.end());
    if (n % 1) {
        if (iterin != input.begin()) {
            --iterin;
            ++n;
        } else if (iterin != input.end()) {
            ++iterin;
            ++n;
        } else
            LOG4_THROW("Odd input count " << n << " is not allowed.");
    }

    arma::mat A = arma::zeros(n, n * 2 * K);
#pragma omp parallel for collapse(2) schedule(static, 1 + n * K / std::thread::hardware_concurrency())
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < K; ++j)
            A(i, 2 * j + i * 2 * K) = 1.;

    arma::mat H = arma::zeros(n * 2 * K, n * 2 * K);
    for (size_t i = 0; i < n; ++i) {
        if (i == 0) {
            for (size_t j = 0; j < K; ++j) {
                H(2 * j, 2 * j) += 1.;
                H(2 * j + 1, 2 * j + 1) += 1.;
            }
        } else {
#pragma omp parallel for
            for (size_t j = 0; j < K; j++) {
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

#pragma omp parallel
#pragma omp single nowait
    {
        while (iterin != input.end()) {
            f.rows(even_ixs) = -phase_cos.rows(K_ixs) % soln.rows(even_ixs) + phase_sin.rows(K_ixs) % soln.rows(odd_ixs);
            f.rows(odd_ixs) = -phase_sin.rows(K_ixs) % soln.rows(even_ixs) - phase_cos.rows(K_ixs) % soln.rows(odd_ixs);
#pragma omp taskwait
            const datamodel::DataRow *p_row = iterin->get();
            soln = arma::solve(H, -A.t() * (-arma::solve(A * arma::solve(H, A.t()), A * arma::solve(H, f) + p_row->get_value(in_colix))) - f);
#pragma omp task firstprivate(p_row)
            decon.emplace_back(std::make_shared<datamodel::DataRow>(
                    p_row->get_value_time(), timenow, p_row->get_tick_volume(), row_values.memptr(), row_values.n_elem));
            ++iterin;
        }
    }
}


void
fast_cvmd::inverse_transform(
        const std::vector<double> &decon,
        std::vector<double> &recon,
        const size_t padding) const
{
    LOG4_WARN("Use reconstruct method in decon queue service for hybrid decons.");
    const size_t input_size = decon.size() / levels;
    if (recon.size() != input_size) recon.resize(input_size, 0);
#pragma omp parallel for
    for (size_t t = 0; t < input_size; ++t) {
        recon[t] = 0;
        for (size_t l = 0; l < levels; l += 2)
            recon[t] += decon[t + l * input_size];
    }
}


size_t
fast_cvmd::get_residuals_length(const std::string &decon_queue_table_name)
{
    LOG4_DEBUG("Getting residuals length for " << decon_queue_table_name << " " << levels << " levels.");
    const auto vmd_freq_iter = vmd_frequencies.find({decon_queue_table_name, levels});
    return vmd_freq_iter == vmd_frequencies.end() || vmd_freq_iter->second.phase_cos.empty() ? (size_t) std::pow(levels / 2, 4) : 0;
}


}
