//
// Created by zarko on 7/5/22.
//

#include <armadillo>
#include <memory>
#include <vector>
#include "common/defines.h"
#include "fast_cvmd.hpp"
#include "util/math_utils.hpp"
#include "common/compatibility.hpp"
#include "common/Logging.hpp"
#include "common/parallelism.hpp"
#include "model/DeconQueue.hpp"
#include "model/DataRow.hpp"


namespace svr {
namespace vmd {

inline namespace {

auto
compute_cos_sin(
        const arma::vec &omega,
        const double step = DEFAULT_PHASE_STEP)
{
    const auto step_omega = 2. * M_PI * step * omega;
    const arma::vec phase_cos = arma::cos(step_omega);
    const arma::vec phase_sin = arma::sin(step_omega);
    LOG4_DEBUG("Omega " << omega << ", phase cos " << phase_cos << ", phase sin " << phase_sin << ", step " << step << ", alpha bins " << C_default_alpha_bins << ", tau fidelity "
                        << TAU_FIDELITY << ", max VMD iterations " << MAX_VMD_ITERATIONS << ", tolerance " << CVMD_TOL);
    return std::tuple{phase_cos, phase_sin};
}

}

const business::t_iqscaler fast_cvmd::C_no_scaler = [](const double v) -> double { return v; };

fast_cvmd::fast_cvmd(const size_t _levels) :
    spectral_transform(std::string("cvmd"), _levels),
    levels(_levels),
    K(_levels / 2),
    f(_levels),
    A(_levels),
    H(arma::eye(_levels, _levels)),
    even_ixs(arma::regspace<arma::uvec>(0, 2, _levels - 2)),
    odd_ixs(arma::regspace<arma::uvec>(1, 2, _levels - 1)),
    K_ixs(arma::regspace<arma::uvec>(0, K - 1)),
    row_values(_levels * 2, arma::fill::zeros),
    soln(row_values.memptr() + _levels, _levels, false, true),
    timenow(boost::posix_time::second_clock::local_time())
{
    if (levels < 2 or levels % 2) LOG4_THROW("Invalid number of levels " << levels);
    A.cols(odd_ixs).zeros();
    A.cols(even_ixs).ones();
}


bool fast_cvmd::initialized(const std::string &decon_queue_table_name)
{
    return not vmd_frequencies.empty() and vmd_frequencies.find({decon_queue_table_name, levels}) != vmd_frequencies.end();
}


inline void
calc_omega(arma::vec &omega, const size_t k, const arma::vec &freqs, const arma::cx_mat &u_hat_plus, const size_t T)
{
    const arma::vec norm_u_hat_plus = common::norm<double>(u_hat_plus.submat(T / 2, k, T - 1, k));
    omega[k] = arma::accu(freqs.rows(T / 2, T - 1) % norm_u_hat_plus) / arma::accu(norm_u_hat_plus);
}

void
fast_cvmd::initialize(const datamodel::datarow_crange &input, const size_t input_column_index, const std::string &decon_queue_table_name, const business::t_iqscaler &scaler)
{
    if (input.distance() < 1) LOG4_THROW("Illegal input size " << input.distance());
    LOG4_DEBUG("Initializing omega on " << input.distance() << " rows, for " << levels << " levels, " << decon_queue_table_name << " table.");
    const auto freq_key = freq_key_t(decon_queue_table_name, levels);
    if (vmd_frequencies.find(freq_key) != vmd_frequencies.end()) return;

#ifdef EMOS_OMEGAS
    {
        const arma::colvec omega = {
                1.000000000000000e-12, 4.830917874396136e-08, 9.661835748792271e-08, 2.415458937198068e-07, 6.038647342995169e-07,
                1.207729468599034e-06, 2.415458937198068e-06, 1.207729468599034e-05, 3.472222222222222e-05, 6.944444444444444e-05,
                1.388888888888889e-04, 2.777777777777778e-04, 5.555555555555556e-04, 1.111111111111111e-03, 3.333333333333333e-03,
                1.666666666666667e-02};
        const auto [phase_cos, phase_sin] = compute_cos_sin(omega, DEFAULT_PHASE_STEP);
        vmd_frequencies.emplace(freq_key, fcvmd_frequency_outputs{phase_cos, phase_sin});
        return;
    }
#endif

    //Initialize u, w, lambda, n
    const size_t save_T = input.distance();
    size_t T = save_T;
    if (T & 1) LOG4_THROW("Input data size " << T << " must be even!");

    //create the mirrored signal
    arma::vec f_init(2 * T);
#pragma omp parallel for num_threads(adj_threads(T)) schedule(static, 1 + T / std::thread::hardware_concurrency())
    for (size_t i = 0; i < T; ++i) {
        if (i < T / 2) {
            f_init[i] = scaler(input[T / 2 - 1 - i]->at(input_column_index));
            f_init[3 * T / 2 + i] = scaler(input[T - 1 - i]->at(input_column_index));
        }
        f_init[T / 2 + i] = scaler(input[i]->at(input_column_index));
    }
    T = 2 * T;
    const arma::vec Alpha(K, arma::fill::value(save_T * C_default_alpha_bins * 1e-5)); //provision to use different alpha for the different frequences, at a later stage
    const double fs = 1. / double(save_T); //step

    /* outputs */
    arma::mat u(T, K);
    // Spectral Domain discretization
    const arma::vec freqs = arma::regspace(1, T) / double(T) - .5 - 1. / double(T);
    arma::vec omega(K);
    arma::cx_vec f_hat_plus = common::fftshift(common::matlab_fft(f_init));
    f_hat_plus.rows(0, T / 2 - 1).zeros();

    switch (C_freq_init_type) {
        case 1:
            omega = arma::regspace(0, K - 1) * .5 / K;
            break;

        case 2: {
            arma::colvec real_ran(K);
            const auto log_fs = std::log(fs);
            const auto log_05_fs = std::log(0.5) - log_fs;
#pragma omp parallel for num_threads(adj_threads(K / 2))
            for (size_t i = 0; i < K; ++i) real_ran[i] = std::exp(log_fs + log_05_fs * common::randouble());
            real_ran = arma::sort(real_ran);
#pragma omp parallel for num_threads(adj_threads(K / 2))
            for (size_t i = 0; i < K; ++i) omega[i] = real_ran[i];
            break;
        }
        case 0:
            omega.zeros(); // Was commented out originally!
            break;

        default:
            LOG4_THROW("Initialization type should be 0, 1 or 2!");
    }
    if (HAS_DC == 1) omega[0] = 0; //if DC component, then first frequency is 0!
    /* completed initialization of frequencies */

    arma::cx_vec lambda_hat(freqs.n_elem); // keeping track of the evolution of lambda_hat with the iterations
    arma::cx_mat u_hat_plus(T, K);
    arma::cx_mat u_hat_plus_old(arma::size(u_hat_plus));
    double uDiff = CVMD_TOL + std::numeric_limits<double>::epsilon(); //tol + eps must be different from just tol, that is why eps should not be too small
    arma::cx_vec sum_uk(T);
    size_t n_iter = 0;
    while (n_iter < MAX_VMD_ITERATIONS && uDiff > CVMD_TOL) {
        //% update first mode accumulator
        sum_uk += u_hat_plus_old.col(K - 1) - u_hat_plus_old.col(0);

        //% update spectrum of first mode through Wiener filter of residuals
        u_hat_plus.col(0) = (f_hat_plus - sum_uk - .5 * lambda_hat) / (1. + Alpha[0] * arma::pow(freqs - omega[0], 2));

        //when DC is true, the first frequency can not be changed. that is why this cycle only for DC false.
        if (!HAS_DC) calc_omega(omega, 0, freqs, u_hat_plus, T);

        // update of any other mode (k from 1 to K-1), after the first
#pragma omp parallel for schedule(static, 1) num_threads(adj_threads(K - 1)) ordered
        for (size_t k = 1; k < K; ++k) {
            // accumulator
#pragma omp ordered
            sum_uk += u_hat_plus.col(k - 1) - u_hat_plus_old.col(k);
            // mode spectrum
            u_hat_plus.col(k) = (f_hat_plus - sum_uk - .5 * lambda_hat) / (1. + Alpha[k] * arma::pow(freqs - omega[k], 2));
            // re-compute center frequencies
            calc_omega(omega, k, freqs, u_hat_plus, T);
        }
        // Dual ascent
        lambda_hat += TAU_FIDELITY * (arma::accu(u_hat_plus.row(0)) - f_hat_plus);
        // loop counter
        ++n_iter;

        // compute uDiff
        uDiff = arma::accu(arma::sum(common::norm<double>(u_hat_plus - u_hat_plus_old)) / arma::sum(common::norm(u_hat_plus_old)));
        u_hat_plus_old = u_hat_plus;
    }

#ifdef FAST_CVMD_POSTPROCESSING_AND_CLEANUP
    //%------ Postprocessing and cleanup
    //% discard empty space if converged early - this step is not used here
    //N = min(N,n);
    //omega = omega_plus(1:N,:);
    // - this not needed, since we do not keep all omega, only the latest!

    // Signal reconstruction
    arma::cx_mat u_hat(T, K);
    u_hat.rows(T / 2, T - 1) = u_hat_plus.rows(T / 2, T - 1);

    // here it is not clear why, but T/2 is set both above and below. Instead, the 0 is set to conj of T-1
#pragma omp parallel for collapse(2) num_threads(adj_threads(K * T / 2)) schedule(static, 1 + K * T / (2 * std::thread::hardware_concurrency()))
    for (size_t i = 0; i < T / 2; ++i)
        for (size_t k = 0; k < K; ++k)
            u_hat(T / 2 - i, k) = std::conj(u_hat_plus(T / 2 + i, k));

    u_hat.row(0) = arma::conj(u_hat.row(T - 1));

    arma::mat u_big(K, freqs.n_elem);
#pragma omp parallel for num_threads(adj_threads(K)) schedule(static, 1)
    for (size_t k = 0; k < K; ++k)
        u_big.row(k) = arma::trans(arma::real(common::matlab_ifft(common::ifftshift(u_hat.col(k)))));
    u = arma::zeros(K, T / 2);
#pragma omp parallel for num_threads(adj_threads(K * T / 2)) collapse(2) schedule(static, 1 + K * T / (2 * std::thread::hardware_concurrency()))
    for (size_t i = 0; i < T / 2; ++i)
        for (size_t k = 0; k < K; ++k)
            u(k, i) = u_big(k, T / 4 + i);
    // recompute spectrum
    // clear u_hat;
    u_hat = arma::cx_mat(K, T / 2);
#pragma omp parallel for num_threads(adj_threads(K)) schedule(static, 1)
    for (size_t k = 0; k < K; ++k) u_hat.row(k) = common::fftshift(common::matlab_fft(u.row(k)));
#endif
    const auto [phase_cos, phase_sin] = compute_cos_sin(omega, DEFAULT_PHASE_STEP);
    vmd_frequencies.emplace(freq_key, fcvmd_frequency_outputs{phase_cos, phase_sin});
}


void
fast_cvmd::transform(
        const data_row_container &input,
        datamodel::DeconQueue &decon,
        const size_t in_colix,
        const size_t test_offset,
        const business::t_iqscaler &scaler)
{
    if (input.empty()) {
        LOG4_ERROR("Input empty!");
        return;
    }

    // VMD
    if (!initialized(decon.get_table_name())) {
        auto start_cvmd_init = input.begin() + std::max<ssize_t>(0, input.size() - CVMD_INIT_LEN - test_offset);
        auto end_cvmd_init = input.end() - test_offset;
        if (std::distance(start_cvmd_init, end_cvmd_init) % 2) ++start_cvmd_init; // Ensure even distance
        PROFILE_EXEC_TIME(initialize(
                datamodel::datarow_crange{start_cvmd_init, end_cvmd_init, input}, in_colix, decon.get_table_name(), scaler), "CVMD init");
    }

    const auto found_freqs = vmd_frequencies.find({decon.get_table_name(), levels});
    if (vmd_frequencies.empty() or found_freqs == vmd_frequencies.end())
        LOG4_THROW("VMD frequencies for " << decon.get_table_name() << " " << levels << " not found!");

    const auto &phase_cos = found_freqs->second.phase_cos;
    const auto &phase_sin = found_freqs->second.phase_sin;

    if (K != phase_sin.size() || K != phase_cos.size())
        LOG4_THROW("Invalid phase cos size " << arma::size(phase_cos) << " or phase sin size " << arma::size(phase_sin) << ", should be " << K);

    data_row_container::const_iterator iterin;
    if (decon.empty()) {
        iterin = input.begin();
        soln[0] = scaler((**iterin)[in_colix]);
    } else {
        iterin = lower_bound_back(input, decon.back()->get_value_time());
        while ((**iterin).get_value_time() <= decon.back()->get_value_time() && iterin != input.end()) ++iterin;
        if (iterin == input.end()) {
            LOG4_WARN("No input data newer than " << decon.back()->get_value_time() << " to deconstruct.");
            return;
        }
        memcpy(soln.memptr(), decon.back()->get_values().data() + levels, levels * sizeof(double));
    }

    const auto in_ct = std::distance(iterin , input.end());
    const auto prev_decon_ct = decon.size();
    decon.get_data().resize(prev_decon_ct + in_ct);
#pragma omp parallel for num_threads(adj_threads(in_ct)) schedule(static, 1 + in_ct / std::thread::hardware_concurrency())
    for (ssize_t t = 0; t < in_ct; ++t) {
        const auto p_in_row = *(iterin + t - prev_decon_ct);
        decon[prev_decon_ct + t] = ptr<datamodel::DataRow>(p_in_row->get_value_time(), timenow, p_in_row->get_tick_volume(), levels * 2);
    }

    f.rows(even_ixs) = -phase_cos.rows(K_ixs) % soln.rows(even_ixs) + phase_sin.rows(K_ixs) % soln.rows(odd_ixs);
    f.rows(odd_ixs) = -phase_sin.rows(K_ixs) % soln.rows(even_ixs) - phase_cos.rows(K_ixs) % soln.rows(odd_ixs);
#pragma omp parallel num_threads(adj_threads(3))
#pragma omp single
    for (size_t t = prev_decon_ct; t < decon.size() && iterin != input.end(); ++iterin, ++t)
    {
#define ASOLVE(M, N) arma::solve(M, (N), C_arma_solver_opts)
        soln = ASOLVE(H, -A.t() * (-ASOLVE(A * ASOLVE(H, A.t()), A * ASOLVE(H, f) + scaler((**iterin)[in_colix]))) - f);
#pragma omp task
        memcpy(decon[t]->get_values().data() + levels, soln.mem, levels * sizeof(double));
#pragma omp task
        f.rows(even_ixs) = -phase_cos.rows(K_ixs) % soln.rows(even_ixs) + phase_sin.rows(K_ixs) % soln.rows(odd_ixs);
        f.rows(odd_ixs) = -phase_sin.rows(K_ixs) % soln.rows(even_ixs) - phase_cos.rows(K_ixs) % soln.rows(odd_ixs);
#pragma omp taskwait
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
#pragma omp parallel for num_threads(adj_threads(input_size))
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
}
