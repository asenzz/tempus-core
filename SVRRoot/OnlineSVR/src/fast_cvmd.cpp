//
// Created by zarko on 7/5/22.
//
// #define ORIG_VMD

#ifdef ORIG_VMD

#include "vmd.hpp"

#endif

#include <armadillo>
#include <limits>
#include <memory>
#include <vector>
#include "common/defines.h"
#include "fast_cvmd.hpp"
#include "util/math_utils.hpp"
#include "common/compatibility.hpp"
#include "common/logging.hpp"
#include "common/parallelism.hpp"
#include "model/DeconQueue.hpp"
#include "model/DataRow.hpp"


namespace svr {
namespace vmd {

tbb::concurrent_map<freq_key_t, fcvmd_frequency_outputs> fast_cvmd::get_vmd_frequencies() const
{
    return vmd_frequencies;
}

void fast_cvmd::set_vmd_frequencies(const tbb::concurrent_map<freq_key_t, fcvmd_frequency_outputs> &_vmd_frequencies)
{
    vmd_frequencies = _vmd_frequencies;
}


fcvmd_frequency_outputs fast_cvmd::compute_cos_sin(const arma::vec &omega, const double step)
{
    const auto step_omega = 2. * M_PI * step * omega;
    const arma::vec phase_cos = arma::cos(step_omega);
    const arma::vec phase_sin = arma::sin(step_omega);
    LOG4_DEBUG("Omega " << omega << ", phase cos " << phase_cos << ", phase sin " << phase_sin << ", step " << step << ", alpha bins " << C_default_alpha_bins
                        << ", tau fidelity " << TAU_FIDELITY << ", max VMD iterations " << MAX_VMD_ITERATIONS << ", tolerance " << CVMD_TOL);
    return {phase_cos, phase_sin};
}

fast_cvmd::fast_cvmd(const unsigned int levels_) :
        spectral_transform(std::string("cvmd"), levels_),
        levels(levels_),
        K(levels_ / 2),
        A(levels_),
        H(arma::eye(levels_, levels_)),
        even_ixs(arma::regspace<arma::uvec>(0, 2, levels_ - 2)),
        odd_ixs(arma::regspace<arma::uvec>(1, 2, levels_ - 1)),
        f(levels_),
#ifdef VMD_ONLY
        row_values(levels_),
        soln(row_values.memptr(), levels_, false, true),
#else
        row_values(levels_ * 2),
        soln(row_values.memptr() + levels_, levels_, false, true),
#endif
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
calc_omega(arma::vec &omega, const unsigned int k, const arma::vec &freqs, const arma::cx_mat &u_hat_plus, const unsigned int T)
{
    const auto T_2 = T / 2;
    const auto T_1 = T - 1;
    const arma::vec norm_u_hat_plus = common::norm<double>(u_hat_plus.submat(T_2, k, T_1, k));
    omega[k] = arma::accu(freqs.rows(T_2, T_1) % norm_u_hat_plus) / arma::accu(norm_u_hat_plus);
}

#if 1

void
fast_cvmd::initialize(const datamodel::datarow_crange &input, const unsigned int input_column_index, const std::string &decon_queue_table_name,
                      const datamodel::t_iqscaler &scaler)
{
    if (input.distance() < 1) LOG4_THROW("Illegal input size " << input.distance());
    LOG4_DEBUG("Initializing omega on " << input.distance() << " rows, for " << levels << " levels, " << decon_queue_table_name << " table.");
    const auto freq_key = freq_key_t(decon_queue_table_name, levels);
    if (vmd_frequencies.find(freq_key) != vmd_frequencies.end()) return;

#ifdef EMOS_OMEGAS
    static const arma::vec emos_omega = {
            1.000000000000000e-12, 4.830917874396136e-08, 9.661835748792271e-08, 2.415458937198068e-07, 6.038647342995169e-07,
            1.207729468599034e-06, 2.415458937198068e-06, 1.207729468599034e-05, 3.472222222222222e-05, 6.944444444444444e-05,
            1.388888888888889e-04, 2.777777777777778e-04, 5.555555555555556e-04, 1.111111111111111e-03, 3.333333333333333e-03,
            1.666666666666667e-02};
    if (K > emos_omega.size()) LOG4_THROW("K " << K << " higher than Emo's omegas " << emos_omega.size());
    arma::vec omega(K);
    const size_t ratio_K = emos_omega.size() / K;
    for (size_t i = 0; i < emos_omega.size(); i += ratio_K) omega[i / ratio_K] = emos_omega[i];
    vmd_frequencies.emplace(freq_key, compute_cos_sin(omega, DEFAULT_PHASE_STEP));
#else
    //Initialize u, w, lambda, n
    const unsigned save_T = input.distance();
    auto T = save_T;
    if (T & 1) LOG4_THROW("Input data size " << T << " must be even!");

    //create the mirrored signal
    arma::vec f_init(2 * T);
    const auto T_2 = T / 2;
    const auto T_3_2 = 3 * T / 2;
    OMP_FOR_i(T) {
        if (i < T / 2) {
            f_init[i] = scaler(input[T_2 - 1 - i]->at(input_column_index));
            f_init[T_3_2 + i] = scaler(input[T - 1 - i]->at(input_column_index));
        }
        f_init[T_2 + i] = scaler(input[i]->at(input_column_index));
    }
    T = 2 * T;
    const arma::vec Alpha(K, arma::fill::value(save_T * C_default_alpha_bins * 1e-5)); // provision to use different alpha for the different frequences, at a later stage
    const double fs = 1. / double(save_T); // step

    /* outputs */
    arma::mat u(T, K);
    // Spectral Domain discretization
    const arma::vec freqs = arma::regspace(1, T) / double(T) - .5 - 1. / double(T);
    arma::cx_vec f_hat_plus = arma::shift(arma::conj(arma::fft(f_init)), T / 2);
    f_hat_plus.rows(0, T / 2 - 1).zeros();

    arma::vec omega(K);
    switch (C_freq_init_type) {
        case 1:
            omega = arma::regspace(0, K - 1) * .5 / K;
            break;

        case 2: {
            const auto log_fs = std::log(fs);
            omega = arma::sort(arma::exp(log_fs + (std::log(0.5) - log_fs) * arma::vec(K, arma::fill::randn)));
            break;
        }
        case 0:
            // Leave omega zeros
            break;

        default:
            LOG4_THROW("Initialization type should be 0, 1 or 2!");
    }
    if (HAS_DC == 1) omega[0] = 0; // if DC component, then first frequency is 0!
    /* completed initialization of frequencies */

    arma::cx_vec lambda_hat(freqs.n_elem); // keeping track of the evolution of lambda_hat with the iterations
    arma::cx_mat u_hat_plus(T, K);
    arma::cx_mat u_hat_plus_old(arma::size(u_hat_plus));
    double uDiff = CVMD_TOL + CVMD_EPS; // tol + eps must be different from just tol, that is why eps should not be too small
    arma::cx_vec sum_uk(T);

    unsigned n_iter = 0;
    while (n_iter < MAX_VMD_ITERATIONS && uDiff > CVMD_TOL) {
        //% update first mode accumulator
        sum_uk += u_hat_plus_old.col(K - 1) - u_hat_plus_old.col(0);

        //% update spectrum of first mode through Wiener filter of residuals
        u_hat_plus.col(0) = (f_hat_plus - sum_uk - .5 * lambda_hat) / (1. + Alpha[0] * arma::pow(freqs - omega[0], 2));

        //when DC is true, the first frequency can not be changed. that is why this cycle only for DC false.
        if (!HAS_DC) calc_omega(omega, 0, freqs, u_hat_plus, T);

        // update of any other mode (k from 1 to K - 1), after the first
#pragma omp simd
        for (size_t k = 1; k < K; ++k) {
            // accumulator
            sum_uk += u_hat_plus.col(k - 1) - u_hat_plus_old.col(k);
            // mode spectrum
            u_hat_plus.col(k) = (f_hat_plus - sum_uk - .5 * lambda_hat) / (1. + Alpha[k] * arma::pow(freqs - omega[k], 2));
            // re-compute center frequencies
            calc_omega(omega, k, freqs, u_hat_plus, T);
        }
        // Dual ascent
        lambda_hat += TAU_FIDELITY * (arma::accu(u_hat_plus.row(0)) - f_hat_plus);

        ++n_iter;

        // compute uDiff
        uDiff = arma::accu(arma::sum(common::norm<double>(u_hat_plus - u_hat_plus_old)) / arma::sum(common::norm(u_hat_plus_old)));
        u_hat_plus_old = u_hat_plus;
    }

#ifdef CVMD_POSTPROCESSING_AND_CLEANUP
    //%------ Postprocessing and cleanup
    //% discard empty space if converged early - this step is not used here
    //N = min(N,n);
    //omega = omega_plus(1:N,:);
    // - this not needed, since we do not keep all omega, only the latest!

    // Signal reconstruction
    arma::cx_mat u_hat(T, K);
    u_hat.rows(T / 2, T - 1) = u_hat_plus.rows(T / 2, T - 1);

    // here it is not clear why, but T/2 is set both above and below. Instead, the 0 is set to conj of T-1
#pragma omp parallel for collapse(2) num_threads(adj_threads(K * T / 2)) schedule(static, 1 + K * T / (2 * C_n_cpu))
    for (size_t i = 0; i < T / 2; ++i)
        for (size_t k = 0; k < K; ++k)
            u_hat(T / 2 - i, k) = std::conj(u_hat_plus(T / 2 + i, k));

    u_hat.row(0) = arma::conj(u_hat.row(T - 1));

    arma::mat u_big(K, freqs.n_elem);
#pragma omp parallel for num_threads(adj_threads(K)) schedule(static, 1)
    for (size_t k = 0; k < K; ++k)
        u_big.row(k) = arma::trans(arma::real(common::matlab_ifft(common::ifftshift(u_hat.col(k)))));
    u = arma::zeros(K, T / 2);
    OMP_FOR(K * T / 2)
    for (size_t i = 0; i < T / 2; ++i)
        for (size_t k = 0; k < K; ++k)
            u(k, i) = u_big(k, T / 4 + i);
    // recompute spectrum
    // clear u_hat;
    u_hat = arma::cx_mat(K, T / 2);
#pragma omp parallel for num_threads(adj_threads(K)) schedule(static, 1)
    for (size_t k = 0; k < K; ++k) u_hat.row(k) = common::fftshift(common::matlab_fft(u.row(k)));
#endif
    vmd_frequencies.emplace(freq_key, compute_cos_sin(omega, DEFAULT_PHASE_STEP));
#endif
}

#endif

void
fast_cvmd::transform(
        const data_row_container &input,
        datamodel::DeconQueue &decon,
        const unsigned int in_colix,
        const unsigned int test_offset,
        const datamodel::t_iqscaler &scaler)
{
#ifdef ORIG_VMD

    data_row_container::const_iterator iterin;
    if (decon.empty()) {
        iterin = input.cbegin();
    } else {
        iterin = lower_bound_back(input, decon.back()->get_value_time());
        while ((**iterin).get_value_time() <= decon.back()->get_value_time() && iterin != input.cend()) ++iterin;
        if (iterin == input.cend()) {
            LOG4_WARN("No input data newer than " << decon.back()->get_value_time() << " to deconstruct.");
            return;
        }
    }
    const auto in_ct = std::distance(iterin, input.cend());
#ifdef EIGEN_VMD
    vectord signal(in_ct);

    // Example 1: If you want to get the full results as a 2D matrix of VMD.
    MatrixXd u, omega;
    MatrixXcd u_hat;
    VMD(u, u_hat, omega, iterin, in_ct, in_colix, 50 /* C_default_alpha_bins */, 0 /*TAU_FIDELITY*/, levels, HAS_DC, C_freq_init_type, CVMD_TOL, 2.2204e-16 /* std::numeric_limits<double>::epsilon() */, scaler);
#else
    std::vector<std::vector<std::complex<double>>> u_hat;
    std::vector<std::vector<double>> u;
    std::vector<std::vector<double>> omega;
    std::tie(u, u_hat, omega) = VMD(iterin, in_ct, in_colix, 50 /* C_default_alpha_bins */, 0 /*TAU_FIDELITY*/, levels, HAS_DC, C_freq_init_type, CVMD_TOL, scaler);
#endif
    const auto prev_decon_ct = decon.size();
    decon.get_data().resize(prev_decon_ct + in_ct);
    OMP_FOR(in_ct)
    for (ssize_t t = 0; t < in_ct; ++t) {
        const auto p_in_row = *(iterin + t - prev_decon_ct);
#ifdef VMD_ONLY
        const auto p_row= ptr<datamodel::DataRow>(p_in_row->get_value_time(), timenow, p_in_row->get_tick_volume(), levels);
#else
        const auto p_row = ptr<datamodel::DataRow>(p_in_row->get_value_time(), timenow, p_in_row->get_tick_volume(), 2 * levels);
#endif
        decon[prev_decon_ct + t] = p_row;
#ifdef VMD_ONLY
        for (unsigned l = 0; l < levels; ++l) p_row->at(l) = u[l][t];
#else
        for (unsigned l = 0; l < levels; ++l) p_row->at(levels + l) = u(l, t);
#endif
    }

#else

    if (input.empty()) {
        LOG4_ERROR("Input empty!");
        return;
    }

    // VMD
    if (!initialized(decon.get_table_name())) {
        auto start_cvmd_init = input.cbegin() + std::max<ssize_t>(0, input.size() - CVMD_INIT_LEN - test_offset);
        auto end_cvmd_init = input.cend() - test_offset;
        if (std::distance(start_cvmd_init, end_cvmd_init) % 2) ++start_cvmd_init; // Ensure even distance
        PROFILE_EXEC_TIME(initialize(datamodel::datarow_crange{start_cvmd_init, end_cvmd_init, input}, in_colix, decon.get_table_name(), scaler), "CVMD init");
    }

    const auto found_freqs = vmd_frequencies.find({decon.get_table_name(), levels});
    if (vmd_frequencies.empty() || found_freqs == vmd_frequencies.cend())
        LOG4_THROW("VMD frequencies for " << decon.get_table_name() << " " << levels << " not found!");

    const auto &phase_cos = found_freqs->second.phase_cos;
    const auto &phase_sin = found_freqs->second.phase_sin;
    if (K != phase_sin.size() || K != phase_cos.size())
        LOG4_THROW("Invalid phase cos size " << phase_cos.size() << " or phase sin size " << phase_sin.size() << ", should be " << K);

    const auto levels_size = levels * sizeof(double);
    data_row_container::const_iterator iterin;
    if (decon.empty()) {
        iterin = input.cbegin();
        soln[0] = scaler((**iterin)[in_colix]);
    } else {
        iterin = lower_bound_back(input, decon.back()->get_value_time());
        while ((**iterin).get_value_time() <= decon.back()->get_value_time() && iterin != input.cend()) ++iterin;
        if (iterin == input.cend()) {
            LOG4_WARN("No input data newer than " << decon.back()->get_value_time() << " to deconstruct.");
            return;
        }
#ifdef VMD_ONLY
        memcpy(soln.memptr(), decon.back()->p(), levels_size);
#else
        memcpy(soln.memptr(), decon.back()->p(levels), levels_size);
#endif
    }

    const auto in_ct = std::distance(iterin, input.cend());
    const auto prev_decon_ct = decon.size();
    decon.get_data().resize(prev_decon_ct + in_ct);
#ifdef VMD_ONLY
#define levels_2 levels
#else
    const auto levels_2 = levels * 2;
#endif
    OMP_FOR_i_(in_ct, simd firstprivate(in_ct, prev_decon_ct, levels_2)) {
        const auto p_in_row = *(iterin + i - prev_decon_ct);
        decon[prev_decon_ct + i] = ptr<datamodel::DataRow>(p_in_row->get_value_time(), timenow, p_in_row->get_tick_volume(), levels_2);
    }

    auto f_even = f.rows(even_ixs);
    auto f_odd = f.rows(odd_ixs);
    auto sl_even = soln.rows(even_ixs);
    auto sl_odd = soln.rows(odd_ixs);
    const arma::vec A_t = A.t();
    f_even = -phase_cos % sl_even + phase_sin % sl_odd;
    f_odd = -phase_sin % sl_even - phase_cos % sl_odd;
#define ASOLVE(M, N) arma::solve(M, (N), C_arma_solver_opts)
#pragma omp simd
    for (size_t t = prev_decon_ct; t < decon.size(); ++t) {
        soln = ASOLVE(H, -A_t * (-ASOLVE(A * ASOLVE(H, A_t), A * ASOLVE(H, f) + scaler((**iterin)[in_colix]))) - f);
#ifdef VMD_ONLY
        memcpy(decon[t]->p(), soln.memptr(), levels_size);
#else
        memcpy(decon[t]->p(levels), soln.mem, levels_size);
#endif
        f_even = -phase_cos % sl_even + phase_sin % sl_odd;
        f_odd = -phase_sin % sl_even - phase_cos % sl_odd;
        ++iterin;
    }
#endif // ORIG_VMD
}


void
fast_cvmd::inverse_transform(
        const std::vector<double> &decon,
        std::vector<double> &recon,
        const size_t padding) const
{
    LOG4_WARN("Use reconstruct method in decon queue service for hybrid decons.");
    const size_t input_size = decon.size() / levels;
    if (recon.size() != input_size)
        recon.resize(input_size, 0);

    OMP_FOR(input_size)
    for (size_t t = 0; t < input_size; ++t) {
        recon[t] = 0;
        UNROLL()
        for (size_t l = 0; l < levels; l += 2)
            recon[t] += decon[t + l * input_size];
    }
}


size_t fast_cvmd::get_residuals_length(const std::string &decon_queue_table_name)
{
    LOG4_DEBUG("Getting residuals length for " << decon_queue_table_name << " " << levels << " levels.");
    const auto vmd_freq_iter = vmd_frequencies.find({decon_queue_table_name, levels});
    return vmd_freq_iter == vmd_frequencies.end() || vmd_freq_iter->second.phase_cos.empty() ?
#ifdef VMD_ONLY
    (size_t) std::pow(levels, 4)
#else
    (size_t) std::pow(levels / 2, 4)
#endif
    : 0;
}

size_t fast_cvmd::get_residuals_length(const unsigned levels) noexcept
{
    LOG4_DEBUG("Getting residuals length for " << levels << " levels.");
#ifdef VMD_ONLY
    return std::pow(levels, 4);
#else
    return std::pow(levels / 2, 4);
#endif
}

}
}
