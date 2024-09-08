//
// Created by zarko on 8/8/24.
//

#include <eigen3/Eigen/Core>
#define ARMA_DONT_USE_LAPACK
#define ARMA_DONT_USE_BLAS
#undef ARMA_USE_LAPACK
#undef ARMA_USE_BLAS
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
#include "vmd.hpp"

namespace svr {
namespace vmd {

#if 0

void
fast_cvmd::initialize(const datamodel::datarow_crange &input, const size_t input_column_index, const std::string &decon_queue_table_name,
                      const datamodel::t_iqscaler &scaler)
{
    if (input.distance() < 1) LOG4_THROW("Illegal input size " << input.distance());
    const auto freq_key = freq_key_t(decon_queue_table_name, levels);
    if (vmd_frequencies.find(freq_key) != vmd_frequencies.end()) {
	LOG4_DEBUG("VMD frequencies found.");
	return;
    }

    LOG4_DEBUG("Initializing omega on " << input.distance() << " rows, for " << levels << " levels, " << decon_queue_table_name << " table.");

    /* ---------------------

    Output:
    -------
    u - the collection of decomposed modes (2D double Matrix in Eigen -MatrixXd)
    u_hat - spectra of the modes (2D complex<double> Matrix in Eigen -MatrixXd)
    omega - estimated mode center - frequencies (2D double Matrix in Eigen -MatrixXd)
    -------
    Input:
    -------
    signal - the time domain signal(1D vector) to be decomposed
    alpha - the balancing parameter of the data - fidelity constraint
    tau - time - step of the dual ascent(pick 0 for noise - slack)
    K - the number of modes to be recovered
    DC - true if the first mode is putand kept at DC(0 - freq)
    init - 0 = all omegas start at 0
                        1 = all omegas start uniformly distributed
                        2 = all omegas initialized randomly
    tol - tolerance of convergence criterion; typically around 1e-6

    */

    Eigen::setNbThreads(C_n_cpu);

    const auto saveT = input.distance();
    // ----------Preparations
    // Periodand sampling frequency of input signal
    int T = saveT;
    double fs = 1.0 / T;

    //Extend the signal by mirroring
    vectord f_(2 * T, 0.0);
    /*
    copy(signal.begin(), signal.end(), f.begin() + T / 2);
    for (int i = 0; i < T / 2; i++)
        f[i] = signal[T / 2 - 1 - i];
    for (int i = 3 * T / 2; i < 2 * T; i++)
        f[i] = signal[T + 3 * T / 2 - 1 - i];
    */
    const auto T_2 = T / 2;
    const auto T_3_2 = 3 * T / 2;
    OMP_FOR(T)
    for (size_t i = 0; i < T; ++i) {
        if (i < T / 2) {
            f_[i] = scaler(input[T_2 - 1 - i]->at(input_column_index));
            f_[T_3_2 + i] = scaler(input[T - 1 - i]->at(input_column_index));
        }
        f_[T_2 + i] = scaler(input[i]->at(input_column_index));
    }

    // Time Domain 0 to T (of mirrored signal)
    // Spectral Domain discretization
    T = int(f_.size());
    vectorcd freqs(T, 0.0);
    vectord timevec(T, 0.0);
    for (int i = 0; i < T; i++) {
        timevec[i] = double(i + 1.0) / T;
        freqs[i] = (timevec[i] - 0.5) - double(1 / T);
    }

    // Maximum number of iterations(if not converged yet, then it won't anyway)
    int N = 500;

    // Construct and center f_hat
    vectorcd freqvec(T, 0.0);
    FFT<double> fft;
    fft.fwd(freqvec, f_);
    vectorcd f_hat = circshift(freqvec, T / 2);
    vectorcd f_hat_plus(f_hat.size(), 0.0);
    copy(f_hat.begin() + T / 2, f_hat.end(), f_hat_plus.begin() + T / 2);

    // Calculate Matrix-Column in advance
    MatrixXcd f_hat_plus_Xcd = vector_to_MatrixXcd_in_col(f_hat_plus);
    MatrixXcd freqs_Xcd = vector_to_MatrixXcd_in_col(freqs);

    // Matrix keeping track of every iterant // could be discarded for mem
    MatrixXcd u_hat_plus = MatrixXcd::Zero(K, T);
    MatrixXcd prev_u_hat_plus = u_hat_plus;
    MatrixXcd prev_prev_u_hat_plus = prev_u_hat_plus;

    // Initialization of omega_k
    MatrixXcd omega_plus = MatrixXcd::Zero(N, K);
    vectord tmp;
    switch (C_freq_init_type) {
        case 1:
            for (int i = 0; i < K; i++) {
                omega_plus(0, i) = double(0.5 / K) * (i);
                for (int j = 1; j < N; j++)
                    omega_plus(j, i) = 0.0;
            }
            break;
        case 2:
            tmp = omega_init_method2(K, fs);
            for (int i = 0; i < K; i++) {
                omega_plus(0, i) = tmp[i];
                for (int j = 1; j < N; j++)
                    omega_plus(j, i) = 0.0;
            }
            break;
        default:
            break;
    }

    // If DC mode imposed, set its omega to 0
    if (HAS_DC)
        omega_plus(0, 0) = 0;

    // Start with empty dual variables
    MatrixXcd lambda_hat = MatrixXcd::Zero(1, T);
    MatrixXcd prev_lambda_hat = lambda_hat;

    // Other inits
    double uDiff = CVMD_TOL + CVMD_EPS;//% update step
    int n = 1;// loop counter
    MatrixXcd sum_uk = MatrixXcd::Zero(1, T);
    // Accumulator
    int k;

    // ----------- Main loop for iterative updates
    while (uDiff > CVMD_TOL && n < N) {

        //update first mode accumulator
        k = 0;
        sum_uk = prev_u_hat_plus.row(K - 1) + sum_uk - prev_u_hat_plus.row(0);

        //update spectrum of first mode through Wiener filter of residuals
        MatrixXcd Dividend_vec = f_hat_plus_Xcd - sum_uk - (prev_lambda_hat / 2.0);
        MatrixXcd Divisor_vec = (1 + C_default_alpha_bins *
                                     ((freqs_Xcd.array() - omega_plus(n - 1, k))).array().square());
        u_hat_plus.row(k) = Dividend_vec.cwiseQuotient(Divisor_vec);

        //update first omega if not held at 0
        if (!HAS_DC) {
            std::complex<double> Dividend{0, 0}, Divisor{0, 0}, Addend{0, 0}, Addend_sqrt{0, 0};
            for (int i = 0; i < T - T / 2; i++) {
                Addend_sqrt = abs(u_hat_plus(k, T / 2 + i));
                Addend = Addend_sqrt * Addend_sqrt;
                Divisor += Addend;
                Dividend += freqs[T / 2 + i] * Addend;
            }
            omega_plus(n, k) = Dividend / Divisor;

        }
        // Dual ascent

        auto lambda_hat_lastrow_half = prev_lambda_hat / 2.0;
        for (k = 1; k < K; k++) {
            //accumulator
            sum_uk.noalias() += u_hat_plus.row(k - 1) - prev_u_hat_plus.row(k);

            //mode spectrum
            Dividend_vec = f_hat_plus_Xcd;
            Dividend_vec.noalias() -= sum_uk;         // in-place calculate
            Dividend_vec.noalias() -= lambda_hat_lastrow_half;  // in-place calculate
            Divisor_vec = (1 + C_default_alpha_bins * ((freqs_Xcd.array() - omega_plus(n - 1, k))).array().square());

            u_hat_plus.row(k) = Dividend_vec.cwiseQuotient(Divisor_vec);

            //center frequencies
            std::complex<double> Dividend{0, 0}, Divisor{0, 0}, Addend{0, 0}, Addend_sqrt{0, 0};
            for (int i = 0; i < T - T / 2; i++) {
                Addend_sqrt = abs(u_hat_plus(k, T / 2 + i));
                Addend = Addend_sqrt * Addend_sqrt;
                Divisor += Addend;
                Dividend += freqs[T / 2 + i] * Addend;
            }
            omega_plus(n, k) = Dividend / Divisor;
        }

        lambda_hat = prev_lambda_hat + TAU_FIDELITY * (u_hat_plus.colwise().sum() - f_hat_plus_Xcd);
        n++;

        std::complex<double> acc{CVMD_EPS, 0};
        for (int i = 0; i < K; i++) {
            MatrixXcd tmpm = prev_u_hat_plus.row(i) - prev_prev_u_hat_plus.row(i);
            tmpm = (tmpm * (tmpm.adjoint()));
            acc = acc + tmpm(0, 0) / double(T);

        }
        uDiff = abs(acc);

        prev_prev_u_hat_plus = prev_u_hat_plus;
        prev_u_hat_plus = u_hat_plus;

        prev_lambda_hat = lambda_hat;
    }

    //Postprocessing and cleanup
    //Discard empty space if converged early
    N = std::min(N, n);
    const VectorXd omega = omega_plus.row(N - 1).real();
    arma::vec aomega(omega.size(), arma::fill::none);
    for (int i = 0; i < omega.size(); ++i) aomega[i] = omega[i];
    vmd_frequencies.emplace(freq_key, compute_cos_sin(aomega, DEFAULT_PHASE_STEP));
}

#endif

}
}
