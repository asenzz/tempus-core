//
// Created by zarko on 7/2/24.
//

#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <algorithm>
#include <random>
#include <numeric>

#include "vmd.hpp"
#include "util/math_utils.hpp"
#include "IQScalingFactorService.hpp"


#ifdef EIGEN_VMD

using namespace Eigen;
using namespace std;

#endif


namespace svr {


#ifdef EIGEN_VMD

void printMatrix(const MatrixXd &u)
{
    std::ostringstream out; // use ostringstream to accumulate output
    for (int i = 0; i < u.rows(); i++) {
        for (int j = 0; j < u.cols(); j++)
            out << u(i, j) << ' ';
        out << "\n\n";
    }
    std::cout << out.str(); // output once
}


void VMD(MatrixXd &u, MatrixXcd &u_hat, MatrixXd &omega,
         svr::data_row_container::const_iterator iterin, const size_t saveT, const unsigned input_column_index, const double alpha, const double tau,
         const int K, const int DC, const int init, const double tol, const double eps, const svr::datamodel::t_iqscaler &scaler)
{
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
    // ----------Preparations
    // Period and sampling frequency of input signal
    int T = saveT;
    double fs = 1.0 / T;

    //Extend the signal by mirroring
    vectord f(2 * T, 0.0);
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
        f[T_2 + i] = scaler( (**(iterin + i))[input_column_index] );
        if (i < T / 2) {
            f[i] = scaler( (**(iterin + T_2 - 1 - i))[input_column_index] );
            f[T_3_2 + i] = scaler( (**(iterin + T - 1 - i))[input_column_index]);
        }
    }

    // Time Domain 0 to T (of mirrored signal)
    // Spectral Domain discretization
    T = int(f.size());
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
    fft.fwd(freqvec, f);
    vectorcd f_hat = circshift(freqvec, T / 2);
    vectorcd f_hat_plus(f_hat.size(), 0.0);
    copy(f_hat.begin() + T / 2, f_hat.end(), f_hat_plus.begin() + T / 2);

    // Calculate Matrix-Column in advance
	MatrixXcd f_hat_plus_Xcd = Eigen::Map<Eigen::MatrixXcd>(f_hat_plus.data(), 1, int(f_hat_plus.size()));
	MatrixXcd freqs_Xcd = Eigen::Map<Eigen::MatrixXcd>(freqs.data(), 1, int(freqs.size()));

    // Matrix keeping track of every iterant // could be discarded for mem
    MatrixXcd u_hat_plus = MatrixXcd::Zero(K, T);
    MatrixXcd prev_u_hat_plus = MatrixXcd::Zero(K, T);
    MatrixXcd prev_prev_u_hat_plus = MatrixXcd::Zero(K, T);

    // Initialization of omega_k
    MatrixXcd omega_plus = MatrixXcd::Zero(N, K);
    vectord tmp;
    switch (init) {
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
    if (DC)
        omega_plus(0, 0) = 0;

    // Start with empty dual variables
    MatrixXcd lambda_hat = MatrixXcd::Zero(1, T);
    MatrixXcd prev_lambda_hat = MatrixXcd::Zero(1, T);

    // Other inits
    double uDiff = tol + eps;//% update step
    int n = 1;// loop counter
    MatrixXcd sum_uk = MatrixXcd::Zero(1, T);
    // Accumulator
    int k;

    // ----------- Main loop for iterative updates
    while (uDiff > tol && n < N) {

        //update first mode accumulator
        k = 0;
        sum_uk = prev_u_hat_plus.row(K - 1) + sum_uk - prev_u_hat_plus.row(0);

        //update spectrum of first mode through Wiener filter of residuals
        MatrixXcd Dividend_vec = f_hat_plus_Xcd - sum_uk - (prev_lambda_hat / 2.0);
        MatrixXcd Divisor_vec = (1 + alpha *
                                     ((freqs_Xcd.array() - omega_plus(n - 1, k))).array().square());
        prev_prev_u_hat_plus = prev_u_hat_plus;
        prev_u_hat_plus = u_hat_plus;
        u_hat_plus.row(k).noalias() = Dividend_vec.cwiseQuotient(Divisor_vec);

        //update first omega if not held at 0
        if (!DC) {
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
			MatrixXcd Dividend_vec = f_hat_plus_Xcd;
            Dividend_vec.noalias() -= sum_uk;         // in-place calculate
            Dividend_vec.noalias() -= lambda_hat_lastrow_half;  // in-place calculate
            Divisor_vec = (1 + alpha *
                                         ((freqs_Xcd.array() - omega_plus(n - 1, k))).array().square());

            u_hat_plus.row(k).noalias() = Dividend_vec.cwiseQuotient(Divisor_vec);

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

        prev_lambda_hat = lambda_hat;
        lambda_hat.noalias() = prev_lambda_hat + tau * (u_hat_plus.rowwise().sum() - f_hat_plus_Xcd);
        n++;

        std::complex<double> acc{eps, 0};
        for (int i = 0; i < K; i++) {
            MatrixXcd tmp = prev_u_hat_plus.row(i) - prev_prev_u_hat_plus.row(i);
            tmp = (tmp * (tmp.adjoint()));
            acc = acc + tmp(0, 0) / double(T);

        }
        uDiff = abs(acc);

    }

    //Postprocessing and cleanup
    //Discard empty space if converged early
    N = std::min(N, n);
    omega = omega_plus.topRows(N).real();

    //Signal reconstruction
    u_hat = MatrixXcd::Zero(T, K);
    for (int i = T / 2; i < T; i++)
		for (int k = 0; k < K; k++)
			u_hat(i, k) = u_hat_plus(k, i);

    for (int i = T / 2; i >= 0; i--)
        for (int k = 0; k < K; k++)
            u_hat(i, k) = conj(u_hat_plus(k, T - i - 1));

    u_hat.row(0) = u_hat.row(T - 1).transpose().adjoint();
    u.resize(K, saveT);
    vectord result_col;
	for (int k = 0; k < K; k++) {
        vectorcd u_hat_col = ExtractColFromMatrixXcd(u_hat, k, T);
        u_hat_col = circshift(u_hat_col, int(floor(T / 2)));
        fft.inv(result_col, u_hat_col);
        for (int t = 0; t < saveT; t++)
            u(k, t) = result_col[t + T / 4];
    }

    vectord result_timevec(saveT, 0);
    for (int i = 0; i < saveT; i += 1) {
        result_timevec[i] = double(i + 1) / saveT;
    }

	for (int k = 0; k < K; k++) {
        vectorcd u_row = ExtractRowFromMatrixXd(u, k, saveT);
        fft.inv(result_timevec, u_row);
        u_row = circshift(u_row, saveT / 2);
        for (int t = 0; t < saveT; t++)
            u_hat(t, k) = u_row[t].real();
    }
}

#pragma region Ancillary Functions

vectorcd circshift(vectorcd &data, int offset)
{
    int n = int(data.size());
    if (offset == 0) {
        vectorcd out_data(data);
        return out_data;
    } else {
        offset = (offset > 0) ? n - offset : -offset;        // move to right by offset positions
        vectorcd out_data(data.begin() + offset, data.end());
        out_data.insert(out_data.end(), data.begin(), data.begin() + offset);
        return out_data;
    }
}

vectord omega_init_method2(int K, const double fs)
{
    vectord res(K, 0);
    int N = INT_MAX / 2;
    srand(int(time(NULL)));
    for (int i = 0; i < K; i++) {
        res[i] = exp(log(fs) + (log(0.5) - log(fs)) *
                               (rand() % (N + 1) / (double) (N + 1))
        );
    }
    sort(res.begin(), res.end());
    return res;
}


vectorcd ExtractColFromMatrixXcd(MatrixXcd &Input, const int ColIdx, const int RowNum)
{
    vectorcd Output(RowNum, 0);
    for (int i = 0; i < RowNum; ++i)
        Output[i] = Input(i, ColIdx);
    return Output;
}


vectorcd ExtractRowFromMatrixXd(MatrixXd &Input, const int RowIdx, const int ColNum)
{
    vectorcd Output(ColNum, 0);
    for (int i = 0; i < ColNum; ++i)
        Output[i] = Input(RowIdx, i);
    return Output;
}

#pragma endregion

#else

// Function declaration
std::tuple<std::vector<std::vector<double>>, std::vector<std::vector<std::complex<double>>>, std::vector<std::vector<double>>>
VMD(const svr::data_row_container::const_iterator iterin, const size_t save_T,
    const unsigned input_column_index, double alpha, double tau, int K, bool DC, int init, double tol, const svr::datamodel::t_iqscaler &scaler)
{
    if (save_T % 2) LOG4_THROW("Signal length must be even.");
    // Preparations
    int T = save_T;
    double fs = 1.0 / save_T;

    // Extend the signal by mirroring
    const auto T_2 = T / 2;
    const auto T_3_2 = 3 * T / 2;
    std::vector<double> f(2 * T);
    OMP_FOR_i(T) {
        f[T_2 + i] = scaler( (**(iterin + i))[input_column_index] );
        if (i < T / 2) {
            f[i] = scaler( (**(iterin + T_2 - 1 - i))[input_column_index] );
            f[T_3_2 + i] = scaler( (**(iterin + T - 1 - i))[input_column_index]);
        }
    }

    // Time Domain 0 to T (of mirrored signal)
    T = f.size();
    std::vector<double> t(T);
    OMP_FOR(T)
    for (int i = 0; i < T; ++i) t[i] = (i + 1.0) / T;

    // Spectral Domain discretization
    std::vector<double> freqs(T);
    OMP_FOR(T)
    for (int i = 0; i < T; ++i) freqs[i] = t[i] - 0.5 - 1.0/T;

    // Maximum number of iterations
    int N = 500;

    // Individual alpha for each mode
    std::vector<double> Alpha(K, alpha);

    // Construct and center f_hat
    std::vector<std::complex<double>> f_hat(T);
    OMP_FOR(T)
    for (int i = 0; i < T; ++i) {
        f_hat[i] = std::polar(f[i], -2.0 * M_PI * i * (T/2) / T);
    }
    std::vector<std::complex<double>> f_hat_plus = f_hat;
    std::fill(f_hat_plus.begin(), f_hat_plus.begin() + T/2, 0.0);

    // Initialize u_hat_plus, omega_plus, and lambda_hat
    std::vector<std::vector<std::vector<std::complex<double>>>> u_hat_plus(N, std::vector<std::vector<std::complex<double>>>(T, std::vector<std::complex<double>>(K, std::complex<double>(0))));
    std::vector<std::vector<double>> omega_plus(N, std::vector<double>(K));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(std::log(fs), std::log(0.5));

    switch (init) {
        case 1:
            for (int i = 0; i < K; ++i) omega_plus[0][i] = (0.5/K) * i;
            break;
        case 2:
            for (int i = 0; i < K; ++i) omega_plus[0][i] = std::exp(dis(gen));
            std::sort(omega_plus[0].begin(), omega_plus[0].end());
            break;
        default:
            std::fill(omega_plus[0].begin(), omega_plus[0].end(), 0.0);
    }

    if (DC) omega_plus[0][0] = 0.0;

    std::vector<std::vector<std::complex<double>>> lambda_hat(N, std::vector<std::complex<double>>(T, 0));

    double uDiff = tol + std::numeric_limits<double>::epsilon();
    int n = 0;
    std::vector<std::complex<double>> sum_uk(T);

    // Main loop for iterative updates
    while (uDiff > tol && n < N - 1) {
        // Update first mode
        int k = 0;
        OMP_FOR(T)
        for (int i = 0; i < T; ++i) {
            sum_uk[i] += u_hat_plus[n][i][K-1] - u_hat_plus[n][i][0];
        }

        OMP_FOR(T)
        for (int i = 0; i < T; ++i) {
            u_hat_plus[n+1][i][k] = (f_hat_plus[i] - sum_uk[i] - lambda_hat[n][i]/2.0) /
                                    (1.0 + Alpha[k] * std::pow(freqs[i] - omega_plus[n][k], 2));
        }

        if (!DC) {
            double num = 0.0, den = 0.0;
            OMP_FOR(T/2)
            for (int i = T/2; i < T; ++i) {
                num += freqs[i] * std::norm(u_hat_plus[n+1][i][k]);
                den += std::norm(u_hat_plus[n+1][i][k]);
            }
            omega_plus[n+1][k] = num / den;
        }

        // Update other modes
        for (k = 1; k < K; ++k) {
            OMP_FOR(T)
            for (int i = 0; i < T; ++i) {
                sum_uk[i] += u_hat_plus[n+1][i][k-1] - u_hat_plus[n][i][k];
            }

            for (int i = 0; i < T; ++i) {
                u_hat_plus[n+1][i][k] = (f_hat_plus[i] - sum_uk[i] - lambda_hat[n][i]/2.0) /
                                        (1.0 + Alpha[k] * std::pow(freqs[i] - omega_plus[n][k], 2));
            }

            double num = 0.0, den = 0.0;
            for (int i = T/2; i < T; ++i) {
                num += freqs[i] * std::norm(u_hat_plus[n+1][i][k]);
                den += std::norm(u_hat_plus[n+1][i][k]);
            }
            omega_plus[n+1][k] = num / den;
        }

        // Dual ascent
        OMP_FOR(T)
        for (int i = 0; i < T; ++i) {
            std::complex<double> sum_u = 0.0;
            for (int j = 0; j < K; ++j) {
                sum_u += u_hat_plus[n+1][i][j];
            }
            lambda_hat[n+1][i] = lambda_hat[n][i] + tau * (sum_u - f_hat_plus[i]);
        }

        ++n;

        // Check convergence
        uDiff = std::numeric_limits<double>::epsilon();
        OMP_FOR_(K * T, simd collapse(2))
        for (int i = 0; i < K; ++i) {
            for (int j = 0; j < T; ++j) {
                uDiff += std::norm(u_hat_plus[n][j][i] - u_hat_plus[n-1][j][i]) / T;
            }
        }
        uDiff = std::abs(uDiff);
    }

    // Postprocessing and cleanup
    N = std::min(N, n + 1);
    std::vector<std::vector<double>> omega(N, std::vector<double>(K));
    for (int i = 0; i < N; ++i) {
        omega[i] = omega_plus[i];
    }

    // Signal reconstruction
    std::vector<std::vector<std::complex<double>>> u_hat(T, std::vector<std::complex<double>>(K));
    OMP_FOR(K)
    for (int k = 0; k < K; ++k) {
        for (int i = T/2; i < T; ++i) {
            u_hat[i][k] = u_hat_plus[N-1][i][k];
            u_hat[T-i][k] = std::conj(u_hat_plus[N-1][i][k]);
        }
        u_hat[0][k] = std::conj(u_hat[T-1][k]);
    }

    std::vector<std::vector<double>> u(K, std::vector<double>(T));
    OMP_FOR(K)
    for (int k = 0; k < K; ++k) {
        std::vector<std::complex<double>> temp(T);
        for (int i = 0; i < T; ++i) {
            temp[i] = u_hat[i][k] * std::polar(1.0, 2.0 * M_PI * i * (T/2) / T);
        }
        for (int i = 0; i < T; ++i) {
            u[k][i] = std::real(std::accumulate(temp.begin(), temp.end(), std::complex<double>(0.0)) / double(T));
        }
    }

    // Remove mirror part
    std::vector<std::vector<double>> uo(K, std::vector<double>(alpha));
    OMP_FOR(K)
    for (int k = 0; k < K; ++k) {
        std::copy(u[k].begin() + T/4, u[k].begin() + T/4 + alpha, uo[k].begin());
    }

    // Recompute spectrum
    std::vector<std::vector<std::complex<double>>> u_hat_final(T, std::vector<std::complex<double>>(K));
    OMP_FOR(K)
    for (int k = 0; k < K; ++k) {
        for (int i = 0; i < T; ++i) {
            u_hat_final[i][k] = std::polar(uo[k][std::fmod(i, alpha)], -2.0 * M_PI * i * (double(T)/2) / T);
        }
    }

    return std::make_tuple(uo, u_hat_final, omega);
}

#endif

}
