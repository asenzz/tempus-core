#include <iomanip>
#include <limits>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <vector>
#include <cassert>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <unistd.h>
#include <fcntl.h>
#include <sys/time.h>
//include interface
//#include "nvmd.hpp" 

#include <complex>
//#define ARMA_NO_DEBUG
#define ARMA_ALLOW_FAKE_GCC

#include <armadillo>
#include <fstream>

// maximum number of iterations. 500 seems to be enough

#define MAX_VMD_ITERATIONS 500


static double time1[100] = {0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.};

static double msecs()
{
    struct timespec start;
    long mtime, seconds, useconds;

    //gettimeofday(&start, NULL);
    clock_gettime(CLOCK_MONOTONIC, &start);
    seconds = start.tv_sec;
    useconds = start.tv_nsec;

    double stime = ((seconds) * 1 + (double) useconds / 1000000000.0);


    return stime;
}


double randouble()
{
    static std::once_flag flag1;
    std::call_once(flag1, []() { srand48(9879798797); });
    return drand48();
}

arma::cx_mat matlab_fft(const arma::mat &input)
{
    //armadillo and matlab do fft differently, need to take conjugate
    //use this for real matrix
    return arma::conj(arma::fft(input));
}

arma::cx_mat matlab_fft(const arma::cx_mat &input)
{
    //armadillo and matlab do fft differently, need to take conjugate
    //use this for complex matrix
    return arma::conj(arma::fft(input));
}

arma::cx_mat matlab_ifft(const arma::cx_mat &input)
{
    //armadillo and matlab do fft differently, need to take conjugate
    return arma::ifft(arma::conj(input));
}

arma::cx_mat fftshift(const arma::cx_mat &input)
{
    arma::cx_mat output = input;
    int N = input.n_elem;
    for (int i = 0; i < input.n_elem; i++) {
        output(i) = input((i + N / 2) % N);
    }
    return output;
}

arma::cx_mat ifftshift(const arma::cx_mat &input)
{
    arma::cx_mat output = input;
    int N = input.n_elem;
    for (int i = 0; i < input.n_elem; i++) {
        output(i) = input((i + N - N / 2) % N);
    }
    return output;
}

int compute_cos_sin(const std::vector<double> &omega, std::vector<double> &phase_cos, std::vector<double> &phase_sin, double step)
{
    const size_t levels = omega.size();
    for (size_t i = 0; i < levels; i++) {
        phase_cos[i] = cos(2 * M_PI * step * omega[i]);
        phase_sin[i] = sin(2 * M_PI * step * omega[i]);
    }
}

extern int step_decompose_matrix(const std::vector<double> &phase_cos, const std::vector<double> &phase_sin, size_t len, CPTR(double) values, const std::vector<double> &previous,
                                 std::vector<double> &decomposition);

double loss_func_quadratic(size_t K, size_t len, double *UV, double *cos_phases, double *sin_phases)
{
    double result = 0;
    for (int ii = 0; ii < len - 1; ii++) {
        for (int jj = 0; jj < K; jj++) {
            std::complex<double> left(UV[2 * jj + 0 + ii * 2 * K], UV[2 * jj + 1 + 0 + ii * 2 * K]);
            std::complex<double> right(UV[2 * jj + 0 + (ii + 1) * 2 * K], UV[2 * jj + 1 + 0 + (ii + 1) * 2 * K]);
            std::complex<double> phase(cos_phases[jj], -sin_phases[jj]);
            result += std::norm(left - right * phase);
        }
    }
    return result;
}


int create_solve_problem(size_t K, size_t n, CPTR(double) previous, const double *values, const double *phase_cos, const double *phase_sin, arma::mat &H, arma::mat &f, arma::mat &A, arma::mat &b,
                         double &const_sum, bool do_H)
{
    A = arma::zeros(n, n * 2 * K);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < K; j++) {
            A(i, 2 * j + i * 2 * K) = 1.;
        }
    }
    b = arma::zeros(n, 1);
    for (int i = 0; i < n; i++) {
        b(i, 0) = values[i];
    }
    if (do_H) {
        H = arma::zeros(n * 2 * K, n * 2 * K);
    }
    f = arma::zeros(n * 2 * K, 1);
    const_sum = 0.;
    for (int i = 0; i < n; i++) {
        if (i == 0) {
            for (int j = 0; j < K; j++) {
                double prev_ui = previous[2 * j];
                double prev_vi = previous[2 * j + 1];
                double CC = phase_cos[j];
                double SS = phase_sin[j];
                //%u_i+1, u_i+a1
                if (do_H) {
                    H(2 * j + i * 2 * K, 2 * j + i * 2 * K) += 1.;
                    H(2 * j + 1 + i * 2 * K, 2 * j + 1 + i * 2 * K) += 1.;
                }
                const_sum += prev_ui * prev_ui + prev_vi * prev_vi;
                //%f for u_i+1 u_i
                f(2 * j + i * 2 * K, 0) += -CC * prev_ui; //  SS * prev_vi;
                //%f for u_i+1 v_i
                f(2 * j + i * 2 * K, 0) += SS * prev_vi;

                //%f for v_i+1 u_i
                f(2 * j + 1 + i * 2 * K, 0) += -SS * prev_ui; // -CC * prev_vi;
                //%f for v_i+1 v_i
                f(2 * j + 1 + i * 2 * K, 0) += -CC * prev_vi;
            }
        } else {
            if (do_H) {
                for (int j = 0; j < K; j++) {
                    double CC = phase_cos[j];
                    double SS = phase_sin[j];

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
    return 0;
}


int step_decompose_matrix(const std::vector<double> &phase_cos, const std::vector<double> &phase_sin, size_t len, const double *values, const std::vector<double> &previous,
                          std::vector<double> &decomposition)
{
    const size_t K = phase_cos.size();
    const size_t N = 2 * K;
    decomposition.resize(2 * K * len);

    static arma::mat H;
    arma::mat f, A, b;
    double const_sum;
    static bool once_only = [&]() {
        arma::mat tempf, tempA, tempb;
        create_solve_problem(K, len, previous.data(), values, phase_cos.data(), phase_sin.data(), H, tempf, tempA, tempb, const_sum, true);
        return true;
    }();
    create_solve_problem(K, len, previous.data(), values, phase_cos.data(), phase_sin.data(), H, f, A, b, const_sum, false);

    arma::mat lambda_rhs = A * arma::solve(H, f) + b;
    arma::mat lambda = -arma::solve((A * arma::solve(H, A.t())), lambda_rhs);
    arma::mat solution = arma::solve(H, -A.t() * lambda - f);
    std::memcpy(decomposition.data(), solution.memptr(), 2 * K * len * sizeof(double));
    return 0;

}


//internal
int vmd_decomp(const arma::mat &signal, double alpha, double tau, int K, int DC, int init, double tol,
        /*outputs */ arma::mat &u, arma::cx_mat &u_hat, arma::mat &omega_plus)
{
    //Initialize u, w, lambda, n
    //
    size_t save_T = signal.n_elem;
    std::cout << " Inside " << std::endl;
    size_t T = save_T;
    if (T & 1 != 0) abort();//must be even.
    //create the mirrored signal
    arma::mat f_mirror = arma::zeros(1, 2 * T);
#pragma omp parallel for
    for (int i = 0; i < T / 2; i++) {
        f_mirror(0, i) = signal(0, T / 2 - 1 - i);
    }
    std::cout << " Inside2 " << std::endl;
#pragma omp parallel for
    for (int i = 0; i < T; i++) {
        f_mirror(0, T / 2 + i) = signal(0, i);
    }
#pragma omp parallel for
    for (int i = 0; i < T / 2; i++) {
        f_mirror(0, 3 * T / 2 + i) = signal(0, T - 1 - i);
    }
    arma::mat f = f_mirror;
    T = 2 * T;

    arma::mat Alpha = alpha * arma::ones(1, K);//provision to use different alpha for the different frequences, at a later stage
    std::cout << " Inside3 " << std::endl;

    double fs = 1. / (double) save_T;//step

    const int N_max = MAX_VMD_ITERATIONS;//max number of iterations

    u = arma::zeros(T, K);
    double eps = 2.220446049250313e-16;
    //this may or may not work instead. double eps=std::numeric_limits<T>::epsilon();

    //% Time Domain 0 to T (of mirrored signal)
    arma::mat t(1, T);
#pragma omp parallel for
    for (int i = 0; i < T; i++) {
        t(0, i) = ((double) i + 1.) / (double) T;
    }
    //% Spectral Domain discretization
    arma::mat freqs = t - 0.5 - 1. / (double) T;

    std::cout << " Inside4 " << std::endl;


    omega_plus = arma::zeros(K, 1);

    arma::cx_mat f_hat = fftshift(matlab_fft(f));

    arma::cx_mat f_hat_plus = f_hat;

#pragma omp parallel for
    for (int i = 0; i < T / 2; i++) {
        f_hat_plus(0, i) = 0.;
    }
    std::cout << " Inside5 " << std::endl;
    switch (init) {
        case 1: {
            for (int i = 0; i < K; i++) {
                omega_plus(i, 0) = 0.5 / K * i;
            }
            break;
        }
        case 2: {
            arma::mat real_ran(K, 1);
            for (int i = 0; i < K; i++) {
                real_ran(i, 0) = exp(log(fs) + (log(0.5) - log(fs)) * randouble());
            }
            real_ran = arma::sort(real_ran);
            for (int i = 0; i < K; i++) {
                omega_plus(i, 0) = real_ran(i, 0);
            }
            break;
        }
        case 0: {
            for (int i = 0; i < K; i++) {
                omega_plus(i, 0) = 0.;
            }
            break;
        }
        default: {
            std::cout << "Init should be 0,1 or 2!" << std::endl;
            abort();
        }
    }
    if (DC == 1) {//if DC component, then first frequency is 0!
        omega_plus(0, 0) = 0;
    }
    std::cout << " Inside6 " << std::endl;
    /* completed initialization of frequencies*/

    //arma::cx_mat lambda_hat( arma::zeros(N_max+1, freqs.n_elem), arma::zeros(N_max+1, freqs.n_elem));//keeping track of the evolution of lambda_hat with the iterations
    arma::cx_mat lambda_hat(arma::zeros(1, freqs.n_elem), arma::zeros(1, freqs.n_elem));//keeping track of the evolution of lambda_hat with the iterations

    arma::cx_mat u_hat_plus(arma::zeros(T, K), arma::zeros(T, K));
    arma::cx_mat u_hat_plus_old = u_hat_plus;

    double uDiff = tol + eps;//tol+eps must be different from just tol, that is why eps should not be too small

    arma::cx_mat sum_uk(arma::zeros(1, T), arma::zeros(1, T));
    int n_iter = 0;

    while ((n_iter < N_max) && (uDiff > tol)) {
        std::cout << n_iter << " " << N_max << std::endl;


        //% update first mode accumulator
        int k = 0;
#pragma omp parallel for
        for (int i = 0; i < T; i++) {
            sum_uk(0, i) = u_hat_plus_old(i, K - 1) + sum_uk(0, i) - u_hat_plus_old(i, 0);
        }
        //% update spectrum of first mode through Wiener filter of residuals
#pragma omp parallel for
        for (int i = 0; i < T; i++) {
            u_hat_plus(i, k) = (f_hat_plus(0, i) - sum_uk(0, i) - lambda_hat(0, i) / 2.) / (1. + Alpha(0, k) * std::pow((freqs(0, i) - omega_plus(k, 0)), 2));
        }

        if (DC == 0) {
            //when DC==1, the first frequency can not be changed. that is why this cycle only for DC==0.
            double sum_up = 0.;
            double sum_down = 0.;
#pragma omp parallel for reduction(+:sum_up, sum_down)
            for (int i = 0; i < T / 2; i++) {
                sum_up += freqs(0, T / 2 + i) * std::norm(u_hat_plus(T / 2 + i, k));
                sum_down += std::norm(u_hat_plus(T / 2 + i, k));
            }
            omega_plus(0, 0) = sum_up / sum_down;
        }

        //update of any other mode (k from 1 to K-1), after the first
        for (k = 1; k < K; k++) {
            // accumulator
#pragma omp parallel for
            for (int i = 0; i < T; i++) {
                sum_uk(0, i) = u_hat_plus(i, k - 1) + sum_uk(0, i) - u_hat_plus_old(i, k);
            }
            // mode spectrum
#pragma omp parallel for
            for (int i = 0; i < T; i++) {
                u_hat_plus(i, k) = (f_hat_plus(0, i) - sum_uk(0, i) - lambda_hat(0, i) / 2.) / (1. + Alpha(0, k) * std::pow(freqs(0, i) - omega_plus(k, 0), 2));
            }
            //re-compute center frequencies
            double sum_up = 0.;
            double sum_down = 0.;
#pragma omp parallel for reduction(+:sum_up, sum_down)
            for (int i = 0; i < T / 2; i++) {
                //std::cout << "freq " << i << " " <<  std::norm(u_hat_plus(T/2+i,k))<< std::endl;
                sum_up += freqs(0, T / 2 + i) * std::norm(u_hat_plus(T / 2 + i, k));
                sum_down += std::norm(u_hat_plus(T / 2 + i, k));
            }
            omega_plus(k, 0) = sum_up / sum_down;
        }
        // Dual ascent
        arma::cx_mat sum_u_hat_plus = arma::sum(u_hat_plus.t());

#pragma omp parallel for
        for (int i = 0; i < T; i++) {
            lambda_hat(0, i) = lambda_hat(0, i) + tau * (sum_u_hat_plus(0, i) - f_hat_plus(0, i));
        }
        // loop counter
        n_iter++;

        //compute uDiff
        uDiff = 0.;
        for (int i = 0; i < K; i++) {
            double s = 0.;
            double s_norm_old = 0.;
//can be parallelized with reduction for s and s_norm_old #pragma omp parallel for default(shared) reduction(+:s,s_norm_old)
            for (int j = 0; j < T; j++) {
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
    arma::cx_mat zmat(arma::zeros(T, K), arma::zeros(T, K));
    u_hat = zmat;
    for (int i = 0; i < T / 2; i++) {
        for (int k = 0; k < K; k++) {
            u_hat(T / 2 + i, k) = u_hat_plus(T / 2 + i, k);
        }
    }
    // here it is not clear why, but T/2 is set both above and below. Instead, the 0 is set to conj of T-1
    for (int i = 0; i < T / 2; i++) {
        for (int k = 0; k < K; k++) {
            u_hat(T / 2 - i, k) = std::conj(u_hat_plus(T / 2 + i, k));
        }
    }
    for (int k = 0; k < K; k++) {
        u_hat(0, k) = std::conj(u_hat(T - 1, k));
    }

    arma::mat u_big = arma::zeros(K, t.n_elem);
    for (int k = 0; k < K; k++) {
        u_big.rows(k, k) = arma::trans(arma::real(matlab_ifft(ifftshift(u_hat.cols(k, k)))));
    }
    u = arma::zeros(K, T / 2);
    for (int i = 0; i < T / 2; i++) {
        for (int k = 0; k < K; k++) {
            u(k, i) = u_big(k, T / 4 + i);
        }
    }
    //recompute spectrum
    //clear u_hat;
    u_hat = arma::cx_mat(arma::zeros(K, T / 2), arma::zeros(K, T / 2));
    for (int k = 0; k < K; k++) {
        u_hat.rows(k, k) = fftshift(matlab_fft(u.rows(k, k)));
        // should work without conj. not clear why conj needed.
    }
    return 0;
}

//interface without armadillo
//
//
int vmd_decomp(int T, const double *signal, double alpha, double tau, int K, int DC, int init, double tol,
        /*outputs */ double *decomposition_u,/*complex*/ double *decomposition_u_hat, double *decomposition_omega)
{
    /*caller must allocate K*T*sizeof(double) for decomposition_u, 2*K*T*sizeof(double) for decomposition_u_hat (which is complex), K*sizeof(double) for decomposition_omega_plus*/

    arma::mat x_signal(1, T);
    for (int i = 0; i < T; i++) {
        x_signal(0, i) = signal[i];
    }
    arma::mat u;
    arma::cx_mat u_hat;
    arma::mat omega;
    vmd_decomp(x_signal, alpha, tau, K, DC, init, tol, u, u_hat, omega);
    for (int k = 0; k < K; k++) {
        for (int i = 0; i < T; i++) {
            decomposition_u[k * T + i] = u(k, i);
        }
    }
    for (int k = 0; k < K; k++) {
        for (int i = 0; i < T; i++) {
            decomposition_u_hat[2 * (k * T + i) + 0] = std::real(u_hat(k, i));
            decomposition_u_hat[2 * (k * T + i) + 1] = std::imag(u_hat(k, i));
        }
    }
    for (int k = 0; k < K; k++) {
        decomposition_omega[k] = omega(k, 0);
    }
    return 0;
}

//This is bad!
int find_best_omega_l1_fft(int N, int from, double &omega, std::vector<double> &U, std::vector<double> &V)
{
    if (N % 2 == 1) N--;

    arma::mat u_data(1, N);
    arma::mat v_data(1, N);
    for (int i = 0; i < N; i++) {
        u_data(0, i) = U[i];
        v_data(0, i) = V[i];
    }

    arma::cx_mat data(u_data, v_data);
    double sum_up = 0.;
    double sum_down = 0.;

    arma::cx_mat data_fft = matlab_fft(data);
    for (int i = 0; i < N / 2; i++) {
        sum_up += (double) i / (double) N * std::norm(data_fft(0, i));
        sum_down += std::norm(data_fft(0, i));
    }
    omega = sum_up / sum_down;
    std::vector<double> norms(N, 0.);
    for (int i = 0; i < N / 2; i++) {
        norms[i] = std::norm(data_fft(0, i));
    }

    omega = -(double) (std::distance(std::max_element(norms.begin(), norms.end()), norms.begin())) / (double) N;
}


int test_main(int argc, char *argv[])
{
    int T = 200;//number of timesteps , must be even number!
    arma::mat x_signal(1, T);
    for (int i = 0; i < T; i++) {
        double x = (double) i / (double) T;
        //x_signal(0,i)=0.1*x+sin(2*M_PI*x*5)+0.2*cos(2*M_PI*x*3)+0.10*sin(2*M_PI*x*7)+0.021*randouble();
        x_signal(0, i) = cos(2 * M_PI * x * 5) + 0.2 * cos(2 * M_PI * x * 3) + 0.10 * cos(2 * M_PI * x * 7) + 0.000021 * x * x;
        //test signal, has 3 frequencies
    }
    //std::cout << x_signal << std::endl;

    /* constants*/
    const double alpha = 2000.;//bands
    const double tau = 0;//fidelity - 0 means no strict enforcement of decomposition
    const int DC = 0; //has a DC component or not.
    const int K = 3;//number of modes/frequencies. because of DC=1 we use 1 more than the natural number of frequencies (3 in the signal above).
    const int init = 1;//kind of initialization, let's use 1 for now.
    const double tol = 1e-7;//some tolerance
    /* end constants*/
    /* outptuts*/
    arma::mat u;
    arma::cx_mat u_hat;
    arma::mat omega;
    /* end outptuts*/
    const bool print_some_output = true;
    vmd_decomp(x_signal, alpha, tau, K, DC, init, tol, u, u_hat, omega);


    return 0;
}

int find_omega(int K, const std::vector<double> &timeseries, std::vector<double> &omega)
{
    omega.resize(K);
    size_t T = timeseries.size();
    const double *signal = timeseries.data();
    double alpha = 2000.;
    double tau = 0.001;
    std::vector<double> mtimeseries(timeseries);
    if (T % 2 == 1) {
        mtimeseries.erase(mtimeseries.begin());
        T--;
    }
    assert(T % 2 == 0);
    int DC = 0;
    int init = 1;
    double tol = 1.e-06;
    std::vector<double> decomposition_u(K * T * 2);
    std::vector<double> decomposition_u_hat(K * T * 2 * 2);
    std::vector<double> decomposition_omega(K);
    vmd_decomp(T, signal, alpha, tau, K, DC, init, tol, decomposition_u.data(), decomposition_u_hat.data(), decomposition_omega.data());

    for (int i = 0; i < K; i++) {
        omega[i] = decomposition_omega[i];
    }
    std::cout << "omega: " << std::endl;
    for (int i = 0; i < K; i++) {
        std::cout << omega[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}


