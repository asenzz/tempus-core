//
// Created by zarko on 19.12.20 ?..
//
#include <iostream>
#include <algorithm>
#include <cmath>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <unistd.h>
#include <complex>
#include "deprecated/cvmd.hpp"

#include "common/Logging.hpp"
#include "common/gpu_handler.hpp"


// #include "osqp.h"
#undef MAX_ITER
#include "common.hpp"

// maximum number of iterations. 500 seems to be enough

#define MAX_VMD_ITERATIONS 500
namespace svr {
#if 0

struct cvmd_frequency_outputs {
    arma::mat       u;
    arma::cx_mat    u_hat;
    arma::mat       omega;
    std::vector<double> last_y = {};
//    OSQPSolver *p_solver = nullptr;
};


OSQPSettings *init_osqp_settings(const svr::common::gpu_context &gtx)
{
    /* Set default settings */
    auto settings = (OSQPSettings *) malloc(sizeof(OSQPSettings));
    if (settings) osqp_set_default_settings(settings);
    settings->polish = 1;
    settings->max_iter = 4000;
    settings->eps_abs = 1.0e-05;
    settings->eps_rel = 1.0e-05;
    settings->alpha = 0.6;

    settings->eps_prim_inf = 1.0e-06;
    settings->eps_dual_inf = 1.0e-06;
    settings->polish_refine_iter = 3;
    settings->deviceId = gtx.id();

    // settings->delta = 1e-06;
#ifdef DEBUG_CVMD
    settings->verbose = 1;
#else
    settings->verbose = 0;
#endif
    return settings;
}

void do_vmd_freqs(
        const arma::mat &signal, const double alpha, const double tau, const int K, const int DC, const int init, const double tol,
        /*outputs */ arma::mat &u, arma::cx_mat &u_hat, arma::mat &omega_plus)
{
    //Initialize u, w, lambda, n
    int save_T = signal.n_elem;

    int T = save_T;
    if (T & 1) LOG4_THROW("Input data size " << T << " must be even!");

    //create the mirrored signal
    arma::mat f_mirror = arma::zeros(1,2 * T);

    __omp_pfor_i(0, T/2, f_mirror(0, i) = signal(0, T / 2 - 1 - i))
    __omp_pfor_i(0, T, f_mirror(0, T / 2 + i) = signal(0, i))
    __omp_pfor_i(0, T/2, f_mirror(0, 3 * T / 2 + i) = signal(0, T-1-i))

    arma::mat f = f_mirror;
    T = 2 * T;

    arma::mat Alpha = alpha * arma::ones(1, K);//provision to use different alpha for the different frequences, at a later stage

    const double fs = 1./ double(save_T);//step

    const int N_max = MAX_VMD_ITERATIONS;//max number of iterations

    u = arma::zeros(T,K);
    const double eps = 2.220446049250313e-16;
    //this may or may not work instead. double eps=std::numeric_limits<T>::epsilon();

    // Time Domain 0 to T (of mirrored signal)
    arma::mat t(1, T);
    __omp_pfor_i(0, T, t(0,i) = (double(i) + 1.) / double(T))
    // Spectral Domain discretization
    arma::mat freqs = t - 0.5 - 1. / double(T);
    omega_plus = arma::zeros(K, 1);
    arma::cx_mat f_hat = common::fftshift(common::matlab_fft(f));
    arma::cx_mat f_hat_plus = f_hat;

    __omp_pfor_i(0, T / 2, f_hat_plus(0, i) = 0)

    switch (init) {
        case 1:{
            __omp_pfor_i(0, K, omega_plus(i, 0) = .5 / K * i);
            break;
        }
        case 2:{
            arma::mat real_ran (K,1);
            __omp_pfor_i(0, K, real_ran(i, 0) = exp(log(fs) + (log(0.5) - log(fs)) * common::randouble()));
            real_ran = arma::sort(real_ran);
            __omp_pfor_i(0, K, omega_plus(i, 0) = real_ran(i, 0));
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
        omega_plus(0,0) = 0;
    }
    /* completed initialization of frequencies */
    arma::cx_mat lambda_hat( arma::zeros(1, freqs.n_elem), arma::zeros(1, freqs.n_elem)); // keeping track of the evolution of lambda_hat with the iterations
    arma::cx_mat u_hat_plus(arma::zeros(T,K), arma::zeros(T,K));
    arma::cx_mat u_hat_plus_old = u_hat_plus;

#ifdef DEBUG_CVMD
    double uDiff{tol+eps};
#else
    double uDiff = tol + eps; //tol+eps must be different from just tol, that is why eps should not be too small
#endif

    arma::cx_mat sum_uk(arma::zeros(1,T),arma::zeros(1, T));
    int n_iter = 0;
#ifdef DEBUG_CVMD
    while  ((n_iter < N_max) && (uDiff > tol)) {
#else
    while(n_iter < N_max && uDiff > tol) {
#endif
        //% update first mode accumulator
        __omp_pfor_i(0, T, sum_uk(0, i) = u_hat_plus_old(i, K-1) + sum_uk(0, i) - u_hat_plus_old(i, 0));

        //% update spectrum of first mode through Wiener filter of residuals
        int k = 0;
        __omp_pfor_i(0, T,
            u_hat_plus(i, k) = (f_hat_plus(0,i) - sum_uk(0,i) - lambda_hat(0,i)/2.)/(1.+Alpha(0,k) * std::pow((freqs(0,i) - omega_plus[k]), 2));
        )

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
            double clk_sum_up = 0;
            double clk_sum_down = 0;
            for (int i=0; i < T / 2; ++i) {
                const auto norm_uhat_plus = std::norm(u_hat_plus(T / 2 + i, k));
                clk_sum_up += freqs(0,T/2+i) * norm_uhat_plus;
                clk_sum_down += norm_uhat_plus;
            }
            omega_plus(0, 0) = clk_sum_up / clk_sum_down;
#endif
        }

        // update of any other mode (k from 1 to K-1), after the first
        for (k=1; k < K; ++k) {
            // accumulator
            __omp_pfor_i(0, T, sum_uk(0, i) = u_hat_plus(i, k - 1) + sum_uk(0, i) - u_hat_plus_old(i, k))
            // mode spectrum
            __omp_pfor_i(0, T, u_hat_plus(i, k) = (f_hat_plus(0, i) - sum_uk(0, i) - lambda_hat(0, i)/2.)/(1.+Alpha(0, k)*std::pow(freqs(0, i) - omega_plus[k], 2)))

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
            omega_plus(k, 0) = clk_sum_up / clk_sum_down;
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
    arma::cx_mat zmat (arma::zeros(T, K), arma::zeros(T, K));
    u_hat = zmat;
    __omp_pfor_i(0, T / 2,
        for (int k = 0; k < K; ++k)
            u_hat(T / 2 + i, k) = u_hat_plus(T / 2 + i, k) )
    // here it is not clear why, but T/2 is set both above and below. Instead, the 0 is set to conj of T-1
    __omp_pfor_i(0, T / 2, for (int k=0; k < K; ++k) u_hat(T / 2 - i, k) = std::conj(u_hat_plus(T / 2 + i, k)) )
    for (int k = 0; k < K; ++k) u_hat(0, k) = std::conj(u_hat(T - 1, k));

    arma::mat u_big = arma::zeros(K, t.n_elem);
    __omp_pfor(k, 0, K, u_big.rows(k, k) = arma::trans(arma::real(common::matlab_ifft(common::ifftshift(u_hat.cols(k, k))))); )
    u = arma::zeros(K,T/2);
    __omp_pfor_i(0, T / 2, for (int k = 0; k < K; ++k) u(k, i) = u_big(k, T / 4 + i); )
    //recompute spectrum
    //clear u_hat;
    u_hat = arma::cx_mat(arma::zeros(K,T/2),arma::zeros(K,T/2));
    __omp_pfor(k, 0, K, u_hat.rows(k, k) = common::fftshift(common::matlab_fft(u.rows(k,k))) )
    omega_plus /= OMEGA_DIVISOR;

    LOG4_TRACE("Omega is " << omega_plus);
}

// TODO Add column and table name
cvmd::cvmd(const size_t _levels)
        : spectral_transform(std::string("cvmd"), _levels), levels(_levels)
{
    if (levels < 2 or levels % 2) LOG4_THROW("Invalid number of levels " << levels);
}

cvmd::~cvmd()
{
    uninitialize(true);
}

cvmd_frequency_outputs
calculate_vmd_frequencies(const std::vector<double> &input, const size_t levels_half)
{
    LOG4_DEBUG("Calculating VMD frequencies for " << levels_half << " half levels.");
    arma::rowvec x_signal;
    if (input.size() % 2) LOG4_WARN("Odd signal length " << input.size() << ", trimming first value.");
    x_signal = input.size() % 2 ? arma::rowvec(&*std::next(input.begin()), input.size() - 1) : input;
    x_signal *= CVMD_INPUT_MULTIPLIER;

    /* constants */
    const double alpha = ALPHA_BINS;//bands
    const double tau = TAU_FIDELITY;//fidelity - 0 means no strict enforcement of decomposition. In reality we should use something like 0.1 or higher.
    const int DC = 0; //has a DC component or not.
    const int K = levels_half; //number of modes/frequencies. because of DC=1 we use 1 more than the natural number of frequencies (3 in the signal above).
    const int init = 1; //kind of initialization, let's use 1 for now.
    const double tol = 1e-7; //some tolerance
    /* end constants*/
    /* outputs*/
    arma::mat u;
    arma::cx_mat u_hat;
    arma::mat omega;
    /* end outputs*/

    do_vmd_freqs(x_signal, alpha, tau, K, DC, init, tol, u, u_hat, omega);
#ifdef DEBUG_CVMD
    {
        std::stringstream ss_input;
        for (size_t i = 0; i < T; ++i) ss_input << x_signal(0, i) << ", ";
        LOG4_FILE("cvmd_input.csv", ss_input.str());
    }
    {
        std::stringstream ss_omega;
        for (size_t i = 0; i < omega.n_rows; ++i) ss_omega << omega(i, 0) << ", ";
        LOG4_FILE("cvmd_omega.csv", ss_omega.str());
    }
#endif

    return {u, u_hat, omega};
}

double
object_compute(
        int N, int levels,
        std::vector<std::vector<double>> &f_u_mult, std::vector<std::vector<double>> &f_v_mult,
        double *test_sol, double *f_lambda, bool use_left,
        const std::vector<double> &left_u_values, const std::vector<double> &left_v_values)
{
    double object0=0.;
    for(int i=(use_left?-1:0);i<N-1;i++){
        double s0=0.;
        for (int j=0 ; j<levels; ++j){
            double um1 = (use_left && (i==-1)) ? left_u_values[j] : test_sol[2*(i+j*N)+0 ];
            double vm1 = (use_left && (i==-1)) ? left_v_values[j] : test_sol[2*(i+j*N)+1 ];
            int ip1=i + 1;

            double u = test_sol[2*(ip1+j*N) + 0];
            double v = test_sol[2*(ip1+j*N) + 1];

            std::complex<double> xm1{um1, vm1};
            std::complex<double> x{u, v};
            std::complex<double> xm1_adjusted = xm1 * std::complex<double>(f_u_mult[j][i+(use_left?1:0)], f_v_mult[j][i+(use_left?1:0)]);
            std::complex<double> x_adjusted = x * std::complex<double>(f_u_mult[j][i+1+(use_left?1:0)], f_v_mult[j][i+1+(use_left?1:0)]);

            double diff_norm = std::norm( x_adjusted - xm1_adjusted);
            double quant = f_lambda[j]*diff_norm;
            s0+=quant;
        }
        object0+=s0;
    }
    return object0;
}

//interface without armadillo
int
vmd_decomp(const size_t T, const double *signal, double alpha, const double tau, const size_t K, int DC, int init , double tol,
        /*outputs */ double* decomposition_u ,/*complex*/ double*decomposition_u_hat, double * decomposition_omega)
{
    /*caller must allocate K*T*sizeof(double) for decomposition_u, 2*K*T*sizeof(double) for decomposition_u_hat (which is complex), K*sizeof(double) for decomposition_omega_plus*/

    arma::mat x_signal(1, T);
    for (size_t i=0;i<T;i++) {
        x_signal(0,i)=signal[i];
    }
    arma::mat u;
    arma::cx_mat u_hat;
    arma::mat omega;
    do_vmd_freqs(x_signal, alpha, tau, K, DC, init, tol, u, u_hat, omega);
    /*cilk_*/for(size_t k=0;k<K;k++){
        for(size_t i=0;i<T;i++){
            decomposition_u[k*T+i] = u(k,i);
        }
    }
    /*cilk_*/for (size_t k=0;k<K;k++){
        for(size_t i=0;i<T;i++){
            decomposition_u_hat[2*(k*T+i)+0] = std::real(u_hat(k,i));
            decomposition_u_hat[2*(k*T+i)+1] = std::imag(u_hat(k,i));
        }
    }
    /*cilk_*/for(size_t k=0;k<K;k++){
        decomposition_omega[k]=omega(k,0);
    }
    return 0;
}

void reset_solver(OSQPSolver *p_solver)
{
    OSQPVectorf_set_scalar(p_solver->work->x_prev, 0);
    OSQPVectorf_set_scalar(p_solver->work->z_prev, 0);
    OSQPVectorf_set_scalar(p_solver->work->z, 0);
    OSQPVectorf_set_scalar(p_solver->work->D_temp, 0);
    OSQPVectorf_set_scalar(p_solver->work->D_temp_A, 0);
    OSQPVectorf_set_scalar(p_solver->work->E_temp, 0);
    OSQPVectorf_set_scalar(p_solver->work->xz_tilde, 0);
    OSQPVectorf_set_scalar(p_solver->work->xtilde_view, 0);
    OSQPVectorf_set_scalar(p_solver->work->ztilde_view, 0);
    OSQPVectorf_set_scalar(p_solver->work->delta_y, 0);
    OSQPVectorf_set_scalar(p_solver->work->Atdelta_y, 0);
    OSQPVectorf_set_scalar(p_solver->work->delta_x, 0);
    OSQPVectorf_set_scalar(p_solver->work->Pdelta_x, 0);
    OSQPVectorf_set_scalar(p_solver->work->Adelta_x, 0);
    OSQPVectorf_set_scalar(p_solver->work->scaling->D, 0);
    OSQPVectorf_set_scalar(p_solver->work->scaling->Dinv, 0);
    OSQPVectorf_set_scalar(p_solver->work->scaling->E, 0);
    OSQPVectorf_set_scalar(p_solver->work->scaling->Einv, 0);
    p_solver->work->linsys_solver->reset(p_solver->work->linsys_solver);
}

void
do_problem(
#ifdef USE_CUDA
        OSQPSolver *&p_solver, const OSQPSettings *p_settings,
#endif
        const size_t levels_half,
        const std::vector<std::vector<double>> &phase_series,
        const std::vector<double> &x_series,
        const size_t N, // Number of time steps does not need to equal x_series.size()
#ifdef DEBUG_CVMD
        const std::vector<std::vector<double>> &u_series, const std::vector<std::vector<double>> &v_series,
#endif
        std::vector<std::vector<double>> &decon,
        cvmd_frequency_outputs &found_freqs,
        const std::vector<double> &prev_decon = {}
)
{
    if (not prev_decon.empty() and x_series.size() != 1)
        THROW_EX_FS(std::invalid_argument, "Online VMD should only be done with 1 row per call!"); // TODO For now, improve

    std::vector<std::vector<double>> f_u_mult(levels_half), f_v_mult(levels_half);
    //const std::vector<double>  f_lambda(levels_half, 1.);
    /*
    for(size_t i = 0; i < levels_half;i++){
        //f_lambda[i]=pow(0.5,2*i); this was good
        f_lambda[i]=1.;
    }
    */

#if 0
    std::vector<std::vector<double>>  f_mu(levels_half);
    for (size_t i = 0; i<levels_half; ++i) {
        f_mu[i].resize(N, 0.);
        /*
        for(int j = 0; j<N; ++j){
            f_mu[i][j]=0.;//not used for now
        }
         */
    }
#endif
    const auto levels = 2 * levels_half;
    /*cilk_*/for(size_t l = 0; l < levels_half; ++l) {
        f_u_mult[l].resize(N + 1); // 2*M_PI*omega
        f_v_mult[l].resize(N + 1);
    }
    // Phase_series is computed on basis of the omega. In fact, it is not necessary to know the exact phase at a
    // given point, knowing the omega, i.e., difference in phase, is enough for this computation
    // further enhancement can be achieved with varying the phases or the differences between phases - for future work.
    /*cilk_*/for(size_t l = 0; l < levels_half; ++l) {
        for (size_t t = 0; t < N + 1; ++t) {
            f_u_mult[l][t] = cos(-phase_series[t][l]);
            f_v_mult[l][t] = sin(-phase_series[t][l]);
        }
    };

    std::vector<double> B_matrix_val;
    std::vector<int> B_matrix_i;
    std::vector<int> B_matrix_j;

    // u and v are the unknowns,
    // total number is levels_half * 2 * N
    //
    const size_t total_number = levels * N;
    for (size_t cntr = 0; cntr < total_number; ++cntr){
        const int timestep = (cntr / 2 ) % N; //which timestep
        const int level = (cntr / 2) / N;
        double val;
        if ((size_t) timestep == N-1) {
            B_matrix_i.push_back(cntr);
            B_matrix_j.push_back(cntr);
            val = 1.;//1.*f_lambda[level]+1.*f_mu[level][timestep];
            B_matrix_val.push_back(val);
            continue;
        }

        if (cntr  % 2 == 0) {//u
            B_matrix_i.push_back(cntr);
            B_matrix_j.push_back(cntr);
            val = f_u_mult[level][timestep]*f_u_mult[level][timestep] + f_v_mult[level][timestep]*f_v_mult[level][timestep];
            //val+= 1.*f_mu[level][timestep];
            if (timestep > 0 or not prev_decon.empty()) { //changed here to be able to deal with left values
                val += 1.; //this should be simply 1., no need to compute cos^2+sin^2
            }
            B_matrix_val.push_back(val);//*f_lambda[level]);
            B_matrix_i.push_back(cntr);//itself
            B_matrix_j.push_back(cntr+2);//u+1 for this u
            val = -2 * f_u_mult[level][timestep+1]*f_u_mult[level][timestep];
            val += -2 * f_v_mult[level][timestep+1]*f_v_mult[level][timestep];
            B_matrix_val.push_back(val*0.5);//f_lambda[level]/2.);//due to the symmetry
            B_matrix_i.push_back(cntr);//itself
            B_matrix_j.push_back(cntr+3);//v+1 for this u
            val = 2* f_u_mult[level][timestep]*f_v_mult[level][timestep+1];
            val += -2* f_v_mult[level][timestep]*f_u_mult[level][timestep+1];
            B_matrix_val.push_back(val*0.5);//f_lambda[level]/2);//symmetry
        } else {
            B_matrix_i.push_back(cntr);
            B_matrix_j.push_back(cntr);
            val = f_v_mult[level][timestep] * f_v_mult[level][timestep] + f_u_mult[level][timestep] * f_u_mult[level][timestep];//==1.
            //val+= 1.*f_mu[level][timestep];
            if (timestep > 0 or not prev_decon.empty()) {//changed here to be able to deal with left values
                val += 1.;
            }
            B_matrix_val.push_back(val);//*f_lambda[level]);
            /*
             with the corresponding u - is 0 by conjugates
             */
            B_matrix_i.push_back(cntr);//itself
            B_matrix_j.push_back(cntr+1);//u+1 for this v
            val = 2 * f_u_mult[level][timestep+1] * f_v_mult[level][timestep];
            val += -2 * f_v_mult[level][timestep+1] * f_u_mult[level][timestep];
            B_matrix_val.push_back(val * 0.5);//f_lambda[level]/2.);//symmetry
            B_matrix_i.push_back(cntr);//itself
            B_matrix_j.push_back(cntr + 2);//v+1 for this v
            val = -2 * f_v_mult[level][timestep] * f_v_mult[level][timestep+1] ;
            val += -2 * f_u_mult[level][timestep] * f_u_mult[level][timestep+1];
            B_matrix_val.push_back(val * 0.5);//f_lambda[level]/2.);//symmetry
        }
    }
    std::vector<int> index(B_matrix_j.size());
    std::iota(index.begin(),index.end(),0); //Initializing
    std::sort(index.begin(),index.end(), [&](int i, int j) { return B_matrix_j[i]<B_matrix_j[j]; } );

    std::vector<c_int> C_matrix_i(B_matrix_i.size());
    std::vector<c_float> C_matrix_val(B_matrix_val.size()); // P_x_vector
    std::vector<c_int> C_matrix_j;//empty
    int col = -1;
    for (size_t i = 0; i < B_matrix_i.size(); ++i) {
        if (B_matrix_j[index[i]] != col){
            if ((col - B_matrix_j[index[i]])!=-1)
                THROW_EX_FS(std::runtime_error, "CVMD Error " << col << " " << B_matrix_j[index[i]]);
            col = B_matrix_j[index[i]];
            C_matrix_j.push_back(i);
        }
        C_matrix_i[i]=B_matrix_i[index[i]];
        C_matrix_val[i]=B_matrix_val[index[i]];
    }
    C_matrix_j.push_back(B_matrix_i.size());

    //c_int P_nnz = C_matrix_val.size();
    c_int n = total_number;

    //c_int *P_p = C_matrix_j.data();
    //c_int *P_i = C_matrix_i.data();
    //c_float *P_x = C_matrix_val.data();
    //the real parts of the first elements are to be fixed to their matlab generated decomposed u
    //for this fixing we need conditions, to set them equal to the matlab results.
    //for the corresponding sums, we can not fix them to the x_series , because matlab does not solve exactly

    c_int A_nnz = levels_half * N; //this does not change , only the number of equations change. instead of one, we have many separate
    std::vector<c_int> pre_A_i_vector(A_nnz);
    std::vector<c_float> pre_A_x_vector(A_nnz,1.);
    std::vector<c_int> pre_A_j_vector(A_nnz);
    /*cilk_*/for(size_t i = 0; i < (decltype(i))A_nnz; ++i) {
        const int timestep = i % N  ; //which timestep
        const int level = i / N ;
        pre_A_i_vector[i] = timestep;
        //pre_A_x_vector [i]=1.; - already done
        pre_A_j_vector[i] = (2 * timestep + 0) + 2 * N * level;
        //std::cout <<  "preA " << pre_A_i_vector[i] << " "<< pre_A_j_vector[i]<< std::endl;
    };

    std::vector<c_int> A_i_vector(A_nnz);
    std::vector<c_float> A_x_vector(A_nnz,1.);
    std::vector<c_int> A_p_vector;

    index.resize(A_nnz);
    std::iota(index.begin(), index.end(),0); //Initializing
    std::sort(index.begin(), index.end(),[&](int i,int j){return pre_A_j_vector[i]<pre_A_j_vector[j];} );

    col = -1;
    for (size_t i = 0; i < pre_A_i_vector.size(); ++i) {
        if (pre_A_j_vector[index[i]] != col) {
            for (int filler = col+1; filler < pre_A_j_vector[index[i]]; ++filler) {
                A_p_vector.push_back(i);
            }
            col = pre_A_j_vector[index[i]];
            A_p_vector.push_back(i);
        }
        A_i_vector[i]=pre_A_i_vector[index[i]];
        A_x_vector[i]=pre_A_x_vector[index[i]];
    }

    A_p_vector.push_back(A_nnz);
    A_p_vector.push_back(A_nnz);

    //c_int *A_i = A_i_vector.data();
    //c_float *A_x = A_x_vector.data();
    //c_int *A_p = A_p_vector.data();
    std::vector<c_float> lo_vector(N);
    std::vector<c_float> up_vector(N);
    /*cilk_*/for(size_t i = 0; i < N; ++i) {
        lo_vector[i] = CVMD_INPUT_MULTIPLIER * x_series[i] - EPSI_TOL_VMD;
        up_vector[i] = CVMD_INPUT_MULTIPLIER * x_series[i] + EPSI_TOL_VMD;
    };
    //c_float *lo = lo_vector.data();
    //c_float *up = up_vector.data();
    std::vector<c_float> q_vector(total_number,0.);

    // second place where there is difference because of left values
    if (not prev_decon.empty()) {
        std::vector<double> cos_delta_phi(levels_half, 0.);
        std::vector<double> sin_delta_phi(levels_half, 0.);
        /*cilk_*/for (size_t l = 0; l < levels_half; ++l){
            // in principle fromN-1 and fromN should be used, but sometimes function can be invoked with fromN equal to 0. Only the difference in phases at the start is needed here.
            cos_delta_phi[l] = cos(phase_series[1][l] - phase_series[0][l]); // in principle fromN-1 and fromN should be used, but sometimes function can be invoked with fromN =0.
            sin_delta_phi[l] = sin(phase_series[1][l] - phase_series[0][l]); // TODO Verify with Emanouil
        }
        /*cilk_*/for (size_t t = 0; t < total_number; ++t) {
            if ((t / 2) % N == 0) {
                const auto level = (t / 2) / N;
                if (t % 2 == 0) {
                    q_vector[t] = -2. * prev_decon[2 * level] * cos_delta_phi[level] + 2. * prev_decon[2 * level + 1] * sin_delta_phi[level];
                } else {
                    q_vector[t] = -2. * prev_decon[2 * level] * sin_delta_phi[level] - 2. * prev_decon[2 * level + 1] * cos_delta_phi[level];
                }
                q_vector[t] = q_vector[t] / 2.;
            }
        }
    }

    //c_float *q = q_vector.data();

#ifdef DEBUG_CVMD
    {  /* little testing perhaps */
        std::vector<double> test_sol(total_number);
        for(size_t i=0;i<total_number;i++){
            int timestep = (i / 2 ) % N  ; //which timestep
            int u_or_v =  i  % 2;
            int level = (i / 2) / N ;
            if (u_or_v==0){
                if (fromN+timestep < u_series[level].size()){
                    test_sol[i]=u_series[level][fromN+timestep];
                }else{
                    test_sol[i]=u_series[level][ u_series[level].size()-1];
                }
            }else{
                if (fromN+timestep < v_series[level].size()){
                    test_sol[i]=v_series[level][fromN+timestep];
                }else{
                    test_sol[i]=v_series[level][ v_series[level].size()-1];
                }
            }
        }
#if 0
        {
            std::stringstream ss;
            for (const auto i: test_sol) ss << ", " << i;
            LOG4_FILE("cvmd_ptest_sol.txt", ss.str().c_str());
        }
        {
            std::stringstream ss;
            for (size_t j = 0; j < M; ++j) {
                ss << "\nLevel " << j;
                for (const auto i: u_series[j]) ss << ", " << i;
            }
            LOG4_FILE("cvmd_u_series.txt", ss.str().c_str());
        }
        {
            std::stringstream ss;
            for (size_t j = 0; j < M; ++j) {
                ss << "\nLevel " << j;
                for (const auto i: f_u_mult[j]) ss << ", " << i;
            }
            LOG4_FILE("cvmd_f_u_mult.txt", ss.str().c_str());
        }
        {
            std::stringstream ss;
            for (size_t j = 0; j < M; ++j) {
                ss << "\nLevel " << j;
                for (const auto i: f_v_mult[j]) ss << ", " << i;
            }
            LOG4_FILE("cvmd_f_v_mult.txt", ss.str().c_str());
        }
        {
            std::stringstream ss;
            for (const auto i: B_matrix_val) ss << ", " << i;
            LOG4_FILE("cvmd_B_matrix_val.txt", ss.str().c_str());
        }
        {
            std::stringstream ss;
            for (const auto i: B_matrix_i) ss << ", " << i;
            LOG4_FILE("cvmd_B_matrix_i.txt", ss.str().c_str());
        }
        {
            std::stringstream ss;
            for (const auto i: B_matrix_j) ss << ", " << i;
            LOG4_FILE("cvmd_B_matrix_j.txt", ss.str().c_str());
        }
#endif
        double object0=object_compute( N, levels_half, f_u_mult,f_v_mult, test_sol.data(),f_lambda.data(), use_left, left_u_values, left_v_values);
        std::cout << "object0" << object0<<std::endl;


        //how to multiply as x' * P  * x + q*x

        double object1=0;
        std::vector<double> p_times_x(total_number,0.);
        for(size_t i=0;i<total_number;i++){
            for(int elem_ptr = P_p[i];elem_ptr < P_p[i+1];elem_ptr++){
                double val_elem = P_x[elem_ptr];
                int    row_elem = P_i[elem_ptr];
                if (row_elem > (int) i) throw std::runtime_error(svr::common::formatter() << "row_elem > i" << row_elem << " > " << i);
                double product = val_elem * test_sol [i];

                if (row_elem < (int) i){
                    p_times_x [row_elem] += product;
                    product = val_elem * test_sol[row_elem];
                    p_times_x [i] += product;
                }else{
                    p_times_x [row_elem] += product;
                }
            }
        }
        for(int i=0;i<(int) total_number;i++){
            object1+= test_sol[i]*  p_times_x [i];
        }
        if (use_left){
            //double s0=0.;
            for(size_t i=0;i<total_number;i++){
                //int timestep = (i / 2 ) % N  ; //which timestep. In reality if timestep is >=1, the q_vector is 0
                //int u_or_v =  i  % 2;
                int level = (i / 2) / N ;

                object1+= test_sol[i]* q_vector[i]*f_lambda[level]*2;//be careful - this had to be multiplied by 2. to make object0 and object1 equal.
            }
            for(size_t i=0;i<levels_half;i++){
                object1+=pow(left_u_values[i],2)*f_lambda[i];
                object1+=pow(left_v_values[i],2)*f_lambda[i];
            }
        }
        std::cout << "object1 must be the same as object0 - this is just to see the matrix is right"<<std::endl;
        std::cout << "object1" << object1<<std::endl;
        if (fabs(object1-object0) > 0.00000001)
            throw std::runtime_error(svr::common::formatter() <<
                    "Object1 must be the same as object0 - this is just to see the matrix is right object1 " <<
                    object1 << " object0 " << object0);
#if 0
        for(int i=0;i<total_number;i++){
            int timestep = (i / 2 ) % N  ;
            int u_or_v =  i  % 2;
            int level = (i / 2) / N ;
            if (u_or_v ==0) {
                test_sol[i]=x_series[timestep]/levels_half;
            }else{
                test_sol[i]=1.;//just for the test, !=0
            }
        }
        //multiply by A
        std::vector<double> a_times_x(num_equations,0.);
        for(int i=0;i<total_number;i++){
            for(int elem_ptr = A_p[i];elem_ptr < A_p[i+1];elem_ptr++){
                double val_elem = A_x[elem_ptr];
                int    row_elem = A_i[elem_ptr];
                double product = val_elem * test_sol [i];
                a_times_x [row_elem] += product;
            }
        }
#endif // 0
    }
#endif // DEBUG_CVMD

#ifdef USE_CUDA
    csc *P = nullptr, *A = nullptr;
    //bool cleanup_solver = false;
    if (p_solver == nullptr or prev_decon.empty()) {
    //    cleanup_solver = true;

        /* Workspace, p_settings, matrices */
        P = (csc *) malloc(sizeof(csc));
        A = (csc *) malloc(sizeof(csc));

        /* Populate matrices */
        csc_set_data(A, N, n, A_nnz, A_x_vector.data(), A_i_vector.data(), A_p_vector.data());
        csc_set_data(P, n, n, C_matrix_val.size(), C_matrix_val.data(), C_matrix_i.data(), C_matrix_j.data());

        if (not p_settings) LOG4_THROW("OSQP settings not initialized.");
        /* Setup workspace */
        if (p_solver) osqp_cleanup(p_solver);
        const auto osqp_err = osqp_setup(&p_solver, P, q_vector.data(), A, lo_vector.data(), up_vector.data(), N, n, p_settings);
        if (osqp_err) THROW_EX_FS(std::runtime_error, "OSQP setup terminated with error " << osqp_err);
    } else if (not prev_decon.empty()) {
        osqp_update_P_A(p_solver, C_matrix_val.data(), C_matrix_i.data(), C_matrix_val.size(), A_x_vector.data(), A_i_vector.data(), A_x_vector.size());
        osqp_update_lin_cost(p_solver, q_vector.data());
        osqp_update_bounds(p_solver, lo_vector.data(), up_vector.data());
        //if (found_freqs.last_y.empty()) LOG4_WARN("Last Y is empty!");
        osqp_warm_start(p_solver, prev_decon.data(), found_freqs.last_y.empty() ? OSQP_NULL : found_freqs.last_y.data());
    }

/* TODO - 1/2. not implemented - careful if adding q */
    /* Solve Problem */
    //PROFILE_EXEC_TIME(osqp_solve(p_solver), "osqp_solve of size " << C_matrix_val.size() << " " << A_x_vector.size());
    osqp_solve(p_solver);

    c_float *real_solution = p_solver->solution->x;
    if (decon.size() != N) decon.resize(N);
    //we write inside of it
    for (size_t t = 0; t < N; ++t) { // timestep
        if (t == N - 1 or prev_decon.empty()) {
            decon[t].resize(levels);
            /*cilk_*/for (size_t l = 0; l < levels_half; ++l) {
                decon[t][2 * l] = real_solution[2 * t + 0 + 2 * l * N];
                decon[t][2 * l + 1] = real_solution[2 * t + 1 + 2 * l * N];
            }
        }
    }

#ifdef FAST_CVMD
    if (not prev_decon.empty() or N > 1) {
        if (found_freqs.last_y.size() != levels) found_freqs.last_y.resize(levels);
        //BOOST_ASSERT(sizeof(double) == sizeof(c_float));
        memcpy(found_freqs.last_y.data(), p_solver->solution->y + 2 * (N - 1) * levels, found_freqs.last_y.size() * sizeof(double));
    }

/* Clean workspace */
    if (prev_decon.empty())
#endif
    {
        osqp_cleanup(p_solver);
        p_solver = nullptr;
    }
    if (A) {
        free(A);
        free(P);
    }
#else // Do it on CPU // TODO Fix build and test!
    // Workspace structures
    OSQPWorkspace *work = nullptr;
    auto p_settings = (OSQPSettings *)c_malloc(sizeof(OSQPSettings));
    auto data     = (OSQPData *)c_malloc(sizeof(OSQPData));

    // Populate data
    if (data) {
        data->n = n;
        data->m = m;
        data->P = csc_matrix(data->n, data->n, P_nnz, P_x, P_i, P_p);
        data->q = q;
        data->A = csc_matrix(data->m, data->n, A_nnz, A_x, A_i, A_p);
        data->l = lo;
        data->u = up;
    }

    // Define solver p_settings as default
    if (p_settings) {
        osqp_set_default_settings(p_settings);
#ifdef DEBUG_CVMD
        p_settings->verbose = 1;
#else
        p_settings->verbose = 0;
#endif
        p_settings->polish = 0;
        p_settings->max_iter = 4000;
        p_settings->eps_abs = 1.0e-05;
        p_settings->eps_rel = 1.0e-05;
        p_settings->alpha=0.6;
        p_settings->eps_prim_inf = 1.0e-06;
        p_settings->eps_dual_inf = 1.0e-06;
    }

    // Setup workspace
    exitflag = osqp_setup(&work, data, p_settings);
    if (exitflag) throw std::runtime_error(svr::common::formatter() << "OSQP terminated with error " << exitflag);
    // Solve Problem
    osqp_solve(work);
    OSQPSolution *sol = work->solution;
    c_float *real_solution = sol->x;

    sol_u_series.resize(levels_half);
    sol_v_series.resize(levels_half);
    //we write inside of it
    /* cilk_ */ for (size_t i = 0; i < levels_half; ++i){
        sol_u_series[i].resize(N);
        sol_v_series[i].resize(N);
        /* cilk_ */ for (size_t j=0;j<N;++j) {//timestep
            if (j == N-1 || !use_left){
                sol_u_series[i][j] = real_solution[2 * j + 0 + 2 * i * N];
                sol_v_series[i][j] = real_solution[2 * j + 1 + 2 * i * N];
            }
        }
    }

    // Cleanup
    if (data) {
        if (data->A) c_free(data->A);
        if (data->P) c_free(data->P);
        c_free(data);
    }
    if (p_settings) c_free(p_settings);
#endif // USE_CUDA
}

bool cvmd::initialized(const std::string &decon_queue_table_name)
{
    const auto freq_key = freq_key_t(decon_queue_table_name, levels);
    return not vmd_frequencies.empty() and vmd_frequencies.find(freq_key) != vmd_frequencies.end();
}

#define SOLVER_FILE_NAME SAVE_OUTPUT_LOCATION << "/" << std::get<0>(f.first) << "_solver.state"

void cvmd::load_solvers_file()
{
#if 0
    for (const auto &f: vmd_frequencies)
        osqp_load_solver(f.second.p_solver, std::string{svr::common::formatter() << SOLVER_FILE_NAME}.c_str());
#endif
}

void cvmd::initialize(const std::vector<double> &input, const std::string &decon_queue_table_name, const bool load_solvers)
{
    LOG4_DEBUG("Initializing omega on " << input.size() << " rows, for " << levels << " levels, " << decon_queue_table_name << " table.");
    const auto freq_key = freq_key_t(decon_queue_table_name, levels);
    static std::mutex mx;
    if (vmd_frequencies.empty() or vmd_frequencies.find(freq_key) == vmd_frequencies.end()) {
        const auto freqs = calculate_vmd_frequencies(input, levels / 2);
        std::scoped_lock lg(mx);
        vmd_frequencies[freq_key] = freqs;
    }
    if (load_solvers) load_solvers_file();
}

void cvmd::uninitialize(const bool save)
{
#if 0
    if (save)
        for (const auto &f: vmd_frequencies)
            osqp_save_solver(f.second.p_solver, std::string{svr::common::formatter() << SOLVER_FILE_NAME}.c_str());
#endif

    vmd_frequencies.clear();
}

void cvmd::transform(
        const std::vector<double> &input,
        std::vector<std::vector<double>> &decon,
        const size_t padding = 0)
{
    THROW_EX_FS(std::logic_error, "Not implemented!");
}

// Online VMD transform
void cvmd::transform(
        const std::vector<double> &input,
        std::vector<std::vector<double>> &decon,
        const std::string &decon_queue_table_name,
        const std::vector<double> &prev_decon,
        std::vector<std::vector<double>> &phase_series,
        const std::vector<double> &start_phase)
{
    if (vmd_frequencies.empty()) LOG4_THROW("VMD frequencies empty!");
    const auto found_freqs = vmd_frequencies.find(freq_key_t(decon_queue_table_name, levels));
    if (found_freqs == vmd_frequencies.end())
        LOG4_THROW("VMD frequencies not found!");
    const size_t levels_half = levels / 2;
    if (start_phase.size() != levels_half)
        LOG4_THROW("Phase start size is not correct " << start_phase.size());

    const size_t T = input.size();
    LOG4_DEBUG("Online input size " << T << ", levels count " << levels_half);

    const auto &omega = found_freqs->second.omega;
    if (phase_series.size() != T + 1) phase_series.resize(T + 1);
    /*cilk_*/for (size_t t = 1; t < T + 1; ++t) {
        phase_series[t].resize(levels_half);
    };
    phase_series[0] = start_phase; // This is correct with the current setup and not: + omega[l] * 2. * M_PI;
    /*cilk_*/for (size_t l = 0; l < levels_half; ++l) {
        for (size_t t = 1; t < T + 1; ++t) {
            phase_series[t][l] = phase_series[t - 1][l] + omega[l] * 2. * M_PI;
        }
    };

#ifdef FASTCVMD
    const OSQPSettings *p_osqp_settings = nullptr;
#else
    svr::common::gpu_context gtx;
    const OSQPSettings *p_osqp_settings = init_osqp_settings(gtx);
#endif
    OSQPSolver *p_solver = nullptr;
    if (not decon.empty()) decon.clear();
    std::vector<std::vector<double>> decon_slice(1, std::vector<double>(levels));
    do_problem(
            p_solver /* found_freqs->second.p_solver */, p_osqp_settings,
            levels_half, {phase_series[0], phase_series[1]}, {input[0]}, 1,
            decon_slice, found_freqs->second, prev_decon);
    decon.emplace_back(decon_slice[0]);
    for (size_t t = 1; t < T; ++t) {
        do_problem(
                p_solver /* found_freqs->second.p_solver */, p_osqp_settings,
                levels_half, {phase_series[t], phase_series[t + 1]}, {input[t]}, 1,
                decon_slice, found_freqs->second, decon[t - 1]);
        decon.emplace_back(decon_slice[0]);
    }
    osqp_cleanup(p_solver);

#if 0
    {
        static size_t call_ct;
        std::stringstream ss;
        std::stringstream ss_file_name;
        ss_file_name << "cvmd_decon_online_" << call_ct++ << ".csv";
        for (size_t i = 0; i < decon.size(); ++i) { // i = timesteps
            for (size_t j = 0; j < decon[i].size(); ++j) // j = levels
                ss << ", " << decon[i][j];
            ss << std::endl;
        }
        LOG4_FILE(ss_file_name.str().c_str(), ss.str().c_str());
    }
#endif
    if (p_osqp_settings) free((void *) p_osqp_settings);
}

#ifdef SAVE_DECON

size_t
load_decon(
        std::vector<std::vector<double>> &decon,
        std::vector<std::vector<double>> &phase_series,
        const std::string &table_name,
        const size_t levels_half)
{
    std::ifstream ifs(common::formatter() << SAVE_OUTPUT_LOCATION << table_name << "_decon_values_phases.tsv");
    if (ifs.fail()) return 0;
    LOG4_DEBUG("Loading " << levels_half << " levels of data for decon queue " << table_name << " from file.");
    std::string line;
    size_t t = 0;
    phase_series.clear();
    decon.clear();
    while (std::getline(ifs, line))
    {
        std::vector<std::string> row_values;
        common::split(line, '\t', row_values);
        if (row_values.size() < 3 * levels_half) continue;
        size_t col_ix =  0;
        decon.emplace_back(std::vector<double>(levels_half * 2));
        phase_series.emplace_back(std::vector<double>(levels_half));
        try {
            for (size_t l = 0; l < levels_half * 2; ++l)
                decon.back()[l] = stold(row_values[col_ix++]);
            for (size_t l = 0; l < levels_half; ++l)
                phase_series.back()[l] = stold(row_values[col_ix++]);
        } catch (const std::exception &ex) {
            LOG4_ERROR("Failed parsing " << row_values[--col_ix] << " last good row " << t);
        }
        ++t;
    }
    LOG4_DEBUG("Loaded " << t << " values");
    return t;
}

#endif

#define MAX_GPU_SIZE 65536

// Batch transform
void cvmd::transform(
        const std::vector<double> &input,
        std::vector<std::vector<double>> &decon,
        const std::string &table_name,
        std::vector<std::vector<double>> &phase_series,
        const size_t count_from_start)
{
    if (vmd_frequencies.empty()) THROW_EX_FS(std::runtime_error, "Empty VMD frequencies!");

    const auto T = input.size();
    const size_t levels_half = levels / 2;
    const size_t R = get_residuals_length("__DUMMY__") > MAX_GPU_SIZE ? MAX_GPU_SIZE : get_residuals_length("__DUMMY__"); // Limit imposed by GPU RAM

    if (T < R) THROW_EX_FS(std::invalid_argument, "Input size " << T << " must contain at least " << R << " rows.");

    const auto found_freqs = vmd_frequencies.find({table_name, levels});
    if (found_freqs == vmd_frequencies.end())
        THROW_EX_FS(std::runtime_error, "VMD frequencies not found!");

    const auto &omega = found_freqs->second.omega;

    LOG4_DEBUG("Input size " << T << ", levels count " << levels_half << ", residuals length " << R << " count from start " << count_from_start << ", omega " << omega);

#ifndef USE_CUDA
    const auto &decomposition_u_series = vmd_frequencies[freq_key].u;
#endif

    // levels should be organized
    // u 0
    // v 0
    // u 1
    // v 1
    // 32 levels for 16 frequencies

    // Now do it batch and then online, by first computing without left values for half of the time and then continue with online
    svr::common::gpu_context gtx;
    auto p_osqp_settings = init_osqp_settings(gtx);

#ifdef SAVE_DECON
    auto loaded_decon = load_decon(decon, phase_series, table_name, levels_half);
    if (loaded_decon > T) {
        LOG4_ERROR("Loaded decon " << loaded_decon << " is larger than expected " << T << ", discarding loaded values.");

        decon.clear();
        phase_series.clear();
        loaded_decon = 0;
    }
#else
    size_t loaded_decon = 0;
#endif

    if (loaded_decon < 1 or phase_series.size() != T + 1 or phase_series[0].size() != levels_half) {
        LOG4_DEBUG("Initializing phase series of length " << T + 1 << ", half levels " << levels_half);

        std::vector<double> phase_start(levels_half, 0.);
        size_t t_phase_start = 0;
        if (phase_series.size() > 0 and phase_series[0].size() == levels_half) {
            LOG4_DEBUG("Taking loaded phase series start.");
            t_phase_start = phase_series.size();
            for (size_t l = 0; l < levels_half; ++l) phase_start[l] = phase_series[t_phase_start - 1][l];
        } else if (count_from_start) {
            /*cilk_*/for (size_t l = 0; l < levels_half; ++l) {
                for (size_t t = 0; t < count_from_start; ++t) {
                    phase_start[l] += omega[l] * 2. * M_PI;
                }
            }
        }

        if (phase_series.size() != T + 1) phase_series.resize(T + 1);
        // u_series will be initialized, v_series - not!
        /*cilk_*/for(size_t t = t_phase_start; t < phase_series.size(); ++t) {
            phase_series[t].resize(levels_half);
        }
        /*cilk_*/for (size_t l = 0; l < levels_half; ++l) {
            phase_series[t_phase_start][l] = phase_start[l] + omega(l, 0) * 2. * M_PI;
        }

        /* cilk_ */ for (size_t l = 0; l < levels_half;  ++l) {
            for (size_t t = t_phase_start + 1; t < phase_series.size(); ++t) {
                phase_series[t][l] = phase_series[t - 1][l] + omega(l, 0) * 2. * M_PI;
            }
        };
    }
#ifdef SAVE_DECON
    if (loaded_decon) goto __loaded_decon;
#endif

#ifdef USE_CUDA
    PROFILE_EXEC_TIME(do_problem(
            found_freqs->second.p_solver, p_osqp_settings, levels_half, phase_series, input, R, decon, found_freqs->second),
                    "Tail transform of " << levels << " levels");
#else // USE_CUDA
    // since CPU osqp does not work for the matrix that appears in B, we have to simply use the matlab result and continue from there
    // there is some danger when N is even to have some slight miss, should be checked.
    /* cilk_ */ for (size_t l = 0; l < levels_half; ++l) {
        /* cilk_ */ for (size_t t = 0; t < T; ++t) {
            sol_u_series[l][t] = decomposition_u_series[t + levels_half * l]; // u_series[i][j];
            sol_v_series[l][t] = 0.; // should be v_series[i][j] somehow, but because the way matlab code is done, v_series comes out as 0, so we can't do anything different here.
        }
    }
    // let's make sure the sum is exactly right, not approximately.
    /* cilk_ */ for (size_t t = 0; t < R; ++t) {
        cilk::reducer_opadd<double> u_sum{0.};
        /* cilk_ */ for (size_t l = 0; l < levels_half; ++l) {
            u_sum += sol_u_series[l][t];
        }
        /* cilk_ */ for (size_t l = 0; l < levels_half; ++l) {
            sol_u_series[l][t] = sol_u_series[l][t] * input[t] / u_sum.get_value();
        }
    }
    // the first 1000 2000 after this part should be skipped for training, some time is needed for the decomposition
    // to adjust itself.
#endif // USE_CUDA

#ifdef SAVE_DECON
__loaded_decon:
    std::ofstream of;
    of.open(common::formatter() << SAVE_OUTPUT_LOCATION << table_name << "_decon_values_phases.tsv",
            std::ofstream::out | std::ofstream::app);
    of.precision(std::numeric_limits<double>::max_digits10);
    if (loaded_decon < 1) {
        for (size_t t = 0; t < R; ++t) {
            for (size_t l = 0; l < levels; ++l) of << decon[t][l] << "\t";
            for (size_t l = 0; l < levels_half; ++l) of << phase_series[t][l] << "\t";
            of << std::endl;
        }
    }
#endif

    // Now do online VMD
    std::vector<std::vector<double>> decon_slice(1, std::vector<double>(levels));
    for (size_t t = R > loaded_decon ? R : loaded_decon; t < T; ++t) {
        LOG4_TRACE("Doing problem " << t << " values " << input[t] << " phases " << svr::common::deep_to_string(phase_series[t+1]));
        do_problem(
                found_freqs->second.p_solver, p_osqp_settings,
                levels_half, {phase_series[t], phase_series[t + 1]}, {input[t]}, 1,
                decon_slice, found_freqs->second, decon.back());
        decon.emplace_back(decon_slice[0]);
#ifdef SAVE_DECON
        for (size_t l = 0; l < levels; ++l) of << decon[t][l] << "\t";
        for (size_t l = 0; l < levels_half; ++l) of << phase_series[t][l] << "\t";
        of << std::endl;
#endif
    }
#ifdef SAVE_DECON
    of.flush();
    of.close();
#endif

#if 0 // DEBUG
    {
        static size_t call_ct;
        std::stringstream ss;
        std::stringstream ss_file_name;
        ss_file_name << "cvmd_decon_batch_" << call_ct++ << ".csv";
        for (size_t i = 0; i < decon.size(); ++i) { // i = timesteps
            for (size_t j = 0; j < decon[i].size(); ++j) // j = levels
                ss << ", " << decon[i][j];
            ss << std::endl;
        }
        LOG4_FILE(ss_file_name.str().c_str(), ss.str().c_str());
    }
#endif
    free(p_osqp_settings);
}

void
cvmd::inverse_transform(
        const std::vector<double> &decon,
        std::vector<double> &recon,
        const size_t padding) const
{
    const size_t input_size = decon.size() / levels;
    recon = std::vector<double>(input_size, 0.);
    __omp_pfor(t, 0, input_size,
        recon[t] = 0;
        for (size_t l = 0; l < levels / 2; ++l) recon[t] += decon[t + 2 * l * input_size];
        recon[t] = recon[t] / CVMD_INPUT_MULTIPLIER;
    )
}

size_t cvmd::get_residuals_length(const std::string &decon_queue_table_name)
{
    LOG4_DEBUG("Getting residuals length for " << decon_queue_table_name << " " << levels << " levels.");
    const auto freq_key = freq_key_t(decon_queue_table_name, levels);
    const auto vmd_freq_iter = vmd_frequencies.find(freq_key);
    if (vmd_freq_iter != vmd_frequencies.end()) return 0; // Already initialized, in online mode we have no tail length
    else {
        size_t res_len = std::pow(levels / 2, 4);
        if (res_len % 2) ++res_len;
        return res_len;
    }
}

// TODO Test
std::vector<double> cvmd::calculate_phases(const size_t count_from_start, const std::string &decon_queue_table_name) const
{
    LOG4_INFO("Recalculating phases for " << decon_queue_table_name << ", starting " << count_from_start << ", levels " << levels);

    const auto freq_key = freq_key_t(decon_queue_table_name, levels);
    const auto vmd_freq_iter = vmd_frequencies.find(freq_key);

    if (vmd_freq_iter == vmd_frequencies.end()) LOG4_THROW("Decon queue " << decon_queue_table_name << " levels " << levels << " omega not found.");

    const auto omega = vmd_freq_iter->second.omega;
    std::vector<double> phase_start(levels / 2, 0.);
    /* cilk_ */ for(size_t l = 0; l < levels / 2; ++l) {
        for (size_t t = 0; t < count_from_start; ++t) {
            phase_start[l] += omega[l] * 2. * M_PI;
        }
    }

    LOG4_DEBUG("Returning " << svr::common::deep_to_string(phase_start));
    return phase_start;
}

#endif
}
