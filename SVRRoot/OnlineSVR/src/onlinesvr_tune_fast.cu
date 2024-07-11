//
// Created by zarko on 4/29/24.
//
#include <ipp/ipp.h>
#include <cstring>
#include <mkl.h>
#include <armadillo>
#include <cublas_v2.h>
#include <driver_types.h>
#include <limits>
#include <magma_d.h>
#include <mkl_cblas.h>
#include <thread>
#include <magma_v2.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cooperative_groups.h>
#include <tuple>
#include <thrust/async/reduce.h>
#include "common/compatibility.hpp"
#include "common/cuda_util.cuh"
#include "common/logging.hpp"
#include "common/gpu_handler.tpp"
#include "common/constants.hpp"
#include "cuqrsolve.cuh"
#include "cuda_path.hpp"
#include "onlinesvr.hpp"
#include "matmul.cuh"
#include "util/math_utils.hpp"
#include "appcontext.hpp"
#include "pprune.hpp"

namespace svr {
namespace datamodel {

__global__ void G_div_inplace(double *__restrict__ x, const double a, const size_t n)
{
    CUDA_STRIDED_FOR_i(n) x[i] /= a;
}

__global__ void G_mul_inplace(double *__restrict__ x, const double a, const size_t n)
{
    CUDA_STRIDED_FOR_i(n) x[i] *= a;
}

__global__ void G_add_inplace(const double *__restrict__ x, double *__restrict__ y, const size_t n)
{
    CUDA_STRIDED_FOR_i(n) y[i] += x[i];
}


double cu_calc_gamma(const double *Z, const double L_mean, const double train_len, const size_t n_elem, const cudaStream_t &stm)
{
    const auto Z_mm = solvers::mean(Z, n_elem, stm);
    const auto g = kernel::path::calc_g(train_len, Z_mm, L_mean);
    LOG4_TRACE("Mean Z " << Z_mm << ", mean L " << L_mean << ", n " << train_len << ", gamma " << g);
    return g;
}

double OnlineMIMOSVR::calc_qgamma(const double Z_mean, const double Z_minmax, const double L_mean, const double L_minmax, const double train_len, const double q)
{
    const double q_inc = q + 1;
    const auto Z_mm = (q * Z_mean + Z_minmax) / q_inc;
    const auto L_mm = (q * L_mean + L_minmax) / q_inc;
    const auto g = kernel::path::calc_g(train_len, Z_mm, L_mm);
    LOG4_TRACE("Zmm " << Z_mm << ", Lmm " << L_mm << ", n " << train_len << ", gamma " << g << ", quantile " << q);
    return g;
}

std::tuple<double, double> cu_calc_minmax_gamma(const double *Z, const mmm_t &train_L_m, const double train_len, const size_t Z_n_elem, const cudaStream_t &stm)
{
    const auto [Z_mean, Z_min, Z_max] = solvers::meanminmax(Z, Z_n_elem, stm);
    return {OnlineMIMOSVR::calc_qgamma(Z_mean, Z_min, train_L_m.mean, train_L_m.min, train_len, C_gamma_variance),
            OnlineMIMOSVR::calc_qgamma(Z_mean, Z_max, train_L_m.mean, train_L_m.max, train_len, C_gamma_variance)};
}


// error = abs(labels_train - K_train * solved)
__global__ void G_absdif(const double *__restrict__ labels_train, double *__restrict__ error_mat, const size_t mn)
{
    CUDA_STRIDED_FOR_i(mn) {
        error_mat[i] = abs(labels_train[i] - error_mat[i]);
    }
}

// work = abs(j_test_labels - (j_K_test * best_solution - svr_epsilon))
__global__ void G_pred_absdif_inplace(const double *__restrict__ j_test_labels, double *__restrict__ work, const double svr_epsilon, const size_t test_len_n)
{
    CUDA_STRIDED_FOR_i(test_len_n) work[i] = abs(j_test_labels[i] - work[i] + svr_epsilon);
}

__global__ void G_set_diag(double *__restrict__ K, const double d, const size_t m)
{
    CUDA_STRIDED_FOR_i(m) K[i * m + i] = d;
}


__global__ void
G_copy_submat(const double *__restrict in, double *out, const size_t in_m, const size_t out_m, const size_t start_offset, const size_t out_mn)
{
    CUDA_STRIDED_FOR_i(out_mn) out[i] = in[start_offset + i / out_m * in_m + i % out_m];
}

void copy_submat(const double *in, double *out, const size_t in_m, const size_t out_start_m, const size_t out_start_n, const size_t out_end_m,
                 const size_t out_end_n, const cudaStream_t &strm)
{
    const auto out_m = out_end_m - out_start_m;
    const size_t out_mn = out_m * (out_end_n - out_start_n);
    const auto start_offset = out_start_n * in_m + out_start_m;
    G_copy_submat<<<CUDA_THREADS_BLOCKS(out_mn), 0, strm>>>(in, out, in_m, out_m, start_offset, out_mn);
}

std::tuple<double, double, double, t_param_preds::t_predictions_ptr>
OnlineMIMOSVR::cuvalidate(const double lambda, const double gamma_param, const size_t lag, const arma::mat &tune_cuml, const arma::mat &train_cuml,
                          const arma::mat &tune_label_chunk, const mmm_t &train_L_m, const double labels_sf, magma_queue_t ma_queue)
{
    constexpr unsigned irwls_iters = 3; // PROPS.get_online_learn_iter_limit();
    constexpr unsigned magma_iters = 10; // C_rbt_iter

    constexpr double one = 1;
    constexpr double oneneg = -1;
    constexpr double zero = 0;

    const size_t train_len = train_cuml.n_cols;
    const size_t m = tune_cuml.n_cols;
    const auto mm = m * m;
    const auto mm_size = mm * sizeof(double);
    const size_t n = tune_label_chunk.n_cols;
    const auto K_train_len = train_len * train_len;
    const auto K_train_size = K_train_len * sizeof(double);
    const auto train_len_n = train_len * n;
    const auto train_n_size = train_len_n * sizeof(double);
    assert(tune_label_chunk.n_rows == m);
    assert(train_cuml.n_rows == tune_cuml.n_rows);

    const size_t dim = tune_cuml.n_rows / lag;
    const uint32_t lag_tile_width = _CEILDIV(lag, common::C_cu_tile_width);
    double *d_train_cuml, *d_K_train;
    const auto custream = magma_queue_get_cuda_stream(ma_queue);
    cu_errchk(cudaMallocAsync((void **) &d_train_cuml, train_cuml.n_elem * sizeof(double), custream));
    cu_errchk(cudaMemcpyAsync(d_train_cuml, train_cuml.mem, train_cuml.n_elem * sizeof(double), cudaMemcpyHostToDevice, custream));
    cu_errchk(cudaMallocAsync((void **) &d_K_train, K_train_size, custream));
    kernel::path::G_kernel_xy<<<CUDA_THREADS_BLOCKS_2D(train_len), 0, custream>>>(train_len, train_len, train_cuml.n_rows, lag, dim, lag_tile_width, lambda, d_train_cuml, d_train_cuml, d_K_train);
    cu_errchk(cudaFreeAsync(d_train_cuml, custream));
    const auto [mingamma, maxgamma] = cu_calc_minmax_gamma(d_K_train, train_L_m, train_len, K_train_len, custream);
    const auto gamma = gamma_param * (maxgamma - mingamma) + mingamma;
    solvers::G_kernel_from_distances_inplace<<<CUDA_THREADS_BLOCKS(K_train_len), 0, custream>>>(d_K_train, K_train_len, DIST(gamma));
    const auto epsco = 1. - solvers::mean(d_K_train, K_train_len, custream);

    double *d_K_tune, *d_tune_cuml;
    cu_errchk(cudaMallocAsync((void **) &d_tune_cuml, tune_cuml.n_elem * sizeof(double), custream));
    cu_errchk(cudaMemcpyAsync(d_tune_cuml, tune_cuml.mem, tune_cuml.n_elem * sizeof(double), cudaMemcpyHostToDevice, custream));
    cu_errchk(cudaMallocAsync((void **) &d_K_tune, mm_size, custream));
    kernel::path::G_kernel_xy<<<CUDA_THREADS_BLOCKS_2D(m), 0, custream>>>(m, m, tune_cuml.n_rows, lag, dim, lag_tile_width, lambda, DIST(gamma), d_tune_cuml, d_tune_cuml, d_K_tune);
    cu_errchk(cudaFreeAsync(d_tune_cuml, custream));

    double total_score = 0;
    assert(m - C_test_len - train_len == 0);
    double *d_K_epsco, *d_tune_labels;
    cu_errchk(cudaMallocAsync((void **) &d_K_epsco, mm_size, custream));
    cu_errchk(cudaMemcpyAsync(d_K_epsco, d_K_tune, mm_size, cudaMemcpyDeviceToDevice, custream));
    G_set_diag<<<CUDA_THREADS_BLOCKS(m), 0, custream>>>(d_K_epsco, epsco, m);
    cu_errchk(cudaMallocAsync((void **) &d_tune_labels, m * n * sizeof(double), custream));
    cu_errchk(cudaMemcpyAsync(d_tune_labels, tune_label_chunk.mem, m * n * sizeof(double), cudaMemcpyHostToDevice, custream));

    double *j_K_tune, *j_K_epsco, *j_K_test, *j_solved, *j_train_error, *j_test_error, *j_train_labels, *j_test_labels, *sum_rows_ones, *j_left, *d_best_weights;
    cu_errchk(cudaMallocAsync((void **) &d_best_weights, train_n_size, custream));
    cu_errchk(cudaMallocAsync((void **) &j_K_tune, K_train_size, custream));
    cu_errchk(cudaMallocAsync((void **) &j_left, K_train_size, custream));
    cu_errchk(cudaMallocAsync((void **) &j_K_epsco, K_train_size, custream));
    cu_errchk(cudaMallocAsync((void **) &j_K_test, C_test_len * train_len * sizeof(double), custream));
    cu_errchk(cudaMallocAsync((void **) &j_solved, train_n_size, custream));
    cu_errchk(cudaMallocAsync((void **) &j_train_error, train_n_size, custream));
    cu_errchk(cudaMallocAsync((void **) &j_test_error, C_test_len * n * sizeof(double), custream));
    cu_errchk(cudaMallocAsync((void **) &j_train_labels, train_n_size, custream));
    if (PROPS.get_recombine_parameters()) cu_errchk(cudaMallocAsync((void **) &j_test_labels, C_test_len * n * sizeof(double), custream));
    cu_errchk(cudaMallocAsync((void **) &sum_rows_ones, train_n_size, custream));
    thrust::fill(thrust::cuda::par.on(custream), sum_rows_ones, sum_rows_ones + train_len_n, one);

    magma_int_t info;
    const auto iters_mul = common::C_itersolve_range / double(irwls_iters);
    auto p_predictions = new t_param_preds::t_predictions;
    p_predictions->fill(nullptr);
    const auto cublas_H = magma_queue_get_cublas_handle(ma_queue);
#pragma unroll C_max_j
    for (size_t j = 0; j < C_max_j; ++j) {
        const auto train_start = j * C_slide_skip;
        const auto train_end = train_start + train_len;
        const auto test_start = train_end;
        const auto test_len = m - test_start;
        const auto test_len_n = test_len * n;
        const auto test_n_size = test_len_n * sizeof(double);
        if (test_len > train_len) LOG4_THROW("Test " << test_len << " length should be smaller than train dimension " << train_len);
        LOG4_TRACE("Try " << j << ", train start " << train_start << ", train final " << train_end << ", test start " <<
                          test_start << ", test len " << test_len << ", train len " << train_len << ", current score " << total_score);
        auto best_solve_score = std::numeric_limits<double>::infinity();
        unsigned best_iter = 0;
        // Cublas and Magma buggy work with submatrices LDA, copying to separate matrix is a must
        copy_submat(d_K_tune, j_K_tune, m, train_start, train_start, train_end, train_end, custream);
        copy_submat(d_K_tune, j_K_test, m, test_start, train_start, m, train_end, custream);
        copy_submat(d_K_epsco, j_K_epsco, m, train_start, train_start, train_end, train_end, custream);
        copy_submat(d_tune_labels, j_train_labels, m, train_start, 0, train_end, n, custream);
        solvers::solve_hybrid(
                j_K_epsco, n, train_len, j_solved, magma_iters, C_rbt_threshold, ma_queue, irwls_iters, j_train_labels, train_n_size, j_train_error, custream,
                cublas_H, j_K_tune, labels_sf, train_len_n, best_solve_score, best_iter, d_best_weights, K_train_len, j_left, info, iters_mul);
        double predict_score;
        if (PROPS.get_recombine_parameters()) {
            cb_errchk(cublasDgemm(cublas_H, CUBLAS_OP_N, CUBLAS_OP_N,
                                  test_len, n, train_len, &one, j_K_test, test_len, d_best_weights, train_len, &zero, j_test_error, test_len));
            copy_submat(d_tune_labels, j_test_labels, m, test_start, 0, m, n, custream);
            predict_score = solvers::unscaled_distance(j_test_labels, j_test_error, labels_sf, test_len_n, custream);
            if (!p_predictions->at(j)) p_predictions->at(j) = new arma::mat(test_len, n);
            cu_errchk(cudaMemcpyAsync(p_predictions->at(j)->memptr(), j_test_error, test_n_size, cudaMemcpyDeviceToHost, custream));
            cu_errchk(cudaStreamSynchronize(custream));
            *p_predictions->at(j) *= labels_sf;
        } else {
            copy_submat(d_tune_labels, j_test_error, m, test_start, 0, m, n, custream);
            cb_errchk(cublasDgemm(cublas_H, CUBLAS_OP_N, CUBLAS_OP_N,
                                  test_len, n, train_len, &one, j_K_test, test_len, d_best_weights, train_len, &oneneg, j_test_error, test_len));
#if 0 // Started crashing for no reason
            cb_errchk(cublasDasum(cublas_H, test_len_n, j_test_error, 1, &predict_score));
            predict_score *= labels_sf.get_labels_factor() / double(test_len_n);
#endif
            predict_score = labels_sf * solvers::meanabs(j_test_error, test_len_n, custream);
        }
        LOG4_TRACE("Try " << j << ", IRWLS best iteration " << best_iter << ", kernel dimensions " << train_len << "x" << train_len << ", delta " <<
                          common::C_itersolve_delta << ", range " << common::C_itersolve_range << ", solution " << train_len << "x" << n << ", test " << test_len << "x"
                          << n << ", score " << best_solve_score << ", former total score " << total_score);
        total_score += predict_score;
    }
    cu_errchk(cudaFreeAsync(d_K_train, custream));
    cu_errchk(cudaFreeAsync(d_tune_labels, custream));
    cu_errchk(cudaFreeAsync(d_K_epsco, custream));
    cu_errchk(cudaFreeAsync(d_K_tune, custream));
    cu_errchk(cudaFreeAsync(d_best_weights, custream));
    cu_errchk(cudaFreeAsync(j_left, custream));
    cu_errchk(cudaFreeAsync(j_solved, custream));
    cu_errchk(cudaFreeAsync(j_K_epsco, custream));
    cu_errchk(cudaFreeAsync(j_K_tune, custream));
    cu_errchk(cudaFreeAsync(j_K_test, custream));
    cu_errchk(cudaFreeAsync(j_train_error, custream));
    cu_errchk(cudaFreeAsync(j_test_error, custream));
    cu_errchk(cudaFreeAsync(j_train_labels, custream));
    if (PROPS.get_recombine_parameters()) cu_errchk(cudaFreeAsync(j_test_labels, custream));
    cu_errchk(cudaFreeAsync(sum_rows_ones, custream));
    return {total_score == common::C_bad_validation ? total_score : total_score / double(C_max_j), gamma, epsco, p_predictions};
}

#if 0
// MAGMA batched solver is broken, do not use this method
std::tuple<double, double, double, t_param_preds::t_predictions_ptr>
OnlineMIMOSVR::cuvalidate_batched(const double lambda, const double gamma_param, const size_t lag, const arma::mat &tune_cuml, const arma::mat &train_cuml,
                          const arma::mat &tune_label_chunk, const mmm_t &train_L_m, const datamodel::DQScalingFactor &labels_sf, magma_queue_t ma_queue,
                          omp_lock_t *p_ratio_l, double &solve_predict_ratio)
{
    constexpr unsigned irwls_iters = 3; // PROPS.get_online_learn_iter_limit();

    constexpr double one = 1;
    constexpr double oneneg = -1;
    constexpr double zero = 0;

    const size_t train_len = train_cuml.n_cols;
    const size_t m = tune_cuml.n_cols;
    const auto mm = m * m;
    const auto mm_size = mm * sizeof(double);
    const size_t n = tune_label_chunk.n_cols;
    const auto K_train_len = train_len * train_len;
    const auto K_train_size = K_train_len * sizeof(double);
    const auto train_len_n = train_len * n;
    const auto train_n_size = train_len_n * sizeof(double);
    assert(tune_label_chunk.n_rows == m);
    assert(train_cuml.n_rows == tune_cuml.n_rows);

    const size_t dim = tune_cuml.n_rows / lag;
    const size_t lag_tile_width = std::ceil(float(lag) / float(common::C_cu_tile_width));
    double *d_train_cuml, *d_K_train;
    const auto custream = magma_queue_get_cuda_stream(ma_queue);
    cu_errchk(cudaMallocAsync((void **) &d_train_cuml, train_cuml.n_elem * sizeof(double), custream));
    cu_errchk(cudaMemcpyAsync(d_train_cuml, train_cuml.mem, train_cuml.n_elem * sizeof(double), cudaMemcpyHostToDevice, custream));
    cu_errchk(cudaMallocAsync((void **) &d_K_train, K_train_size, custream));
    kernel::path::G_kernel_xx<<<CUDA_THREADS_BLOCKS_2D(train_len), 0, custream>>>(train_len, train_cuml.n_rows, lag, dim, lag_tile_width, lambda, d_train_cuml, d_K_train);
    cu_errchk(cudaFreeAsync(d_train_cuml, custream));
    // const auto mingamma = cu_calc_gamma(d_K_train, train_L_m.mean, train_len, K_train_len, custream);
    const auto [mingamma, maxgamma] = cu_calc_minmax_gamma(d_K_train, train_L_m, train_len, K_train_len, custream);
    const auto gamma = gamma_param * (maxgamma - mingamma) + mingamma;
    solvers::G_kernel_from_distances_inplace<<<CUDA_THREADS_BLOCKS(K_train_len), 0, custream>>>(d_K_train, K_train_len, DIST(gamma));
    const auto epsco = 1. - solvers::mean(d_K_train, K_train_len, custream);

    double *d_K_tune, *d_tune_cuml;
    cu_errchk(cudaMallocAsync((void **) &d_tune_cuml, tune_cuml.n_elem * sizeof(double), custream));
    cu_errchk(cudaMemcpyAsync(d_tune_cuml, tune_cuml.mem, tune_cuml.n_elem * sizeof(double), cudaMemcpyHostToDevice, custream));
    cu_errchk(cudaMallocAsync((void **) &d_K_tune, mm_size, custream));
    kernel::path::G_kernel_xx<<<CUDA_THREADS_BLOCKS_2D(m), 0, custream>>>(m, tune_cuml.n_rows, lag, dim, lag_tile_width, lambda, DIST(gamma), d_tune_cuml, d_K_tune);
    cu_errchk(cudaFreeAsync(d_tune_cuml, custream));

    double total_score = 0;
    assert(m - C_test_len - train_len == 0);
    double *d_K_epsco, *d_tune_labels;
    cu_errchk(cudaMallocAsync((void **) &d_K_epsco, mm_size, custream));
    cu_errchk(cudaMemcpyAsync(d_K_epsco, d_K_tune, mm_size, cudaMemcpyDeviceToDevice, custream));
    G_set_diag<<<CUDA_THREADS_BLOCKS(m), 0, custream>>>(d_K_epsco, epsco, m);
    cu_errchk(cudaMallocAsync((void **) &d_tune_labels, m * n * sizeof(double), custream));
    cu_errchk(cudaMemcpyAsync(d_tune_labels, tune_label_chunk.mem, m * n * sizeof(double), cudaMemcpyHostToDevice, custream));

    std::array<double *, C_max_j> j_K_tune, j_K_epsco, j_K_test, j_solved, j_train_error, j_test_error, j_test_labels, j_left, j_best_weights;
    for (unsigned j = 0; j < C_max_j; ++j) {
        cu_errchk(cudaMallocAsync((void **) &j_best_weights[j], train_n_size, custream));
        cu_errchk(cudaMallocAsync((void **) &j_K_tune[j], K_train_size, custream));
        cu_errchk(cudaMallocAsync((void **) &j_left[j], K_train_size, custream));
        cu_errchk(cudaMallocAsync((void **) &j_K_epsco[j], K_train_size, custream));
        cu_errchk(cudaMallocAsync((void **) &j_K_test[j], C_test_len * train_len * sizeof(double), custream));
        cu_errchk(cudaMallocAsync((void **) &j_solved[j], train_n_size, custream));
        cu_errchk(cudaMallocAsync((void **) &j_train_error[j], train_n_size, custream));
        cu_errchk(cudaMallocAsync((void **) &j_test_error[j], C_test_len * n * sizeof(double), custream));
        if (PROPS.get_recombine_parameters()) cu_errchk(cudaMallocAsync((void **) &j_test_labels[j], C_test_len * n * sizeof(double), custream));
    }
    double *sum_rows_ones;
    cu_errchk(cudaMallocAsync((void **) &sum_rows_ones, train_n_size, custream));
    thrust::fill(thrust::cuda::par.on(custream), sum_rows_ones, sum_rows_ones + train_len_n, one);

    magma_int_t info;
    const auto iters_mul = common::C_itersolve_range / double(irwls_iters);
    auto p_predictions = new t_param_preds::t_predictions;
    p_predictions->fill(nullptr);
    const auto cublas_H = magma_queue_get_cublas_handle(ma_queue);
    std::array<unsigned, C_max_j> train_start, train_end, test_start, test_len, test_len_n, test_n_size;
#pragma unroll C_max_j
    for (size_t j = 0; j < C_max_j; ++j) {
        train_start[j] = j * C_slide_skip;
        train_end[j] = train_start[j] + train_len;
        test_start[j] = train_end[j];
        test_len[j] = m - test_start[j];
        test_len_n[j] = test_len[j] * n;
        test_n_size[j] = test_len_n[j] * sizeof(double);
        if (test_len[j] > train_len) LOG4_THROW("Test " << test_len[j] << " length should be smaller than train dimension " << train_len);
        LOG4_TRACE("Try " << j << ", train start " << train_start[j] << ", train final " << train_end[j] << ", test start " <<
                          test_start[j] << ", test len " << test_len[j] << ", train len " << train_len);
        copy_submat(d_K_tune, j_K_tune[j], m, train_start[j], train_start[j], train_end[j], train_end[j], custream);
        copy_submat(d_K_tune, j_K_test[j], m, test_start[j], train_start[j], m, train_end[j], custream);
        copy_submat(d_K_epsco, j_K_epsco[j], m, train_start[j], train_start[j], train_end[j], train_end[j], custream);
        copy_submat(d_tune_labels, j_solved[j], m, train_start[j], 0, train_end[j], n, custream);
    }

    ma_errchk(magma_dgesv_rbt_batched(train_len, n, j_K_epsco.data(), train_len, j_solved.data(), train_len, &info, C_max_j, ma_queue));
    std::array<double, C_max_j> best_solve_score;
    best_solve_score.fill(std::numeric_limits<double>::infinity());
    std::array<unsigned, C_max_j> best_iter;
    best_iter.fill(0);
#pragma unroll common::C_default_online_iter_limit
    for (size_t i = 1; i < irwls_iters + 1; ++i) {
#pragma unroll C_max_j
        for (unsigned j = 0; j < C_max_j; ++j) copy_submat(d_tune_labels, j_train_error[j], m, train_start[j], 0, train_end[j], n, custream);
        cb_errchk(cublasDgemmBatched(cublas_H, CUBLAS_OP_N, CUBLAS_OP_N,
                              train_len, n, train_len, &one, j_K_tune.data(), train_len, j_solved.data(), train_len, &oneneg, j_train_error.data(), train_len, C_max_j));
#pragma unroll C_max_j
        for (unsigned j = 0; j < C_max_j; ++j) {
            const auto solve_score = labels_sf.get_labels_factor() * solvers::irwls_op1(j_train_error[j], train_len_n, custream);
            if (!std::isnormal(solve_score)) {
                LOG4_ERROR("Score not normal " << solve_score << " for try " << j << ", iteration " << i << ", epsco " << epsco << ", gamma " << gamma << ", lambda "
                                               << lambda << ", previous score " << total_score << ", train len " << train_len << ", tune len " << m);
                total_score = common::C_bad_validation;
                goto __bail;
            } else if (solve_score < best_solve_score[j]) {
                // LOG4_TRACE("Try " << j << ", IRWLS iteration " << i << ", kernel dimensions " << train_len << "x" << train_len << ", former best score " <<
                //                best_solve_score << ", new best score " << solve_score << ", improvement " << 100. * (1. - solve_score / best_solve_score) << " pct.");
                best_solve_score[j] = solve_score;
                best_iter[j] = i;
                cu_errchk(cudaMemcpyAsync(j_best_weights[j], j_solved[j], train_n_size, cudaMemcpyDeviceToDevice, custream));
            }
        }
        if (i == irwls_iters) break;
        const double iter_additive = common::C_itersolve_delta / (double(i) * iters_mul);
/*#pragma unroll C_max_j
        for (unsigned j = 0; j < C_max_j; ++j)
            G_irwls_op2<<<CUDA_THREADS_BLOCKS(K_train_len), 0, custream>>>(
                    j_train_error[j], d_K_epsco + train_start[j], m, d_tune_labels + train_start[j], j_left[j], j_solved[j], iter_additive, train_len, train_len_n, K_train_len);
                    */
        ma_errchk(magma_dgesv_rbt_batched(train_len, n, j_left.data(), train_len, j_solved.data(), train_len, &info, C_max_j, ma_queue));
    }
    std::array<double, C_max_j> predict_score;
#pragma unroll C_max_j
    for (unsigned j = 0; j < C_max_j; ++j) {
        if (PROPS.get_recombine_parameters()) {
            cb_errchk(cublasDgemm(cublas_H, CUBLAS_OP_N, CUBLAS_OP_N,
                                  test_len[j], n, train_len, &one, j_K_tune[j], train_len, j_best_weights[j], train_len, &zero, j_test_error[j], test_len[j]));
            copy_submat(d_tune_labels, j_test_labels[j], m, test_start[j], 0, m, n, custream);
            predict_score[j] = solvers::unscaled_distance(
                    j_test_labels[j], j_test_error[j], labels_sf.get_labels_factor(), labels_sf.get_dc_offset_labels(), test_len_n[j], custream);
            if (!p_predictions->at(j)) p_predictions->at(j) = new arma::mat(test_len[j], n);
            cu_errchk(cudaMemcpyAsync(p_predictions->at(j)->memptr(), j_test_error[j], test_n_size[j], cudaMemcpyDeviceToHost, custream));
            *p_predictions->at(j) *= labels_sf.get_labels_factor();
        } else {
            copy_submat(d_tune_labels, j_test_error[j], m, test_start[j], 0, m, n, custream);
            cb_errchk(cublasDgemm(cublas_H, CUBLAS_OP_N, CUBLAS_OP_N,
                                  test_len[j], n, train_len, &one, j_K_test[j], test_len[j], j_best_weights[j], train_len, &oneneg, j_test_error[j], test_len[j]));
#if 0 // Started crashing for no reason
            cb_errchk(cublasDasum(cublas_H, test_len_n, j_test_error, 1, &predict_score));
            predict_score *= labels_sf.get_labels_factor() / double(test_len_n);
#endif
            predict_score[j] = labels_sf.get_labels_factor() * solvers::meanabs(j_test_error[j], test_len_n[j], custream);
        }
#if 0
        if (!std::isnormal(solve_predict_ratio)) {
            omp_set_lock(p_ratio_l);
            if (!std::isnormal(solve_predict_ratio)) solve_predict_ratio = predict_score / best_solve_score / 2.;
            omp_unset_lock(p_ratio_l);
        }
#endif
        // predict_score = 100. * predict_score / mae_lk[j]; // 100. * (predict_score + best_solve_score * solve_predict_ratio) / L_mean / 2.;
        LOG4_TRACE("Try " << j << ", IRWLS best iteration " << best_iter[j] << ", kernel dimensions " << train_len << "x" << train_len << ", delta " <<
                          common::C_itersolve_delta << ", range " << common::C_itersolve_range << ", solution " << train_len << "x" << n << ", test " << test_len[j] << "x"
                          << n << ", score " << best_solve_score[j] << ", former total score " << total_score << " MAPE");
        total_score += predict_score[j];
        cu_errchk(cudaFreeAsync(j_best_weights[j], custream));
        cu_errchk(cudaFreeAsync(j_left[j], custream));
        cu_errchk(cudaFreeAsync(j_solved[j], custream));
        cu_errchk(cudaFreeAsync(j_K_epsco[j], custream));
        cu_errchk(cudaFreeAsync(j_K_tune[j], custream));
        cu_errchk(cudaFreeAsync(j_K_test[j], custream));
        cu_errchk(cudaFreeAsync(j_train_error[j], custream));
        cu_errchk(cudaFreeAsync(j_test_error[j], custream));
        if (PROPS.get_recombine_parameters()) cu_errchk(cudaFreeAsync(j_test_labels[j], custream));
    }
    __bail:
    cu_errchk(cudaFreeAsync(d_K_train, custream));
    cu_errchk(cudaFreeAsync(d_tune_labels, custream));
    cu_errchk(cudaFreeAsync(d_K_epsco, custream));
    cu_errchk(cudaFreeAsync(d_K_tune, custream));
    cu_errchk(cudaFreeAsync(sum_rows_ones, custream));
    return {total_score == common::C_bad_validation ? total_score : total_score / double(C_max_j), gamma, epsco, p_predictions};
}
#endif

}
}
