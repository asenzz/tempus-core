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
#include "common/gpu_handler.hpp"
#include "common/constants.hpp"
#include "cuqrsolve.cuh"
#include "cuda_path.cuh"
#include "onlinesvr.hpp"
#include "matmul.cuh"
#include "util/math_utils.hpp"
#include "appcontext.hpp"
#include "pprune.hpp"
#include "new_path_kernel.cuh"

namespace svr {
namespace datamodel {

// TODO Fix Bug in debug build getting zero valued kernel matrices inside this function
std::tuple<double, double, double, t_param_preds::t_predictions_ptr>
OnlineMIMOSVR::cuvalidate(
        const double lambda, const double gamma_bias, const double tau, const unsigned lag,
        const arma::mat &tune_cuml, const arma::mat &train_cuml,
        const arma::mat &tune_features_t, const arma::mat &train_features_t,
        const arma::mat &tune_label_chunk, const arma::mat &train_label_chunk,
        const arma::mat &tune_W, const arma::mat &train_W, const solvers::mmm_t &train_L_m, const double labels_sf)
{
    constexpr uint8_t irwls_iters = 4, magma_iters = 4;
    constexpr double one = 1, oneneg = -1, zero = 0;

    const uint32_t m = tune_cuml.n_cols;
    const auto mm = m * m;
    const auto mm_size = mm * sizeof(double);
    const uint32_t n = tune_label_chunk.n_cols;

    const uint32_t train_len = train_cuml.n_cols;
    const auto K_train_len = train_len * train_len;
    const auto K_train_size = K_train_len * sizeof(double);
    const auto train_len_n = train_len * n;
    const auto train_n_size = train_len_n * sizeof(double);

    constexpr auto train_clamp = 0; // C_test_len
    const auto tune_len = train_len - train_clamp;
    const auto tune_len_n = tune_len * n;
    const uint32_t tune_n_size = tune_len_n * sizeof(double);
    const auto K_tune_len = tune_len * tune_len;

    const auto calc_start = 0; // train_len - C_test_len;
    const auto calc_len = train_len - calc_start;
    assert(tune_label_chunk.n_rows == m);
    assert(train_cuml.n_rows == tune_cuml.n_rows);
    assert(m - C_test_len - train_len == 0);
#ifdef NEW_PATH
    constexpr uint8_t streams_per_gpu = 4;
#else
    constexpr uint8_t streams_per_gpu = common::C_default_kernel_max_chunk_len > 6500 ? 2 : 4;
    const uint16_t dim = tune_cuml.n_rows / lag;
#endif
    const uint32_t lag_tile_width = CDIV(lag, common::C_cu_tile_width);

    const common::gpu_context_<streams_per_gpu> ctx;
    cu_errchk(cudaSetDevice(ctx.phy_id()));
    magma_queue_t ma_queue;
    magma_queue_create(ctx.phy_id(), &ma_queue);
    const auto custream = magma_queue_get_cuda_stream(ma_queue);

    const auto d_train_label_chunk = cumallocopy(train_label_chunk, custream);
    const auto train_label_off = d_train_label_chunk + calc_start;
#ifdef NEW_PATH
    const auto d_train_features_t = cumallocopy(train_features_t, custream);
    auto const d_K_train = kernel::cu_compute_path_distances(d_train_features_t, d_train_features_t, train_features_t.n_rows, train_features_t.n_rows, train_features_t.n_cols,
                                                             train_features_t.n_cols, lag, lambda, 0, custream);
    cu_errchk(cudaFreeAsync(d_train_features_t, custream));
    const auto K_train_off = d_K_train + calc_start * train_len + calc_start;
    const auto gamma_train = solvers::cu_calc_gamma(K_train_off, train_len, train_label_off, calc_len, n, gamma_bias, custream);
    solvers::G_kernel_from_distances_I<<<CU_BLOCKS_THREADS(K_train_len), 0, custream>>>(d_K_train, K_train_len, gamma_train);
    const auto epsco = cu_calc_epsco(K_train_off, train_label_off, calc_len, n, train_len, custream);
#else
    double *d_K_train;
    const auto d_train_cuml = cumallocopy(train_cuml, custream);
    cu_errchk(cudaMallocAsync((void **) &d_K_train, K_train_size, custream));
    kernel::path::cu_kernel_xy(train_len, train_len, train_cuml.n_rows, lag, dim, lag_tile_width, lambda, tau, d_train_cuml, d_train_cuml, d_K_train, custream);
    cu_errchk(cudaFreeAsync(d_train_cuml, custream));
    const auto K_train_off = d_K_train + calc_start * train_len + calc_start;
    const auto gamma_train = solvers::cu_calc_gamma(K_train_off, train_len, train_label_off, calc_len, n, gamma_bias, custream);
    solvers::G_kernel_from_distances_I<<<CU_BLOCKS_THREADS(K_train_len), 0, custream>>>(d_K_train, K_train_len, gamma_train);
    if (train_W.n_elem) {
        RPTR(double) d_train_W = cumallocopy(train_W, custream);
        thrust::transform(thrust::cuda::par.on(custream), d_K_train, d_K_train + K_train_len, d_train_W, d_K_train, thrust::multiplies<double>());
        cu_errchk(cudaFreeAsync(d_train_W, custream));
    }
    const auto epsco = solvers::cu_calc_epsco(K_train_off, train_label_off, calc_len, n, train_len, custream);
#endif
    cu_errchk(cudaFreeAsync(d_train_label_chunk, custream));
    cu_errchk(cudaFreeAsync(d_K_train, custream));

    double *d_K_epsco, *j_solved, *j_work, *j_K_work, *d_best_weights, *d_Kz_tune;
    cu_errchk(cudaMallocAsync((void **) &d_K_epsco, mm_size, custream));
    cu_errchk(cudaMallocAsync((void **) &d_best_weights, train_n_size, custream));
    cu_errchk(cudaMallocAsync((void **) &j_K_work, K_train_size, custream));
    cu_errchk(cudaMallocAsync((void **) &j_solved, train_n_size, custream));
    cu_errchk(cudaMallocAsync((void **) &j_work, train_n_size, custream));
#ifdef INSTANCE_WEIGHTS
    RPTR(double) d_tune_W = cumallocopy(tune_W, custream);
    constexpr double *d_tune_W = nullptr;
#endif

    t_param_preds::t_predictions_ptr p_predictions;
    if (PROPS.get_recombine_parameters()) {
        p_predictions = new t_param_preds::t_predictions;
        p_predictions->fill(nullptr);
    } else p_predictions = nullptr;

#ifdef NEW_PATH
    const auto d_tune_features_t = cumallocopy(tune_features_t, custream);
    d_Kz_tune = kernel::cu_compute_path_distances(
            d_tune_features_t, d_tune_features_t, tune_features_t.n_rows, tune_features_t.n_rows, tune_features_t.n_cols, tune_features_t.n_cols, lag, lambda, 0, custream);
    cu_errchk(cudaFreeAsync(d_tune_features_t, custream));
#else
    auto const d_tune_cuml = cumallocopy(tune_cuml, custream);
    cu_errchk(cudaMallocAsync((void **) &d_Kz_tune, mm_size, custream));
    kernel::path::cu_kernel_xy(m, m, tune_cuml.n_rows, lag, dim, lag_tile_width, lambda, tau, gamma_train, d_tune_cuml, d_tune_cuml, d_Kz_tune, custream);
    cu_errchk(cudaFreeAsync(d_tune_cuml, custream));
#endif
    const auto d_tune_label_chunk = cumallocopy(tune_label_chunk, custream);
    magma_int_t info;
    const double iters_mul = common::C_itersolve_range / irwls_iters;
    const auto cublas_H = magma_queue_get_cublas_handle(ma_queue);
    double total_score = 0;
    cu_errchk(cudaMemcpyAsync(d_K_epsco, d_Kz_tune, mm_size, cudaMemcpyDeviceToDevice, custream));
#ifdef INSTANCE_WEIGHTS
    if (d_tune_W)
        solvers::G_augment_K<<<CU_BLOCKS_THREADS_2D(m), 0, custream>>>(d_K_epsco, d_tune_W, epsco, m);
    else
#endif
        solvers::G_set_diag<<<CU_BLOCKS_THREADS(m), 0, custream>>>(d_K_epsco, epsco, m);

    UNROLL(C_max_j)
    for (DTYPE(C_max_j) j = 0; j < C_max_j; ++j) {
        const auto train_start = train_clamp + j * C_slide_skip;
        const auto train_end = train_start + tune_len;
        const auto test_start = train_end;
        const auto test_len = m - test_start;
        const auto test_len_n = test_len * n;
        const auto test_n_size = test_len_n * sizeof(double);
        assert(test_len <= train_len);
#ifndef NDEBUG
        LOG4_TRACE("Try " << j << ", train start " << train_start << ", train final " << train_end << ", test start " <<
                          test_start << ", test len " << test_len << ", train len " << train_len << ", current score " << total_score);
#endif
        const auto best_solve_score = solvers::solve_hybrid(
                d_K_epsco + train_start + train_start * m, n, tune_len, j_solved, magma_iters, C_rbt_threshold, ma_queue, irwls_iters, d_tune_label_chunk + train_start,
                tune_n_size, j_work, custream, cublas_H, d_Kz_tune + train_start + train_start * m, labels_sf, tune_len_n, d_best_weights, K_tune_len, j_K_work, info,
                iters_mul, m);
        if (best_solve_score == common::C_bad_validation) {
            LOG4_ERROR("Bad validation at slide " << j);
            total_score = common::C_bad_validation;
            break;
        }
        double predict_score;
        if (PROPS.get_recombine_parameters()) {
            cb_errchk(cublasDgemm(cublas_H, CUBLAS_OP_N, CUBLAS_OP_N,
                                  test_len, n, train_len, &one, d_Kz_tune + test_start + train_start * m, m, d_best_weights, train_len, &zero, j_work, train_len));
            predict_score = solvers::unscaled_distance(d_tune_label_chunk + test_start, j_work, labels_sf, test_len, n, m, custream);
            if (!p_predictions->at(j)) p_predictions->at(j) = new arma::mat(test_len, n);
            cu_errchk(cudaMemcpyAsync(p_predictions->at(j)->memptr(), j_work, test_n_size, cudaMemcpyDeviceToHost, custream));
            cu_errchk(cudaStreamSynchronize(custream));
            *p_predictions->at(j) *= labels_sf;
        } else {
            copy_submat(d_tune_label_chunk, j_work, m, test_start, 0, m, n, tune_len, cudaMemcpyDeviceToDevice, custream);
            cb_errchk(cublasDgemm(cublas_H, CUBLAS_OP_N, CUBLAS_OP_N,
                                  test_len, n, tune_len, &one, d_Kz_tune + test_start + train_start * m, m, d_best_weights, tune_len, &oneneg, j_work, tune_len));
#if 0 // Started crashing for no reason
            cu_errchk(cudaStreamSynchronize(custream));
            cb_errchk(cublasDasum(cublas_H, test_len_n, j_work, 1, &predict_score));
            predict_score *= labels_sf / double(test_len_n);
#else
            predict_score = labels_sf * solvers::meanabs(j_work, test_len_n, custream);
#endif
        }
        LOG4_TRACE("Try " << j << " kernel dimensions " << train_len << "x" << train_len << ", delta " << common::C_itersolve_delta << ", range " << common::C_itersolve_range <<
            ", solution " << train_len << "x" << n << ", test " << test_len << "x" << n << ", score " << best_solve_score << ", former total score " << total_score);
        total_score += predict_score;
    }
#ifdef INSTANCE_WEIGHTS
    if (d_tune_W) cu_errchk(cudaFreeAsync(d_tune_W, custream));
#endif
    cu_errchk(cudaFreeAsync(d_K_epsco, custream));
    cu_errchk(cudaFreeAsync(d_Kz_tune, custream));
    cu_errchk(cudaFreeAsync(j_K_work, custream));
    cu_errchk(cudaFreeAsync(j_solved, custream));
    cu_errchk(cudaFreeAsync(j_work, custream));
    cu_errchk(cudaFreeAsync(d_best_weights, custream));
    cu_errchk(cudaFreeAsync(d_tune_label_chunk, custream));
    magma_queue_destroy(ma_queue);
    return {total_score / C_max_j, gamma_train, epsco, p_predictions};
}

#if 0
// MAGMA batched solver is broken, do not use this method
std::tuple<double, double, double, t_param_preds::t_predictions_ptr>
cuvalidate_batched(const double lambda, const double gamma_param, const unsigned lag, const arma::mat &tune_cuml, const arma::mat &train_cuml,
                                  const arma::mat &tune_label_chunk, const solvers::mmm_t &train_L_m, const double labels_sf, magma_queue_t ma_queue)
{
    constexpr unsigned irwls_iters = 3; // PROPS.get_online_learn_iter_limit();

    constexpr double one = 1;
    constexpr double oneneg = -1;
    constexpr double zero = 0;

    const unsigned train_len = train_cuml.n_cols;
    const unsigned m = tune_cuml.n_cols;
    const auto mm = m * m;
    const auto mm_size = mm * sizeof(double);
    const unsigned n = tune_label_chunk.n_cols;
    const auto K_train_len = train_len * train_len;
    const auto K_train_size = K_train_len * sizeof(double);
    const auto train_len_n = train_len * n;
    const auto train_n_size = train_len_n * sizeof(double);
    assert(tune_label_chunk.n_rows == m);
    assert(train_cuml.n_rows == tune_cuml.n_rows);

    const unsigned dim = tune_cuml.n_rows / lag;
    const unsigned lag_tile_width = cdiv(lag, common::C_cu_tile_width);
    double *d_train_cuml, *d_K_train;
    const auto custream = magma_queue_get_cuda_stream(ma_queue);
    cu_errchk(cudaMallocAsync((void **) &d_train_cuml, train_cuml.n_elem * sizeof(double), custream));
    cu_errchk(cudaMemcpyAsync(d_train_cuml, train_cuml.mem, train_cuml.n_elem * sizeof(double), cudaMemcpyHostToDevice, custream));
    cu_errchk(cudaMallocAsync((void **) &d_K_train, K_train_size, custream));
    kernel::path::cu_kernel_xy(train_len, train_len, train_cuml.n_rows, lag, dim, lag_tile_width, lambda, d_train_cuml, d_train_cuml, d_K_train, custream);
    cu_errchk(cudaFreeAsync(d_train_cuml, custream));
    const auto [mingamma, maxgamma] = solvers::cu_calc_minmax_gamma(d_K_train, train_L_m, train_len, K_train_len, custream);
    const auto gamma = gamma_param * (maxgamma - mingamma) + mingamma;
    solvers::G_kernel_from_distances_I<<<CU_BLOCKS_THREADS(K_train_len), 0, custream>>>(d_K_train, K_train_len, gamma);
    const auto epsco = 1. - solvers::mean(d_K_train, K_train_len, custream);

    double *d_K_tune, *d_tune_cuml;
    cu_errchk(cudaMallocAsync((void **) &d_tune_cuml, tune_cuml.n_elem * sizeof(double), custream));
    cu_errchk(cudaMemcpyAsync(d_tune_cuml, tune_cuml.mem, tune_cuml.n_elem * sizeof(double), cudaMemcpyHostToDevice, custream));
    cu_errchk(cudaMallocAsync((void **) &d_K_tune, mm_size, custream));
    kernel::path::cu_kernel_xy(m, m, tune_cuml.n_rows, lag, dim, lag_tile_width, lambda, gamma, d_tune_cuml, d_tune_cuml, d_K_tune, custream);
    cu_errchk(cudaFreeAsync(d_tune_cuml, custream));

    double total_score = 0;
    assert(m - C_test_len - train_len == 0);
    double *d_K_epsco, *d_tune_label_chunk;
    cu_errchk(cudaMallocAsync((void **) &d_K_epsco, mm_size, custream));
    cu_errchk(cudaMemcpyAsync(d_K_epsco, d_K_tune, mm_size, cudaMemcpyDeviceToDevice, custream));
    solvers::G_set_diag<<<CU_BLOCKS_THREADS(m), 0, custream>>>(d_K_epsco, &epsco, m); // TODO Fix
    cu_errchk(cudaMallocAsync((void **) &d_tune_label_chunk, m * n * sizeof(double), custream));
    cu_errchk(cudaMemcpyAsync(d_tune_label_chunk, tune_label_chunk.mem, m * n * sizeof(double), cudaMemcpyHostToDevice, custream));

    magma_int_t info;
    const auto iters_mul = common::C_itersolve_range / double(irwls_iters);
    auto const p_predictions = new t_param_preds::t_predictions;
    p_predictions->fill(nullptr);
    const auto cublas_H = magma_queue_get_cublas_handle(ma_queue);
    std::array<double *, C_max_j> j_K_tune, j_K_epsco, j_solved, j_train_error, j_test_error, j_left, j_best_weights;
    std::array<unsigned, C_max_j> train_start, train_end, test_start, test_len, test_len_n, test_n_size;
#pragma vectorise C_max_j
    for (unsigned j = 0; j < C_max_j; ++j) {
        train_start[j] = j * C_slide_skip;
        train_end[j] = train_start[j] + train_len;
        test_start[j] = train_end[j];
        test_len[j] = m - test_start[j];
        test_len_n[j] = test_len[j] * n;
        test_n_size[j] = test_len_n[j] * sizeof(double);
        if (test_len[j] > train_len) LOG4_THROW("Test " << test_len[j] << " length should be smaller than train dimension " << train_len);
        LOG4_TRACE("Try " << j << ", train start " << train_start[j] << ", train final " << train_end[j] << ", test start " <<
                          test_start[j] << ", test len " << test_len[j] << ", train len " << train_len);
        cu_errchk(cudaMallocAsync((void **) &j_best_weights[j], train_n_size, custream));
        cu_errchk(cudaMallocAsync((void **) &j_left[j], K_train_size, custream));
        cu_errchk(cudaMallocAsync((void **) &j_solved[j], train_n_size, custream));
        cu_errchk(cudaMallocAsync((void **) &j_train_error[j], train_n_size, custream));
        cu_errchk(cudaMallocAsync((void **) &j_test_error[j], test_len[j] * n * sizeof(double), custream));
        j_K_tune[j] = d_K_tune + train_start[j] + train_start[j] * m;
        j_K_epsco[j] = d_K_epsco + train_start[j] + train_start[j] * m;
        copy_submat(d_tune_label_chunk, j_solved[j], m, train_start[j], 0, train_end[j], n, train_len, cudaMemcpyDeviceToDevice, custream);
    }

    cu_errchk(cudaStreamSynchronize(custream));
    ma_errchk(magma_dgesv_rbt_batched(train_len, n, j_K_epsco.data(), m, j_solved.data(), train_len, &info, C_max_j, ma_queue));

    std::array<double, C_max_j> best_solve_score;
    best_solve_score.fill(std::numeric_limits<double>::infinity());
    std::array<unsigned, C_max_j> best_iter;
    best_iter.fill(0);
    UNROLL(irwls_iters)
    for (unsigned i = 1; i < irwls_iters + 1; ++i) {
        UNROLL(C_max_j)
        for (unsigned j = 0; j < C_max_j; ++j) copy_submat(d_tune_label_chunk, j_train_error[j], m, train_start[j], 0, train_end[j], n, train_len, cudaMemcpyDeviceToDevice, custream);
        cb_errchk(cublasDgemmBatched(cublas_H, CUBLAS_OP_N, CUBLAS_OP_N, train_len, n, train_len, &one, j_K_tune.data(), m, j_solved.data(), train_len, &oneneg,
                                     j_train_error.data(), train_len, C_max_j));
        UNROLL(C_max_j)
        for (unsigned j = 0; j < C_max_j; ++j) {
            const auto solve_score = labels_sf * solvers::irwls_op1(j_train_error[j], train_len_n, custream);
            if (!std::isnormal(solve_score))
                LOG4_THROW("Score not normal " << solve_score << " for try " << j << ", iteration " << i << ", epsco " << epsco << ", gamma " << gamma << ", lambda "
                                               << lambda << ", previous score " << total_score << ", train len " << train_len << ", tune len " << m);
            else if (solve_score < best_solve_score[j]) {
#ifndef NDEBUG
                LOG4_TRACE("Try " << j << ", IRWLS iteration " << i << ", kernel dimensions " << train_len << "x" << train_len << ", former best score " <<
                                best_solve_score << ", new best score " << solve_score << ", improvement " << 100. * (1. - solve_score / best_solve_score[j]) << " pct.");
#endif
                best_solve_score[j] = solve_score;
                best_iter[j] = i;
                cu_errchk(cudaMemcpyAsync(j_best_weights[j], j_solved[j], train_n_size, cudaMemcpyDeviceToDevice, custream));
                // cu_errchk(cudaStreamSynchronize(custream));
            }
        }
        if (i == irwls_iters) break;
        const double iter_additive = common::C_itersolve_delta / (double(i) * iters_mul);
        UNROLL(C_max_j)
        for (unsigned j = 0; j < C_max_j; ++j)
            solvers::G_irwls_op2<<<CU_BLOCKS_THREADS(K_train_len), 0, custream>>>(j_train_error[j], d_K_epsco + train_start[j] + train_start[j] * m, m,
                d_tune_label_chunk + train_start[j], j_left[j], j_solved[j], iter_additive, train_len, train_len_n, K_train_len);
        cu_errchk(cudaStreamSynchronize(custream));
        ma_errchk(magma_dgesv_rbt_batched(train_len, n, j_left.data(), train_len, j_solved.data(), train_len, &info, C_max_j, ma_queue));
    }
    std::array<double, C_max_j> predict_score;
    UNROLL(C_max_j)
    for (unsigned j = 0; j < C_max_j; ++j) {
        if (PROPS.get_recombine_parameters()) {
            cb_errchk(cublasDgemm(cublas_H, CUBLAS_OP_N, CUBLAS_OP_N,
                                  test_len[j], n, train_len, &one, j_K_tune[j], train_len, j_best_weights[j], train_len, &zero, j_test_error[j], test_len[j]));
            predict_score[j] = solvers::unscaled_distance(d_tune_label_chunk + test_start[j], j_test_error[j], labels_sf, test_len[j], n, m, custream);
            if (!p_predictions->at(j)) p_predictions->at(j) = new arma::mat(test_len[j], n);
            cu_errchk(cudaMemcpyAsync(p_predictions->at(j)->memptr(), j_test_error[j], test_n_size[j], cudaMemcpyDeviceToHost, custream));
            *p_predictions->at(j) *= labels_sf;
        } else {
            copy_submat(d_tune_label_chunk, j_test_error[j], m, test_start[j], 0, m, n, train_len, cudaMemcpyDeviceToDevice, custream);
            cb_errchk(cublasDgemm(cublas_H, CUBLAS_OP_N, CUBLAS_OP_N, test_len[j], n, train_len, &one, d_K_tune + test_start[j] + train_start[j] * m, m, j_best_weights[j],
                                  train_len, &oneneg, j_test_error[j], test_len[j]));
#if 0 // Started crashing for no reason
            cb_errchk(cublasDasum(cublas_H, test_len_n, j_test_error, 1, &predict_score));
            predict_score *= labels_sf / double(test_len_n);
#else
            predict_score[j] = labels_sf * solvers::meanabs(j_test_error[j], test_len_n[j], custream);
#endif
        }
        // predict_score = 100. * predict_score / mae_lk[j]; // 100. * (predict_score + best_solve_score * solve_predict_ratio) / L_mean / 2.;
        LOG4_TRACE("Try " << j << ", IRWLS best iteration " << best_iter[j] << ", kernel dimensions " << train_len << "x" << train_len << ", delta " <<
                          common::C_itersolve_delta << ", range " << common::C_itersolve_range << ", solution " << train_len << "x" << n << ", test " << test_len[j] << "x"
                          << n << ", score " << best_solve_score[j] << ", former total score " << total_score << " MAPE");
        total_score += predict_score[j];
        cu_errchk(cudaFreeAsync(j_best_weights[j], custream));
        cu_errchk(cudaFreeAsync(j_left[j], custream));
        cu_errchk(cudaFreeAsync(j_solved[j], custream));
        cu_errchk(cudaFreeAsync(j_train_error[j], custream));
        cu_errchk(cudaFreeAsync(j_test_error[j], custream));
    }
    cu_errchk(cudaFreeAsync(d_K_train, custream));
    cu_errchk(cudaFreeAsync(d_tune_label_chunk, custream));
    cu_errchk(cudaFreeAsync(d_K_epsco, custream));
    cu_errchk(cudaFreeAsync(d_K_tune, custream));
    return {total_score == common::C_bad_validation ? total_score : total_score / double(C_max_j), gamma, epsco, p_predictions};
}
#endif

}
}
