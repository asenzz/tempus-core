//
// Created by zarko on 4/29/24.
//
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

namespace svr {
namespace datamodel {

__global__ void G_div_inplace(double *__restrict__ x, const double a, const size_t n)
{
    CUDA_STRIDED_FOR_i(n) x[i] /= a;
}

__global__ void G_mul_inplace(double *__restrict__ x, const double a, const size_t n)
{
    CUDA_STRIDED_FOR_i(n) x[i] = std::round(x[i] * a);
}

__global__ void G_add_inplace(const double *__restrict__ x, double *__restrict__ y, const size_t n)
{
    CUDA_STRIDED_FOR_i(n) y[i] += x[i];
}


double calc_mingamma(const double *Z, const double mean_L, const double train_len, const size_t n_elem, const cudaStream_t &mag_strm)
{
    const auto mean_Z = solvers::sum(Z, n_elem, mag_strm) / double(n_elem);
    const auto res = std::sqrt(train_len * mean_Z / (2. * (train_len - mean_L)));
    LOG4_TRACE("Mean Z " << mean_Z << ", mean L " << mean_L << ", ncols " << train_len << ", gamma " << res);
    return res;
}

// error = abs(labels_train - K_train * solved)
__global__ void G_absdif(
        const double *__restrict__ labels_train, double *__restrict__ error_mat, const size_t mn)
{
    CUDA_STRIDED_FOR_i(mn) {
        error_mat[i] = abs(labels_train[i] - error_mat[i]);
    }
}

__global__ void G_sum_rows(const double *__restrict__ input, double *__restrict__ output, const size_t m, const size_t mn)
{
    CUDA_STRIDED_FOR_i(m) {
        output[i] = 0;
#ifndef __GNUC__
#pragma unroll
#endif
        for (size_t j = i; j < mn; j += m) output[i] += input[j];
    }
}

// left = mult * arma::trans(arma::ones(arma::size(mult))) % d_K_epsco
__global__ void G_iter_op2(
        const double *__restrict__ mult_sum, const double *__restrict__ d_K_epsco, double *__restrict__ left, const size_t train_len, const size_t K_train_len)
{
    CUDA_STRIDED_FOR_i(K_train_len)
        left[i] = mult_sum[i % train_len] * d_K_epsco[i];
}

// work = abs(j_labels_test - (j_K_test * best_solution - svr_epsilon))
__global__ void G_pred_absdif_inplace(const double *__restrict__ j_labels_test, double *__restrict__ work, const double svr_epsilon, const size_t test_len_n)
{
    CUDA_STRIDED_FOR_i(test_len_n) {
        work[i] = abs(j_labels_test[i] - work[i] + svr_epsilon);
    }
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

#if 0
template<typename T_cuml>
std::tuple<double, double, double>
cuvalidate_batched(const double gamma_variance, const double labels_mean, const double lambda, const double gamma_param, const double svr_epsilon,
                   const double all_labels_meanabs, const size_t train_len, const size_t lag, const T_cuml &tune_cuml, const T_cuml &train_cuml,
                   const arma::mat &chunk_labels,
                   const size_t iters)
{
    const auto m = tune_cuml->front().n_cols;
    const auto mm = m * m;
    const auto mm_size = mm * sizeof(double);
//    const auto n = chunk_labels.n_cols;
//    const auto K_train_len = train_len * train_len;
//    const auto K_train_size = K_train_len * sizeof(double);
    // const auto train_n_size = train_len * n * sizeof(double);
    if (chunk_labels.n_rows != m) LOG4_THROW("chunk_labels.n_rows != m" << chunk_labels.n_rows << " " << m);
    common::gpu_context ctx;
    const auto gpu_id = ctx.phy_id();
    cu_errchk(cudaSetDevice(gpu_id));
    // cudaStream_t cub_stream = nullptr;
    magma_queue_t magma_queue;
    magma_queue_create(gpu_id, &magma_queue);
    /*
    if (!magma_queue) LOG4_THROW("Failed creating MAGMA queue.");
    cublasHandle_t cublas_h;
    cb_errchk(cublasCreate(&cublas_h)); //magma_queue_get_cublas_handle(magma_queue);
    cb_errchk(cublasSetPointerMode(cublas_h, CUBLAS_POINTER_MODE_DEVICE));
    // cb_errchk(cublasSetMathMode(cublas_h, CUBLAS_TENSOR_OP_MATH));
    //cb_errchk(cublasGetStream(cublas_h, &cub_stream));
    */
    //const auto Kz_train = prepare_Z(lag, lambda, train_cuml, cub_stream);
    arma::mat h_K_train(train_len, train_len);
    // kernel::path::cu_distances_xx(train_len, train_len, train_len, lag, train_cuml->front().mem, lambda, h_K_train.memptr(), gpu_id);

    // cu_errchk(cudaMemcpy(h_Kz_train.memptr(), Kz_train, K_train_size, cudaMemcpyDeviceToHost));
    const auto mingamma = OnlineMIMOSVR::calc_gamma(h_K_train, labels_mean);
    // const auto mingamma = calc_mingamma(Kz_train, labels_mean, double(train_len), K_train_len, cub_stream);
    const auto gamma = (gamma_param * gamma_variance + 1. - gamma_variance) * mingamma;
    const auto gamma_div = 2. * gamma * gamma;
    solvers::kernel_from_distances_inplace(h_K_train.memptr(), train_len, train_len, gamma, gpu_id);
    // solvers::G_kernel_from_distances_inplace<<<CUDA_THREADS_BLOCKS(K_train_len)>>>(Kz_train, K_train_len, gamma_div);
    const auto epsco = OnlineMIMOSVR::calc_epsco(h_K_train);
    //const auto epsco = solvers::sum(Kz_train, K_train_len, cub_stream) / double(K_train_len);
    h_K_train.clear();
    // cu_errchk(cudaFreeAsync(Kz_train, cub_stream));

    arma::mat h_K_tune(m, m);
    // kernel::path::cu_distances_xx(m, m, m, lag, tune_cuml->front().mem, lambda, h_K_tune.memptr(), gpu_id);
    solvers::kernel_from_distances_inplace(h_K_tune.memptr(), m, m, gamma, gpu_id);
    // auto K_tune = prepare_K(lag, gamma, lambda, tune_cuml, cub_stream);
    // cu_errchk(cudaMemcpy(h_K_tune.memptr(), K_tune, mm_size, cudaMemcpyDeviceToHost));
    // cu_errchk(cudaFreeAsync(K_tune, cub_stream));
    arma::mat h_K_epsco = h_K_tune;
    h_K_epsco.diag().fill(epsco);

    double score = 0;
    const ssize_t start_point = m - C_emo_test_len - train_len;

    std::deque<arma::mat> h_K_epsco_batch(C_emo_max_j), h_K_train_batch(C_emo_max_j), h_solutions, h_labels_train_batch(C_emo_max_j);

#pragma omp unroll
    for (size_t j = 0; j < C_emo_max_j; ++j) {
        const size_t train_start = j * C_emo_slide_skip;
        const size_t train_final = train_start + train_len - 1;
        LOG4_TRACE("Try " << j << ", start point " << start_point << ", train start " << train_start << ", train final " << train_final <<
            ", train len " << train_len << ", current score " << score);

        h_K_epsco_batch[j] = h_K_epsco.submat(start_point + train_start, start_point + train_start, start_point + train_final, start_point + train_final);
        h_K_train_batch[j] = h_K_tune.submat(start_point + train_start, start_point + train_start, start_point + train_final, start_point + train_final);
        h_labels_train_batch[j] = chunk_labels.rows(start_point + train_start, start_point + train_final);
    }

    h_solutions = OnlineMIMOSVR::solve_batched_irwls(h_K_epsco_batch, h_K_train_batch, h_labels_train_batch, iters, magma_queue, gpu_id);
    release_cont(h_K_epsco_batch);
    release_cont(h_K_train_batch);
    release_cont(h_labels_train_batch);

#pragma omp unroll
    for (size_t j = 0; j < C_emo_max_j; ++j) {
        const size_t train_start = j * C_emo_slide_skip;
        const size_t train_final = train_start + train_len - 1;
        const size_t test_start = train_final;
        const size_t test_len = m - start_point - test_start - 1; // -1 if arma
        LOG4_TRACE("Try " << j << ", start point " << start_point << ", test start " << test_start << ", test len " << test_len << ", current score " << score);
        const auto this_score = common::meanabs<double>(
                chunk_labels.rows(start_point + test_start, chunk_labels.n_rows - 1) -
                h_K_tune.submat(start_point + test_start, start_point + train_start, h_K_tune.n_rows - 1, start_point + train_final) *
                    h_solutions[j] + svr_epsilon) / all_labels_meanabs;

        if (!std::isnormal(this_score)) {
            LOG4_ERROR(
                    "Score not normal " << this_score << " for epsco " << epsco << ", svr epsilon " << svr_epsilon << ", gamma " << gamma << ", lambda " << lambda <<
                                        ", meanabs labels " << all_labels_meanabs << ", previous score " << score);
            score = C_bad_validation;
        } else
            score += this_score;
    }

    magma_queue_destroy(magma_queue);

    return {score, gamma, epsco};
}
#endif

std::tuple<double, double, double, std::array<arma::mat *, C_emo_max_j> *, double *> OnlineMIMOSVR::cuvalidate(
        const double gamma_variance, const double labels_mean, const double lambda, const double gamma_param, const double svr_epsilon,
        const double all_labels_meanabs, const size_t train_len, const size_t lag,
        const arma::mat &tune_cuml, const arma::mat &train_cuml, const arma::mat &chunk_labels,
        const size_t iters, double &test_train_mape_ratio, bool &mape_ratio_unset, omp_lock_t *p_mape_l, const size_t gpu_id)
{
#ifdef VALGRIND_BUILD
    return std::tuple{1., 1., 1.};
#endif
    const size_t m = tune_cuml.n_cols;
    const auto mm = m * m;
    const auto mm_size = mm * sizeof(double);
    const size_t n = chunk_labels.n_cols;
    // const size_t mn = m * n;
    const auto K_train_len = train_len * train_len;
    const auto K_train_size = K_train_len * sizeof(double);
    const auto train_len_n = train_len * n;
    const auto train_n_size = train_len_n * sizeof(double);
    // const auto neg_epsilon = -svr_epsilon;
    assert(chunk_labels.n_rows == m);
    assert(train_cuml.n_rows == tune_cuml.n_rows);

    cu_errchk(cudaSetDevice(gpu_id));
    magma_queue_t ma_queue;
    magma_queue_create(gpu_id, &ma_queue);
    const auto custream = magma_queue_get_cuda_stream(ma_queue);

    const size_t dim = train_cuml.n_rows / lag;
    const size_t lag_tile_width = std::ceil(double(lag) / double(CUDA_TILE_WIDTH));
    double *d_train_cuml, *d_K_train;
    cu_errchk(cudaMallocAsync((void **) &d_train_cuml, train_cuml.n_elem * sizeof(double), custream));
    cu_errchk(cudaMemcpyAsync(d_train_cuml, train_cuml.mem, train_cuml.n_elem * sizeof(double), cudaMemcpyHostToDevice, custream));
    cu_errchk(cudaMallocAsync((void **) &d_K_train, K_train_size, custream));
    kernel::path::G_kernel_xx<<<CUDA_THREADS_BLOCKS_2D(train_len), 0, custream>>>(train_len, train_cuml.n_rows, lag, dim, lag_tile_width, lambda, d_train_cuml, d_K_train);

    const auto mingamma = calc_mingamma(d_K_train, labels_mean, double(train_len), K_train_len, custream);
    cu_errchk(cudaFreeAsync(d_train_cuml, custream));
    const auto gamma = (gamma_param * gamma_variance + 1. - gamma_variance) * mingamma;
    const auto gamma_2_2 = 2. * gamma * gamma;
    solvers::G_kernel_from_distances_inplace<<<CUDA_THREADS_BLOCKS(K_train_len), 0, custream>>>(d_K_train, K_train_len, gamma_2_2);
    const auto epsco = 1. - solvers::sum(d_K_train, K_train_len, custream) / double(K_train_len);
    // cu_errchk(cudaFreeAsync(d_K_train, custream)); // We keep it in case we need it later

    double *d_K_tune, *d_tune_cuml;
    cu_errchk(cudaMallocAsync((void **) &d_tune_cuml, tune_cuml.n_elem * sizeof(double), custream));
    cu_errchk(cudaMemcpyAsync(d_tune_cuml, tune_cuml.mem, tune_cuml.n_elem * sizeof(double), cudaMemcpyHostToDevice, custream));
    cu_errchk(cudaMallocAsync((void **) &d_K_tune, mm_size, custream));
    kernel::path::G_kernel_xx<<<CUDA_THREADS_BLOCKS_2D(m), 0, custream>>>(m, tune_cuml.n_rows, lag, dim, lag_tile_width, lambda, gamma_2_2, d_tune_cuml, d_K_tune);
    cu_errchk(cudaFreeAsync(d_tune_cuml, custream));

    double total_score = 0;
    const ssize_t start_point = m - C_emo_test_len - train_len;
    assert (start_point >= 0);
    double *d_K_epsco;
    cu_errchk(cudaMallocAsync((void **) &d_K_epsco, mm_size, custream));
    cu_errchk(cudaMemcpyAsync(d_K_epsco, d_K_tune, mm_size, cudaMemcpyDeviceToDevice, custream));
    arma::mat h_K(m, m);
    cu_errchk(cudaMemcpyAsync(h_K.memptr(), d_K_tune, mm_size, cudaMemcpyDeviceToHost, custream));
    cu_errchk(cudaFreeAsync(d_K_tune, custream));
    G_set_diag<<<CUDA_THREADS_BLOCKS(m), 0, custream>>>(d_K_epsco, epsco, m);

    const auto labels_start = chunk_labels.mem + start_point;
    arma::mat h_K_epsco(m, m), h_j_K(train_len, train_len), h_j_K_epsco(train_len, train_len), h_error_mat(train_len, n),
        h_j_train_labels(train_len, n), h_solved(train_len, n), h_mult(train_len, n), h_left(train_len, train_len), h_right(train_len, n);
    cu_errchk(cudaHostRegister(h_K_epsco.memptr(), mm_size, cudaHostRegisterDefault));
    cu_errchk(cudaHostRegister(h_solved.memptr(), train_n_size, cudaHostRegisterDefault));
    cu_errchk(cudaHostRegister(h_left.memptr(), K_train_size, cudaHostRegisterDefault));
    cu_errchk(cudaHostRegister(h_right.memptr(), train_n_size, cudaHostRegisterDefault));
    double *j_K_epsco, *d_left, *d_solved;
    cu_errchk(cudaMemcpyAsync(h_K_epsco.memptr(), d_K_epsco, mm_size, cudaMemcpyDeviceToHost, custream));
    cu_errchk(cudaMallocAsync((void **) &d_solved, train_n_size, custream));
    cu_errchk(cudaMallocAsync((void **) &d_left, K_train_size, custream));
    cu_errchk(cudaMallocAsync((void **) &j_K_epsco, K_train_size, custream));
    magma_int_t info;
    const double iter_div = common::C_itersolve_range / double(iters);
    const arma::mat sum_rows_ones(n, train_len, arma::fill::ones);
    auto p_predictions = new std::array<arma::mat *, C_emo_max_j>;
    memset(p_predictions, 0, C_emo_max_j * sizeof(double));
#pragma omp unroll
    for (size_t j = 0; j < C_emo_max_j; ++j)
        if (total_score != C_bad_validation) {
            const auto train_start = j * C_emo_slide_skip;
            const auto train_end = train_start + train_len;
            const auto test_start = train_end;
            const auto test_len = m - start_point - test_start;
            const unsigned test_len_n = test_len * n;
            const auto test_n_size = test_len_n * sizeof(double);
//            const unsigned K_test_len = test_len * train_len;
            if (test_len > train_len) LOG4_THROW("Test " << test_len << " length should be smaller than train dimension " << train_len);

            LOG4_TRACE("Try " << j << ", start point " << start_point << ", train start " << train_start << ", train final " << train_end << ", test start " <<
                              test_start << ", test len " << test_len << ", train len " << train_len << ", current score " << total_score << ", start point ");

            auto best_score = std::numeric_limits<double>::infinity();
            size_t best_iter = 0;

            cb_errchk(cublasSetMatrixAsync(train_len, n, sizeof(double), labels_start + train_start, m, d_solved, train_len, custream));
            copy_submat(d_K_epsco, j_K_epsco, m, start_point + train_start, start_point + train_start, start_point + train_end, start_point + train_end, custream);
            ma_errchk(magma_dgesv_rbt_q(MagmaTrue, train_len, n, j_K_epsco, train_len, d_solved, train_len, &info, ma_queue));
            h_j_K = h_K.submat(start_point + train_start, start_point + train_start, start_point + train_end - 1, start_point + train_end - 1);
            h_j_K_epsco = h_K_epsco.submat(start_point + train_start, start_point + train_start, start_point + train_end - 1, start_point + train_end - 1);
            h_j_train_labels = chunk_labels.rows(start_point + train_start, start_point + train_end - 1);
            const arma::mat h_K_test = h_K.submat(start_point + test_start, start_point + train_start, m - 1, start_point + train_end - 1);
            const arma::mat labels_test = chunk_labels.rows(start_point + test_start, m - 1);
            arma::mat h_error_test(arma::size(labels_test));
#ifndef __GNUC__
#pragma unroll
#endif
            for (size_t i = 1; i < iters + 1; ++i) {
                cu_errchk(cudaMemcpyAsync(h_solved.memptr(), d_solved, train_n_size, cudaMemcpyDeviceToHost, custream));
                // memcpy(h_error_test.memptr(), labels_test.mem, test_n_size);
                memcpy(h_error_mat.memptr(), h_j_train_labels.mem, train_n_size);
                cu_errchk(cudaDeviceSynchronize());
                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                            train_len, n, train_len, 1, h_j_K.mem, train_len, h_solved.mem, train_len, -1, h_error_mat.memptr(), train_len);
                const auto train_mape = cblas_dasum(train_len_n, h_error_mat.mem, 1) / train_len_n / all_labels_meanabs;
                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                            test_len, n, train_len, 1, h_K_test.mem, test_len, h_solved.mem, train_len, 0, h_error_test.memptr(), test_len);
                auto p_test_preds = new arma::mat(arma::size(labels_test));
                memcpy(p_test_preds->memptr(), h_error_test.mem, test_n_size);
                cblas_daxpy(test_len_n, -1, labels_test.mem, 1, h_error_test.memptr(), 1);
                const auto test_mape = cblas_dasum(test_len_n, h_error_test.mem, 1) / test_len_n / all_labels_meanabs;
                if (mape_ratio_unset) {
                    omp_set_lock(p_mape_l);
                    if (mape_ratio_unset) {
                        test_train_mape_ratio = test_mape / train_mape;
                        mape_ratio_unset = true;
                    }
                    omp_unset_lock(p_mape_l);
                }
                const auto this_score = test_mape + train_mape * test_train_mape_ratio;

                /* if (!std::isnormal(this_score)) LOG4_WARN("Score not normal " << this_score << ", iteration " << i); else */
                if (this_score < best_score) {
                    LOG4_TRACE("IRWLS iteration " << i << ", train MAPE " << train_mape << ", test MAPE " << test_mape << ", kernel dimensions " << train_len << "x" <<
                                      train_len << ", former best score " << best_score << ", new best score " << this_score);
                    best_score = this_score;
                    best_iter = i;
                    if (p_predictions->at(j)) delete p_predictions->at(j);
                    p_predictions->at(j) = p_test_preds;
                } else delete p_test_preds;
                if (i == iters) break;
                h_mult = arma::sqrt(h_error_mat + common::C_itersolve_delta / (double(i) * iter_div));
                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                            train_len, train_len, n, 1, h_mult.mem, train_len, sum_rows_ones.mem, n, 0, h_left.memptr(), train_len);
                // h_left %= h_j_K_epsco;
                vdMul(K_train_len, h_left.mem, h_j_K_epsco.mem, h_left.memptr());
                // h_left = h_mult * sum_rows_ones % h_j_K_epsco;
                // h_right = h_j_train_labels % h_mult;
                vdMul(train_len_n, h_j_train_labels.mem, h_mult.mem, h_right.memptr());
                cu_errchk(cudaMemcpyAsync(d_left, h_left.mem, K_train_size, cudaMemcpyHostToDevice, custream));
                cu_errchk(cudaMemcpyAsync(d_solved, h_right.mem, train_n_size, cudaMemcpyHostToDevice, custream));
                ma_errchk(magma_dgesv_rbt_q(MagmaTrue, train_len, n, d_left, train_len, d_solved, train_len, &info, ma_queue));
            }

            LOG4_TRACE("IRWLS best iteration " << best_iter << ", kernel dimensions " << train_len << "x" << train_len << ", delta " << common::C_itersolve_delta <<
               ", range " << common::C_itersolve_range << ", solution " << train_len << "x" << n << ", test " << test_len << "x" << n << ", score " << best_score << ", former total score " << total_score);
            if (!std::isnormal(best_score)) {
                LOG4_ERROR("Score not normal " << best_score << " for epsco " << epsco << ", svr epsilon " << svr_epsilon << ", gamma " << gamma << ", lambda " << lambda
                    << ", meanabs labels " << all_labels_meanabs << ", previous score " << total_score);
                total_score = C_bad_validation;
            } else
                total_score += best_score;
        }
    cu_errchk(cudaHostUnregister(h_K_epsco.memptr()));
    cu_errchk(cudaHostUnregister(h_solved.memptr()));
    cu_errchk(cudaHostUnregister(h_left.memptr()));
    cu_errchk(cudaHostUnregister(h_right.memptr()));
    cu_errchk(cudaFreeAsync(d_solved, custream));
    cu_errchk(cudaFreeAsync(d_left, custream));
    cu_errchk(cudaFreeAsync(j_K_epsco, custream));
    cu_errchk(cudaFreeAsync(d_K_epsco, custream));
    cu_errchk(cudaDeviceSynchronize());
#if 0
    arma::mat h_K_epsco = h_K_tune;
    h_K_epsco.diag() += epsco;

    magmaDouble_ptr d_a, d_b;
    ma_errchk(magma_dmalloc(&d_a, train_len * train_len));
    ma_errchk(magma_dmalloc(&d_b, train_len * n));
#pragma omp unroll
    for (size_t j = 0; j < C_emo_max_j; ++j)
        if (score != C_bad_validation) {

            const size_t train_start = j * C_emo_slide_skip;
            const size_t train_final = train_start + train_len - 1;
            const size_t test_start = train_final;
            const size_t test_len = m - start_point - test_start - 1; // -1 if arma
            LOG4_TRACE("Try " << j << ", start point " << start_point << ", train start " << train_start << ", train final " << train_final << ", test start " <<
                              test_start << ", test len " << test_len << ", train len " << train_len << ", current score " << score);

            const auto test_labels = chunk_labels.rows(start_point + test_start, chunk_labels.n_rows - 1);
            const auto preds =
                    h_K_tune.submat(start_point + test_start, start_point + train_start, h_K_tune.n_rows - 1, start_point + train_final)
                    * OnlineMIMOSVR::solve_irwls(
                            h_K_epsco.submat(start_point + train_start, start_point + train_start, start_point + train_final, start_point + train_final),
                            h_K_tune.submat(start_point + train_start, start_point + train_start, start_point + train_final, start_point + train_final),
                            chunk_labels.rows(start_point + train_start, start_point + train_final),
                            iters, magma_queue, d_a, d_b) - svr_epsilon;
            const auto this_score = common::meanabs<double>(test_labels - preds) / all_labels_meanabs;

            if (!std::isnormal(this_score)) {
                LOG4_ERROR(
                        "Score not normal " << this_score << " for epsco " << epsco << ", svr epsilon " << svr_epsilon << ", gamma " << gamma << ", lambda " << lambda <<
                                            ", meanabs labels " << all_labels_meanabs << ", previous score " << score);
                score = C_bad_validation;
            } else
                score += this_score;
        }
    ma_errchk(magma_free(d_a));
    ma_errchk(magma_free(d_b));
#endif
    magma_queue_destroy(ma_queue);
    return {total_score / double(C_emo_max_j), gamma, epsco, p_predictions, d_K_train};
}


}
}
