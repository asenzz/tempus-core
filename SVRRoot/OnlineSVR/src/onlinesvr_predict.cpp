//
// Created by zarko on 11/11/21.
//

//#define BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS 1 // Otherwise Valgrind crashes on init
#include <armadillo>
#include <boost/math/distributions/students_t.hpp>
#include <mutex>
#include <cmath>
#include <tuple>
#include <unordered_set>
#include <vector>
#include "common/parallelism.hpp"
#include "common/gpu_handler.hpp"
#include "util/math_utils.hpp"
#include "onlinesvr.hpp"
#include "cuqrsolve.hpp"

namespace svr {


arma::mat OnlineMIMOSVR::init_predict_kernel_matrix(
        const datamodel::SVRParameters &svr_parameters,
        const arma::mat &x_train,
        const arma::mat &x_predict,
        const bpt::ptime &predicted_time)
{
    const size_t predict_num_chunks = std::ceil(double(x_predict.n_rows) / double(C_max_predict_chunk_size));
    const size_t predict_chunk_size = x_predict.n_rows / predict_num_chunks;
    arma::mat predict_K(x_predict.n_rows, x_train.n_rows);

    switch (svr_parameters.get_kernel_type()) {
        case datamodel::kernel_type_e::PATH:
#pragma omp parallel for schedule(static, 1) num_threads(common::gpu_handler::get().get_max_running_gpu_threads_number())
            for (size_t i = 0; i < predict_num_chunks; ++i) {
                arma::mat *p_Zy;
                const size_t start_row = i * predict_chunk_size;
                const size_t end_row = i == predict_num_chunks - 1 ? x_predict.n_rows - 1 : (i + 1) * predict_chunk_size - 1;
                PROFILE_EXEC_TIME(
                        p_Zy = prepare_Zy(svr_parameters, x_train.t(), x_predict.rows(start_row, end_row).t(), predicted_time),
                        "Prepare Zy, params " << svr_parameters << ", start row " << start_row << ", end row " << end_row);
                arma::mat predict_K_chunk(arma::size(*p_Zy));
                solvers::kernel_from_distances(predict_K_chunk.memptr(), p_Zy->mem, p_Zy->n_rows, p_Zy->n_cols, svr_parameters.get_svr_kernel_param());
                predict_K.rows(start_row, end_row) = predict_K_chunk;
                delete p_Zy;
            }
            break;

        default:
            LOG4_ERROR("Kernel type " << size_t(svr_parameters.get_kernel_type()) << " not implemented!");
    }
    LOG4_DEBUG("Predict kernel matrix of size " << arma::size(predict_K) << ", trained features " << arma::size(x_train) << ", predict features " << arma::size(x_predict));
    return predict_K;
}

arma::mat OnlineMIMOSVR::predict(const arma::mat &x_predict, const bpt::ptime &pred_time, const bool masked) const
{
    if (is_manifold()) return manifold_predict(x_predict);

    arma::mat prediction;
#pragma omp parallel for collapse(2) schedule(static, 1) num_threads(adj_threads(ixs.size() * main_components.size()))
    for (size_t i = 0; i < ixs.size(); ++i) {
        for (size_t k = 0; k < main_components.size(); ++k) {
            const auto &com = *main_components[k];
            const arma::mat chunk_predict_K = init_predict_kernel_matrix(get_params(i), p_features->rows(ixs.at(i)), x_predict, pred_time);
            const auto chunk_weights = masked ? com.chunk_weights.at(i) % com.weights_mask[i] : com.chunk_weights.at(i);
            arma::mat multiplicated(chunk_predict_K.n_rows, chunk_weights.n_cols);
#pragma omp parallel for collapse(2) num_threads(adj_threads(chunk_predict_K.n_rows * chunk_weights.n_cols))
            for (size_t r = 0; r < chunk_predict_K.n_rows; ++r)
                for (size_t c = 0; c < chunk_weights.n_cols; ++c)
                    multiplicated(r, c) = arma::as_scalar(chunk_predict_K.row(r) * chunk_weights.col(c) - com.epsilon);
            multiplicated *= com.chunks_weight[i];
#pragma omp critical
            prediction = prediction.empty() ? multiplicated : prediction + multiplicated;
        }
    }
    return labels_scaling_factor * prediction / double(ixs.size() * main_components.size());
}


arma::mat OnlineMIMOSVR::single_chunk_predict(const arma::mat &x_predict, const bpt::ptime &pred_time, const size_t chunk_ix) const
{
    return init_predict_kernel_matrix(get_params(chunk_ix), p_features->rows(ixs.at(chunk_ix)), x_predict, pred_time) * main_components.front()->chunk_weights.at(chunk_ix) * main_components.front()->chunks_weight.at(chunk_ix);
}


std::pair<matrix_ptr, matrix_ptr> OnlineMIMOSVR::produce_residuals() const
{
    if (ixs.size() < 2) {
        LOG4_ERROR("At least two chunks are needed to produce residuals.");
        return {};
    }

    arma::vec divisors(p_labels->n_rows, arma::fill::ones);
    arma::mat residuals(arma::size(*p_labels), arma::fill::zeros);
    const auto all_ixs = arma::regspace<arma::uvec>(0, p_labels->n_rows - 1);
    for (size_t chunk_ix = 0; chunk_ix < ixs.size(); ++chunk_ix) {
        auto excluded_chunk_ixs = all_ixs;
        excluded_chunk_ixs.shed_rows(ixs.at(chunk_ix));
        divisors.rows(excluded_chunk_ixs) /= double(main_components.size());
        for (const auto &com: main_components)
            residuals.rows(excluded_chunk_ixs) += p_labels->rows(excluded_chunk_ixs) - com->epsilon -
                init_predict_kernel_matrix(get_params(chunk_ix), p_features->rows(ixs.at(chunk_ix)), p_features->rows(excluded_chunk_ixs), bpt::not_a_date_time) *
                  com->chunk_weights[chunk_ix] * com->chunks_weight[chunk_ix];
    }
    residuals *= divisors;
    const auto res_rows = arma::find(residuals != 0);
    return {std::make_shared<arma::mat>(residuals.rows(res_rows)), std::make_shared<arma::mat>(p_features->rows(res_rows))};
}

} // namespace svr {