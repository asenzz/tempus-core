//
// Created by zarko on 11/11/21.
//

//#define BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS 1 // Otherwise Valgrind crashes on init
#include <armadillo>
#include <boost/math/distributions/students_t.hpp>
#include <mutex>
#include <cmath>
#include <cstddef>
#include <unordered_set>
#include <vector>
#include "common/parallelism.hpp"
#include "common/gpu_handler.hpp"
#include "common/exceptions.hpp"
#include "util/math_utils.hpp"
#include "onlinesvr.hpp"
#include "cuda_path.hpp"
#include "kernel_factory.hpp"
#include "cuqrsolve.hpp"

namespace svr {


arma::mat OnlineMIMOSVR::init_predict_kernel_matrix(
        const SVRParameters &svr_parameters,
        const arma::mat &x_train,
        const arma::mat &x_predict,
        const OnlineMIMOSVR_ptr &p_manifold,
        const bpt::ptime &predicted_time)
{
    arma::mat predict_kernel_matrix;
    switch (svr_parameters.get_kernel_type()) {
        case kernel_type_e::DEEP_PATH:
        case kernel_type_e::DEEP_PATH2: {
            if (!p_manifold) LOG4_THROW("Manifold model not ready!");
            predict_kernel_matrix = init_predict_manifold_matrix(x_train, x_predict, p_manifold);
            break;
        }

        case kernel_type_e::PATH: {
            // predict_kernel_matrix.set_size(arma::size(Zy));
            // solvers::kernel_from_distances(predict_kernel_matrix.memptr(), Zy.memptr(), Zy.n_rows, svr_parameters.get_svr_kernel_param());
            arma::mat *p_Zy;
            PROFILE_EXEC_TIME(p_Zy = prepare_Zy(svr_parameters, x_train.t(), x_predict.t(), predicted_time), "Prepare Zy, params " << svr_parameters.to_string() << ", trained features " << arma::size(x_train) << ", predict features " << arma::size(x_predict));
            predict_kernel_matrix = 1. - *p_Zy / (2. * std::pow<double>(svr_parameters.get_svr_kernel_param(), 2));
            delete p_Zy;
            break;
        }
        default:
            LOG4_DEBUG("Reintroduce!");
    }

    if (predict_kernel_matrix.empty()) THROW_EX_FS(std::logic_error, "Test kernel matrix is empty!");
    LOG4_TRACE("Predict kernel matrix of size " << arma::size(predict_kernel_matrix));
    return predict_kernel_matrix;
}


arma::mat
OnlineMIMOSVR::init_dist_predict_kernel_matrix(
        const SVRParameters &svr_parameters,
        const arma::mat &x_train,
        const arma::mat &x_predict,
        const OnlineMIMOSVR_ptr &p_manifold)
{
    // Don't use slow and buggy !
    const size_t num_chunks = x_train.n_rows / CHUNK_SIZE;
    if (true /*x_predict.n_rows != 1 or num_chunks < 3*/) {
        LOG4_DEBUG("Too small or only for single predict row " << arma::size(x_predict) << " dist predict matrix works!");
        return init_predict_kernel_matrix(svr_parameters, x_train, x_predict, p_manifold, bpt::not_a_date_time);
    }
    // Slower!
    const size_t chunk_size = x_train.n_rows / num_chunks;
    arma::mat all_predict_kernel_matrix(x_predict.n_rows, x_train.n_rows, arma::fill::zeros);
    __omp_pfor(chunk_ix, 0, num_chunks,
        const auto chunk_ixs = arma::linspace<arma::uvec>(chunk_ix * chunk_size, chunk_ix == num_chunks - 1 ? x_train.n_rows - 1 : (chunk_ix + 1) * chunk_size - 1);
        const auto predict_kernel_matrix = init_predict_kernel_matrix(svr_parameters, x_train.rows(chunk_ixs), x_predict, p_manifold, bpt::not_a_date_time);
        all_predict_kernel_matrix.cols(chunk_ixs) = predict_kernel_matrix;
    )
    return all_predict_kernel_matrix;
}


arma::mat
accumulate_chunks(
        const std::vector<unsigned> &outliers, const std::vector<size_t> &min_ixs, const std::vector<size_t> &max_ixs,
        const std::vector<arma::mat> &chunk_predicted, const SVRParameters &svr_parameters)
{
    arma::mat start_mat = arma::zeros(arma::size(chunk_predicted.front()));
#pragma omp parallel for
    for (size_t i = 0; i < outliers.size(); ++i) {
        size_t counter = 0;
        for (size_t j = 0; j < chunk_predicted.size(); ++j) {
            if (((outliers[i] & 1u) == 0 or j != min_ixs[i]) and ((outliers[i] & 2u) == 0 or j != max_ixs[i])) {
                ++counter;
                start_mat.row(i) = start_mat.row(i) + chunk_predicted[j].row(i);
            } else if (outliers[i] != 0) {
                LOG4_WARN(
                        "Outlier found of type " << outliers[i] << " at chunk " << j << " value " << chunk_predicted[j](i, chunk_predicted[j].n_cols - 1) << " params "
                         << svr_parameters.get_svr_kernel_param() << " " << svr_parameters.get_svr_kernel_param2());
            }
        }
        start_mat.row(i) = start_mat.row(i) / double(counter);
    }
    return start_mat;
}

std::vector<unsigned>
outlier_test(
        const arma::mat &data,
        const double alpha,
        std::vector<size_t> &min_ixs,
        std::vector<size_t> &max_ixs)
{
    min_ixs.resize(data.n_rows, 0);
    max_ixs.resize(data.n_rows, 0);

    std::vector<unsigned> outliers(data.n_rows, 0);
    if (data.n_rows < 1) {
        LOG4_DEBUG("Data is empty!");
        return outliers;
    }
    if (data.row(0).n_elem < 3) {
        __omp_pfor_i(0, data.n_rows,
            min_ixs[i] = index_min(data.row(i));
            max_ixs[i] = index_max(data.row(i));
            outliers[i] = 0;
        )
        return outliers;
    }

    const boost::math::students_t distrib(data.n_cols - 2);
    const double ddata_size = data.n_cols;
    const double t_level = quantile(distrib, alpha / ddata_size);
    const double G_limit_min = (ddata_size - 1) / sqrt(ddata_size) * sqrt(std::pow(t_level, 2) / (ddata_size - 2 + std::pow(t_level, 2)));
    const double G_limit_max = (ddata_size - 1) / sqrt(ddata_size) * sqrt(std::pow(t_level, 2) / (ddata_size - 2 + std::pow(t_level, 2)));

#pragma omp parallel for
    for (size_t i = 0; i < data.n_rows; ++i) {
        const auto data_row = data.row(i);
        const double data_mean = arma::mean(data_row);
        const double data_stddev = arma::stddev(data_row);

        const double min_value = min(data_row);
        const double max_value = max(data_row);
        min_ixs[i] = index_min(data_row);
        max_ixs[i] = index_max(data_row);

        const double G_min = (data_mean - min_value) / data_stddev;
        const double G_max = (max_value - data_mean) / data_stddev;

        if (G_min > G_limit_min) {
            outliers[i] = 1;
            if (G_max > G_limit_max)
                outliers[i] = 3;
        } else if (G_max > G_limit_max)
            outliers[i] = 2;
    }
    return outliers;
}


void log_model(const svr::OnlineMIMOSVR &m, const std::vector<arma::mat> &chunk_kernels)
{
    static size_t call_ct = 0;
    svr::common::armd::serialize_mat(
            svr::common::formatter() << "/mnt/faststore/problematic_level_" << m.get_svr_parameters().get_decon_level() << "_" << m.get_svr_parameters().get_input_queue_column_name()
                                     << "_learning_matrix.txt", *m.p_features, "learning_matrix");
    svr::common::armd::serialize_mat(
            svr::common::formatter() << "/mnt/faststore/problematic_level_" << m.get_svr_parameters().get_decon_level() << "_" << m.get_svr_parameters().get_input_queue_column_name()
                                     << "_reference_matrix.txt", *m.p_labels, "reference_matrix");
    const auto &component = m.main_components.at(0);
    const std::string KERNEL_STR("_kernel_chunk_call" + std::to_string(call_ct));
    const std::string WEIGHT_STR("_weight_chunk_call" + std::to_string(call_ct));
    const std::string IXS_STR("_ixs_call" + std::to_string(call_ct));
    __omp_pfor_i(0, m.ixs.size(),
                 std::stringstream ker;
        std::stringstream wei;
        std::stringstream ixs;
        ker << "/mnt/faststore/problematic_level_" << m.get_svr_parameters().get_decon_level() << "_" << m.get_svr_parameters().get_input_queue_column_name() << "_ix" << i << KERNEL_STR;
        wei << "/mnt/faststore/problematic_level_" << m.get_svr_parameters().get_decon_level() << "_" << m.get_svr_parameters().get_input_queue_column_name() << "_ix" << i << WEIGHT_STR;
        ixs << "/mnt/faststore/problematic_level_" << m.get_svr_parameters().get_decon_level() << "_" << m.get_svr_parameters().get_input_queue_column_name() << "_ix" << i << IXS_STR;
        svr::common::armd::serialize_mat(wei.str(), component.chunk_weights[i], wei.str());
        svr::common::armd::serialize_mat(ker.str(), chunk_kernels[i], ker.str());
        svr::common::armd::serialize_vec(ixs.str(), m.ixs[i], ixs.str());
    )
    ++call_ct;
}

std::vector<size_t> OnlineMIMOSVR::sort_indexes(const std::vector<double> &v)
{

    std::vector<size_t> idx(v.size());
    iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) { return v[i1] < v[i2]; });

    return idx;
}

#ifdef DEBUG_PREDICTIONS
std::map<std::pair<size_t, std::string>, arma::mat> prev_pred;
#endif

arma::mat OnlineMIMOSVR::chunk_predict(const arma::mat &x_predict, const bpt::ptime &pred_time) const
{
#ifdef PSEUDO_PREDICT_LEVEL
    if (p_svr_parameters->get_decon_level() == PSEUDO_PREDICT_LEVEL)
        return p_labels->row(p_labels->n_rows - 1);
#endif
#ifdef DEBUG_PREDICTIONS // Debug log to file
    if (get_svr_parameters().get_decon_level() == C_logged_level) {
        static size_t call_ct = 0;
        std::stringstream ss_file_name;
        ss_file_name << "/mnt/faststore/level_" << this->get_svr_parameters().get_decon_level() << "_" << p_svr_parameters.get_input_queue_column_name() << "_chunk_predict_" << call_ct++ << ".csv";
        x_predict.save(ss_file_name.str(), arma::csv_ascii);
    }
#endif
    if (ixs.size() == 1 && main_components.size() == 1)
        return init_predict_kernel_matrix(*p_svr_parameters, *p_features, x_predict, p_manifold, pred_time) * main_components.begin()->second.total_weights;

    std::vector<size_t> min_ixs;
    std::vector<size_t> max_ixs;
    std::vector<arma::mat> predicted;
    arma::mat start_mat = arma::zeros(x_predict.n_rows, multistep_len);
    std::vector<arma::mat> chunk_predicted;

    std::mutex mx;
    std::vector<double> mae_total(ixs.size(), 0.);
    for (auto &kv: main_components) __omp_pfor_i(0, mae_total.size(), mae_total[i] += kv.second.mae_chunk_values[i])
    auto sorted_ix = sort_indexes(mae_total);
    std::unordered_set<size_t> best_ixs(sorted_ix.begin(), sorted_ix.size() < 2 ? sorted_ix.end() : sorted_ix.begin() + std::round(sorted_ix.size() / BEST_PREDICT_CHUNKS_DIVISOR));
    for (auto &kv: main_components) {
#pragma omp parallel for schedule(static, 1)
        for (size_t i = 0; i < ixs.size(); ++i) {
            if (best_ixs.find(i) != best_ixs.end()) {
                arma::mat multiplicated(arma::size(p_labels->rows(ixs[i])));
                if (p_svr_parameters->is_manifold()) {
                    const arma::mat chunk_kernel_matrix = init_predict_kernel_matrix(*p_svr_parameters, x_predict, p_features->rows(ixs[i]), p_manifold, pred_time);
                    multiplicated = arma::mean(chunk_kernel_matrix + p_labels->rows(ixs[i]));
                } else {
                    const arma::mat chunk_kernel_matrix = init_predict_kernel_matrix(*p_svr_parameters, p_features->rows(ixs[i]), x_predict, p_manifold, pred_time);
#pragma omp parallel for
                    for (size_t col_ix = 0; col_ix < kv.second.chunk_weights[i].n_cols; ++col_ix)
                        multiplicated.col(col_ix) = kv.second.epsilon + chunk_kernel_matrix * kv.second.chunk_weights[i].col(col_ix);
                }
#pragma omp critical(add_chunk_predicted)
                chunk_predicted.emplace_back(multiplicated);
            }
        }

        arma::mat sum_steps = arma::zeros(x_predict.n_rows, chunk_predicted.size());
        __tbb_pfor_i(0, chunk_predicted.size(),
            // last_step.col(i) = chunk_predicted[i].col(multistep_len - 1);
            // Replacing last column check only for sum of all columns
            //arma::colvec sum_of_multistep_predictions = arma::sum(chunk_predicted[i], 1);
            sum_steps.col(i) = arma::colvec(arma::sum(chunk_predicted[i], 1)); // TODO Is this basically transpose?
        )
        std::vector<unsigned> outlier_result = outlier_test(sum_steps, OUTLIER_ALPHA, min_ixs, max_ixs);
        arma::mat accumulated = accumulate_chunks(outlier_result, min_ixs, max_ixs, chunk_predicted, *p_svr_parameters);
#ifdef OUTLIER_TEST
        if (accumulated.max() > SANE_PREDICT or accumulated.min() < -SANE_PREDICT or accumulated.has_nan() or accumulated.has_inf()) {
            LOG4_ERROR("Accumulated predict abs values are bigger than " << SANE_PREDICT << " max " << accumulated.max() << " min " << accumulated.min());
/*
            static size_t call_ct = 0;
            //svr::common::armd::serialize_mat(svr::common::formatter() << "/mnt/faststore/problematic_xtest_level_" << p_svr_parameters.get_decon_level() << "_" << p_svr_parameters.get_input_queue_column_name() << "_call_" << call_ct << ".txt", x_predict, "Xtest");
            ++call_ct;
            //log_model(*this, chunk_kernel_matrixes);
*/
            THROW_EX_FS(svr::common::bad_prediction, "Predicted values don't look sane. " << accumulated << " parameters " << p_svr_parameters->to_string());
        }
#endif
//#pragma omp critical(predict_add)
        predicted.emplace_back(accumulated);
    }
    arma::mat pred = (std::accumulate(predicted.begin(), predicted.end(), start_mat) / double(predicted.size())) * labels_scaling_factor;
#ifdef DEBUG_PREDICTIONS // Debug log to file
    const auto prev_pred_find = prev_pred.find({p_svr_parameters.get_decon_level(), p_svr_parameters.get_input_queue_column_name()});
    if (prev_pred_find != prev_pred.end()) {
        LOG4_DEBUG("level " << p_svr_parameters.get_decon_level() << " column " << p_svr_parameters.get_input_queue_column_name() << " prev_pred_find " << prev_pred_find->second(0, 0) << " pred " << pred(0, 0) << " prev label " << p_labels->at(p_labels->n_rows - 1, 0) << " prev prev label " << p_labels->at(p_labels->n_rows - 2, 0));
        const auto diff1 = p_labels->at(p_labels->n_rows - 1, 0) - prev_pred_find->second(0, 0);
        const auto diff2 = p_labels->at(p_labels->n_rows - 1, 0) - p_labels->at(p_labels->n_rows - 2, 0);
        LOG4_DEBUG("level " << p_svr_parameters.get_decon_level() << " column " << p_svr_parameters.get_input_queue_column_name() << " prev label - prev_pred_find " << diff1 << " prev label - prev prev label" << diff2 << " prev pred is wronger " << (std::abs(diff1) > std::abs(diff2)));
    }
    prev_pred[{p_svr_parameters.get_decon_level(), p_svr_parameters.get_input_queue_column_name()}] = pred;
    if (get_svr_parameters().get_decon_level() == C_logged_level) {
        static size_t call_ct = 0;
        std::stringstream ss_file_name;
        ss_file_name << "/mnt/faststore/level_" << p_svr_parameters.get_decon_level() << "_" << p_svr_parameters.get_input_queue_column_name() << "_chunk_predict_output_" << call_ct++ << ".csv";
        pred.save(ss_file_name.str(), arma::csv_ascii);
    }
#endif
    LOG4_TRACE("Level " << p_svr_parameters->get_decon_level() << " predicted " << pred);
    return pred;
}


arma::mat OnlineMIMOSVR::single_chunk_predict(const arma::mat &x_predict, const bpt::ptime &pred_time, const size_t chunk_ix) const
{
    return init_predict_kernel_matrix(*p_svr_parameters, p_features->rows(ixs[chunk_ix]), x_predict, p_manifold, pred_time) * main_components.begin()->second.chunk_weights[chunk_ix];
}

} // namespace svr {