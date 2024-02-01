//
// Created by zarko on 11/11/21.
//

//#define BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS 1 // Otherwise Valgrind crashes on init
#include <armadillo>
#include <boost/math/distributions/students_t.hpp>
#include <mutex>
#include <cmath>
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
    arma::mat predict_kernel_matrix;
    switch (svr_parameters.get_kernel_type()) {
        case datamodel::kernel_type_e::PATH: {
            arma::mat *p_Zy;
            PROFILE_EXEC_TIME(
                    p_Zy = prepare_Zy(svr_parameters, x_train.t(), x_predict.t(), predicted_time),
                    "Prepare Zy, params " << svr_parameters.to_string() << ", trained features " << arma::size(x_train) << ", predict features " << arma::size(x_predict));
            predict_kernel_matrix.set_size(arma::size(*p_Zy));
            solvers::kernel_from_distances(predict_kernel_matrix.memptr(), p_Zy->mem, p_Zy->n_rows, p_Zy->n_cols, svr_parameters.get_svr_kernel_param());
            // predict_kernel_matrix = 1. - *p_Zy / (2. * std::pow<double>(svr_parameters.get_svr_kernel_param(), 2));
            delete p_Zy;
            break;
        }
        default:
            LOG4_ERROR("Kernel type " << size_t(svr_parameters.get_kernel_type()) << " not implemented!");
    }

    if (predict_kernel_matrix.empty()) THROW_EX_FS(std::logic_error, "Predict kernel matrix is empty!");
    LOG4_DEBUG("Predict kernel matrix of size " << arma::size(predict_kernel_matrix));
    return predict_kernel_matrix;
}


arma::mat
accumulate_chunks(
        const std::deque<unsigned> &outliers, const std::deque<size_t> &min_ixs, const std::deque<size_t> &max_ixs,
        const std::deque<arma::mat> &chunk_predicted, const datamodel::SVRParameters &svr_parameters)
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
                         << svr_parameters.to_string());
            }
        }
        start_mat.row(i) = start_mat.row(i) / double(counter);
    }
    return start_mat;
}

std::deque<unsigned>
outlier_test(
        const arma::mat &data,
        const double alpha,
        std::deque<size_t> &min_ixs,
        std::deque<size_t> &max_ixs)
{
    min_ixs.resize(data.n_rows, 0);
    max_ixs.resize(data.n_rows, 0);

    std::deque<unsigned> outliers(data.n_rows, 0);
    if (data.n_rows < 1) {
        LOG4_DEBUG("Data is empty!");
        return outliers;
    }
    if (data.n_cols < 3) {
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


void log_model(const svr::OnlineMIMOSVR &m, const std::deque<arma::mat> &chunk_kernels)
{
    static size_t call_ct = 0;
    svr::common::armd::serialize_mat(
            svr::common::formatter() << "/mnt/faststore/problematic_level_" << m.get_params().get_decon_level() << "_" << m.get_params().get_input_queue_column_name()
                                     << "_learning_matrix.txt", *m.p_features, "learning_matrix");
    svr::common::armd::serialize_mat(
            svr::common::formatter() << "/mnt/faststore/problematic_level_" << m.get_params().get_decon_level() << "_" << m.get_params().get_input_queue_column_name()
                                     << "_reference_matrix.txt", *m.p_labels, "reference_matrix");
    const auto &component = m.main_components.at(0);
    const std::string KERNEL_STR("_kernel_chunk_call" + std::to_string(call_ct));
    const std::string WEIGHT_STR("_weight_chunk_call" + std::to_string(call_ct));
    const std::string IXS_STR("_ixs_call" + std::to_string(call_ct));
    __omp_pfor_i(0, m.ixs.size(),
        std::stringstream ker;
        std::stringstream wei;
        std::stringstream ixs;
        ker << "/mnt/faststore/problematic_level_" << m.get_params().get_decon_level() << "_" << m.get_params().get_input_queue_column_name() << "_ix" << i << KERNEL_STR;
        wei << "/mnt/faststore/problematic_level_" << m.get_params().get_decon_level() << "_" << m.get_params().get_input_queue_column_name() << "_ix" << i << WEIGHT_STR;
        ixs << "/mnt/faststore/problematic_level_" << m.get_params().get_decon_level() << "_" << m.get_params().get_input_queue_column_name() << "_ix" << i << IXS_STR;
        svr::common::armd::serialize_mat(wei.str(), component->chunk_weights[i], wei.str());
        svr::common::armd::serialize_mat(ker.str(), chunk_kernels[i], ker.str());
        svr::common::armd::serialize_vec(ixs.str(), m.ixs[i], ixs.str());
    )
    ++call_ct;
}

#ifdef DEBUG_PREDICTIONS
std::map<std::pair<size_t, std::string>, arma::mat> prev_pred;
#endif

arma::mat OnlineMIMOSVR::predict(const arma::mat &x_predict, const bpt::ptime &pred_time) const
{
    if (is_manifold()) return manifold_predict(x_predict);

#ifdef PSEUDO_PREDICT_LEVEL
    if (get_params().get_decon_level() == PSEUDO_PREDICT_LEVEL)
        return p_labels->row(p_labels->n_rows - 1);
#endif
#ifdef DEBUG_PREDICTIONS // Debug log to file
    if (get_param_set().get_decon_level() == C_logged_level) {
        static size_t call_ct = 0;
        std::stringstream ss_file_name;
        ss_file_name << "/mnt/faststore/level_" << this->get_param_set().get_decon_level() << "_" << get_params().get_input_queue_column_name() << "_chunk_predict_" << call_ct++ << ".csv";
        x_predict.save(ss_file_name.str(), arma::csv_ascii);
    }
#endif
    if (ixs.size() == 1 && main_components.size() == 1 && x_predict.n_rows == 1 && main_components.front()->total_weights.n_cols == 1)
        return main_components.front()->epsilon + init_predict_kernel_matrix(get_params(), *p_features, x_predict, pred_time) * main_components.front()->total_weights;

    arma::mat prediction;
    size_t ct = 0;
#pragma omp parallel for collapse(2)
    for (size_t i = 0; i < ixs.size(); ++i) {
        for (size_t k = 0; k < main_components.size(); ++k) {
            const auto &com = *main_components[k];
            const arma::mat chunk_kernel_matrix = init_predict_kernel_matrix(get_params(), p_features->rows(ixs.at(i)), x_predict, pred_time);
            arma::mat multiplicated(chunk_kernel_matrix.n_rows, com.chunk_weights.at(i).n_cols);
#pragma omp parallel for collapse(2)
            for (size_t r = 0; r < chunk_kernel_matrix.n_rows; ++r)
                for (size_t c = 0; c < com.chunk_weights.at(i).n_cols; ++c)
                    multiplicated(r, c) = arma::as_scalar(com.epsilon + chunk_kernel_matrix.row(r) * com.chunk_weights.at(i).col(c));
            if (prediction.empty())
                prediction = multiplicated;
            else
                prediction += multiplicated;
            prediction *= main_components[k]->chunks_weight[i];
            ++ct;
        }
    }
    return labels_scaling_factor * prediction / double(ct);
}


arma::mat OnlineMIMOSVR::single_chunk_predict(const arma::mat &x_predict, const bpt::ptime &pred_time, const size_t chunk_ix) const
{
    return init_predict_kernel_matrix(get_params(chunk_ix), p_features->rows(ixs.at(chunk_ix)), x_predict, pred_time) * main_components.front()->chunk_weights.at(chunk_ix);
}

} // namespace svr {