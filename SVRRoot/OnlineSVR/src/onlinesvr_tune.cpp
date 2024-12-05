//
// Created by zarko on 9/29/22.
//

#include "DQScalingFactorService.hpp"
#include <boost/date_time/posix_time/ptime.hpp>
#include <utility>
#include <vector>
#include <cublas_v2.h>
#include <armadillo>
#include <exception>
#include <limits>
#include <cmath>
#include <deque>
#include "firefly.hpp"
#include "firefly.hpp"
#include "appcontext.hpp"
#include "recombine_parameters.cuh"


namespace svr {
namespace datamodel {

double OnlineMIMOSVR::get_gamma_range_variance(const unsigned train_len)
{
    return 1. / std::sqrt(train_len);
}

#if 0
auto
coot_eval_score(const datamodel::SVRParameters &params, const arma::mat &K, const arma::mat &labels, const arma::mat &last_knowns, const size_t train_len,
           const double meanabs_labels)
{
    const ssize_t start_point_K = K.n_rows - C_test_len - train_len;
    const ssize_t start_point_labels = labels.n_rows - C_test_len - train_len;
    if (start_point_K < 0 || start_point_labels < 0 || labels.n_rows != K.n_rows || labels.n_rows != last_knowns.n_rows)
        LOG4_THROW("Shorter K " << start_point_K << " or labels " << start_point_labels << " for K " << arma::size(K) << ", labels " << arma::size(labels)
                                << ", last-knowns " << arma::size(last_knowns));
    auto p_out_preds = ptr<std::deque<arma::mat>>(C_max_j);
    auto p_out_labels = ptr<std::deque<arma::mat>>(C_max_j);
    auto p_out_last_knowns = ptr<std::deque<arma::mat>>(C_max_j);
    arma::mat K_epsco = K;
    K_epsco.diag() += 1 / params.get_svr_C();
    double score = 0;
    try {
#pragma omp parallel for simd reduction(+:score) schedule(static, 1) num_threads(adj_threads(C_max_j))
        for (size_t j = 0; j < C_max_j; ++j) {
            const size_t x_train_start = j * C_slide_skip;
            const size_t x_train_final = x_train_start + train_len - 1;
            const size_t x_test_start = x_train_final + 1;
            LOG4_TRACE(
                    "Try " << j << ", K " << arma::size(K) << ", start point labels " << start_point_labels << ", start point K " << start_point_K << ", train start " <<
                           x_train_start << ", train final " << x_train_final << ", test start " << x_test_start << ", test final is mat end, train len " << train_len
                           << ", labels " << arma::size(labels) << ", current score " << score);
#ifdef TWO_SIDED_VALIDATION
            const size_t x_test_final = x_test_start + EMO_TEST_LEN - j * EMO_SLIDE_SKIP - 1;
            double this_score = arma::mean(arma::abs(arma::vectorise(K.submat(K.n_rows - 1 - start_point_K - x_test_final, K.n_rows - 1 - start_point_K - x_train_final,
                                                                        K.n_rows - 1 - start_point_K - x_test_start, K.n_rows - 1 - start_point_K - x_train_start)
                                                               * OnlineMIMOSVR::solve_dispatch(
                                                                                               K.submat(K.n_rows - 1 - start_point_K - x_train_final,
                                                                                                        K.n_rows - 1 - start_point_K - x_train_final,
                                                                                                        K.n_rows - 1 - start_point_K - x_train_start,
                                                                                                        K.n_rows - 1 - start_point_K - x_train_start),
                                                                                               labels.rows(labels.n_rows - 1 - start_point_labels - x_train_final,
                                                                                                           labels.n_rows - 1 - start_point_labels - x_train_start),
                                                                                               PROPS.get_online_learn_iter_limit(), false)
                                                               - labels.rows(labels.n_rows - 1 - start_point_labels - x_test_final,
                                                                             labels.n_rows - 1 - start_point_labels - x_test_start)))) / meanabs_labels;
#endif
            p_out_labels->at(j) = labels.rows(start_point_labels + x_test_start, labels.n_rows - 1);
            p_out_last_knowns->at(j) = last_knowns.rows(start_point_labels + x_test_start, last_knowns.n_rows - 1);

            const common::gpu_context ctx;
            const auto gpu_id = ctx.phy_id();
            if (cudaSetDevice(gpu_id) != cudaSuccess) LOG4_THROW("cudaSetDevice " << gpu_id << " failed");
            coot::init_rt(coot::coot_backend_t::CUDA_BACKEND, false, 0, gpu_id);
            p_out_preds->at(j) =
                    K.submat(start_point_K + x_test_start, start_point_K + x_train_start, K.n_rows - 1, start_point_K + x_train_final)
                    * coot::conv_to<arma::mat>::from(OnlineMIMOSVR::solve_irwls(
                            coot::conv_to<coot::mat>::from(
                                    K_epsco.submat(start_point_K + x_train_start, start_point_K + x_train_start, start_point_K + x_train_final, start_point_K + x_train_final)),
                            coot::conv_to<coot::mat>::from(
                                    K.submat(start_point_K + x_train_start, start_point_K + x_train_start, start_point_K + x_train_final, start_point_K + x_train_final)),
                            coot::conv_to<coot::mat>::from(labels.rows(start_point_labels + x_train_start, start_point_labels + x_train_final)),
                            PROPS.get_online_learn_iter_limit(), gpu_id)) - params.get_svr_epsilon();
            const auto this_score = common::meanabs<double>(p_out_labels->at(j) - p_out_preds->at(j)) / meanabs_labels;
            if (!std::isnormal(this_score)) LOG4_THROW("Score not normal for " << params);
            score += this_score;
        }
    } catch (const std::exception &ex) {
        LOG4_ERROR(
                "Error solving matrix, K " << arma::size(K) << ", " << start_point_K << ", " << start_point_labels << ", " << train_len << ", error "
                                           << ex.what());
#pragma omp parallel for simd schedule(static, 1) num_threads(adj_threads(C_max_j))
        for (size_t j = 0; j < C_max_j; ++j) {
            p_out_preds->at(j) = arma::mat(p_out_preds->at(j).n_rows - j * C_slide_skip + train_len, labels.n_cols, arma::fill::value(common::C_bad_validation));
            p_out_labels->at(j) = p_out_preds->at(j);
            p_out_last_knowns->at(j) = p_out_preds->at(j);
        }
        score = common::C_bad_validation;
    }
    return std::tuple(p_out_preds, p_out_labels, p_out_last_knowns, score);
}
#endif


#if 0 // Adaptive grid
        constexpr unsigned C_grid_depth = 6; // Tune
        constexpr double C_grid_range_div = 10;
        double range_min_lambda = C_tune_range_min_lambda, range_max_lambda = C_tune_range_max_lambda;
        for (size_t grid_level_lambda = 0; grid_level_lambda < C_grid_depth; ++grid_level_lambda) {
            const double range_lambda = range_max_lambda - range_min_lambda;
            std::deque<double> lambdas;
            for (double r = range_min_lambda; r < range_max_lambda; r += range_lambda / C_grid_range_div) lambdas.emplace_back(r);
#pragma omp parallel for schedule(static, 1) num_threads(adj_threads(std::min<size_t>(lambdas.size(), TUNE_THREADS)))
            for (const double lambda: lambdas) {
                auto lambda_params = *p_template_chunk_params;
                lambda_params.set_svr_kernel_param2(lambda);
                const auto p_Ztune = prepare_Z(lambda_params, tune_feature_chunks_t[chunk_ix]);
                const auto p_Ztrain = prepare_Z(lambda_params, train_feature_chunks_t[chunk_ix]);
                const auto mingamma = calc_gamma(*p_Ztrain, tune_labels_mean);
                double range_min_gamma = C_tune_range_min_gammamulti, range_max_gamma = C_tune_range_max_gammamulti;
                for (size_t grid_level_gamma = 0; grid_level_gamma < C_grid_depth; ++grid_level_gamma) {
                    const double range_gamma = range_max_gamma - range_min_gamma;
                    const double range_half_gamma = range_gamma * .5;
                    std::deque<double> gamma_multis;
                    for (double r = range_min_gamma; r < range_max_gamma; r += range_gamma / C_grid_range_div) gamma_multis.emplace_back(r);
                    PROFILE_EXEC_TIME(validate_gammas(mingamma, *p_Ztune, *p_Ztrain, gamma_multis, lambda_params, chunk_ix, *p_chunk_preds, &predictions_l),
                                      "Validate " << gamma_multis << " gammas " << lambda_params);
                    const auto best_gamma_multi = (**p_chunk_preds->cbegin()).p_params->get_svr_kernel_param() / mingamma;
                    const double gamma_range_tightening = std::abs(best_gamma_multi - range_half_gamma) / range_half_gamma;
                    range_min_gamma = std::max(C_tune_range_min_gammamulti, best_gamma_multi - range_half_gamma * gamma_range_tightening);
                    range_max_gamma = range_min_gamma + 2. * range_half_gamma * gamma_range_tightening;
                }
                LOG4_DEBUG("Crass gamma pass score " << (**p_chunk_preds->cbegin()).score << ", mingamma " << mingamma << ", best parameters " << *(**p_chunk_preds->cbegin()).p_params);
            }
            const auto best_lambda = (**p_chunk_preds->cbegin()).p_params->get_svr_kernel_param2();
            const double range_half_lambda = range_lambda * .5;
            const double lambda_range_tightening = std::abs(best_lambda - range_half_lambda) / range_half_lambda;
            range_min_lambda = std::max(C_tune_range_min_lambda, best_lambda - range_half_lambda * lambda_range_tightening);
            range_max_lambda = range_min_lambda + 2. * range_half_lambda * lambda_range_tightening;
        }
#endif

void
OnlineMIMOSVR::recombine_params(const unsigned chunkix, const unsigned stepix)
{
    LOG4_BEGIN();

    datamodel::t_level_tuned_parameters *const p_tune_predictions = ccache().recombine_go(*this, chunkix);
    if (!p_tune_predictions) {
        LOG4_DEBUG("Skipping recombine parameters.");
        return;
    }
    const auto colct = p_dataset->get_model_count();
    const auto levct = p_dataset->get_spectral_levels();
    constexpr size_t grad_level = 0; // Only gradient level 0 supported
    const auto max_num_combos = std::pow<double>(double(common::C_tune_keep_preds), double(colct));
    const uint64_t num_combos = (uint64_t) std::min<double>(common::C_num_combos, max_num_combos);
    const double filter_combos = double(max_num_combos) / double(num_combos);
    std::vector<t_param_preds_cu> params_preds(colct * common::C_tune_keep_preds);
#pragma omp parallel for simd collapse(2) num_threads(adj_threads(colct))
    for (uint32_t colix = 0; colix < colct; ++colix) {
        for (uint32_t rowix = 0; rowix < common::C_tune_keep_preds; ++rowix) {
            params_preds[rowix * colct + colix].params_ix = rowix;
            const auto levix = business::ModelService::to_level_ix(colix, levct);
            const auto p_sf = business::DQScalingFactorService::find(scaling_factors, 0, chunkix, grad_level, stepix, levix, false, true);
            auto &tp = p_tune_predictions->at(levix).param_pred[rowix];
UNROLL(C_max_j)
            for (uint32_t j = 0; j < C_max_j; ++j) {
                const uint32_t elto = C_tune_min_validation_window + (C_max_j - j - 1) * C_slide_skip;
UNROLL()
                for (uint32_t el = 0; el < elto; ++el) {
                    params_preds[rowix * colct + colix].predictions[j][el] = arma::mean(tp.p_predictions->at(j)->row(el));
                    params_preds[rowix * colct + colix].labels[j][el] = arma::mean(p_tune_predictions->at(levix).labels[j].row(el));
                    params_preds[rowix * colct + colix].last_knowns[j][el] = arma::mean(p_tune_predictions->at(levix).last_knowns[j].row(el));

                    LOG4_TRACE("Row " << rowix << ", J " << j << ", col " << colix << ", prediction " << params_preds[rowix * colct + colix].predictions[j][el]
                          << ", label " << params_preds[rowix * colct + colix].labels[j][el] << ", last known " << params_preds[rowix * colct + colix].last_knowns[j][el]);
                }
            }
            tp.free();
        }
    }

    const uint32_t rows_gpu = common::gpu_handler_1::get().get_max_gpu_data_chunk_size() / 2 / colct / (2 * common::C_cu_block_size) * (2 * common::C_cu_block_size);
    auto best_score = std::numeric_limits<double>::infinity();
    std::vector<uint8_t> best_params_ixs(colct, uint8_t(0));
        LOG4_DEBUG("Predictions filtered out " << filter_combos << ", total prediction rows " << num_combos << ", rows per GPU " << rows_gpu << ", column count " << colct
                                           << ", limit num combos " << common::C_num_combos);

    // const auto start_time = std::chrono::steady_clock::now();
    t_omp_lock best_score_l;
#pragma omp parallel for schedule(static, 1) num_threads(common::gpu_handler_1::get().get_max_gpu_threads())
    for (uint64_t start_row_ix = 0; start_row_ix < num_combos; start_row_ix += rows_gpu) {
        // if (best_score != std::numeric_limits<double>::max()) continue;
        // if (std::chrono::steady_clock::now() - start_time > std::chrono::minutes(45)) continue;
        const uint64_t end_row_ix = std::min<uint64_t>(start_row_ix + rows_gpu, num_combos);
        const uint64_t chunk_rows_ct = end_row_ix - start_row_ix;
        arma::uchar_mat combos(chunk_rows_ct, colct);
UNROLL()
        for (uint32_t colix = 0; colix < colct; ++colix)
            combos.col(colix) = arma::conv_to<arma::uchar_colvec>::from(common::mod<double>(
                    arma::regspace(start_row_ix, end_row_ix - 1) / std::pow<double>(common::C_tune_keep_preds, colct - colix) / filter_combos, common::C_tune_keep_preds));

        double chunk_best_score;
        std::vector<uint8_t> chunk_best_params_ixs(colct, 0);
        PROFILE_EXEC_TIME(recombine_parameters(chunk_rows_ct, colct, combos.mem, params_preds.data(), &chunk_best_score, chunk_best_params_ixs.data()),
                          "Recombine chunk " << chunk_rows_ct << "x" << colct << ", added set of size " << unsigned(common::C_tune_keep_preds)
                                             << ", filter out " << filter_combos - 1 << " combinations, start row " << start_row_ix << ", end row " << end_row_ix
                                             << ", score " << chunk_best_score);
        RELEASE_CONT(combos);
        best_score_l.set();
        if (chunk_best_score < best_score) {
            best_score = chunk_best_score;
            best_params_ixs = chunk_best_params_ixs;
            LOG4_DEBUG("Found best score " << best_score << ", " << 100. * best_score / C_slides_len << " pct direction error, indexes "
                                           << common::to_string(chunk_best_params_ixs));
        }
        best_score_l.unset();
    }
    RELEASE_CONT(params_preds);

    auto p_ensemble = p_dataset->get_ensemble(column_name);
#pragma omp parallel for schedule(static, 1) num_threads(adj_threads(colct))
    for (uint32_t colix = 0; colix < colct; ++colix) {
        const uint32_t levix = business::ModelService::to_level_ix(colix, p_dataset->get_spectral_levels());
        const auto &params = p_tune_predictions->at(levix).param_pred[best_params_ixs[colix]].params;
        auto p_model = p_ensemble->get_model(levix, stepix)->get_gradient(grad_level);
        if (p_model->is_manifold()) p_model = p_model->get_manifold();
        p_model->set_params(params, chunkix);
        p_model->chunks_score[chunkix].first = p_model->chunks_score[chunkix].second = best_score;
    }

    LOG4_END();
}


}// datamodel
} // svr
