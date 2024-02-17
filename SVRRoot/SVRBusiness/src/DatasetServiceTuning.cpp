//
// Created by zarko on 6/24/21.
//

#include <cmath>
#include <limits>
#include <algorithm>
#include <common/rtp_thread_pool.hpp>
#include <bitset>

#include "DQScalingFactorService.hpp"
#include "appcontext.hpp"
#include "onlinesvr.hpp"
#include "DAO/DatasetDAO.hpp"
#include "DatasetService.hpp"
#include "InputQueueService.hpp"
#include "recombine_parameters.cuh"
#include "ModelService.hpp"


namespace svr {
namespace business {


void
DatasetService::recombine_params(
        const datamodel::Dataset_ptr &p_dataset, datamodel::Ensemble_ptr &p_ensemble, t_tuned_parameters &tune_predictions,
        const size_t chunk_ix, const size_t grad_level)
{
    const uint32_t colct = p_dataset->get_model_count();
    if (std::any_of(std::execution::par_unseq, tune_predictions.begin(), tune_predictions.end(), [&](auto &tp){ return tp.second[chunk_ix].empty(); })) {
        LOG4_WARN("Predictions not complete.");
        return;
    }

    const auto &scaling_factors = p_dataset->get_dq_scaling_factors();
    const auto max_num_combos = std::pow<double>(double(common::C_tune_keep_preds), double(colct));
    const uint64_t num_combos = std::min<double>(common::C_num_combos, max_num_combos);
    const double filter_combos = double(max_num_combos) / double(num_combos);
    std::vector<t_param_preds_cu> params_preds(colct * common::C_tune_keep_preds);
#pragma omp parallel for collapse(2) num_threads(adj_threads(colct))
    for (uint32_t colix = 0; colix < colct; ++colix) {
        for (uint32_t rowix = 0; rowix < common::C_tune_keep_preds; ++rowix) {
            params_preds[rowix * colct + colix].params_ix = rowix;
            const auto levix = ModelService::to_level_ix(colix);
            const auto &tp = tune_predictions[levix][grad_level][chunk_ix] ^ rowix;
#pragma omp parallel for collapse(2) num_threads(adj_threads(EMO_MAX_J))
            for (uint32_t j = 0; j < EMO_MAX_J; ++j) {
                for (uint32_t el = 0; el < EMO_TUNE_VALIDATION_WINDOW; ++el) {
                    params_preds[rowix * colct + colix].predictions[j][el] = business::DQScalingFactorService::unscale(
                            arma::mean(tp->p_predictions->at(j).row(el)), levix, tp->p_params->get_input_queue_column_name(), scaling_factors);
                    params_preds[rowix * colct + colix].labels[j][el] = business::DQScalingFactorService::unscale(
                            arma::mean(tp->p_labels->at(j).row(el)), levix, tp->p_params->get_input_queue_column_name(), scaling_factors);
                    params_preds[rowix * colct + colix].last_knowns[j][el] = business::DQScalingFactorService::unscale(
                            arma::mean(tp->p_last_knowns->at(j).row(el)), levix, tp->p_params->get_input_queue_column_name(), scaling_factors);
                    LOG4_TRACE("Row " << rowix << ", J " << j << ", col " << colix << ", prediction " << params_preds[rowix * colct + colix].predictions[j][el]
                        << ", label " << params_preds[rowix * colct + colix].labels[j][el] << ", last known " << params_preds[rowix * colct + colix].last_knowns[j][el]);
                }
            }
        }
    }

    const uint32_t rows_gpu = common::gpu_handler::get().get_max_gpu_data_chunk_size() / 2 / colct / (2 * CUDA_BLOCK_SIZE) * (2 * CUDA_BLOCK_SIZE);
    auto best_score = std::numeric_limits<double>::max();
    std::vector<uint8_t> best_params_ixs(colct, uint8_t(0));
    LOG4_DEBUG("Predictions filtered out " << filter_combos << ", total prediction rows " << num_combos << ", rows per GPU " << rows_gpu << ", column count " << colct << ", limit num combos " << common::C_num_combos);

    // const auto start_time = std::chrono::steady_clock::now();
#pragma omp parallel for schedule(static, 1) num_threads(common::gpu_handler::get().get_max_running_gpu_threads_number())
    for (uint64_t start_row_ix = 0; start_row_ix < num_combos; start_row_ix += rows_gpu) {
        // if (best_score != std::numeric_limits<double>::max()) continue;
        // if (std::chrono::steady_clock::now() - start_time > std::chrono::minutes(45)) continue;
        const uint64_t end_row_ix = std::min<uint64_t>(start_row_ix + rows_gpu, num_combos);
        const uint64_t chunk_rows_ct = end_row_ix - start_row_ix;
        const arma::colvec colixs = arma::linspace<arma::colvec>(double(start_row_ix), double(end_row_ix), chunk_rows_ct);
        arma::uchar_mat combos(chunk_rows_ct, colct);
#pragma omp parallel for schedule(static, 1) num_threads(adj_threads(colct))
        for (uint32_t colix = 0; colix < colct; ++colix)
            combos.col(colix) = arma::conv_to<arma::uchar_colvec>::from(common::mod<arma::colvec>(
                    colixs / std::pow<double>(common::C_tune_keep_preds, colct - colix) / filter_combos, double(common::C_tune_keep_preds)));

        double chunk_best_score;
        std::vector<uint8_t> chunk_best_params_ixs(colct, 0);
        PROFILE_EXEC_TIME(recombine_parameters(chunk_rows_ct, colct, combos.mem, params_preds.data(), &chunk_best_score, chunk_best_params_ixs.data()),
                      "Recombine chunk " << chunk_rows_ct << "x" << colct << ", added set of size " << unsigned(common::C_tune_keep_preds)
                         << ", filter out " << filter_combos - 1 << " combinations, start row " << start_row_ix << ", end row " << end_row_ix << ", score " << chunk_best_score);
        decltype(combos){}.swap(combos);
#pragma omp critical
        {
           if (chunk_best_score < best_score) {
                best_score = chunk_best_score;
                best_params_ixs = chunk_best_params_ixs;
                LOG4_DEBUG("Found best score " << best_score << ", " << 100. * best_score / EMO_MAX_J / EMO_TUNE_VALIDATION_WINDOW << " pct direction error, indexes "
                                               << common::to_string(chunk_best_params_ixs));
            }
        }
    }
    decltype(params_preds){}.swap(params_preds);

#pragma omp parallel for schedule(static, 1) num_threads(adj_threads(colct))
    for (uint32_t colix = 0; colix < colct; ++colix) {
        const uint32_t levix = (colix >= 16 ? (colix + 1) : colix) * 2;
        p_ensemble->get_model(levix)->set_params(tune_predictions[levix][grad_level][chunk_ix].begin()->get()->p_params);
    }
}

#ifdef MANIFOLD_TEST

void
DatasetService::process_dataset_test_tune(
        datamodel::Dataset_ptr &p_dataset,
        datamodel::Ensemble_ptr &p_ensemble)
{
    LOG4_DEBUG("Offline test offset " << MANIFOLD_TEST_VALIDATION_WINDOW << ", " << *p_ensemble);

    const auto levct = p_dataset->get_transformation_levels();
    const auto p_label_decon = p_ensemble->get_decon_queue();
    const auto p_label_aux_decon = DeconQueueService::find_decon_queue(
            p_ensemble->get_aux_decon_queues(), p_ensemble->get_aux_decon_queue()->get_input_queue_table_name(), p_label_decon->get_input_queue_column_name());
    const datamodel::datarow_range aux_labels_range(p_label_aux_decon->get_data());
    const auto aux_feats_range = [&p_ensemble](){
        std::deque<datamodel::datarow_range> r;
        for (const auto &dq: p_ensemble->get_aux_decon_queues()) r.emplace_back(dq->get_data());
        return r;
    }();
    std::deque<bpt::ptime> label_times;
    const size_t half_levct = levct / 2;

    t_tuned_parameters tune_predictions;
    tbb::concurrent_map<size_t, arma::mat> level_features, level_labels, level_last_knowns;
    arma::vec recon_predicted, recon_actual, recon_last_knowns; // Single value prices
    const auto actual_prices = datamodel::datarow_crange{svr::business::EnsembleService::get_start(
            p_dataset->get_input_queue()->get_data(),
            MANIFOLD_TEST_VALIDATION_WINDOW,
            p_dataset->get_last_modeled_time(),
            p_dataset->get_input_queue()->get_resolution()),
                  p_dataset->get_input_queue()->end(),
                  *p_dataset->get_input_queue()};
    const auto input_colix = p_dataset->get_input_queue()->get_value_column_index(p_dataset->get_ensemble()->get_column_name());
#pragma omp parallel for num_threads(adj_threads(common::gpu_handler::get().get_max_running_gpu_threads_number())) schedule(static, 1)
    for (const auto &p_model: p_ensemble->get_models()) {
        const auto levix = p_model->get_decon_level();
        const auto p_head_params = p_model->get_params_ptr();
        APP.model_service.get_training_data(
                level_features[levix],
                level_labels[levix],
                level_last_knowns[levix],
                label_times,
                {svr::business::EnsembleService::get_start(
                        p_label_decon->get_data(),
                        p_head_params->get_svr_decremental_distance() + EMO_TUNE_TEST_SIZE + MANIFOLD_TEST_VALIDATION_WINDOW,
                        p_model->get_last_modeled_value_time(),
                        p_dataset->get_input_queue()->get_resolution()),
                 p_label_decon->end(),
                 p_label_decon->get_data()},
                aux_labels_range,
                aux_feats_range,
                p_head_params->get_lag_count(),
                common::get_adjacent_indexes(levix, p_head_params->get_svr_adjacent_levels_ratio(), p_dataset->get_transformation_levels()),
                p_dataset->get_max_lookback_time_gap(),
                levix,
                p_dataset->get_aux_input_queues().empty() ? p_dataset->get_input_queue()->get_resolution() : p_dataset->get_aux_input_queue()->get_resolution(),
                p_model->get_last_modeled_value_time(),
                p_dataset->get_input_queue()->get_resolution(),
                p_model->get_multiout());

        APP.dq_scaling_factor_service.scale(
                p_dataset, p_ensemble->get_aux_decon_queues(), p_head_params, level_features[levix], level_labels[levix], level_last_knowns[levix]);

        const size_t full_validation_sz = p_head_params->get_svr_decremental_distance() + EMO_TUNE_TEST_SIZE;
        if (tune_predictions[levix].empty()) tune_predictions[levix].resize(1);
        if (!p_head_params->get_svr_kernel_param()) PROFILE_EXEC_TIME(OnlineMIMOSVR::tune(
                tune_predictions[levix].front(), p_model->get_param_set(), level_features[levix].rows(0, full_validation_sz - 1),
                level_labels[levix].rows(0, full_validation_sz - 1),
                level_last_knowns[levix].rows(0, full_validation_sz - 1), p_dataset->get_chunk_size()), "Tune kernel params for model " << levix);
    }

#pragma omp parallel for schedule(static, 1) num_threads(adj_threads(tune_predictions.begin()->second.size()))
    for (size_t chunk_ix = 0; chunk_ix < tune_predictions.begin()->second.size(); ++chunk_ix)
        PROFILE_EXEC_TIME(recombine_params(p_dataset, p_ensemble, tune_predictions, chunk_ix, 0),
                          "Recombine parameters for " << half_levct - 1 << " models, chunk " << chunk_ix);

#pragma omp parallel for num_threads(common::gpu_handler::get().get_max_running_gpu_threads_number()) schedule(static, 1)
    for (size_t levix = 0; levix < levct; levix += 2) {
        if (levix == half_levct) continue;
        const auto p_model = p_ensemble->get_model(levix);
        const size_t full_validation_sz = p_model->get_params().get_svr_decremental_distance() + EMO_TUNE_TEST_SIZE;
        APP.model_service.train(p_model,
                                std::make_shared<arma::mat>(level_features[levix].rows(0, full_validation_sz - 1)),
                                std::make_shared<arma::mat>(level_labels[levix].rows(0, full_validation_sz - 1)));

        for (const auto &p: p_model->get_param_set()) {
            APP.svr_parameters_service.remove(p);
            APP.svr_parameters_service.save(p);
        }

        const auto [level_mae, level_mape, validation_predicted, validation_actuals, level_lin_mape, validation_last_knowns] =
                ModelService::future_validate(full_validation_sz, *p_model->get_gradients().front(), level_features[levix], level_labels[levix],
                                              level_last_knowns[levix], label_times, p_dataset->get_dq_scaling_factors(), p_ensemble->get_column_name());

#pragma omp critical
        {
            recon_predicted = recon_predicted.size() != validation_predicted.size() ? validation_predicted : recon_predicted + validation_predicted;
            recon_actual = recon_actual.size() != validation_actuals.size() ? validation_actuals : recon_actual + validation_actuals;
            recon_last_knowns = recon_last_knowns.size() != validation_last_knowns.size() ? validation_last_knowns : recon_last_knowns + validation_last_knowns;
        }
    }

    double mae = 0, mae_lk = 0, recon_mae = 0;
    size_t pos_mae = 0, pos_direct = 0;
    const auto compared_values_ct = std::min<size_t>(recon_actual.size(), recon_predicted.size());
    for (size_t i = 0; i < compared_values_ct; ++i) {
        const double cur_mae = std::abs(recon_predicted[i] - recon_actual[i]);
        const double cur_mae_lk = std::abs(recon_last_knowns[i] - recon_actual[i]);
        const double cur_alpha_pct = 100. * (cur_mae_lk / cur_mae - 1.);
        const double cur_recon_error = std::abs(recon_actual[i] - actual_prices[i]->at(input_colix));
        if (cur_mae < cur_mae_lk) {
            LOG4_DEBUG("Positive alpha " << cur_alpha_pct << " pct., at " << i);
            ++pos_mae;
        }
        if (std::signbit(recon_predicted[i] - recon_last_knowns[i]) == std::signbit(recon_actual[i] - recon_last_knowns[i])) {
            LOG4_DEBUG("Direction correct at " << i);
            ++pos_direct;
        }
        mae += cur_mae;
        mae_lk += cur_mae_lk;
        recon_mae += cur_recon_error;
        const auto cml_alpha_pct = 100. * (mae_lk / mae - 1.);
        if (mae < mae_lk) LOG4_DEBUG("Positive cumulative alpha at " << i << ", " << cml_alpha_pct << " pct.");
        const auto i_ct = double(i + 1);
        const auto i_valtime = label_times[label_times.size() - compared_values_ct + i];
        LOG4_DEBUG(
                "Position " << i_ct <<
                            ", price time " << i_valtime <<
                            ", actual price " << recon_actual[i] <<
                            ", predicted price " << recon_predicted[i] <<
                            ", last known " << recon_last_knowns[i] <<
                            ", total MAE " << mae / i_ct <<
                            ", total MAE last known " << mae_lk / i_ct <<
                            ", positive directions " << 100. * double(pos_direct) / i_ct <<
                            " pct., positive errors " << 100. * double(pos_mae) / i_ct <<
                            " pct., current MAE " << cur_mae <<
                            ", current MAE last known " << cur_mae_lk <<
                            ", predicted movement " << recon_predicted[i] - recon_last_knowns[i] <<
                            ", actual movement " << recon_actual[i] - recon_last_knowns[i] <<
                            ", current alpha " << cur_alpha_pct << " pct."
                            ", cumulative alpha " << cml_alpha_pct << " pct."
                            ", recon error " << cur_recon_error <<
                            ", average recon error " << 100. * recon_mae / i_ct <<  " pct.");
        if (i < common::C_forecast_focus)
            APP.request_service.save(std::make_shared<datamodel::MultivalResponse>(0, 0, i_valtime, p_label_decon->get_input_queue_column_name(), recon_predicted[i]));
        const auto actual_price = lower_bound(p_dataset->get_input_queue()->get_data(), i_valtime)->get();
        if (actual_price->get_value(0) != recon_actual[i])
            LOG4_WARN("Difference at " << i_valtime << " between actual price " << actual_price->get_value(0) << " and recon price " << recon_actual[i] << ", is " <<
                        actual_price->get_value(0) - recon_actual[i]);
    }
    mae /= double(compared_values_ct);
    mae_lk /= double(compared_values_ct);
    const double labels_meanabs = common::meanabs(recon_actual);
    const double mape = 100. * mae / labels_meanabs;
    const double mape_lk = 100. * mae_lk / labels_meanabs;
    const double alpha_pct = 100. * (mape_lk / mape - 1.);
    LOG4_INFO(
            "Total MAE of " << compared_values_ct << " compared values is " << mae << ", MAPE is " << mape << " pct., last known MAE " << mae_lk <<
            ", last known MAPE " << mape_lk << ", alpha " << alpha_pct << " pct., positive direction " << 100. * double(pos_direct) / double(compared_values_ct) <<
            " pct., positive error " << 100. * double(pos_mae) / double(compared_values_ct) << " pct.");
}

#endif

}
}
