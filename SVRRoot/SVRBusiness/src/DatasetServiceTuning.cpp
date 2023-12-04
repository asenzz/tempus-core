//
// Created by zarko on 6/24/21.
//

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <algorithm>
#include <common/rtp_thread_pool.hpp>
#include <common/thread_pool.hpp>
#include <bitset>
#include <memory_resource>

#include "appcontext.hpp"
#include "common/constants.hpp"
#include "common/defines.h"
#include "onlinesvr.hpp"
#include "util/ValidationUtils.hpp"
#include "util/TimeUtils.hpp"
#include "spectral_transform.hpp"
#include "model/User.hpp"
#include "SVRParametersService.hpp"
#include "DAO/DatasetDAO.hpp"
#include "DatasetService.hpp"
#include "InputQueueService.hpp"
#include "recombine_parameters.cuh"
#include "common/gpu_handler.hpp"


using namespace svr::common;
using namespace svr::datamodel;
using namespace svr::context;


namespace svr {
namespace business {


void
DatasetService::recombine_params(
        predictions_t &tune_predictions,
        std::vector<SVRParameters_ptr> &ensemble_params,
        const size_t half_levct,
        const std::vector<double> &scale_label,
        const std::vector<double> &dc_offset)
{
    if (tune_predictions.empty()) return;
    bool empty_preds = true;
    for (const auto &tp: tune_predictions) empty_preds &= tp.empty();
    if (empty_preds) return;

#ifdef SEPARATE_PREDICTIONS_BY_COST
    // Combined epscos take index 0
    for (auto &tp_level: tune_predictions) {
        if (tp_level.empty()) continue;
        for (const auto &key_tp: tp_level) {
            if (key_tp.second.empty()) continue;
            tp_level[0].insert(key_tp.second.begin(), key_tp.second.end());
        }
    }
#endif

    std::vector<SVRParameters_ptr> this_ensemble_params(ensemble_params.size());
    double best_score = recombine_params(tune_predictions, this_ensemble_params, half_levct, scale_label, dc_offset, 0);
    ensemble_params = this_ensemble_params;
    LOG4_DEBUG("Found best score " << best_score << " for epsco 0.");
#ifdef SEPARATE_PREDICTIONS_BY_COST
    for (const auto epsco: rough_epscos_grid) {
        const uint64_t epsco_key = 1e6 * epsco;
        const auto score = recombine_params(tune_predictions, this_ensemble_params, half_levct, scale_label, dc_offset, epsco_key);
        if (score <= best_score) {
            best_score = score;
            ensemble_params = this_ensemble_params;
            LOG4_DEBUG("Found best score " << best_score << " for epsco " << epsco);
        }
    }
#endif
}

double
DatasetService::recombine_params(
        predictions_t &tune_predictions,
        std::vector<SVRParameters_ptr> &ensemble_params,
        const size_t half_levct,
        const std::vector<double> &scale_label,
        const std::vector<double> &dc_offset,
        const uint64_t epsco_key)
{
    bool empty_preds = true;
    for (const auto &tp: tune_predictions)
        if (tp.find(epsco_key) != tp.end())
            empty_preds &= tp.at(epsco_key).empty();
    if (empty_preds)
        return std::numeric_limits<double>::quiet_NaN();

#if 1
#ifdef EVENING_FACTOR
#pragma omp parallel for
    for (size_t levix = 0; levix < half_levct * 2; levix += 2) {
        if (levix == half_levct || tune_predictions[levix].find(epsco_key) == tune_predictions[levix].end()) {
            LOG4_DEBUG("Level " << levix << " epsco key " << epsco_key << " not found or ignored.");
            continue;
        }
        for (auto &tp: tune_predictions[levix][epsco_key]) {
            for (size_t j = 0; j < EMO_MAX_J; ++j) {
                tp->p_predictions->at(j) *= (scale_label[levix] + EVENING_FACTOR) / (EVENING_FACTOR + 1.);
                tp->p_labels->at(j) *= (scale_label[levix] + EVENING_FACTOR) / (EVENING_FACTOR + 1.);
                tp->p_last_knowns->at(j) *= (scale_label[levix] + EVENING_FACTOR) / (EVENING_FACTOR + 1.);
                if (!levix) {
                    tp->p_predictions->at(j) += dc_offset[levix] / (EVENING_FACTOR + 1.);
                    tp->p_labels->at(j) += dc_offset[levix] / (EVENING_FACTOR + 1.);
                    tp->p_last_knowns->at(j) += dc_offset[levix] / (EVENING_FACTOR + 1.);
                }
            }
        }
    }
#endif
#endif

    const auto max_num_combos = std::pow<double>(double(C_tune_keep_preds), double(half_levct - 1));
    const uint64_t num_combos = std::min<double>(C_num_combos, max_num_combos);
    const double filter_combos = double(max_num_combos) / double(num_combos);
    const uint32_t colct = half_levct - 1;
    std::vector<t_param_preds_cu> params_preds(colct * C_tune_keep_preds);
#pragma omp parallel for collapse(2)
    for (uint32_t colix = 0; colix < colct; ++colix) {
        for (uint32_t rowix = 0; rowix < C_tune_keep_preds; ++rowix) {
            params_preds[rowix * colct + colix].params_ix = rowix;
            const auto &tp = std::next(tune_predictions[(colix >= 16 ? (colix + 1) : colix) * 2][epsco_key].begin(), rowix)->get();
            for (uint32_t j = 0; j < EMO_MAX_J; ++j) {
                for (uint32_t el = 0; el < EMO_TUNE_VALIDATION_WINDOW; ++el) {
                    params_preds[rowix * colct + colix].predictions[j][el] = arma::mean(tp->p_predictions->at(j).row(el));
                    params_preds[rowix * colct + colix].labels[j][el] = arma::mean(tp->p_labels->at(j).row(el));
                    params_preds[rowix * colct + colix].last_knowns[j][el] = arma::mean(tp->p_last_knowns->at(j).row(el));
                    // LOG4_TRACE("Row " << rowix << ", J " << j << ", col " << colix << ", prediction " << params_preds[rowix * colct + colix].predictions[j][el] << ", label " << params_preds[rowix * colct + colix].labels[j][el] << ", last known " << params_preds[rowix * colct + colix].last_knowns[j][el]);
                }
            }
        }
    }

    const uint32_t rows_gpu = common::gpu_handler::get_instance().get_max_gpu_data_chunk_size() / 2 / colct / (2 * CUDA_BLOCK_SIZE) * (2 * CUDA_BLOCK_SIZE);
    auto best_score = std::numeric_limits<double>::max();
    std::vector<uint8_t> best_params_ixs(colct, uint8_t(0));
#ifdef EVENING_FACTOR
    LOG4_DEBUG("Recombining epsco " << double(epsco_key) / 1e6 << ", predictions filtered out " << filter_combos - 1 << ", total prediction rows " << num_combos <<
                ", rows per GPU " << rows_gpu << ", column count " << colct << ", limit num combos " << common::C_num_combos << ", evening factor " << EVENING_FACTOR);
#else
    LOG4_DEBUG("Predictions filtered out " << filter_combos << ", total prediction rows " << num_combos << ", rows per GPU " << rows_gpu << ", column count " << colct << ", limit num combos " << common::C_num_combos);
#endif

    // const auto start_time = std::chrono::steady_clock::now();
#pragma omp parallel for num_threads(common::gpu_handler::get_instance().get_max_running_gpu_threads_number())
    for (uint64_t start_row_ix = 0; start_row_ix < num_combos; start_row_ix += rows_gpu) {
        // if (best_score != std::numeric_limits<double>::max()) continue;
        // if (std::chrono::steady_clock::now() - start_time > std::chrono::minutes(45)) continue;
        const uint64_t end_row_ix = std::min<uint64_t>(start_row_ix + rows_gpu, num_combos);
        const uint64_t chunk_rows_ct = end_row_ix - start_row_ix;
#if 0
        std::vector<uint8_t> combos(chunk_rows_ct * colct);
#pragma omp parallel for simd collapse(2) schedule(static, 0x10000) num_threads(common::gpu_handler::get_instance().get_max_running_gpu_threads_number() > std::thread::hardware_concurrency() ? std::thread::hardware_concurrency() : 1 + std::max<uint32_t>(1, std::thread::hardware_concurrency() - common::gpu_handler::get_instance().get_max_running_gpu_threads_number()) / common::gpu_handler::get_instance().get_max_running_gpu_threads_number())
        for (uint32_t colix = 0; colix < colct; ++colix)
            for (uint64_t row_ix = start_row_ix; row_ix < end_row_ix; ++row_ix)
                combos[(row_ix - start_row_ix) * colct + colix] = std::fmod(std::round(double(row_ix) / (std::pow<double>(C_tune_keep_preds, colct - colix) / filter_combos)), double(C_tune_keep_preds));
#endif
        const arma::colvec colixs = arma::linspace<arma::colvec>(start_row_ix, end_row_ix, chunk_rows_ct);
        arma::uchar_mat combos(chunk_rows_ct, colct);
#pragma omp parallel for
        for (uint32_t colix = 0; colix < colct; ++colix)
            combos.col(colix) = arma::conv_to<arma::uchar_colvec>::from(common::mod<arma::colvec>(colixs / std::pow<double>(C_tune_keep_preds, colct - colix) / filter_combos, double(C_tune_keep_preds)));

        double chunk_best_score;
        std::vector<uint8_t> chunk_best_params_ixs(colct, 0);
        PROFILE_EXEC_TIME(recombine_parameters(chunk_rows_ct, colct, combos.memptr(), params_preds.data(), &chunk_best_score, chunk_best_params_ixs.data()),
                      "Recombine chunk " << chunk_rows_ct << "x" << half_levct - 1 << ", added set of size " << unsigned(C_tune_keep_preds)
                         << ", filter out " << filter_combos - 1 << " combinations, start row " << start_row_ix << ", end row " << end_row_ix << ", score " << chunk_best_score << ", epsco " << double(epsco_key) / 1e6);
        decltype(combos){}.swap(combos);
#pragma omp critical
        {
           if (chunk_best_score < best_score) {
                best_score = chunk_best_score;
                best_params_ixs = chunk_best_params_ixs;
                LOG4_DEBUG("Found best score " << best_score << ", " << 100. * best_score / EMO_MAX_J / EMO_TUNE_VALIDATION_WINDOW << " pct direction error, indexes "
                            << deep_to_string(chunk_best_params_ixs) << ", epsco " << double(epsco_key) / 1e6);
            }
        }
    }
    decltype(params_preds){}.swap(params_preds);

    for (uint32_t colix = 0; colix < colct; ++colix) {
        const uint32_t levix = (colix >= 16 ? (colix + 1) : colix) * 2;
#ifndef EVENING_FACTOR
        ensemble_params[levix] = tune_predictions[levix].begin()->get()->p_params;
#else
        if (tune_predictions[levix].empty()) continue;
        ensemble_params[levix] = std::next(tune_predictions[levix][epsco_key].begin(), best_params_ixs[colix])->get()->p_params;
#endif
    }
    if (epsco_key) {
#pragma omp parallel for
        for (auto &tune_prediction: tune_predictions) {
            if (tune_prediction.empty()) continue;
            for (auto &tune_params: tune_prediction[epsco_key])
                tune_params->clear();
        }
    }
    return best_score;
}


void DatasetService::process_dataset_test_tune(
        Dataset_ptr &p_dataset,
        Ensemble_ptr &p_ensemble,
        std::vector<SVRParameters_ptr> &ensemble_params,
        std::vector<std::vector<matrix_ptr>> &features,
        std::vector<std::vector<matrix_ptr>> &labels,
        std::vector<std::vector<bpt::ptime>> &last_row_time,
        const size_t lev_ct,
        const size_t ens_ix)
{
    LOG4_DEBUG("Offline test offset " << MANIFOLD_TEST_VALIDATION_WINDOW << ", " << p_ensemble->to_string());

    bool train_ret = true;
    if (p_ensemble->get_aux_decon_queues().size() > 1) LOG4_ERROR("More than one aux decon queue is not supported!");
    const std::string save_column_name = p_ensemble->get_decon_queue()->get_input_queue_column_name();
    const auto p_decon = p_ensemble->get_decon_queue();
    const auto p_aux_decon = p_ensemble->get_aux_decon_queue(0);
#ifdef CACHED_FEATURE_ITER
    APP.model_service.aux_decon_hint = p_aux_decon->get_data().begin(); // lower_bound(p_aux_decon->get_data(), p_decon->get_data()[std::max<size_t>(0, p_decon->get_data().size() - p_dataset->get_max_lag_count() - p_dataset->get_max_decrement() - EMO_TUNE_VALIDATION_WINDOW - MANIFOLD_TEST_VALIDATION_WINDOW - DATA_LUFTA)]->get_value_time());
#endif
    const auto resolution_factor = double(p_dataset->get_input_queue()->get_resolution().total_microseconds()) / double(p_dataset->get_aux_input_queue(0)->get_resolution().total_microseconds());
    std::vector<bpt::ptime> label_times;
    const size_t half_levct = lev_ct / 2;
    std::vector<arma::mat> level_features(half_levct), level_labels(half_levct), level_last_knowns(half_levct);
    std::vector<double> predicted_values, actual_values, last_knowns;
    std::mutex mx;
    predictions_t tune_predictions(lev_ct);
    std::vector<double> level_lin_pred_values, level_predicted_values, level_actual_values, scale_label(lev_ct, 1.), dc_offset(lev_ct, 0.);
    __tbb_spfor(levix, 0, lev_ct, 2,
        if (levix == half_levct) continue;
        const auto half_levix = levix / 2;
        const auto p_model = p_ensemble->get_model(levix);
        auto p_params = ensemble_params[levix];
        train_ret = APP.model_service.get_training_data(
                level_features[half_levix],
                level_labels[half_levix],
                level_last_knowns[half_levix],
                label_times,
                { svr::business::EnsembleService::get_start(
                        p_decon->get_data(),
                        p_params->get_svr_decremental_distance() + EMO_TUNE_TEST_SIZE + MANIFOLD_TEST_VALIDATION_WINDOW,
                        0,
                        p_model->get_last_modeled_value_time(),
                        p_dataset->get_input_queue()->get_resolution()),
                        p_decon->get_data().end() - MODEL_TRAIN_OFFSET,
                        p_decon->get_data() },
                p_aux_decon->get_data(),
                p_params->get_lag_count(),
                p_model->get_learning_levels(),
                p_dataset->get_max_lookback_time_gap(),
                levix,
                resolution_factor,
                p_model->get_last_modeled_value_time(),
                p_dataset->get_input_queue()->get_resolution());
        APP.dq_scaling_factor_service.scale(p_dataset, p_aux_decon, p_params, p_model->get_learning_levels(), level_features[half_levix], level_labels[half_levix], level_last_knowns[half_levix]);
        const size_t full_validation_sz = p_params->get_svr_decremental_distance() + EMO_TUNE_TEST_SIZE;
        if (!p_params->get_svr_kernel_param())
            PROFILE_EXEC_TIME(OnlineMIMOSVR::tune_kernel_params(
                tune_predictions[levix], p_params, level_features[half_levix].rows(0, full_validation_sz - 1), level_labels[half_levix].rows(0, full_validation_sz - 1),
                    level_last_knowns[half_levix].rows(0, full_validation_sz - 1)), "Tune kernel params for model " << p_params->get_decon_level());
        scale_label[levix] = p_dataset->get_dq_scaling_factor_labels(p_aux_decon->get_input_queue_table_name(), p_aux_decon->get_input_queue_column_name(), levix);
        dc_offset[levix] = levix ? 0 : p_dataset->get_dq_scaling_factor_labels(p_aux_decon->get_input_queue_table_name(), p_aux_decon->get_input_queue_column_name(), DC_DQ_SCALING_FACTOR);
    )

    PROFILE_EXEC_TIME(recombine_params(tune_predictions, ensemble_params, half_levct, scale_label, dc_offset), "Recombine parameters");

    __tbb_spfor(levix, 0, lev_ct, 2,
        if (levix == half_levct) continue;
        const auto half_levix = levix / 2;
        const auto p_model = p_ensemble->get_model(levix);
        const auto p_params = ensemble_params[levix];
        const size_t full_validation_sz = p_params->get_svr_decremental_distance() + EMO_TUNE_TEST_SIZE;
        svr::OnlineMIMOSVR svr_model(
                p_params,
                std::make_shared<arma::mat>(level_features[half_levix].rows(0, full_validation_sz - 1)),
                std::make_shared<arma::mat>(level_labels[half_levix].rows(0, full_validation_sz - 1)),
                false, svr::matrices_ptr{}, false, svr::MimoType::single, level_labels[half_levix].n_cols);
        APP.svr_parameters_service.remove(p_params);
        APP.svr_parameters_service.save(p_params);

        while (label_times.size() != level_labels[half_levix].n_rows) usleep(10);
        double level_mae, level_mape, level_lin_mape;
        std::vector<double> level_lin_pred_values, level_predicted_values, level_actual_values;
        std::tie(level_mae, level_mape, level_predicted_values, level_actual_values, level_lin_mape, level_lin_pred_values) =
                OnlineMIMOSVR::future_validate(full_validation_sz, svr_model, level_features[half_levix], level_labels[half_levix], level_last_knowns[half_levix], label_times, false, scale_label[levix], dc_offset[levix]);

        const std::scoped_lock lk(mx);
        if (predicted_values.size() != level_predicted_values.size()) predicted_values.resize(level_predicted_values.size(), 0.);
        if (actual_values.size() != level_actual_values.size()) actual_values.resize(level_actual_values.size(), 0.);
        if (last_knowns.size() != level_lin_pred_values.size()) last_knowns.resize(level_lin_pred_values.size(), 0.);
        for (size_t i = 0; i < std::min<size_t>(level_predicted_values.size(), level_actual_values.size()); ++i) {
            predicted_values[i] += level_predicted_values[i];
            actual_values[i] += level_actual_values[i];
            last_knowns[i] += level_lin_pred_values[i];
        }
    )

    double mae = 0, mae_lk = 0;
    size_t pos_mae = 0, pos_direct = 0;
    const auto compared_values_ct = std::min<size_t>(actual_values.size(), predicted_values.size());
    for (size_t i = 0; i < compared_values_ct; ++i) {
        const double cur_mae = std::abs(predicted_values[i] - actual_values[i]);
        const double cur_mae_lk = std::abs(last_knowns[i] - actual_values[i]);
        const double cur_alpha_pct = 100. * (cur_mae_lk / cur_mae - 1.);
        if (cur_mae < cur_mae_lk) {
            LOG4_DEBUG("Positive alpha " << cur_alpha_pct << " pct., at " << i);
            ++pos_mae;
        }
        if (std::signbit(predicted_values[i] - last_knowns[i]) == std::signbit(actual_values[i] - last_knowns[i])) {
            LOG4_DEBUG("Direction correct at " << i);
            ++pos_direct;
        }
        mae += cur_mae;
        mae_lk += cur_mae_lk;
        const auto cml_alpha_pct = 100. * (mae_lk / mae - 1.);
        if (mae < mae_lk) LOG4_DEBUG("Positive cumulative alpha at " << i << ", " << cml_alpha_pct << " pct.");
        const auto i_ct = double(i + 1);
        const auto i_valtime = label_times[label_times.size() - compared_values_ct + i];
        LOG4_DEBUG(
                "Position " << i_ct <<
                            ", price time " << i_valtime <<
                            ", actual price " << actual_values[i] <<
                            ", predicted price " << predicted_values[i] <<
                            ", last known " << last_knowns[i] <<
                            ", total MAE " << mae / i_ct <<
                            ", total MAE last known " << mae_lk / i_ct <<
                            ", positive directions " << 100. * double(pos_direct) / i_ct <<
                            " pct., positive errors " << 100. * double(pos_mae) / i_ct <<
                            " pct., current MAE " << cur_mae <<
                            ", current MAE last known " << cur_mae_lk <<
                            ", predicted movement " << predicted_values[i] - last_knowns[i] <<
                            ", actual movement " << actual_values[i] - last_knowns[i] <<
                            ", current alpha " << cur_alpha_pct << " pct."
                            ", cumulative alpha " << cml_alpha_pct << " pct.");
        if (i < common::C_forecast_focus)
            APP.request_service.save(
                    std::make_shared<MultivalResponse>(0, 0, i_valtime, save_column_name, predicted_values[i]));
        const auto actual_price = lower_bound(p_dataset->get_input_queue()->get_data(), i_valtime)->get();
        if (actual_price->get_value(0) != actual_values[i])
            LOG4_WARN("Difference at " << i_valtime << " between actual price " << actual_price->get_value(0) << " and recon price " << actual_values[i] << ", is " << actual_price->get_value(0) - actual_values[i]);
    }
    mae /= double(compared_values_ct);
    mae_lk /= double(compared_values_ct);
#ifdef EMO_DIFF
    const arma::mat recon_actuals = common::operator+(actual_values, p_last_knowns);
    const double labels_meanabs = common::meanabs(actual_values);
#else
    const double labels_meanabs = common::meanabs(actual_values);
#endif
    const double mape = 100. * mae / labels_meanabs;
    const double mape_lk = 100. * mae_lk / labels_meanabs;
    const double alpha_pct = 100. * (mape_lk / mape - 1.);
    LOG4_INFO(
            "Total MAE of " << compared_values_ct << " compared values is " << mae << ", MAPE is " << mape << " pct., last known MAE " << mae_lk <<
            ", last known MAPE " << mape_lk << ", alpha " << alpha_pct << " pct., positive direction " << 100. * double(pos_direct) / double(compared_values_ct) << " pct., positive error " << 100. * double(pos_mae) / double(compared_values_ct) << " pct.");
}

}
}
