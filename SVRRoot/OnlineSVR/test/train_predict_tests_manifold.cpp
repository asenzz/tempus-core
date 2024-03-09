//
// Created by zarko on 21/05/19.
//
#include <cmath>
#include <cstdlib>
#include <cstddef>
#include <gtest/gtest.h>
#include "EnsembleService.hpp"
#include "IQScalingFactorService.hpp"
#include "ModelService.hpp"
#include "appcontext.hpp"
#include "DQScalingFactorService.hpp"
#include "model/Priority.hpp"
#include "test_harness.hpp"
#include "common/compatibility.hpp"
#include "onlinesvr.hpp"
#include "common/defines.h"
#include "common/constants.hpp"
#include "model/SVRParameters.hpp"


using namespace svr;

// #define TEST_MANIFOLD
#define TEST_ACTUAL_DATA

namespace {
#ifdef INTEGRATION_TEST
const unsigned C_test_validation_window = INTEGRATION_TEST_VALIDATION_WINDOW;
#else
constexpr unsigned C_test_validation_window = 0;
#endif

constexpr unsigned C_test_decrement = 4300;
constexpr unsigned C_test_lag = DEFAULT_SVRPARAM_LAG_COUNT / 2;
const unsigned C_test_length = C_test_decrement + EMO_TEST_LEN + C_test_validation_window;
const std::string C_test_input_table_name = "q_svrwave_test_xauusd_avg_3600";
const std::string C_test_aux_input_table_name = "q_svrwave_test_xauusd_avg_1";
constexpr unsigned C_test_levels = 1;

}

#ifdef INTEGRATION_TEST
TEST(manifold_tune_train_predict, basic_integration)
{
    omp_set_nested(true);
    omp_set_max_active_levels(10 * std::thread::hardware_concurrency());
    svr::context::AppContext::init_instance("../config/app.config");
    auto p_dataset = ptr<datamodel::Dataset>(
            0xDeadBeef, "test_dataset", "test_user", C_test_input_table_name, std::deque{C_test_aux_input_table_name}, datamodel::Priority::Normal, "",
            common::C_default_gradient_count, common::C_kernel_default_max_chunk_size, PROPS.get_multistep_len(), C_test_levels, "cvmd", DEFAULT_FEATURES_MAX_TIME_GAP);

    business::EnsembleService::init_ensembles(p_dataset, false);
    auto p_decon = p_dataset->get_ensemble()->get_decon_queue();
    auto p_decon_aux = p_dataset->get_ensemble()->get_aux_decon_queue();
#pragma omp parallel num_threads(adj_threads(1 + p_dataset->get_aux_input_queues().size()))
#pragma omp single
    {
#pragma omp task
        {
            PROFILE_EXEC_TIME(APP.input_queue_service.load(*p_dataset->get_input_queue()), "Loading " << p_dataset->get_input_queue()->get_table_name());
            APP.iq_scaling_factor_service.prepare(*p_dataset, *p_dataset->get_input_queue(), false);
            business::DeconQueueService::deconstruct(p_dataset, p_dataset->get_input_queue(), p_decon);
        }
#pragma omp task
        {
            PROFILE_EXEC_TIME(APP.input_queue_service.load(*p_dataset->get_aux_input_queue()), "Loading " << p_dataset->get_aux_input_queue()->get_table_name());
            APP.iq_scaling_factor_service.prepare(*p_dataset, *p_dataset->get_aux_input_queue(), false);
            business::DeconQueueService::deconstruct(p_dataset, p_dataset->get_aux_input_queue(), p_decon_aux);
        }
    }

    std::deque<bpt::ptime> times;
    arma::mat recon_predicted, recon_last_knowns, recon_actual;
    matrix_ptr p_common_train_features;
    bool features_scaled = false;
    arma::mat dummy_mat;
    OMP_LOCK(recon_l)
#pragma omp parallel for schedule(static, 1) num_threads(adj_threads(p_dataset->get_transformation_levels()))
    for (size_t l = 0; l < p_dataset->get_transformation_levels(); l += 2) {
        if (l == p_dataset->get_half_levct()) continue;
        auto p_model = p_dataset->get_ensemble()->get_model(l);
        if (!p_model) LOG4_THROW("Model not found!");
        auto p_params = p_model->get_head_params();
        LOG4_DEBUG("Preparing model " << *p_model << " parameters " << *p_params);
#ifdef TEST_MANIFOLD
        p_params->set_kernel_type(datamodel::kernel_type_e::DEEP_PATH);
#endif
#ifdef TEST_ACTUAL_DATA
        p_params->set_svr_decremental_distance(C_test_decrement);
        p_params->set_lag_count(C_test_lag);

        const auto [p_all_features, p_all_labels, p_all_last_knowns] = business::ModelService::get_training_data(
                times, *p_dataset, *p_dataset->get_ensemble(), *p_model, C_test_length);
#else
        auto p_all_features = ptr<arma::mat>(C_test_length, TEST_LAG, arma::fill::randn);
        auto p_all_labels = ptr<arma::mat>(C_test_length, 1, arma::fill::randn);
        auto p_all_last_knowns = ptr<arma::mat>(C_test_length, 1, arma::fill::randn);
        std::deque<bpt::ptime> times(C_test_length, bpt::second_clock::local_time());
#endif
        LOG4_DEBUG("All features size " << arma::size(*p_all_features) << ", test length " << C_test_length);
#pragma omp critical
        {
            const auto feat_levels = common::get_adjacent_indexes(p_params->get_decon_level(), p_params->get_svr_adjacent_levels_ratio(), p_dataset->get_transformation_levels());
            const auto missing_sf = business::DQScalingFactorService::check(p_dataset->get_ensemble()->get_aux_decon_queues(), *p_params, feat_levels, p_dataset->get_dq_scaling_factors());
            if (features_scaled)  business::DQScalingFactorService::scale(*p_dataset, {p_decon_aux}, *p_params, missing_sf, dummy_mat, *p_all_labels, *p_all_last_knowns);
            else {
                business::DQScalingFactorService::scale(*p_dataset, {p_decon_aux}, *p_params, missing_sf, *p_all_features, *p_all_labels, *p_all_last_knowns);
                features_scaled = true;
            }
            if (!p_common_train_features)
                p_common_train_features = otr<arma::mat>(p_all_features->rows(p_all_features->n_rows - C_test_length, p_all_features->n_rows - C_test_validation_window - 1));
        }
        p_model->get_gradient()->batch_train(
                p_common_train_features,
                otr<arma::mat>(p_all_labels->rows(p_all_labels->n_rows - C_test_length, p_all_labels->n_rows - C_test_validation_window - 1)),
                otr<arma::vec>(p_all_last_knowns->rows(p_all_last_knowns->n_rows - C_test_length, p_all_last_knowns->n_rows - C_test_validation_window - 1)),
                times[times.size() - C_test_validation_window - 1]);
        auto [predict_mae_level, predict_mape_level, predicted_level, validation_level, mape_lk_level, validation_lk_level] =
                business::ModelService::future_validate(
                        p_all_labels->n_rows - C_test_validation_window, *p_model->get_gradient(),
                        *p_all_features, *p_all_labels, *p_all_last_knowns, times,
                        p_dataset->get_dq_scaling_factors(), false, p_dataset->get_transformation_levels() < MIN_LEVEL_COUNT);
        omp_set_lock(&recon_l);
        recon_predicted = recon_predicted.empty() ? predicted_level : recon_predicted + predicted_level;
        recon_actual = recon_actual.empty() ? validation_level : recon_actual + validation_level;
        recon_last_knowns = recon_last_knowns.empty() ? validation_lk_level : recon_last_knowns + validation_lk_level;
        omp_unset_lock(&recon_l);
    }

    const auto p_iqsf = p_dataset->get_iq_scaling_factor(p_dataset->get_aux_input_queue()->get_table_name(), p_dataset->get_ensemble()->get_column_name());
    recon_predicted = recon_predicted * p_iqsf->get_scaling_factor() + p_iqsf->get_dc_offset();
    recon_last_knowns = recon_last_knowns * p_iqsf->get_scaling_factor() + p_iqsf->get_dc_offset();
    recon_actual = recon_actual * p_iqsf->get_scaling_factor() + p_iqsf->get_dc_offset();

    double mae = 0, mae_lk = 0, recon_mae = 0;
    size_t pos_mae = 0, pos_direct = 0;
    const auto compared_values_ct = std::min<size_t>(recon_actual.size(), recon_predicted.size());
    for (size_t i = 0; i < compared_values_ct; ++i) {
        const double cur_mae = std::abs(recon_predicted[i] - recon_actual[i]);
        const double cur_mae_lk = std::abs(recon_last_knowns[i] - recon_actual[i]);
        const double cur_alpha_pct = 100. * (cur_mae_lk / cur_mae - 1.);
        const auto i_valtime = times[times.size() - compared_values_ct + i];
        const auto &actual_price = **lower_bound(p_dataset->get_input_queue()->get_data(), i_valtime);
        const double cur_recon_error = std::abs(recon_actual[i] - actual_price[0]);
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
                ", current alpha " << cur_alpha_pct << " pct., cumulative alpha " << cml_alpha_pct
                << " pct., recon error " << cur_recon_error << ", average recon error " << 100. * recon_mae / i_ct << " pct.");
        if (i < common::C_forecast_focus)
            APP.request_service.save(ptr<datamodel::MultivalResponse>(0, 0, i_valtime, p_decon->get_input_queue_column_name(), recon_predicted[i]));
        if (cur_recon_error)
            LOG4_WARN("Difference at " << i_valtime << " between actual price " << actual_price[0] << " and recon price " << recon_actual[i] << ", is " <<
                                       actual_price[0] - recon_actual[i]);
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
