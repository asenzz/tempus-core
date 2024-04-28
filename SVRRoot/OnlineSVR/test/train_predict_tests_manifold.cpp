//
// Created by zarko on 21/05/19.
//

#include "common/defines.h"

#ifdef INTEGRATION_TEST

#include <cmath>
#include <cstdlib>
#include <omp.h>
#include <gtest/gtest.h>
#include "EnsembleService.hpp"
#include "IQScalingFactorService.hpp"
#include "ModelService.hpp"
#include "appcontext.hpp"
#include "DQScalingFactorService.hpp"
#include "model/Priority.hpp"
#include "common/compatibility.hpp"
#include "common/constants.hpp"

using namespace svr;

// #define TEST_MANIFOLD
#define TEST_ACTUAL_DATA

namespace {

constexpr unsigned C_test_decrement = 2 * DEFAULT_SVRPARAM_DECREMENT_DISTANCE;
constexpr unsigned C_test_lag = DEFAULT_SVRPARAM_LAG_COUNT;
const unsigned C_test_length = C_test_decrement + C_emo_test_len + common::INTEGRATION_TEST_VALIDATION_WINDOW;
const std::string C_test_input_table_name = "q_svrwave_test_xauusd_avg_3600";
const std::string C_test_aux_input_table_name = "q_svrwave_test_xauusd_avg_1";
constexpr unsigned C_test_levels = 2 * MIN_LEVEL_COUNT;
constexpr unsigned C_test_gradient_count = common::C_default_gradient_count;

}

TEST(manifold_tune_train_predict, basic_integration)
{
    omp_set_nested(true);
    // omp_set_max_active_levels(20 * std::thread::hardware_concurrency());
    svr::context::AppContext::init_instance("../config/app.config");
    auto p_dataset = ptr<datamodel::Dataset>(
            0xDeadBeef, "test_dataset", "test_user", C_test_input_table_name, std::deque{C_test_aux_input_table_name}, datamodel::Priority::Normal, "",
            C_test_gradient_count, common::C_default_kernel_max_chunk_size, PROPS.get_multistep_len(), C_test_levels, "cvmd", DEFAULT_FEATURES_MAX_TIME_GAP);

    business::EnsembleService::init_ensembles(p_dataset, false);
    auto p_decon = p_dataset->get_ensemble()->get_decon_queue();
    auto p_decon_aux = p_dataset->get_ensemble()->get_aux_decon_queue();
#pragma omp parallel num_threads(adj_threads(1 + p_dataset->get_aux_input_queues().size()))
#pragma omp single
    {
#pragma omp task
        {
            const auto table_name = p_dataset->get_aux_input_queue()->get_table_name();
            PROFILE_EXEC_TIME(APP.input_queue_service.load(*p_dataset->get_input_queue()), "Loading " << table_name);
            PROFILE_EXEC_TIME(APP.iq_scaling_factor_service.prepare(*p_dataset, *p_dataset->get_input_queue(), false), "Prepare input scaling factors for " << table_name);
            PROFILE_EXEC_TIME(business::DeconQueueService::deconstruct(*p_dataset, *p_dataset->get_input_queue(), *p_decon), "Deconstruct " << table_name);
        }
#pragma omp task
        {
            const auto table_name = p_dataset->get_aux_input_queue()->get_table_name();
            PROFILE_EXEC_TIME(APP.input_queue_service.load(*p_dataset->get_aux_input_queue()), "Loading " << table_name);
            PROFILE_EXEC_TIME(APP.iq_scaling_factor_service.prepare(*p_dataset, *p_dataset->get_aux_input_queue(), false), "Prepare input scaling factors for " << table_name);
            PROFILE_EXEC_TIME(business::DeconQueueService::deconstruct(*p_dataset, *p_dataset->get_aux_input_queue(), *p_decon_aux), "Deconstruct " << table_name);
        }
    }

    std::deque<bpt::ptime> times;
    arma::mat recon_predicted, recon_last_knowns, recon_actual;
    OMP_LOCK(recon_l)
#pragma omp parallel for schedule(static, 1) num_threads(adj_threads(std::min<size_t>(C_parallel_train_models, p_dataset->get_transformation_levels())))
    for (size_t l = 0; l < p_dataset->get_transformation_levels(); l += 2) {
        if (l == p_dataset->get_half_levct()) continue;
        auto p_ensemble = p_dataset->get_ensemble();
        auto p_model = p_ensemble->get_model(l);
        if (!p_model) LOG4_THROW("Model not found!");
        datamodel::SVRParameters_ptr p_head_params;
        for (auto &p_gradient: p_model->get_gradients())
            for (auto &p_params: p_gradient->get_param_set()) {
                if (!p_head_params) p_head_params = p_params;
#ifdef TEST_MANIFOLD
                p_params->set_kernel_type(datamodel::kernel_type_e::DEEP_PATH);
#endif
#ifdef TEST_ACTUAL_DATA
                p_params->set_svr_decremental_distance(C_test_decrement);
                p_params->set_lag_count(C_test_lag);
            }

        LOG4_DEBUG("Preparing model " << *p_model << " parameters " << *p_head_params);
        const auto [p_level_features, p_level_labels, p_level_last_knowns, level_times] = business::ModelService::get_training_data(
                *p_dataset, *p_dataset->get_ensemble(), *p_model, C_test_length);
        const auto train_start = p_level_labels->n_rows - C_test_length;
        const auto train_end = p_level_labels->n_rows - common::INTEGRATION_TEST_VALIDATION_WINDOW - 1;
#else
        auto p_all_features = ptr<arma::mat>(C_test_length, TEST_LAG, arma::fill::randn);
        auto p_all_labels = ptr<arma::mat>(C_test_length, 1, arma::fill::randn);
        auto p_all_last_knowns = ptr<arma::mat>(C_test_length, 1, arma::fill::randn);
        std::deque<bpt::ptime> times(C_test_length, bpt::second_clock::local_time());
#endif
        LOG4_DEBUG("All features size " << arma::size(*p_level_features) << ", test length " << C_test_length);
        const auto feat_levels = p_head_params->get_adjacent_levels();
        business::ModelService::train(*p_model,
                                      otr<arma::mat>(p_level_features->rows(train_start, train_end)),
                                      otr<arma::mat>(p_level_labels->rows(train_start, train_end)),
                                      otr<arma::vec>(p_level_last_knowns->rows(train_start, train_end)),
                                      level_times[level_times.size() - common::INTEGRATION_TEST_VALIDATION_WINDOW - 1]);
        const auto [predict_mae_level, predict_mape_level, predicted, actual, mape_lk, last_knowns] =
                business::ModelService::validate(
                        p_level_labels->n_rows - common::INTEGRATION_TEST_VALIDATION_WINDOW,
                        *p_dataset, *p_ensemble, *p_model,
                        *p_level_features, *p_level_labels, *p_level_last_knowns, level_times,
                        false, p_dataset->get_transformation_levels() < MIN_LEVEL_COUNT);
        omp_set_lock(&recon_l);
        if (!l) times = std::move(level_times);
        recon_predicted = recon_predicted.empty() ? predicted : recon_predicted + predicted;
        recon_actual = recon_actual.empty() ? actual : recon_actual + actual;
        recon_last_knowns = recon_last_knowns.empty() ? last_knowns : recon_last_knowns + last_knowns;
        omp_unset_lock(&recon_l);
    }

    const auto p_iqsf = p_dataset->get_iq_scaling_factor(p_dataset->get_aux_input_queue()->get_table_name(), p_dataset->get_ensemble()->get_column_name());
    recon_predicted = recon_predicted * p_iqsf->get_scaling_factor() + p_iqsf->get_dc_offset();
    recon_last_knowns = recon_last_knowns * p_iqsf->get_scaling_factor() + p_iqsf->get_dc_offset();
    recon_actual = recon_actual * p_iqsf->get_scaling_factor() + p_iqsf->get_dc_offset();

    double mae = 0, mae_lk = 0, recon_mae = 0;
    size_t positive_mae = 0, pos_direct = 0;
    const auto validated_ct = recon_actual.size();
    for (size_t i = 0; i < validated_ct; ++i) {
        const double cur_mae = std::abs(recon_predicted[i] - recon_actual[i]);
        const double cur_mae_lk = std::abs(recon_last_knowns[i] - recon_actual[i]);
        const double cur_alpha_pct = 100. * (cur_mae_lk / cur_mae - 1.);
        const auto cur_time = times[times.size() - validated_ct + i];
        const auto actual = (**lower_bound(*p_dataset->get_input_queue(), cur_time))[0];
        const double cur_recon_error = std::abs(recon_actual[i] - actual);
        const auto cml_alpha_pct = 100. * (mae_lk / mae - 1.);
        const auto i_div = double(i + 1);

        mae += cur_mae;
        mae_lk += cur_mae_lk;
        recon_mae += cur_recon_error;

        if (mae < mae_lk) LOG4_DEBUG("Positive cumulative alpha at " << i << ", " << cml_alpha_pct << " pct.");
        if (cur_mae < cur_mae_lk) {
            LOG4_DEBUG("Positive alpha " << cur_alpha_pct << " pct., at " << i);
            ++positive_mae;
        }
        if (std::signbit(recon_predicted[i] - recon_last_knowns[i]) == std::signbit(recon_actual[i] - recon_last_knowns[i])) {
            LOG4_DEBUG("Direction correct at " << i);
            ++pos_direct;
        }
        if (cur_recon_error > std::numeric_limits<double>::epsilon())
            LOG4_WARN("Recon differ at " << cur_time << " between actual price " << actual << " and recon price " << recon_actual[i] << ", is "
                                         << actual - recon_actual[i]);

        LOG4_DEBUG("Position " << i <<
                               ", price time " << cur_time <<
                               ", actual price " << recon_actual[i] <<
                               ", predicted price " << recon_predicted[i] <<
                               ", last known " << recon_last_knowns[i] <<
                               ", total MAE " << mae / i_div <<
                               ", total MAE last known " << mae_lk / i_div <<
                               ", positive directions " << 100. * double(pos_direct) / i_div << " pct."
                               ", positive errors " << 100. * double(positive_mae) / i_div << " pct."
                               ", current MAE " << cur_mae <<
                               ", current MAE last known " << cur_mae_lk <<
                               ", predicted movement " << recon_predicted[i] - recon_last_knowns[i] <<
                               ", actual movement " << recon_actual[i] - recon_last_knowns[i] <<
                               ", current alpha " << cur_alpha_pct << " pct., cumulative alpha " << cml_alpha_pct << " pct."
                               ", recon error " << cur_recon_error << ", average recon error " << 100. * recon_mae / i_div << " pct.");
        if (i < common::C_forecast_focus)
            APP.request_service.save(ptr<datamodel::MultivalResponse>(0, 0, cur_time, p_decon->get_input_queue_column_name(), recon_predicted[i]));
    }
    mae /= double(validated_ct);
    mae_lk /= double(validated_ct);
    const double labels_meanabs = common::meanabs(recon_actual);
    const double mape = 100. * mae / labels_meanabs;
    const double mape_lk = 100. * mae_lk / labels_meanabs;
    const double alpha_pct = 100. * (mape_lk / mape - 1.);
    LOG4_INFO("Total MAE of " << validated_ct << " compared values is " << mae << ","
                              " MAPE is " << mape << " pct.,"
                              " last-known MAE " << mae_lk << ","
                              " last-known MAPE " << mape_lk << " pct.,"
                              " alpha " << alpha_pct << " pct.,"
                              " positive direction " << 100. * double(pos_direct) / double(validated_ct) << " pct.,"
                              " positive error " << 100. * double(positive_mae) / double(validated_ct) << " pct.");
}

#endif
