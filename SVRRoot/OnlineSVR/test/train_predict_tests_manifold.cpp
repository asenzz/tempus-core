//
// Created by zarko on 21/05/19.
//

/*
 * \d+ q_svrwave_test_xauusd_avg_1
                               View "public.q_svrwave_test_xauusd_avg_1"
     Column     |            Type             | Collation | Nullable | Default | Storage | Description
----------------+-----------------------------+-----------+----------+---------+---------+-------------
 value_time     | timestamp without time zone |           |          |         | plain   |
 update_time    | timestamp without time zone |           |          |         | plain   |
 tick_volume    | double precision            |           |          |         | plain   |
 xauusd_avg_bid | double precision            |           |          |         | plain   |
View definition:
 SELECT value_time,
    update_time,
    tick_volume,
    xauusd_avg_bid
   FROM q_svrwave_xauusd_avg_1
  ORDER BY value_time DESC
 LIMIT (10000 * MAIN_QUEUE_RES);

 \d+ q_svrwave_test_xauusd_avg_MAIN_QUEUE_RES
                             View "public.q_svrwave_test_xauusd_avg_MAIN_QUEUE_RES"
     Column     |            Type             | Collation | Nullable | Default | Storage | Description
----------------+-----------------------------+-----------+----------+---------+---------+-------------
 value_time     | timestamp without time zone |           |          |         | plain   |
 update_time    | timestamp without time zone |           |          |         | plain   |
 tick_volume    | double precision            |           |          |         | plain   |
 xauusd_avg_bid | double precision            |           |          |         | plain   |
View definition:
 SELECT value_time,
    update_time,
    tick_volume,
    xauusd_avg_bid
   FROM q_svrwave_xauusd_avg_MAIN_QUEUE_RES
  ORDER BY value_time DESC
 LIMIT 10000;

 */

#include "common/defines.h"

#ifdef INTEGRATION_TEST

#include <cmath>
#include <cstdlib>
#include <gtest/gtest.h>
#include <pqxx/pqxx>
#include "EnsembleService.hpp"
#include "IQScalingFactorService.hpp"
#include "ModelService.hpp"
#include "appcontext.hpp"
#include "DQScalingFactorService.hpp"
#include "model/Priority.hpp"
#include "common/compatibility.hpp"
#include "common/constants.hpp"

using namespace svr;

void prepare_test_queue(datamodel::Dataset &dataset, datamodel::InputQueue &input_queue, datamodel::DeconQueue &decon)
{
    static tbb::mutex m;
    const tbb::mutex::scoped_lock l(m);
    const auto table_name = input_queue.get_table_name();
    PROFILE_MSG(APP.input_queue_service.load(input_queue), "Loading " << table_name);
    PROFILE_MSG(APP.iq_scaling_factor_service.prepare(dataset, input_queue, false), "Prepare input scaling factors for " << table_name);
    PROFILE_MSG(business::DeconQueueService::deconstruct(dataset, input_queue, decon), "Deconstruct " << decon.get_table_name());
}

TEST(manifold_tune_train_predict, basic_integration)
{
    LOG4_BEGIN();

    omp_set_nested(true);

    context::AppContext::init_instance("../config/app.config");

    // Save first N forecasts to database for later analysis
    constexpr uint16_t C_save_forecast = 115;
    constexpr auto C_online_validate = false;
#ifdef VALGRIND_BUILD
    constexpr unsigned C_test_decrement = 5;
#else
    const uint32_t C_test_decrement = 2 * PROPS.get_kernel_length() + PROPS.get_shift_limit() + PROPS.get_outlier_slack(); // 14e3 - common::C_integration_test_validation_window;
#endif
#define MAIN_QUEUE_RES 3600
#define STR_MAIN_QUEUE_RES TOSTR(MAIN_QUEUE_RES)
    const auto C_placement_delay = bpt::seconds(2);
    const auto C_test_labels_len_h = C_test_decrement + common::C_integration_test_validation_window;
    const std::string C_test_input_name = "q_svrwave_test_xauusd_avg_";
    const std::string C_test_input_table_name = C_test_input_name + STR_MAIN_QUEUE_RES;
    const std::string C_test_aux_input_table_name = C_test_input_name + "1";
    constexpr uint16_t C_test_levels = 1;
    constexpr auto C_test_gradient_count = common::C_default_gradient_count;
    constexpr auto C_overload_factor = 2; // Load surplus data from database in case rows discarded during preparation
    const auto C_decon_tail = datamodel::Dataset::get_residuals_length(C_test_levels);
    const uint32_t C_max_features_len = datamodel::C_default_svrparam_lag_count * PROPS.get_lag_multiplier() * business::ModelService::get_max_quantisation();
    const uint32_t C_test_data_len_h = C_overload_factor * (C_test_labels_len_h + cdiv(C_decon_tail + C_max_features_len, MAIN_QUEUE_RES));
    const auto C_test_data_len_h_str = std::to_string(C_test_data_len_h);
    constexpr uint32_t C_dataset_id = 0xDeadBeef;
    const auto C_dataset_id_str = std::to_string(C_dataset_id);
    constexpr char C_last_test_time[] = "2025-01-13 01:00:00";

    try {
        pqxx::connection c(PROPS.get_db_connection_string());
        pqxx::work w(c);
        const std::string q =
                "DROP VIEW IF EXISTS " + C_test_aux_input_table_name + "; " \
                "CREATE VIEW " + C_test_aux_input_table_name + " AS SELECT * FROM (SELECT * FROM q_svrwave_xauusd_avg_1 "
                "WHERE value_time < '" + C_last_test_time + "' ORDER BY value_time DESC LIMIT " + C_test_data_len_h_str + " * " STR_MAIN_QUEUE_RES ") ORDER BY value_time ASC; " \
                "DROP VIEW IF EXISTS " + C_test_input_table_name + ";" \
                "CREATE VIEW " + C_test_input_table_name + " AS SELECT * FROM (SELECT * FROM q_svrwave_xauusd_avg_" STR_MAIN_QUEUE_RES \
                     " WHERE value_time < '" + C_last_test_time + "' ORDER BY value_time DESC LIMIT " + C_test_data_len_h_str + ") ORDER BY value_time ASC;" \
                "DELETE FROM w_scaling_factors WHERE dataset_id = " + C_dataset_id_str + ";" \
                "DELETE FROM iq_scaling_factors WHERE dataset_id = " + C_dataset_id_str + ";" \
                "DELETE FROM dq_scaling_factors WHERE model_id IN (SELECT id FROM models WHERE ensemble_id IN (SELECT id FROM ensembles WHERE dataset_id = " + C_dataset_id_str + ")) ;" \
                "DELETE FROM svr_parameters WHERE dataset_id = " + C_dataset_id_str + ";";
        (void) w.exec(q).no_rows();
        w.commit();
    } catch (const std::exception &ex) {
        LOG4_ERROR("Error " << ex.what() << " while preparing test queue.");
        return;
    }

    auto p_dataset = ptr<datamodel::Dataset>(
            C_dataset_id, "test_dataset", "test_user", C_test_input_table_name, std::deque{C_test_aux_input_table_name}, datamodel::Priority::Normal, "",
            C_test_gradient_count, PROPS.get_kernel_length(), PROPS.get_multistep_len(), C_test_levels, "cvmd", common::C_default_features_max_time_gap);

    business::EnsembleService::init_ensembles(p_dataset, false);
// #pragma omp parallel ADJ_THREADS(std::min<uint16_t>(PROPS.get_parallel_models(), p_dataset->get_spectral_levels() * p_dataset->get_multistep())) default(shared)
// #pragma omp single
    {
        // OMP_TASKLOOP_1()
        for (const auto &p_ensemble: p_dataset->get_ensembles()) {
            const auto &column = p_ensemble->get_column_name();
//            const bool is_ask = column.find("_ask") != std::string::npos;
            prepare_test_queue(*p_dataset, *p_dataset->get_input_queue(), *p_ensemble->get_decon_queue());
//            OMP_TASKLOOP_1()
            for (auto &p_aux_decon_queue: p_ensemble->get_aux_decon_queues())
                prepare_test_queue(*p_dataset, *p_dataset->get_aux_input_queue(p_aux_decon_queue->get_input_queue_table_name()), *p_aux_decon_queue);

            data_row_container times;
            arma::mat recon_predicted(common::C_integration_test_validation_window, p_dataset->get_multistep(), arma::fill::zeros),
                    recon_actual(common::C_integration_test_validation_window, p_dataset->get_multistep(), arma::fill::zeros);
            arma::vec recon_last_knowns(common::C_integration_test_validation_window, arma::fill::zeros);
            tbb::mutex recon_l;
            const auto p_iqsf = p_dataset->get_iq_scaling_factor(p_ensemble->get_aux_decon_queue(column)->get_input_queue_table_name(), column);
            LOG4_TRACE("Got scaling factor " << *p_iqsf);
//            OMP_TASKLOOP_1(collapse(2))
            for (uint16_t l = 0; l < p_dataset->get_spectral_levels(); l += LEVEL_STEP)
                for (uint16_t s = 0; s < p_dataset->get_multistep(); ++s)
                    if (l != p_dataset->get_trans_levix()) {
                        auto p_model = p_ensemble->get_model(l, s);
                        if (!p_model) LOG4_THROW("Model not found!");
                        const auto p_head_params = p_model->get_head_params();
                        p_head_params.first->set_svr_decremental_distance(C_test_decrement);
                        p_head_params.second->set_svr_decremental_distance(C_test_decrement);

                        LOG4_DEBUG("Preparing model " << *p_model << " parameters " << *p_head_params.first << ", integration test validation_window " << common::C_integration_test_validation_window);
                        const auto [p_model_features, p_model_labels, p_model_last_knowns, p_weights, p_model_times] =
                                business::ModelService::get_training_data(*p_dataset, *p_ensemble, *p_model, C_test_labels_len_h);
                        assert(p_model_labels->n_rows == C_test_labels_len_h);
                        assert(p_model_times->size() == C_test_labels_len_h);
                        const uint32_t train_start = p_model_labels->n_rows - C_test_labels_len_h;
                        const uint32_t train_end = p_model_labels->n_rows - common::C_integration_test_validation_window - 1;
                        LOG4_DEBUG("All features size " << arma::size(*p_model_features) << ", test length " << C_test_labels_len_h);
                        const auto last_value_time = p_model_times->at(train_end)->get_value_time();
                        business::ModelService::train_batch(*p_model,
                                                            otr<arma::mat>(p_model_features->rows(train_start, train_end)),
                                                            otr<arma::mat>(p_model_labels->rows(train_start, train_end)),
#ifdef INSTANCE_WEIGHTS
                                                            otr<arma::mat>(p_weights->rows(train_start, train_end)),
#else
                                                            nullptr,
#endif
                                                            last_value_time);
                        p_model->set_last_modeled_value_time(last_value_time);
                        p_model->set_last_modified(bpt::second_clock::local_time());

                        const auto [predict_mae_level, predict_mape_level, predicted, actual, mape_lk, last_knowns] =
                                business::ModelService::validate(
                                        p_model_labels->n_rows - common::C_integration_test_validation_window, *p_dataset, *p_ensemble, *p_model,
                                        *p_model_features, *p_model_labels, *p_model_last_knowns, *p_weights, *p_model_times, C_online_validate,
                                        p_dataset->get_spectral_levels() < MIN_LEVEL_COUNT);
                        const tbb::mutex::scoped_lock lk(recon_l);
                        if (times.empty()) times = *p_model_times;
                        recon_predicted.col(s) += predicted;
                        recon_actual.col(s) += actual;
                        if (!s) recon_last_knowns += last_knowns;
                    }
            recon_predicted = arma::mean(recon_predicted, 1);
            recon_actual = arma::mean(recon_actual, 1);

            LOG4_TRACE("Got scaling factor " << *p_iqsf);
            business::IQScalingFactorService::unscale_I(*p_iqsf, recon_predicted);
            business::IQScalingFactorService::unscale_I(*p_iqsf, recon_last_knowns);
            business::IQScalingFactorService::unscale_I(*p_iqsf, recon_actual);
            LOG4_INFO("Total predicted to actual difference " << common::present<double>(recon_actual - recon_predicted) << ", last known to actual difference " <<
                                                              common::present<double>(recon_actual - recon_last_knowns));

            double mae = 0, mae_lk = 0, recon_mae = 0, recon_lk_mae = 0, pips_won = 0, pips_lost = 0, drawdown = 0, max_drawdown = 0;
            uint16_t positive_mae_ct = 0, pos_direct = 0, price_hits = 0;
            const auto validated_ct = recon_actual.size();
            const auto resolution = p_dataset->get_input_queue()->get_resolution();
            const auto horizon_duration = resolution * PROPS.get_prediction_horizon();
            const auto validate_start = times.size() - validated_ct;
            const auto column_ix = p_dataset->get_input_queue()->get_value_column_index(column);
            for (uint16_t i = 0; i < validated_ct; ++i) {
                const auto i_div = i + 1.;
                const auto cur_time = times[validate_start + i]->get_value_time();
                const auto actual = (**lower_bound(*p_dataset->get_input_queue(), cur_time))[column_ix];
                const auto last_known_iter = lower_bound_before(std::as_const(*p_dataset->get_aux_input_queue()), cur_time - horizon_duration);
                const auto last_known = (**last_known_iter)[column_ix];
                const auto actual_move = actual - last_known;
                const auto recon_actual_move = recon_actual[i] - recon_last_knowns[i];
                const auto predicted_move = recon_predicted[i] - recon_last_knowns[i];
                const auto cur_mae = std::abs(recon_predicted[i] - recon_actual[i]);
                const auto cur_mae_lk = std::abs(recon_actual_move);
                const auto cur_alpha_pct = common::alpha(cur_mae_lk, cur_mae);
                mae += cur_mae;
                mae_lk += cur_mae_lk;
                const auto cur_recon_diff = recon_actual[i] - actual;
                const auto cur_recon_error = std::abs(cur_recon_diff);
                const auto cur_recon_lk_error = std::abs(recon_last_knowns[i] - last_known);
                const auto cml_alpha_pct = common::alpha(mae_lk, mae);
                recon_mae += cur_recon_error;
                recon_lk_mae += cur_recon_lk_error;

                if (mae < mae_lk) LOG4_DEBUG("Positive cumulative alpha at " << i << ", " << cml_alpha_pct << "pc");
                if (cur_mae < cur_mae_lk) {
                    LOG4_DEBUG("Positive alpha " << cur_alpha_pct << "pc, at " << i);
                    ++positive_mae_ct;
                }
                const auto sign_predicted_move = std::signbit(predicted_move);
                const auto start_aux_it = lower_bound(last_known_iter, p_dataset->get_aux_input_queue()->cend(), cur_time);
                const auto last_aux_it = lower_bound(start_aux_it, p_dataset->get_aux_input_queue()->cend(), cur_time + resolution);
                const auto placement_it = lower_bound(last_known_iter, p_dataset->get_aux_input_queue()->cend(), cur_time - horizon_duration + C_placement_delay);
                const auto placement_price = ***placement_it;
                const auto last_aux_price = ***std::prev(last_aux_it);
                constexpr auto time_comp = [](const auto &lhs, const auto &rhs) { return lhs->get_value_time() < rhs->get_value_time(); };
                double this_drawdown;
                if (sign_predicted_move /* && !is_ask */ ) { // Sell signal
                    const auto min_price_it = std::min_element(placement_it /* start_aux_it */, last_aux_it, time_comp);
                    const auto max_price = ***std::max_element(placement_it /* start_aux_it */, last_aux_it, time_comp);
                    const auto min_price = ***min_price_it;
                    if (recon_predicted[i] <= placement_price && recon_predicted[i] >= min_price) {
                        ++price_hits;
                        pips_won += placement_price - recon_predicted[i];
                    } else if (last_aux_price < placement_price) {
                        pips_won += placement_price - last_aux_price;
                    } else
                        pips_lost += last_aux_price - placement_price;
                    this_drawdown = std::max(0., max_price - placement_price);
                    LOG4_TRACE("Sell min price " << min_price << ", max price " << max_price << ", placement price " << placement_price);
                } else if (!sign_predicted_move/* && is_ask */) { // Buy signal
                    const auto max_price_it = std::max_element(placement_it /* start_aux_it */, last_aux_it, time_comp);
                    const auto min_price = ***std::min_element(placement_it /* start_aux_it */, last_aux_it, time_comp);
                    const auto max_price = ***max_price_it;
                    if (recon_predicted[i] >= placement_price && recon_predicted[i] <= max_price) {
                        ++price_hits;
                        pips_won += recon_predicted[i] - placement_price;
                    } else if (last_aux_price > placement_price) {
                        pips_won += last_aux_price - placement_price;
                    } else
                        pips_lost += placement_price - last_aux_price;
                    this_drawdown = std::max(0., placement_price - min_price);
                    LOG4_TRACE("Buy min price " << min_price << ", max price " << max_price << ", placement price " << placement_price);
                }
                if (sign_predicted_move == std::signbit(recon_actual_move)) {
                    LOG4_DEBUG("Direction correct at " << i);
                    ++pos_direct;
                }
                if (common::above_eps(cur_recon_error) || common::above_eps(cur_recon_lk_error))
                    LOG4_WARN("Reconstruction difference at " << cur_time << " between actual " << actual << " and recon price " << \
                    recon_actual[i] << " is " << cur_recon_diff << ", last-known price " << last_known << ", recon last-known " << recon_last_knowns[i] << \
                    ", last known difference " << last_known - recon_last_knowns[i]);
                drawdown += this_drawdown;
                MAXAS(max_drawdown, this_drawdown);
                const auto net_pips = pips_won - pips_lost;
                const auto pips_pos = net_pips / i_div;
                const auto drawdown_pos = drawdown / i_div;
                const auto leverage = drawdown_pos > 0 ? std::max(0., pips_pos / drawdown_pos) : pips_pos;
                const auto abs_leverage = max_drawdown > 0 ? std::max(0., net_pips / max_drawdown) : net_pips;
                const auto positive_preds_pc = 100. * positive_mae_ct / i_div;
                LOG4_INFO("Position " << i << ", column " << column << " " << column_ix << \
                       ", price time " << cur_time << \
                       ", actual price " << actual << \
                       ", recon actual price " << recon_actual[i] << \
                       ", predicted price " << recon_predicted[i] << \
                       ", last-known time " << (**last_known_iter).get_value_time() << \
                       ", last known " << last_known << \
                       ", recon last known " << recon_last_knowns[i] << \
                       ", total MAE " << mae / i_div << \
                       ", total MAE last known " << mae_lk / i_div << \
                       ", positive directions " << 100. * pos_direct / i_div << "pc" \
                       ", positive errors " << positive_preds_pc << "pc" \
                       ", current MAE " << cur_mae << \
                       ", current MAE last known " << cur_mae_lk << \
                       ", predicted movement " << predicted_move << \
                       ", actual movement " << actual_move << \
                       ", recon actual movement " << recon_actual_move << \
                       ", current alpha " << cur_alpha_pct << "pc" \
                       ", cumulative alpha " << cml_alpha_pct << "pc" \
                       ", recon error " << cur_recon_error << \
                       ", recon error last-known " << cur_recon_lk_error << \
                       ", recon label MAE " << 100. * recon_mae / i_div << \
                       ", recon last-known MAE " << 100. * recon_lk_mae / i_div << \
                       ", price hits " << 100. * price_hits / i_div << "pc" \
                       ", won " << pips_won << \
                       ", lost " << pips_lost << \
                       ", neto " << net_pips << \
                       ", value per position " << pips_pos << \
                       ", drawdown per position " << drawdown_pos << \
                       ", sum drawdown " << drawdown << \
                       ", max drawdown " << max_drawdown << \
                       ", mean leverage " << leverage << /* net won to average drawdown ratio */ \
                       ", absolute leverage " << abs_leverage << /* net won to maximum drawdown ratio */ \
                       ", trade rating " << abs_leverage * positive_preds_pc * cml_alpha_pct);
                if (i < C_save_forecast && std::isnormal(recon_predicted[i]))
                    APP.request_service.save(ptr<datamodel::MultivalResponse>(0, 0, cur_time, column, recon_predicted[i]));
            }
            mae /= validated_ct;
            mae_lk /= validated_ct;
            const auto labels_meanabs = common::meanabs(recon_actual);
            const auto mape = common::mape(mae, labels_meanabs);
            const auto mape_lk = common::mape(mae_lk, labels_meanabs);
            const auto alpha_pct = common::alpha(mape_lk, mape);
            LOG4_INFO("Total MAE of " << validated_ct << " compared values for ensemble " << column << " is " << mae << "," " MAPE is " << mape << "pc," " last-known MAE " << \
                mae_lk << ", last-known MAPE " << mape_lk << "pc," " alpha " << alpha_pct << "pc," " positive direction " << 100. * double(pos_direct) / double(validated_ct) << \
                "pc, positive error " << 100. * double(positive_mae_ct) / double(validated_ct) << "pc");

        }
    }
}

#endif
