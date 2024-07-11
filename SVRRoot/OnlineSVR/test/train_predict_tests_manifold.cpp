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
 LIMIT (10000 * 3600);

 \d+ q_svrwave_test_xauusd_avg_3600
                             View "public.q_svrwave_test_xauusd_avg_3600"
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
   FROM q_svrwave_xauusd_avg_3600
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

// #define TEST_MANIFOLD
#define TEST_ACTUAL_DATA
#define ONLINE_VALIDATE false

namespace {
#ifdef VALGRIND_BUILD
constexpr unsigned C_test_decrement = 1;
constexpr unsigned C_test_lag = 1;
#else
const unsigned C_test_decrement = common::C_default_kernel_max_chunk_size;
constexpr unsigned C_test_lag = datamodel::C_default_svrparam_lag_count;
#endif

const bpt::time_duration C_placement_delay = bpt::seconds(3);
const unsigned C_test_length = C_test_decrement + C_test_len + common::C_integration_test_validation_window;
const std::string C_test_input_table_name = "q_svrwave_test_xauusd_avg_3600";
const std::string C_test_aux_input_table_name = "q_svrwave_test_xauusd_avg_1";
constexpr unsigned C_test_levels = 1; // 2 * MIN_LEVEL_COUNT;
constexpr unsigned C_test_gradient_count = common::C_default_gradient_count;
}

#define OVERLOAD_FACTOR 2

TEST(manifold_tune_train_predict, basic_integration)
{
    omp_set_nested(true);
    // omp_set_max_active_levels(20 * std::thread::hardware_concurrency());
    svr::context::AppContext::init_instance("../config/app.config");

    try {
        pqxx::connection c(PROPS.get_db_connection_string());
        pqxx::work w(c);
        const std::string q = common::formatter() << "DROP VIEW IF EXISTS q_svrwave_test_xauusd_avg_1; "
                                                     "CREATE VIEW " << C_test_aux_input_table_name
                                                  << " AS SELECT value_time, update_time, tick_volume, xauusd_avg_bid FROM q_svrwave_xauusd_avg_1 "
                                                     "ORDER BY value_time DESC LIMIT (" << C_test_length * OVERLOAD_FACTOR << " * 3600); "
                                                                                                                              "DROP VIEW IF EXISTS q_svrwave_test_xauusd_avg_3600; "
                                                                                                                              "CREATE VIEW " << C_test_input_table_name
                                                  << " AS SELECT value_time, update_time, tick_volume, xauusd_avg_bid FROM q_svrwave_xauusd_avg_3600 "
                                                     "ORDER BY value_time DESC LIMIT " << C_test_length * OVERLOAD_FACTOR;
        w.exec0(q);
        w.commit();
    } catch (const std::exception &ex) {
        LOG4_DEBUG("Error " << ex.what() << " while preparing test queue.");
        return;
    }

    auto p_dataset = ptr<datamodel::Dataset>(
            0xDeadBeef, "test_dataset", "test_user", C_test_input_table_name, std::deque{C_test_aux_input_table_name}, datamodel::Priority::Normal, "",
            C_test_gradient_count, common::C_default_kernel_max_chunk_size, PROPS.get_multistep_len(), C_test_levels, "cvmd", common::C_default_features_max_time_gap);

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
            PROFILE_EXEC_TIME(APP.iq_scaling_factor_service.prepare(*p_dataset, *p_dataset->get_input_queue(), false),
                              "Prepare input scaling factors for " << table_name);
            PROFILE_EXEC_TIME(business::DeconQueueService::deconstruct(*p_dataset, *p_dataset->get_input_queue(), *p_decon), "Deconstruct " << table_name);
        }
#pragma omp task
        {
            const auto table_name = p_dataset->get_aux_input_queue()->get_table_name();
            PROFILE_EXEC_TIME(APP.input_queue_service.load(*p_dataset->get_aux_input_queue()), "Loading " << table_name);
            PROFILE_EXEC_TIME(APP.iq_scaling_factor_service.prepare(*p_dataset, *p_dataset->get_aux_input_queue(), false),
                              "Prepare input scaling factors for " << table_name);
            PROFILE_EXEC_TIME(business::DeconQueueService::deconstruct(*p_dataset, *p_dataset->get_aux_input_queue(), *p_decon_aux), "Deconstruct " << table_name);
        }
    }

    times_ptr p_times;
    arma::mat recon_predicted(common::C_integration_test_validation_window, p_dataset->get_multistep()),
            recon_last_knowns(common::C_integration_test_validation_window, p_dataset->get_multistep()),
            recon_actual(common::C_integration_test_validation_window, p_dataset->get_multistep());
    t_omp_lock recon_l;
#pragma omp parallel for schedule(static, 1) num_threads(adj_threads(std::min<size_t>(C_parallel_train_models, p_dataset->get_transformation_levels() * p_dataset->get_multistep()))) collapse(2)
    for (size_t l = 0; l < p_dataset->get_transformation_levels(); l += 2)
        for (size_t s = 0; s < p_dataset->get_multistep(); ++s)
            if (l != p_dataset->get_half_levct()) {
                auto p_ensemble = p_dataset->get_ensemble();
                auto p_model = p_ensemble->get_model(l, s);
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
                const auto [p_model_features, p_model_labels, p_model_last_knowns, p_model_times] = business::ModelService::get_training_data(
                        *p_dataset, *p_ensemble, *p_model, C_test_length);
                assert(p_model_labels->n_rows == C_test_length);
                assert(p_model_times->size() == C_test_length);
                const size_t train_start = p_model_labels->n_rows - C_test_length;
                const size_t train_end = p_model_labels->n_rows - common::C_integration_test_validation_window - 1;
#else
                auto p_all_features = ptr<arma::mat>(C_test_length, TEST_LAG, arma::fill::randn);
                auto p_all_labels = ptr<arma::mat>(C_test_length, 1, arma::fill::randn);
                auto p_all_last_knowns = ptr<arma::mat>(C_test_length, 1, arma::fill::randn);
                std::deque<bpt::ptime> times(C_test_length, bpt::second_clock::local_time());
#endif
                LOG4_DEBUG("All features size " << arma::size(*p_model_features) << ", test length " << C_test_length);
                const auto feat_levels = p_head_params->get_adjacent_levels();
                const auto last_value_time = p_model_times->back();
                business::ModelService::train_batch(*p_model, otr<arma::mat>(p_model_features->rows(train_start, train_end)),
                            otr<arma::mat>(p_model_labels->rows(train_start, train_end)),
                            otr<arma::vec>(p_model_last_knowns->rows(train_start, train_end)), last_value_time);
                p_model->set_last_modeled_value_time(last_value_time);
                p_model->set_last_modified(bpt::second_clock::local_time());

                const auto [predict_mae_level, predict_mape_level, predicted, actual, mape_lk, last_knowns] =
                        business::ModelService::validate(
                                p_model_labels->n_rows - common::C_integration_test_validation_window,
                                *p_dataset, *p_ensemble, *p_model,
                                *p_model_features, *p_model_labels, *p_model_last_knowns, *p_model_times,
                                ONLINE_VALIDATE, p_dataset->get_transformation_levels() < MIN_LEVEL_COUNT);
                recon_l.set();
                if (!p_times) p_times = p_model_times;
                recon_predicted.col(s) += predicted;
                recon_actual.col(s) += actual;
                recon_last_knowns.col(s) += last_knowns;
                recon_l.unset();
            }
    recon_predicted = arma::mean(recon_predicted, 1);
    recon_actual = arma::mean(recon_actual, 1);
    recon_last_knowns = arma::mean(recon_last_knowns, 1);

    const auto p_iqsf = p_dataset->get_iq_scaling_factor(p_dataset->get_aux_input_queue()->get_table_name(), p_dataset->get_ensemble()->get_column_name());
    recon_predicted = recon_predicted * p_iqsf->get_scaling_factor() + p_iqsf->get_dc_offset();
    recon_last_knowns = recon_last_knowns * p_iqsf->get_scaling_factor() + p_iqsf->get_dc_offset();
    recon_actual = recon_actual * p_iqsf->get_scaling_factor() + p_iqsf->get_dc_offset();

    double mae = 0, mae_lk = 0, recon_mae = 0, recon_lk_mae = 0, pips_won = 0, pips_lost = 0;
    size_t positive_mae = 0, pos_direct = 0, price_hits = 0;
    const auto validated_ct = recon_actual.size();
    const auto resolution = p_dataset->get_input_queue()->get_resolution();
    for (size_t i = 0; i < validated_ct; ++i) {
        const auto i_div = double(i + 1);
        const auto cur_time = p_times->at(p_times->size() - validated_ct + i);
        const auto actual = (**lower_bound(*p_dataset->get_input_queue(), cur_time))[0];
        const auto last_known_iter = lower_bound_back_before(std::as_const(*p_dataset->get_aux_input_queue()), cur_time - resolution * PROPS.get_prediction_offset());
        const auto last_known = (**last_known_iter)[0];
        const double cur_mae = std::abs(recon_predicted[i] - actual);
        const auto diff_actual_last_known = actual - last_known;
        const double cur_mae_lk = std::abs(diff_actual_last_known);
        const double cur_alpha_pct = 100. * (cur_mae_lk / cur_mae - 1.);

        const double cur_recon_error = std::abs(recon_actual[i] - actual);
        const double cur_recon_lk_error = std::abs(recon_last_knowns[i] - last_known);
        const auto cml_alpha_pct = 100. * (mae_lk / mae - 1.);

        mae += cur_mae;
        mae_lk += cur_mae_lk;
        recon_mae += cur_recon_error;
        recon_lk_mae += cur_recon_lk_error;

        if (mae < mae_lk) LOG4_DEBUG("Positive cumulative alpha at " << i << ", " << cml_alpha_pct << "pc");
        if (cur_mae < cur_mae_lk) {
            LOG4_DEBUG("Positive alpha " << cur_alpha_pct << "pc, at " << i);
            ++positive_mae;
        }
        const auto sign_recon_predicted_last_known = std::signbit(recon_predicted[i] - last_known);
        const auto start_aux_iter = lower_bound(last_known_iter, p_dataset->get_aux_input_queue()->cend(), cur_time);
        const auto last_aux_iter = lower_bound(start_aux_iter, p_dataset->get_aux_input_queue()->end(), cur_time + resolution);
        const auto placement_price = (**lower_bound(last_known_iter, p_dataset->get_aux_input_queue()->cend(),
                                                    cur_time - resolution * PROPS.get_prediction_offset() + C_placement_delay))[0];
        const auto last_aux_price = (**std::prev(last_aux_iter))[0];
        if (sign_recon_predicted_last_known) { // Sell signal
            const auto min_price = (**std::min_element(start_aux_iter, last_aux_iter,
                                                       [](const auto lhs, const auto rhs) { return lhs->get_value_time() < rhs->get_value_time(); }))[0];
            if (recon_predicted[i] <= placement_price && recon_predicted[i] >= min_price) {
                ++price_hits;
                pips_won += placement_price - recon_predicted[i];
            } else if (last_aux_price < placement_price) {
                ++price_hits;
                pips_won += placement_price - last_aux_price;
            } else
                pips_lost += last_aux_price - placement_price;
        } else { // Buy signal
            const auto max_price = (**std::max_element(start_aux_iter, last_aux_iter,
                                                       [](const auto lhs, const auto rhs) { return lhs->get_value_time() < rhs->get_value_time(); }))[0];
            if (recon_predicted[i] >= placement_price && recon_predicted[i] <= max_price) {
                ++price_hits;
                pips_won += recon_predicted[i] - placement_price;
            } else if (last_aux_price > placement_price) {
                ++price_hits;
                pips_won += last_aux_price - placement_price;
            } else
                pips_lost += placement_price - last_aux_price;
        }
        if (sign_recon_predicted_last_known == std::signbit(diff_actual_last_known)) {
            LOG4_DEBUG("Direction correct at " << i);
            ++pos_direct;
        }
        if (cur_recon_error > std::numeric_limits<double>::epsilon() || cur_recon_lk_error > std::numeric_limits<double>::epsilon())
            LOG4_WARN("Recon differ at " << cur_time << " between actual price " << actual << " and recon price " << recon_actual[i] << ", is "
                                         << actual - recon_actual[i] << ", last-known price " << last_known << ", recon last-known " << recon_last_knowns[i] <<
                                         ", last known difference " << last_known - recon_last_knowns[i]);

        LOG4_DEBUG("Position " << i <<
                               ", price time " << cur_time <<
                               ", actual price " << actual <<
                               ", recon actual price " << recon_actual[i] <<
                               ", predicted price " << recon_predicted[i] <<
                               ", last known " << last_known <<
                               ", recon last known " << recon_last_knowns[i] <<
                               ", total MAE " << mae / i_div <<
                               ", total MAE last known " << mae_lk / i_div <<
                               ", positive directions " << 100. * double(pos_direct) / i_div << "pc"
                                                                                                ", positive errors " << 100. * double(positive_mae) / i_div << "pc"
                                                                                                                                                               ", current MAE "
                               << cur_mae <<
                               ", current MAE last known " << cur_mae_lk <<
                               ", predicted movement " << recon_predicted[i] - last_known <<
                               ", actual movement " << actual - last_known <<
                               ", current alpha " << cur_alpha_pct << "pc, cumulative alpha " << cml_alpha_pct << "pc"
                                                                                                                  ", recon error " << cur_recon_error <<
                               ", average recon error " << 100. * recon_mae / i_div << "pc" <<
                               ", average recon error last-known " << 100. * recon_lk_mae / i_div << "pc" <<
                               ", price hits " << 100. * double(price_hits) / i_div << "pc" <<
                               ", won PIPs " << pips_won <<
                               ", lost PIPs " << pips_lost <<
                               ", net PIPs " << pips_won - pips_lost <<
                               ", PIPs per position " << (pips_won - pips_lost) / i_div);
        if (i < common::C_forecast_focus && std::isnormal(recon_predicted[i]))
            APP.request_service.save(ptr<datamodel::MultivalResponse>(0, 0, cur_time, p_decon->get_input_queue_column_name(), recon_predicted[i]));
    }
    mae /= double(validated_ct);
    mae_lk /= double(validated_ct);
    const double labels_meanabs = common::meanabs(recon_actual);
    const double mape = 100 * mae / labels_meanabs;
    const double mape_lk = 100 * mae_lk / labels_meanabs;
    const double alpha_pct = 100 * (mape_lk / mape - 1.);
    LOG4_INFO("Total MAE of " << validated_ct << " compared values is " << mae << ","
                                                                                  " MAPE is " << mape << "pc,"
                                                                                                         " last-known MAE " << mae_lk << ","
                                                                                                                                         " last-known MAPE " << mape_lk
                              << "pc,"
                                 " alpha " << alpha_pct << "pc,"
                                                           " positive direction " << 100. * double(pos_direct) / double(validated_ct) << "pc,"
                                                                                                                                         " positive error "
                              << 100. * double(positive_mae) / double(validated_ct) << "pc");
}

#endif
