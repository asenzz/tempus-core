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

namespace {

#ifdef VALGRIND_BUILD
constexpr unsigned C_test_decrement = 1;
constexpr unsigned C_test_lag = 1;
#else
constexpr auto C_online_validate = false;
constexpr auto C_test_decrement = 2 * common::C_default_kernel_max_chunk_len;
constexpr auto C_test_lag = datamodel::C_default_svrparam_lag_count;
#endif
#define MAIN_QUEUE_RES 3600
#define STR_MAIN_QUEUE_RES TOSTR(MAIN_QUEUE_RES)
const auto C_placement_delay = bpt::seconds(3);
const auto C_test_labels_len_h = C_test_decrement + C_test_len + common::C_integration_test_validation_window;
const std::string C_test_input_table_name = "q_svrwave_test_xauusd_avg_" STR_MAIN_QUEUE_RES;
const std::string C_test_aux_input_table_name = "q_svrwave_test_xauusd_avg_1";
constexpr unsigned C_test_levels = MIN_LEVEL_COUNT;
constexpr auto C_test_gradient_count = common::C_default_gradient_count;
constexpr auto C_overload_factor = 1.2;
const auto C_decon_tail = svr::datamodel::Dataset::get_residuals_length(C_test_levels);
const unsigned C_max_features_len = C_test_lag * datamodel::C_features_superset_coef * business::ModelService::C_max_quantisation;
const unsigned C_test_features_len_h = C_overload_factor * (C_test_labels_len_h + cdiv(C_decon_tail + C_max_features_len, MAIN_QUEUE_RES));

}

void prepare_test_queue(datamodel::Dataset &dataset, datamodel::InputQueue &input_queue, datamodel::DeconQueue &decon)
{
    const auto table_name = input_queue.get_table_name();
    PROFILE_EXEC_TIME(APP.input_queue_service.load(input_queue), "Loading " << table_name);
    PROFILE_EXEC_TIME(APP.iq_scaling_factor_service.prepare(dataset, input_queue, false), "Prepare input scaling factors for " << table_name);
    PROFILE_EXEC_TIME(business::DeconQueueService::deconstruct(dataset, input_queue, decon), "Deconstruct " << decon.get_table_name());
}

TEST(manifold_tune_train_predict, basic_integration)
{
    LOG4_BEGIN();

    omp_set_nested(true);
    svr::context::AppContext::init_instance("../config/app.config");

    try {
        pqxx::connection c(PROPS.get_db_connection_string());
        pqxx::work w(c);
        const std::string q = common::formatter() << "DROP VIEW IF EXISTS q_svrwave_test_xauusd_avg_1; "
                                                     "CREATE VIEW " << C_test_aux_input_table_name
                                                  << " AS SELECT value_time, update_time, tick_volume, xauusd_avg_bid FROM q_svrwave_xauusd_avg_1 "
                                                     "ORDER BY value_time DESC LIMIT (" << C_test_features_len_h << " * " << TOSTR(MAIN_QUEUE_RES) << "); "
                                                  "DROP VIEW IF EXISTS q_svrwave_test_xauusd_avg_" STR_MAIN_QUEUE_RES "; "
                                                  "CREATE VIEW " << C_test_input_table_name
                                                  << " AS SELECT value_time, update_time, tick_volume, xauusd_avg_bid FROM q_svrwave_xauusd_avg_" STR_MAIN_QUEUE_RES
                                                     " ORDER BY value_time DESC LIMIT " << C_test_features_len_h;
        w.exec(q).no_rows();
        w.commit();
    } catch (const std::exception &ex) {
        LOG4_DEBUG("Error " << ex.what() << " while preparing test queue.");
        return;
    }

    auto p_dataset = ptr<datamodel::Dataset>(
            0xDeadBeef, "test_dataset", "test_user", C_test_input_table_name, std::deque{C_test_aux_input_table_name}, datamodel::Priority::Normal, "",
            C_test_gradient_count, common::C_default_kernel_max_chunk_len, PROPS.get_multistep_len(), C_test_levels, "cvmd", common::C_default_features_max_time_gap);

    business::EnsembleService::init_ensembles(p_dataset, false);
    auto p_decon = p_dataset->get_ensemble()->get_decon_queue();
    auto p_decon_aux = p_dataset->get_ensemble()->get_aux_decon_queue();
#pragma omp parallel num_threads(adj_threads(1 + p_dataset->get_aux_input_queues().size()))
#pragma omp single
    {
#pragma omp task
        prepare_test_queue(*p_dataset, *p_dataset->get_input_queue(), *p_decon);
#pragma omp task
        prepare_test_queue(*p_dataset, *p_dataset->get_aux_input_queue(), *p_decon_aux);
    }

    times_ptr p_times;
    arma::mat recon_predicted(common::C_integration_test_validation_window, p_dataset->get_multistep()),
            recon_last_knowns(common::C_integration_test_validation_window, p_dataset->get_multistep()),
            recon_actual(common::C_integration_test_validation_window, p_dataset->get_multistep());
    t_omp_lock recon_l;
#pragma omp parallel for schedule(static, 1) collapse(2) \
    num_threads(adj_threads(std::min<unsigned>(C_parallel_train_models, p_dataset->get_spectral_levels() * p_dataset->get_multistep())))
    for (unsigned l = 0; l < p_dataset->get_spectral_levels(); l += LEVEL_STEP)
        for (unsigned s = 0; s < p_dataset->get_multistep(); ++s)
            if (l != p_dataset->get_trans_levix()) {
                auto p_ensemble = p_dataset->get_ensemble();
                auto p_model = p_ensemble->get_model(l, s);
                if (!p_model) LOG4_THROW("Model not found!");
                datamodel::SVRParameters_ptr p_head_params;
                for (auto &p_gradient: p_model->get_gradients())
                    for (auto &p_params: p_gradient->get_param_set()) {
                        if (!p_head_params) p_head_params = p_params;
#ifdef TEST_MANIFOLD
                        p_params->set_kernel_type(datamodel::e_kernel_type::DEEP_PATH);
#endif
#ifdef TEST_ACTUAL_DATA
                        p_params->set_svr_decremental_distance(C_test_decrement);
                        p_params->set_lag_count(C_test_lag);
                    }

                LOG4_DEBUG("Preparing model " << *p_model << " parameters " << *p_head_params);
                const auto [p_model_features, p_model_labels, p_model_last_knowns, p_model_times] =
                        business::ModelService::get_training_data(*p_dataset, *p_ensemble, *p_model, C_test_labels_len_h);
                assert(p_model_labels->n_rows == C_test_labels_len_h);
                assert(p_model_times->size() == C_test_labels_len_h);
                const unsigned train_start = p_model_labels->n_rows - C_test_labels_len_h;
                const unsigned train_end = p_model_labels->n_rows - common::C_integration_test_validation_window - 1;
#else
                auto p_all_features = ptr<arma::mat>(C_test_length, TEST_LAG, arma::fill::randn);
                auto p_all_labels = ptr<arma::mat>(C_test_length, 1, arma::fill::randn);
                auto p_all_last_knowns = ptr<arma::mat>(C_test_length, 1, arma::fill::randn);
                std::deque<bpt::ptime> times(C_test_length, bpt::second_clock::local_time());
#endif
                LOG4_DEBUG("All features size " << arma::size(*p_model_features) << ", test length " << C_test_labels_len_h);
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
                                C_online_validate, p_dataset->get_spectral_levels() < MIN_LEVEL_COUNT);
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

    double mae = 0, mae_lk = 0, recon_mae = 0, recon_lk_mae = 0, pips_won = 0, pips_lost = 0, drawdown = 0;
    unsigned positive_mae_ct = 0, pos_direct = 0, price_hits = 0;
    const auto validated_ct = recon_actual.size();
    const auto resolution = p_dataset->get_input_queue()->get_resolution();
    const auto offset_period = resolution * PROPS.get_prediction_horizon();
    UNROLL()
    for (unsigned i = 0; i < validated_ct; ++i) {
        const auto i_div = i + 1.;
        const auto cur_time = p_times->at(p_times->size() - validated_ct + i);
        const auto actual = recon_actual[i]; // ***lower_bound(*p_dataset->get_input_queue(), cur_time);
        const auto last_known_iter = lower_bound_back_before(std::as_const(*p_dataset->get_aux_input_queue()), cur_time - resolution * PROPS.get_prediction_horizon());
        const auto last_known = ***last_known_iter;
        const auto cur_mae = std::abs(recon_predicted[i] - actual);
        const auto diff_actual_last_known = actual - last_known;
        const auto cur_mae_lk = std::abs(diff_actual_last_known);
        const auto cur_alpha_pct = common::alpha(cur_mae_lk, cur_mae);

        const auto cur_recon_diff = recon_actual[i] - ***lower_bound(*p_dataset->get_input_queue(), cur_time);
        const auto cur_recon_error = std::abs(cur_recon_diff);
        const auto cur_recon_lk_error = std::abs(recon_last_knowns[i] - last_known);
        const auto cml_alpha_pct = common::alpha(mae_lk, mae);

        mae += cur_mae;
        mae_lk += cur_mae_lk;
        recon_mae += cur_recon_error;
        recon_lk_mae += cur_recon_lk_error;

        if (mae < mae_lk) LOG4_DEBUG("Positive cumulative alpha at " << i << ", " << cml_alpha_pct << "pc");
        if (cur_mae < cur_mae_lk) {
            LOG4_DEBUG("Positive alpha " << cur_alpha_pct << "pc, at " << i);
            ++positive_mae_ct;
        }
        const auto sign_recon_predicted_last_known = std::signbit(recon_predicted[i] - last_known);
        const auto start_aux_it = lower_bound(last_known_iter, p_dataset->get_aux_input_queue()->cend(), cur_time);
        const auto last_aux_it = lower_bound(start_aux_it, p_dataset->get_aux_input_queue()->end(), cur_time + resolution);
        const auto placement_it = lower_bound(last_known_iter, p_dataset->get_aux_input_queue()->cend(), cur_time - offset_period + C_placement_delay);
        const auto placement_price = ***placement_it;
        const auto last_aux_price = ***std::prev(last_aux_it);
        constexpr auto time_comp = [](const auto &lhs, const auto &rhs) { return lhs->get_value_time() < rhs->get_value_time(); };
        if (sign_recon_predicted_last_known) { // Sell signal
            const auto min_price_it = std::min_element(placement_it /* start_aux_it */, last_aux_it, time_comp);
            const auto max_price = ***std::max_element(placement_it /* start_aux_it */, last_aux_it, time_comp);
            const auto min_price = ***min_price_it;
            if (recon_predicted[i] <= placement_price && recon_predicted[i] >= min_price) {
                ++price_hits;
                pips_won += placement_price - recon_predicted[i];
            } else if (last_aux_price < placement_price) {
                ++price_hits;
                pips_won += placement_price - last_aux_price;
            } else
                pips_lost += last_aux_price - placement_price;
            drawdown += std::max(0., max_price - placement_price);
            LOG4_TRACE("Sell min price " << min_price << ", max price " << max_price << ", placement price " << placement_price);
        } else { // Buy signal
            const auto max_price_it = std::max_element(placement_it /* start_aux_it */, last_aux_it, time_comp);
            const auto min_price = ***std::min_element(placement_it /* start_aux_it */, last_aux_it, time_comp);
            const auto max_price = ***max_price_it;
            if (recon_predicted[i] >= placement_price && recon_predicted[i] <= max_price) {
                ++price_hits;
                pips_won += recon_predicted[i] - placement_price;
            } else if (last_aux_price > placement_price) {
                ++price_hits;
                pips_won += last_aux_price - placement_price;
            } else
                pips_lost += placement_price - last_aux_price;
            drawdown += std::max(0., placement_price - min_price);
            LOG4_TRACE("Buy min price " << min_price << ", max price " << max_price << ", placement price " << placement_price);
        }
        if (sign_recon_predicted_last_known == std::signbit(diff_actual_last_known)) {
            LOG4_DEBUG("Direction correct at " << i);
            ++pos_direct;
        }
        if (cur_recon_error > std::numeric_limits<double>::epsilon() || cur_recon_lk_error > std::numeric_limits<double>::epsilon())
            LOG4_WARN("Recon differ at " << cur_time << " between actual price " << ***lower_bound(*p_dataset->get_input_queue(), cur_time) << " and recon price " << recon_actual[i] << ", is "
                                         << cur_recon_diff << ", last-known price " << last_known << ", recon last-known " << recon_last_knowns[i] <<
                                         ", last known difference " << last_known - recon_last_knowns[i]);
        const auto pips_pos = (pips_won - pips_lost) / i_div;
        const auto drawdown_pos = drawdown / i_div;
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
                                ", positive errors " << 100. * double(positive_mae_ct) / i_div << "pc"
                                ", current MAE " << cur_mae <<
                                ", current MAE last known " << cur_mae_lk <<
                                ", predicted movement " << recon_predicted[i] - last_known <<
                                ", actual movement " << actual - last_known <<
                                ", current alpha " << cur_alpha_pct << "pc, cumulative alpha " << cml_alpha_pct << "pc"
                                ", recon error " << cur_recon_error <<
                                ", average recon error " << 100. * recon_mae / i_div << "pc" <<
                                ", average recon error last-known " << 100. * recon_lk_mae / i_div << "pc" <<
                                ", price hits " << 100. * double(price_hits) / i_div << "pc" <<
                                ", won " << pips_won <<
                                ", lost " << pips_lost <<
                                ", net " << pips_won - pips_lost <<
                                ", value per position " << pips_pos <<
                                ", drawdown per position " << drawdown_pos <<
                                ", total drawdown " << drawdown <<
                                ", drawdown ratio " << drawdown_pos / pips_pos);
        if (i < common::C_forecast_focus && std::isnormal(recon_predicted[i]))
            APP.request_service.save(ptr<datamodel::MultivalResponse>(0, 0, cur_time, p_decon->get_input_queue_column_name(), recon_predicted[i]));
    }
    mae /= validated_ct;
    mae_lk /= validated_ct;
    const auto labels_meanabs = common::meanabs(recon_actual);
    const auto mape = common::mape(mae, labels_meanabs);
    const auto mape_lk = common::mape(mae_lk, labels_meanabs);
    const auto alpha_pct = common::alpha(mape_lk, mape);
    LOG4_INFO("Total MAE of " << validated_ct << " compared values is " << mae << "," " MAPE is " << mape << "pc," " last-known MAE " << mae_lk << ","
              " last-known MAPE " << mape_lk << "pc," " alpha " << alpha_pct << "pc," " positive direction " << 100. * double(pos_direct) / double(validated_ct) << "pc,"
                " positive error " << 100. * double(positive_mae_ct) / double(validated_ct) << "pc");
}

#endif
