#include "../../OnlineSVR/include/recombine_parameters.cuh"
#include "DatasetService.hpp"
#include "InputQueueService.hpp"
#include "EnsembleService.hpp"
#include "ModelService.hpp"
#include "UserService.hpp"
#include "common/constants.hpp"
#include "include/DaoTestFixture.h"
#include "model/Dataset.hpp"
#include "model/InputQueue.hpp"
#include "model/User.hpp"
#include "util/string_utils.hpp"
#include "util/time_utils.hpp"
#include "DAO/DataSource.hpp"
#include "RequestService.hpp"
#include "DataRowService.hpp"

using namespace svr;

TEST_F(DaoTestFixture, DatasetTuningRecombination)
{
    constexpr uint32_t colct = 31; // levct
    constexpr uint32_t rowct = 34144256;
    const double best_score = std::numeric_limits<double>::max();
    std::vector<t_param_preds_cu> params_preds(colct * common::C_tune_keep_preds);
#if 0
    std::vector<uint8_t> combos(rowct * colct, uint8_t(1));
#endif
    for (uint32_t colix = 0; colix < colct; ++colix) {
#if 0
        combos[.5 * rowct * colct + colix] = 0;
#endif
        params_preds[colix].params_ix = 77;
    }
    arma::uchar_mat combos(rowct, colct, arma::fill::ones);
    combos.row(combos.n_rows / 2).fill(0);
    std::vector<uint8_t> best_params_ixs(colct, uint8_t(0));
//    PROFILE_INFO(recombine_parameters(rowct, colct, combos.memptr(), params_preds.data(), &best_score, best_params_ixs.data()), "recombine_parameters");
    LOG4_DEBUG("Best score " << best_score << ", best params ixs " << common::to_string(best_params_ixs));
}


TEST_F(DaoTestFixture, DatasetWorkflow)
{
    auto user1 = std::make_shared<datamodel::User>(
            bigint(), "DeconQueueTestUser", "DeconQueueTestUser@email", "DeconQueueTestUser", "DeconQueueTestUser", datamodel::ROLE::ADMIN, datamodel::Priority::High) ;

    aci.user_service.save(user1);

    auto iq = std::make_shared<datamodel::InputQueue>(
            "tableName", "logicalName", user1->get_name(), "description", bpt::seconds(60), bpt::seconds(5), "UTC", std::deque<std::string>{"up", "down", "left", "right"} );
    aci.input_queue_service.save(iq);

    const auto ds = std::make_shared<datamodel::Dataset>(0, "DeconQueueTestDataset", user1->get_user_name(), iq, std::deque<datamodel::InputQueue_ptr>{}
            , datamodel::Priority::Normal, "", common::C_default_residual_coef, 1, common::AppConfig::C_default_kernel_length, PROPS.get_steps(), 4, "sym7");
    ds->set_is_active(true);

    ds->set_max_lookback_time_gap(common::date_time_string_to_seconds("38,21:22:23"));

    aci.dataset_service.save(ds);

    business::DatasetService::UserDatasetPairs dsu;
    aci.dataset_service.update_active_datasets(dsu);
    EXPECT_TRUE(2UL == dsu.size());

    datamodel::Dataset_ptr &p_dataset = dsu[0].p_dataset;
    p_dataset->set_input_queue(
            aci.input_queue_service.get_queue_metadata(
                    p_dataset->get_input_queue()->get_table_name()));

    EXPECT_TRUE(p_dataset->get_max_lookback_time_gap() == bpt::hours(38*24 + 21) + bpt::minutes(22) + bpt::seconds(23) );

    aci.dataset_service.remove(ds);
    aci.input_queue_service.remove(iq);
    aci.user_service.remove(user1);
}

TEST_F(DaoTestFixture, SelectingActiveDatasets)
{
    datamodel::User_ptr user1Low = std::make_shared<datamodel::User>(
            bigint(), "User2016-07-20-Low", "User2016-07-20-Low@dkdk.dld", "User2016-07-20-Low", "User2016-07-20-Low", datamodel::ROLE::ADMIN, datamodel::Priority::Low) ;

    aci.user_service.save(user1Low);

    datamodel::InputQueue_ptr iq1 = std::make_shared<datamodel::InputQueue>(
            "InputQueue1", "InputQueue1", user1Low->get_name(), "InputQueue1", bpt::seconds(60), bpt::seconds(5), "UTC", std::deque<std::string>{"up", "down", "left", "right"} );
    aci.input_queue_service.save(iq1);

    datamodel::Dataset_ptr ds1 = std::make_shared<datamodel::Dataset>(0, "Dataset2016-07-20-Low", user1Low->get_user_name(), iq1, std::deque<datamodel::InputQueue_ptr>{}
            , datamodel::Priority::Low, "", common::C_default_residual_coef, 1, common::AppConfig::C_default_kernel_length, PROPS.get_steps(), 4, "sym7");
    ds1->set_is_active(true);

    aci.dataset_service.save(ds1);

    ////////////////////////////////////////////////////////////////////////////

    business::DatasetService::UserDatasetPairs pairs;
    aci.dataset_service.update_active_datasets(pairs);

    EXPECT_TRUE(2UL == pairs.size());
    auto iter = pairs.begin();
    EXPECT_TRUE(iter->p_dataset->get_dataset_name() == "eurusd"); EXPECT_TRUE(1UL == iter->users.size()); EXPECT_TRUE(iter->users[0]->get_user_name() == "svrwave");
    ++iter;
    EXPECT_TRUE(iter->p_dataset->get_dataset_name() == "Dataset2016-07-20-Low"); EXPECT_TRUE(1UL == iter->users.size()); EXPECT_TRUE(iter->users[0]->get_user_name() == "User2016-07-20-Low");

    ////////////////////////////////////////////////////////////////////////////

    datamodel::User_ptr user2Normal = std::make_shared<datamodel::User>(
            bigint(), "User2016-07-20-Normal", "User2016-07-20-Normal@dkdk.dld", "User2016-07-20-Normal", "User2016-07-20-Normal"
            , datamodel::ROLE::ADMIN, datamodel::Priority::Normal) ;

    aci.user_service.save(user2Normal);

    datamodel::InputQueue_ptr iq2 = std::make_shared<datamodel::InputQueue>(
            "InputQueue2", "InputQueue2", user2Normal->get_name(), "InputQueue2", bpt::seconds(60), bpt::seconds(5), "UTC", std::deque<std::string>{"up", "down", "left", "right"} );
    aci.input_queue_service.save(iq2);

    datamodel::Dataset_ptr ds2 =
            std::make_shared<datamodel::Dataset>(0, "Dataset2016-07-20-Below", user2Normal->get_user_name(), iq1, std::deque<datamodel::InputQueue_ptr>{},
                                                      datamodel::Priority::BelowNormal, "", common::C_default_residual_coef, 1, common::AppConfig::C_default_kernel_length, PROPS.get_steps(), 4, "sym7");
    ds2->set_is_active(true);

    aci.dataset_service.save(ds2);

    ////////////////////////////////////////////////////////////////////////////

    aci.dataset_service.update_active_datasets(pairs);

    EXPECT_TRUE(3UL == pairs.size());

    iter = pairs.begin();
    EXPECT_TRUE(iter->p_dataset->get_dataset_name() == "Dataset2016-07-20-Below"); EXPECT_TRUE(1UL == iter->users.size());EXPECT_TRUE(iter->users[0]->get_user_name() == "User2016-07-20-Normal");
    ++iter;
    EXPECT_TRUE(iter->p_dataset->get_dataset_name() == "eurusd"); EXPECT_TRUE(1UL == iter->users.size()); EXPECT_TRUE(iter->users[0]->get_user_name() == "svrwave");
    ++iter;
    EXPECT_TRUE(iter->p_dataset->get_dataset_name() == "Dataset2016-07-20-Low"); EXPECT_TRUE(1UL == iter->users.size()); EXPECT_TRUE(iter->users[0]->get_user_name() == "User2016-07-20-Low");

    ////////////////////////////////////////////////////////////////////////////

    ASSERT_FALSE( aci.dataset_service.unlink_user_from_dataset( user2Normal, ds1 ) );
    ASSERT_TRUE ( aci.dataset_service.link_user_to_dataset( user2Normal, ds1 ) );

    aci.dataset_service.update_active_datasets(pairs);

    EXPECT_TRUE(3UL == pairs.size());

    iter = pairs.begin();
    ASSERT_EQ(iter->p_dataset->get_dataset_name(), "Dataset2016-07-20-Below");
    ASSERT_EQ(1UL, iter->users.size());
    ASSERT_EQ(iter->users[0]->get_user_name(), "User2016-07-20-Normal");
    ++iter;
    ASSERT_EQ(iter->p_dataset->get_dataset_name(), "Dataset2016-07-20-Low");
    ASSERT_EQ(2UL, iter->users.size());
    ASSERT_EQ(iter->users[0]->get_user_name(), "User2016-07-20-Normal");
    ASSERT_EQ(iter->users[1]->get_user_name(), "User2016-07-20-Low");
    ++iter;
    ASSERT_EQ(iter->p_dataset->get_dataset_name(), "eurusd");
    ASSERT_EQ(1UL, iter->users.size());
    ASSERT_EQ(iter->users[0]->get_user_name(), "svrwave");

    ////////////////////////////////////////////////////////////////////////////

    datamodel::Dataset_ptr ds3 = std::make_shared<datamodel::Dataset>(0, "Dataset2016-07-20-High-3", user2Normal->get_user_name(), iq1, std::deque<datamodel::InputQueue_ptr>{}, datamodel::Priority::High, "", common::C_default_residual_coef, 1, common::AppConfig::C_default_kernel_length, PROPS.get_steps(), 4, "sym7");
    ds3->set_is_active(true);

    aci.dataset_service.save(ds3);

    aci.dataset_service.update_active_datasets(pairs);

    EXPECT_TRUE(4UL == pairs.size());

    iter = pairs.begin();
    ASSERT_EQ(iter->p_dataset->get_dataset_name(), "Dataset2016-07-20-High-3"); ASSERT_EQ(1UL, iter->users.size()); ASSERT_EQ(iter->users[0]->get_user_name(), "User2016-07-20-Normal");
    ++iter;
    ASSERT_EQ(iter->p_dataset->get_dataset_name(), "Dataset2016-07-20-Below"); ASSERT_EQ(1UL, iter->users.size());ASSERT_EQ(iter->users[0]->get_user_name(), "User2016-07-20-Normal");
    ++iter;
    ASSERT_EQ(iter->p_dataset->get_dataset_name(), "Dataset2016-07-20-Low"); ASSERT_EQ(2UL, iter->users.size()); ASSERT_EQ(iter->users[0]->get_user_name(), "User2016-07-20-Normal"); ASSERT_EQ(iter->users[1]->get_user_name(), "User2016-07-20-Low");
    ++iter;
    ASSERT_EQ(iter->p_dataset->get_dataset_name(), "eurusd"); ASSERT_EQ(1UL, iter->users.size());ASSERT_EQ(iter->users[0]->get_user_name(), "svrwave");

    ////////////////////////////////////////////////////////////////////////////

    ASSERT_TRUE( aci.dataset_service.unlink_user_from_dataset( user2Normal, ds1 ) );
    ASSERT_FALSE( aci.dataset_service.unlink_user_from_dataset( user2Normal, ds1 ) );

    aci.dataset_service.update_active_datasets(pairs);

    EXPECT_TRUE(4UL == pairs.size());

    iter = pairs.begin();
    ASSERT_EQ(iter->p_dataset->get_dataset_name(), "Dataset2016-07-20-High-3"); ASSERT_EQ(1UL, iter->users.size()); ASSERT_EQ(iter->users[0]->get_user_name(), "User2016-07-20-Normal");
    ++iter;
    ASSERT_EQ(iter->p_dataset->get_dataset_name(), "Dataset2016-07-20-Below"); ASSERT_EQ(1UL, iter->users.size());ASSERT_EQ(iter->users[0]->get_user_name(), "User2016-07-20-Normal");
    ++iter;
    ASSERT_EQ(iter->p_dataset->get_dataset_name(), "eurusd"); ASSERT_EQ(1UL, iter->users.size());ASSERT_EQ(iter->users[0]->get_user_name(), "svrwave");
    ++iter;
    ASSERT_EQ(iter->p_dataset->get_dataset_name(), "Dataset2016-07-20-Low"); ASSERT_EQ(1UL, iter->users.size()); ASSERT_EQ(iter->users[0]->get_user_name(), "User2016-07-20-Low");

    ////////////////////////////////////////////////////////////////////////////

    auto user3High = std::make_shared<datamodel::User>(
        bigint(), "User2016-07-20-High-3", "User2016-07-20-High-3@dkdk.dld", "User2016-07-20-High-3", "User2016-07-20-High-3", datamodel::ROLE::ADMIN, datamodel::Priority::High);

    aci.user_service.save(user3High);

    ASSERT_TRUE( aci.dataset_service.link_user_to_dataset( user3High, ds1 ) );
    ASSERT_TRUE( aci.dataset_service.link_user_to_dataset( user3High, ds2 ) );
    ASSERT_TRUE( aci.dataset_service.link_user_to_dataset( user3High, ds3 ) );

    aci.dataset_service.update_active_datasets(pairs);

    EXPECT_TRUE(4UL == pairs.size());

    iter = pairs.begin();
    ASSERT_EQ(iter->p_dataset->get_dataset_name(), "Dataset2016-07-20-High-3");
    ASSERT_EQ(2UL, iter->users.size());
    ASSERT_EQ(iter->users[0]->get_user_name(), "User2016-07-20-High-3");
    ASSERT_EQ(iter->users[1]->get_user_name(), "User2016-07-20-Normal");
    ++iter;
    ASSERT_EQ(iter->p_dataset->get_dataset_name(), "Dataset2016-07-20-Below");
    ASSERT_EQ(2UL, iter->users.size());
    ASSERT_EQ(iter->users[0]->get_user_name(), "User2016-07-20-High-3");
    ASSERT_EQ(iter->users[1]->get_user_name(), "User2016-07-20-Normal");
    ++iter;
    ASSERT_EQ(iter->p_dataset->get_dataset_name(), "Dataset2016-07-20-Low");
    ASSERT_EQ(2UL, iter->users.size());
    ASSERT_EQ(iter->users[0]->get_user_name(), "User2016-07-20-High-3");
    ASSERT_EQ(iter->users[1]->get_user_name(), "User2016-07-20-Low");
    ++iter;
    ASSERT_EQ(iter->p_dataset->get_dataset_name(), "eurusd");
    ASSERT_EQ(1UL, iter->users.size());
    ASSERT_EQ(iter->users[0]->get_user_name(), "svrwave");

    ////////////////////////////////////////////////////////////////////////////

    aci.user_service.remove(user3High);
    aci.dataset_service.remove(ds3);

    aci.dataset_service.remove(ds2);
    aci.input_queue_service.remove(iq2);
    aci.user_service.remove(user2Normal);


    aci.dataset_service.remove(ds1);
    aci.input_queue_service.remove(iq1);
    aci.user_service.remove(user1Low);
}

TEST_F(DaoTestFixture, DatasetIntegrationTest)
{
#ifndef INTEGRATION_TEST
    LOG4_FATAL("This test needs to be run on integration build.");
    exit(1);
#endif

    LOG4_BEGIN();

    // Save first N forecasts to database for later analysis
    constexpr uint16_t C_save_forecast = 115;
    constexpr auto C_online_validate = false;
    const uint32_t C_test_decrement = .5 * PROPS.get_kernel_length() + PROPS.get_shift_limit() + PROPS.get_outlier_slack(); // 14e3 - common::C_integration_test_validation_window;
#define MAIN_QUEUE_RES 43200
#define STR_MAIN_QUEUE_RES TOSTR(MAIN_QUEUE_RES)
    const bpt::seconds C_placement_delay(2);
    const auto C_test_labels_len_h = C_test_decrement + common::C_integration_test_validation_window;
    const std::string C_symbol = "xauusd_avg";
    const std::string C_test_input_name = "q_svrwave_test_" + C_symbol + "_";
    const std::string C_input_queue_name = "q_svrwave_" + C_symbol + "_";
    const std::string C_test_input_table_name(C_test_input_name + STR_MAIN_QUEUE_RES);
    const std::string C_test_aux_input_table_name(C_test_input_name + "1");
    constexpr uint16_t C_test_levels = 8; // Spectral levels
    constexpr auto C_test_gradient_count = common::C_default_gradient_count;
    constexpr auto C_overload_factor = 2; // Load surplus data from database in case rows discarded during preparation
    const auto C_decon_tail = datamodel::Dataset::get_residuals_length(C_test_levels);
    const uint32_t C_max_features_len = datamodel::C_default_svrparam_lag_count * PROPS.get_lag_multiplier() * business::ModelService::get_max_quantisation();
    const uint32_t C_test_data_len_h = C_overload_factor * (C_test_labels_len_h + cdiv(C_decon_tail + C_max_features_len, MAIN_QUEUE_RES));
    const auto C_test_data_len_h_str = std::to_string(C_test_data_len_h);
    constexpr uint32_t C_dataset_id = 0xDeadBeef;
    const std::string C_dataset_id_str(std::to_string(C_dataset_id));
    // constexpr char C_last_test_time[] = "2025-07-21 22:29:58";
    constexpr char C_last_test_time[] = "2025-01-24 22:29:58"; // Scott test

    // Prepare database for test
    try {
        const std::string query =
            "INSERT INTO input_queues SELECT '" + C_test_input_table_name + "', '" + C_symbol + "', user_name, description, resolution, legal_time_deviation, timezone, " \
            "value_columns, missing_hours_retention, uses_fix_connection FROM input_queues WHERE table_name = '" + C_input_queue_name + STR_MAIN_QUEUE_RES
            "' and not exists (select 1 from input_queues where table_name = '" + C_test_aux_input_table_name + "');" +

            "INSERT INTO input_queues SELECT '" + C_test_aux_input_table_name + "', '" + C_symbol + "', user_name, description, resolution, legal_time_deviation, timezone, " \
            "value_columns, missing_hours_retention, uses_fix_connection FROM input_queues WHERE table_name = '" + C_input_queue_name + "1" \
            "' and not exists (select 1 from input_queues where table_name = '" + C_test_aux_input_table_name + "');" +

            "DROP VIEW IF EXISTS " + C_test_aux_input_table_name + ";" \

            "CREATE VIEW " + C_test_aux_input_table_name + " AS SELECT * FROM q_svrwave_xauusd_avg_1 "
                                            "WHERE value_time < '" + C_last_test_time + "' ORDER BY value_time DESC LIMIT " + C_test_data_len_h_str + " * " STR_MAIN_QUEUE_RES "; " \

            "DROP VIEW IF EXISTS " + C_test_input_table_name + ";" \

            "CREATE VIEW " + C_test_input_table_name + " AS SELECT * FROM q_svrwave_xauusd_avg_" STR_MAIN_QUEUE_RES \
            " WHERE value_time < '" + C_last_test_time + "' ORDER BY value_time DESC LIMIT " + C_test_data_len_h_str + ";" \

            "DELETE FROM w_scaling_factors WHERE dataset_id = " + C_dataset_id_str + ";" \

            "DELETE FROM iq_scaling_factors WHERE dataset_id = " + C_dataset_id_str + ";" \

            "DELETE FROM dq_scaling_factors WHERE model_id IN (SELECT id FROM models WHERE ensemble_id IN (SELECT id FROM ensembles WHERE dataset_id = " + C_dataset_id_str + ")) ;" \

            "DELETE FROM svr_parameters WHERE dataset_id = " + C_dataset_id_str;

        auto ds = std::make_from_tuple<dao::DataSource>(PROPS.get_connection_arguments());
#ifdef USE_DUCKDB
        if (PROPS.is_duck()) {
            const auto trx = ds.open_file();
            auto res = trx->exec(query);
            duckdb_destroy_result(&res);
        } else
#endif
            const auto trx = ds.open_transaction()->exec(query);
    } catch (const std::exception &ex) {
        LOG4_ERROR("Error " << ex.what() << " while preparing test queue.");
        return;
    }

    // Initialize dataset
    auto p_dataset = ptr<datamodel::Dataset>(
        C_dataset_id, "test_dataset", "test_user", C_test_input_table_name, std::deque{C_test_aux_input_table_name}, datamodel::Priority::Normal, "",
        common::C_default_residual_coef, C_test_gradient_count, PROPS.get_kernel_length(), PROPS.get_steps(), C_test_levels, "cvmd", common::C_default_features_max_time_gap);
    business::EnsembleService::init_ensembles(p_dataset, false);
    const auto nl = business::EnsembleService::get_levels_limit(p_dataset->get_spectral_levels());
    // const auto n_threads = std::min<uint16_t>(PROPS.get_parallel_models(), nl * p_dataset->get_steps());
    // OMP_PAR(n_threads)
    {
        // OMP_TASKLOOP_1()
        for (const auto &p_ensemble: p_dataset->get_ensembles()) {
            const auto &column = p_ensemble->get_column_name();
            // OMP_TASKLOOP_1(collapse(2))
            for (uint16_t l = 0; l < nl; l += LEVEL_STEP)
                for (uint16_t s = 0; s < p_dataset->get_steps(); ++s) {
                    if (l == p_dataset->get_trans_levix()) continue;
                    auto p_model = p_ensemble->get_model(l, s);
                    if (!p_model) LOG4_THROW("Model not found!");

                    const auto p_head_params = p_model->get_head_params();
                    p_head_params.first->set_svr_decremental_distance(C_test_decrement);
                    p_head_params.second->set_svr_decremental_distance(C_test_decrement);
                }
        }
        const auto dataset_train_data = business::DatasetService::process(*p_dataset);
        // OMP_TASKLOOP_1() // To preserve order of processing, do not parallelize
        for (const auto &p_ensemble: p_dataset->get_ensembles()) {
            const auto &column = p_ensemble->get_column_name();
            // const bool is_ask = column.find("_ask") != std::string::npos;
            datamodel::data_row_container times;
            arma::mat recon_predicted(common::C_integration_test_validation_window, p_dataset->get_steps(), arma::fill::zeros),
                recon_predicted_lgbm(common::C_integration_test_validation_window, p_dataset->get_steps(), arma::fill::zeros),
                recon_actual(common::C_integration_test_validation_window, p_dataset->get_steps(), arma::fill::zeros);
            arma::vec recon_last_knowns(common::C_integration_test_validation_window, arma::fill::zeros);
            tbb::mutex recon_mx;
            // OMP_TASKLOOP_1(collapse(2))
            for (uint16_t l = 0; l < nl; l += LEVEL_STEP)
                for (uint16_t s = 0; s < p_dataset->get_steps(); ++s) {
                    if (l == p_dataset->get_trans_levix()) continue;
                    auto p_model = p_ensemble->get_model(l, s);
                    if (!p_model) LOG4_THROW("Model not found!");

                    const auto p_head_params = p_model->get_head_params();
                    LOG4_DEBUG("Preparing model " << *p_model << " parameters " << *p_head_params.first << ", integration test validation_window " << common::C_integration_test_validation_window);
                    const auto [p_model_features, p_model_labels, p_model_last_knowns, p_weights, p_model_times] = dataset_train_data.at(column).at({l, s});
                    assert(p_model_labels->n_rows == C_test_labels_len_h);
                    assert(p_model_times->size() == C_test_labels_len_h);
                    LOG4_DEBUG("All features size " << arma::size(*p_model_features) << ", test length " << C_test_labels_len_h);

                    const auto [predict_mae_level, predict_mape_level, predicted, predicted_lgbm, actual, mape_lk, last_knowns] =
                        business::ModelService::validate(
                            p_model_labels->n_rows - common::C_integration_test_validation_window, *p_dataset, *p_ensemble, *p_model,
                            *p_model_features, *p_model_labels, *p_model_last_knowns, *p_weights, *p_model_times, C_online_validate,
                            p_dataset->get_spectral_levels() < MIN_LEVEL_COUNT);
                    const tbb::mutex::scoped_lock lk(recon_mx);
                    if (times.empty()) times = *p_model_times;
                    recon_predicted.col(s) += predicted;
                    recon_predicted_lgbm.col(s) += predicted_lgbm;
                    recon_actual.col(s) += actual;
                    if (!s) recon_last_knowns += last_knowns;
                }

            recon_predicted = arma::mean(recon_predicted, 1);
            recon_predicted_lgbm = arma::mean(recon_predicted_lgbm, 1);
            recon_actual = arma::mean(recon_actual, 1);

            const auto p_iqsf = p_dataset->get_iq_scaling_factor(p_ensemble->get_aux_decon_queue(column)->get_input_queue_table_name(), column);
            LOG4_TRACE("Input scaling factor " << *p_iqsf);
            business::IQScalingFactorService::unscale_I(*p_iqsf, recon_predicted);
            business::IQScalingFactorService::unscale_I(*p_iqsf, recon_predicted_lgbm);
            business::IQScalingFactorService::unscale_I(*p_iqsf, recon_last_knowns);
            business::IQScalingFactorService::unscale_I(*p_iqsf, recon_actual);
            LOG4_INFO("Total predicted to actual difference " << common::present<double>(recon_actual - recon_predicted) <<
                      ", predicted LGBM to actual difference " << common::present<double>(recon_actual - recon_predicted) <<
                      ", last known to actual difference " << common::present<double>(recon_actual - recon_last_knowns));

            double mae = 0, mae_lgbm = 0, mae_lk = 0, recon_mae = 0, recon_lk_mae = 0, pips_won = 0, pips_lost = 0, drawdown = 0, max_drawdown = 0, pips_won_lgbm = 0, pips_lost_lgbm = 0, drawdown_lgbm = 0, max_drawdown_lgbm = 0;
            uint16_t positive_mae_ct = 0, positive_mae_lgbm_ct = 0, pos_direct = 0, pos_direct_lgbm = 0, price_hits = 0, price_hits_lgbm = 0;
            const auto validated_ct = recon_actual.size();
            const auto resolution = p_dataset->get_input_queue()->get_resolution();
            const auto horizon_duration = resolution * PROPS.get_prediction_horizon();
            const auto validate_start = times.size() - validated_ct;
            const auto column_ix = p_dataset->get_input_queue()->get_value_column_index(column);
            for (uint16_t i = 0; i < validated_ct; ++i) {
                const auto i_div = i + 1.;
                const auto cur_time = times[validate_start + i]->get_value_time();
                const auto actual_it = business::lower_bound(*p_dataset->get_input_queue(), cur_time);
                if (actual_it == p_dataset->get_input_queue()->cend()) {
                    LOG4_ERROR("No actual value found for time " << cur_time << ", skipping.");
                    continue;
                }
                const auto actual = (**actual_it)[column_ix];
                const auto last_known_iter = business::lower_bound_before(std::as_const(*p_dataset->get_aux_input_queue()), cur_time - horizon_duration);
                const auto last_known = (**last_known_iter)[column_ix];
                const auto actual_move = actual - last_known;
                const auto recon_actual_move = recon_actual[i] - recon_last_knowns[i];
                const auto predicted_move = recon_predicted[i] - recon_last_knowns[i];
                const auto predicted_move_lgbm = recon_predicted_lgbm[i] - recon_last_knowns[i];
                const auto cur_mae = std::abs(recon_predicted[i] - recon_actual[i]);
                const auto cur_mae_lgbm = std::abs(recon_predicted_lgbm[i] - recon_actual[i]);
                const auto cur_mae_lk = std::abs(recon_actual_move);
                const auto cur_alpha_pct = common::alpha(cur_mae_lk, cur_mae);
                const auto cur_alpha_lgbm_pct = common::alpha(cur_mae_lk, cur_mae_lgbm);
                mae += cur_mae;
                mae_lgbm += cur_mae_lgbm;
                mae_lk += cur_mae_lk;
                const auto cur_recon_diff = recon_actual[i] - actual;
                const auto cur_recon_error = std::abs(cur_recon_diff);
                const auto cur_recon_lk_error = std::abs(recon_last_knowns[i] - last_known);
                const auto cml_alpha_pct = common::alpha(mae_lk, mae);
                const auto cml_alpha_lgbm_pct = common::alpha(mae_lk, mae_lgbm);
                recon_mae += cur_recon_error;
                recon_lk_mae += cur_recon_lk_error;

                if (mae < mae_lk) LOG4_DEBUG("Positive cumulative alpha at " << i << ", " << cml_alpha_pct << "pc");
                if (mae_lgbm < mae_lk) LOG4_DEBUG("Positive cumulative LGBM alpha at " << i << ", " << cml_alpha_lgbm_pct << "pc");
                if (cur_mae < cur_mae_lk) {
                    LOG4_DEBUG("Positive alpha " << cur_alpha_pct << "pc, at " << i);
                    ++positive_mae_ct;
                }
                if (cur_mae_lgbm < cur_mae_lk) {
                    LOG4_DEBUG("Positive LGBM alpha " << cur_alpha_lgbm_pct << "pc, at " << i);
                    ++positive_mae_lgbm_ct;
                }
                const auto start_aux_it = business::lower_bound(last_known_iter, p_dataset->get_aux_input_queue()->cend(), cur_time);
                const auto last_aux_it = business::lower_bound(start_aux_it, p_dataset->get_aux_input_queue()->cend(), cur_time + resolution);
                const auto placement_it = business::lower_bound(last_known_iter, p_dataset->get_aux_input_queue()->cend(), cur_time - horizon_duration + C_placement_delay);
                const auto placement_price = ***placement_it;
                const auto last_aux_price = ***std::prev(last_aux_it);
                constexpr auto time_comp = [](const auto &lhs, const auto &rhs) { return lhs->get_value_time() < rhs->get_value_time(); };

                const auto is_bear = std::signbit(predicted_move);
                double this_drawdown;
                const auto [min_it, max_it] = std::minmax_element(placement_it /* start_aux_it */, last_aux_it, time_comp);
                const auto min_price = ***min_it;
                const auto max_price = ***max_it;
                if (is_bear /* && !is_ask */ ) { // Sell signal
                    if (recon_predicted[i] <= placement_price && recon_predicted[i] >= min_price) {
                        ++price_hits;
                        pips_won += placement_price - recon_predicted[i];
                    } else if (last_aux_price < placement_price) {
                        pips_won += placement_price - last_aux_price;
                    } else
                        pips_lost += last_aux_price - placement_price;
                    this_drawdown = std::max(0., max_price - placement_price);
                    LOG4_TRACE("Sell min price " << min_price << ", max price " << max_price << ", placement price " << placement_price);
                } else if (!is_bear/* && is_ask */) { // Buy signal
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
                if (is_bear == std::signbit(recon_actual_move)) {
                    LOG4_DEBUG("Direction correct at " << i);
                    ++pos_direct;
                }

                const auto is_bear_lgbm = std::signbit(predicted_move_lgbm);
                double this_drawdown_lgbm;
                if (is_bear_lgbm /* && !is_ask */ ) { // Sell signal
                    if (recon_predicted_lgbm[i] <= placement_price && recon_predicted_lgbm[i] >= min_price) {
                        ++price_hits_lgbm;
                        pips_won_lgbm += placement_price - recon_predicted_lgbm[i];
                    } else if (last_aux_price < placement_price) {
                        pips_won_lgbm += placement_price - last_aux_price;
                    } else
                        pips_lost_lgbm += last_aux_price - placement_price;
                    this_drawdown_lgbm = std::max(0., max_price - placement_price);
                    LOG4_TRACE("Sell LGBM min price " << min_price << ", max price " << max_price << ", placement price " << placement_price);
                } else if (!is_bear_lgbm/* && is_ask */) { // Buy signal
                    if (recon_predicted_lgbm[i] >= placement_price && recon_predicted_lgbm[i] <= max_price) {
                        ++price_hits_lgbm;
                        pips_won_lgbm += recon_predicted_lgbm[i] - placement_price;
                    } else if (last_aux_price > placement_price) {
                        pips_won_lgbm += last_aux_price - placement_price;
                    } else
                        pips_lost_lgbm += placement_price - last_aux_price;
                    this_drawdown_lgbm = std::max(0., placement_price - min_price);
                    LOG4_TRACE("Buy LGBM min price " << min_price << ", max price " << max_price << ", placement price " << placement_price);
                }
                if (is_bear_lgbm == std::signbit(recon_actual_move)) {
                    LOG4_DEBUG("Direction correct at " << i);
                    ++pos_direct_lgbm;
                }
                if (common::above_eps(cur_recon_error) || common::above_eps(cur_recon_lk_error))
                    LOG4_WARN("Reconstruction difference at " << cur_time << " between actual " << actual << " and recon price " << \
                              recon_actual[i] << " is " << cur_recon_diff << ", last-known price " << last_known << ", recon last-known " << recon_last_knowns[i] << \
                              ", last known difference " << last_known - recon_last_knowns[i]);
                drawdown += this_drawdown;
                drawdown_lgbm += this_drawdown_lgbm;
                MAXAS(max_drawdown, this_drawdown);
                const auto net_pips = pips_won - pips_lost;
                const auto net_pips_lgbm = pips_won_lgbm - pips_lost_lgbm;
                const auto pips_pos = net_pips / i_div;
                const auto pips_pos_lgbm = net_pips_lgbm / i_div;
                const auto drawdown_pos = drawdown / i_div;
                const auto drawdown_pos_lgbm = drawdown_lgbm / i_div;
                const auto leverage = drawdown_pos > 0 ? std::max(0., net_pips / drawdown) : pips_pos;
                const auto leverage_lgbm = drawdown_pos_lgbm > 0 ? std::max(0., net_pips_lgbm / drawdown_lgbm) : pips_pos_lgbm;
                const auto abs_leverage = max_drawdown > 0 ? std::max(0., net_pips / max_drawdown) : net_pips;
                const auto abs_leverage_lgbm = max_drawdown > 0 ? std::max(0., net_pips_lgbm / max_drawdown_lgbm) : net_pips_lgbm;
                const auto positive_preds_pc = 100. * positive_mae_ct / i_div;
                const auto positive_preds_lgbm_pc = 100. * positive_mae_lgbm_ct / i_div;
                LOG4_INFO("Position " << i << ", column " << column << " " << column_ix << \
                          ", price time " << cur_time << \
                          ", actual price " << actual << \
                          ", recon actual price " << recon_actual[i] << \
                          ", predicted price " << recon_predicted[i] << \
                          ", predicted price LGBM " << recon_predicted_lgbm[i] << \
                          ", last-known time " << (**last_known_iter).get_value_time() << \
                          ", last known " << last_known << \
                          ", recon last known " << recon_last_knowns[i] << \
                          ", total MAE " << mae / i_div << \
                          ", total MAE LGBM " << mae_lgbm / i_div << \
                          ", total MAE last known " << mae_lk / i_div << \
                          ", positive directions " << 100. * pos_direct / i_div << "pc" \
                          ", positive errors " << positive_preds_pc << "pc" \
                          ", positive directions LGBM " << 100. * pos_direct_lgbm / i_div << "pc" \
                          ", positive errors LGBM " << positive_preds_lgbm_pc << "pc" \
                          ", current MAE " << cur_mae << \
                          ", current MAE LGBM " << cur_mae_lgbm << \
                          ", current MAE last known " << cur_mae_lk << \
                          ", predicted movement " << predicted_move << \
                          ", predicted movement LGBM " << predicted_move_lgbm << \
                          ", actual movement " << actual_move << \
                          ", recon actual movement " << recon_actual_move << \
                          ", current alpha " << cur_alpha_pct << "pc" \
                          ", cumulative alpha " << cml_alpha_pct << "pc" \
                          ", current alpha LGBM " << cur_alpha_lgbm_pct << "pc" \
                          ", cumulative alpha LGBM " << cml_alpha_lgbm_pct << "pc" \
                          ", recon error " << cur_recon_error << \
                          ", recon error last-known " << cur_recon_lk_error << \
                          ", recon label MAE " << recon_mae / i_div << \
                          ", recon last-known MAE " << recon_lk_mae / i_div << \
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
                          ", trade rating " << abs_leverage * positive_preds_pc * cml_alpha_pct << \
                          ", price hits LGBM " << 100. * price_hits_lgbm / i_div << "pc" \
                          ", won LGBM " << pips_won_lgbm << \
                          ", lost LGBM " << pips_lost_lgbm << \
                          ", neto LGBM " << net_pips_lgbm << \
                          ", value per position LGBM " << pips_pos_lgbm << \
                          ", drawdown per position LGBM " << drawdown_pos_lgbm << \
                          ", sum drawdown LGBM " << drawdown_lgbm << \
                          ", max drawdown LGBM " << max_drawdown_lgbm << \
                          ", mean leverage LGBM  " << leverage_lgbm << /* net won to average drawdown ratio */ \
                          ", absolute leverage LGBM " << abs_leverage_lgbm << /* net won to maximum drawdown ratio */ \
                          ", trade rating LGBM " << abs_leverage_lgbm * positive_preds_lgbm_pc * cml_alpha_lgbm_pct);
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
