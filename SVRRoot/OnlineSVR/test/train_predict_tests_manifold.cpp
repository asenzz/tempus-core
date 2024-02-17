//
// Created by zarko on 21/05/19.
//
#include <gtest/gtest.h>
#include "ModelService.hpp"
#include "appcontext.hpp"
#include "DQScalingFactorService.hpp"
#include "SVRParametersService.hpp"
#include "test_harness.hpp"
#include "common/compatibility.hpp"
#include "onlinesvr.hpp"


using namespace svr;

// #define TEST_MANIFOLD
#define TEST_ACTUAL_DATA
#define TEST_VALIDATION_WINDOW MANIFOLD_TEST_VALIDATION_WINDOW
#define TEST_DECREMENT 4300
#define TEST_LAG (DEFAULT_SVRPARAM_LAG_COUNT)
#define TEST_LENGTH (TEST_DECREMENT + EMO_TUNE_TEST_SIZE + TEST_VALIDATION_WINDOW)
#define TEST_DECON_QUEUE_TABLE_NAME "z_q_svrwave_xauusd_avg_3600_100_xauusd_avg_bid"
#define TEST_INPUT_TABLE_NAME "q_svrwave_xauusd_avg_3600"
#define TEST_DECON_INPUT_COLUMN_NAME "xauusd_avg_bid"
#define TEST_DECON_AUX_QUEUE_TABLE_NAME "z_q_svrwave_xauusd_avg_1_100_xauusd_avg_bid"
#define TEST_AUX_INPUT_TABLE_NAME "q_svrwave_xauusd_avg_1"
#define TEST_DECON_AUX_INPUT_COLUMN_NAME "xauusd_avg_bid" // Aux column names should have the same name as the corresponding main column
#define MAIN_INPUT_DATA "/mnt/faststore/repo/tempus-core/SVRRoot/OnlineSVR/test/test_data/xauusd_3600_2023.csv"
#define AUX_INPUT_DATA "/mnt/faststore/repo/tempus-core/SVRRoot/OnlineSVR/test/test_data/xauusd_1_2023.csv"
#define TEST_LIMIT std::numeric_limits<int>::max()

static void load(std::shared_ptr<datamodel::DeconQueue> &p_decon, datamodel::DeconQueue_ptr &p_decon_aux)
{
    std::string csv_line, csv_line_aux;
    std::ifstream input_f(MAIN_INPUT_DATA), input_f_aux(AUX_INPUT_DATA);
    EXPECT_TRUE(input_f.good());
    while (std::getline(input_f, csv_line))
        p_decon->get_data().emplace_back(datamodel::DataRow::load(csv_line));
    EXPECT_TRUE(input_f_aux.good());
    size_t ct = 0;
    while (std::getline(input_f_aux, csv_line_aux) && ct++ < TEST_LIMIT)
        p_decon_aux->get_data().emplace_back(datamodel::DataRow::load(csv_line_aux));
}

TEST(manifold_tune_train_predict, basic_integration)
{
    omp_set_nested(true);
    omp_set_max_active_levels(1000);
    svr::context::AppContext::init_instance("../config/app.config");
    auto p_dataset = std::make_shared<datamodel::Dataset>(100, "test_dataset", "test_user", TEST_INPUT_TABLE_NAME, std::deque{std::string(TEST_AUX_INPUT_TABLE_NAME)});
    business::EnsembleService::init_default_ensembles(p_dataset);
#ifdef TEST_ACTUAL_DATA
    datamodel::InputQueue_ptr p_input, p_input_aux;
    datamodel::DeconQueue_ptr p_decon, p_decon_aux;
#pragma omp parallel num_threads(2)
#pragma omp single
    {
        p_input = p_dataset->get_input_queue();
        p_input_aux = p_dataset->get_aux_input_queue();
        APP.input_queue_service.load_latest(p_input, TEST_LENGTH + 10);
        APP.input_queue_service.load_latest(p_input_aux, p_input->size() * 3600 + TEST_LAG * QUANTIZE_FIXED, p_input->back()->get_value_time());
#pragma omp taskwait
#pragma omp task
        p_decon = APP.decon_queue_service.deconstruct(p_dataset, p_input, TEST_DECON_INPUT_COLUMN_NAME);
        p_decon_aux = APP.decon_queue_service.deconstruct(p_dataset, p_input_aux, TEST_DECON_AUX_INPUT_COLUMN_NAME);
    }

#else
    auto p_decon = std::make_shared<datamodel::DeconQueue>(TEST_DECON_QUEUE_TABLE_NAME, TEST_DECON_INPUT_TABLE_NAME, TEST_DECON_INPUT_COLUMN_NAME, 100, 1);
    auto p_decon_aux = std::make_shared<datamodel::DeconQueue>(TEST_DECON_AUX_QUEUE_TABLE_NAME, TEST_DECON_AUX_INPUT_TABLE_NAME, TEST_DECON_AUX_INPUT_COLUMN_NAME, 100, 1);
#endif
    auto p_head_params = std::make_shared<datamodel::SVRParameters>(0, 100, TEST_INPUT_TABLE_NAME, TEST_DECON_INPUT_COLUMN_NAME, 0, 0, 0, svr::mimo_type_e::single);
    p_head_params->set_svr_decremental_distance(TEST_DECREMENT);
    p_head_params->set_lag_count(TEST_LAG);
#ifdef TEST_MANIFOLD
    p_head_params->set_kernel_type(kernel_type_e::DEEP_PATH);
#endif
#ifdef TEST_ACTUAL_DATA
    auto p_all_features = std::make_shared<arma::mat>();
    auto p_all_labels = std::make_shared<arma::mat>();
    auto p_all_last_knowns = std::make_shared<arma::mat>();
    std::deque<bpt::ptime> times;
    business::ModelService::get_training_data(*p_all_features, *p_all_labels, *p_all_last_knowns, times,
                                              {svr::business::EnsembleService::get_start( // Main labels
                                                      p_decon->get_data(),
                                                      p_head_params->get_svr_decremental_distance() + EMO_TUNE_TEST_SIZE + TEST_VALIDATION_WINDOW,
                                                      bpt::min_date_time,
                                                      bpt::hours(1)),
                                               p_decon->end(),
                                               p_decon->get_data()},
                                              *p_decon_aux,
                                              {datamodel::datarow_range{*p_decon_aux}},
                                              p_head_params->get_lag_count(),
                                              {0},
                                              bpt::hours(92),
                                              0,
                                              bpt::seconds(1),
                                              bpt::min_date_time,
                                              bpt::hours(1),
                                              p_dataset->get_multiout()
    );
#else
    auto p_all_features =       std::make_shared<arma::mat>(TEST_DECREMENT + TEST_VALIDATION_WINDOW + EMO_TUNE_TEST_SIZE, TEST_LAG, arma::fill::randn);
    auto p_all_labels =         std::make_shared<arma::mat>(TEST_DECREMENT + TEST_VALIDATION_WINDOW + EMO_TUNE_TEST_SIZE, 1, arma::fill::randn);
    auto p_all_last_knowns =    std::make_shared<arma::mat>(TEST_DECREMENT + TEST_VALIDATION_WINDOW + EMO_TUNE_TEST_SIZE, 1, arma::fill::randn);
    std::deque<bpt::ptime> times(TEST_DECREMENT + TEST_VALIDATION_WINDOW + EMO_TUNE_TEST_SIZE, bpt::second_clock::local_time());
#endif
    const std::set<size_t> feat_levels = common::get_adjacent_indexes(
            p_head_params->get_decon_level(), p_head_params->get_svr_adjacent_levels_ratio(), p_dataset->get_transformation_levels());
    auto missing_factors = business::DQScalingFactorService::check({p_decon_aux}, p_head_params, feat_levels, p_dataset->get_dq_scaling_factors());
    business::DQScalingFactorService::scale(p_dataset, {p_decon_aux}, p_head_params, missing_factors, *p_all_features, *p_all_labels, *p_all_last_knowns);
    t_gradient_tuned_parameters tune_results;
    LOG4_DEBUG("All features size " << arma::size(*p_all_features) << ", test length " << TEST_LENGTH);
    PROFILE_EXEC_TIME(
            OnlineMIMOSVR::tune(tune_results, datamodel::t_param_set{p_head_params},
                                p_all_features->rows(p_all_features->n_rows - TEST_LENGTH, p_all_features->n_rows - TEST_VALIDATION_WINDOW),
                                p_all_labels->rows(p_all_labels->n_rows - TEST_LENGTH, p_all_labels->n_rows - TEST_VALIDATION_WINDOW),
                                p_all_last_knowns->rows(p_all_last_knowns->n_rows - TEST_LENGTH, p_all_last_knowns->n_rows - TEST_VALIDATION_WINDOW),
                                p_dataset->get_chunk_size()), "Tune parameters");
    OnlineMIMOSVR model(
#ifdef TEST_MANIFOLD
            std::make_shared<t_param_set>(t_param_set{p_head_params}),
#else
            business::SVRParametersService::get_best_params(tune_results),
#endif
            std::make_shared<arma::mat>(p_all_features->rows(p_all_features->n_rows - TEST_LENGTH, p_all_features->n_rows - TEST_VALIDATION_WINDOW - 1)),
            std::make_shared<arma::mat>(p_all_labels->rows(p_all_labels->n_rows - TEST_LENGTH, p_all_labels->n_rows - TEST_VALIDATION_WINDOW - 1)));
    const auto scale_label = p_dataset->get_dq_scaling_factor(p_decon_aux->get_input_queue_table_name(), p_decon_aux->get_input_queue_column_name(), 0);
    (void) business::ModelService::future_validate(
            p_all_labels->n_rows - TEST_VALIDATION_WINDOW, model, *p_all_features, *p_all_labels, *p_all_last_knowns,
            times, p_dataset->get_dq_scaling_factors(), p_decon_aux->get_input_queue_column_name(), false);
}
