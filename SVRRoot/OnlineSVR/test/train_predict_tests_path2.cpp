//
// Created by zarko on 21/05/19.
//
#include <gtest/gtest.h>
#include <ModelService.hpp>
#include <optimizer.hpp>
#include <appcontext.hpp>
#include "kernel_basic_integration_test.hpp"

#define DO_TEST_PSO


#define VALIDATION_WINDOW 500
#define DEFAULT_LAG 650
#define DEFAULT_SVR_DECREMENT 20000
#define DEFAULT_ADJACENT 1
#define DEFAULT_KERNEL kernel_type_e::PATH
#define INPUT_LIMIT (DEFAULT_SVR_DECREMENT + DEFAULT_LAG + VALIDATION_WINDOW)

static const size_t lag = DEFAULT_LAG;

#if 0
static const std::pair<arma::mat, arma::mat>
generate_features_labels(const arma::mat &data)
{
    arma::mat features(data.n_rows - lag, lag);
    arma::mat labels(data.n_rows - lag, 1);
    /* cilk_ */ for (size_t i = lag; i < data.n_rows; ++i) {
        const arma::rowvec row_features = data.rows(i - lag, i - 1).t();
        features.row(i - lag) = row_features;
        labels.row(i - lag) = data.row(i);
    }
    //LOG4_DEBUG("Generated " << arma::size(features) << " features, first row " << features.row(0));
    return {features, labels};
}
#endif

static const svr::matrix_ptr load_file(const std::string &file_name)
{
    svr::matrix_ptr res_arma;
    res_arma->load(file_name);
    return res_arma;
}

static std::vector<double>
run_file(
        const std::string &level_file,
        std::mutex &printout_mutex, std::atomic<size_t> &running_ct)
{
    const svr::matrix_ptr p_features = load_file(svr::common::formatter() << "/mnt/slowstore/var/tmp/features_" << svr::C_logged_level << "_" << 0 << ".out");
    const svr::matrix_ptr p_labels = load_file(svr::common::formatter() << "/mnt/slowstore/var/tmp/labels_" << svr::C_logged_level << "_" << 0 << ".out");
    auto best_parameters = std::make_shared<SVRParameters>(
            0, 0, "test path 2", "test path 2", 0, 0, 0, svr::mimo_type_e::single, 0, 0, 1, 1, DEFAULT_SVR_DECREMENT, DEFAULT_ADJACENT, DEFAULT_KERNEL,
            DEFAULT_LAG);
    {
#ifdef DO_TEST_PSO
//        svr::OnlineMIMOSVR::tune_kernel_params(best_parameters, *p_features, *p_labels);
#else
        best_parameters.set_svr_kernel_param(0.46733608245812541);
        best_parameters.set_svr_kernel_param2(1.2209334891491921);
#endif
    }
#if 0
    std::atomic<double> best_mae = std::numeric_limits<double>::max();
    double best_mape = 0;
    double best_lin_mape = 0;
    const auto start_parameters = best_parameters;
    std::mutex printout_mutex;
    //for (const auto &err_args: errors_args) {
    {
        SVRParameters kernel_svr_parameters(start_parameters);
        LOG4_DEBUG("Kernel gamma " << start_parameters.get_svr_kernel_param() << ", lambda " << start_parameters.get_svr_kernel_param2());
        std::vector<matrices_ptr> kernel_matrices(PROPS.get_slide_count());
        /* cilk_ */ for (size_t slide_ix = 0; slide_ix < PROPS.get_slide_count(); ++slide_ix) {
            const auto slide_start_pos = kernel_svr_parameters.get_svr_decremental_distance() + slide_ix * PROPS.get_future_predict_count() / PROPS.get_slide_count();
            const auto slide_features = p_features_data->rows(slide_start_pos - kernel_svr_parameters.get_svr_decremental_distance(), slide_start_pos - 1);
            // PROFILE_EXEC_TIME(kernel_matrices[slide_ix] = svr::OnlineMIMOSVR::produce_kernel_matrices(kernel_svr_parameters, slide_features), "Produce kernel matrices");
        }
        /* cilk_ */ for (size_t cost_exp = MIN_COST_EXP; cost_exp < MAX_COST_EXP; ++cost_exp) {
            const double cost = std::pow(10, 4 * cost_exp);
            auto cost_svr_parameters = kernel_svr_parameters;
            cost_svr_parameters.set_svr_C(cost);
            cycle_returns_t cycle_returns;
            PROFILE_EXEC_TIME(cycle_returns = train_predict_cycle(cost_svr_parameters.get_svr_decremental_distance(), p_labels_data, p_features_data, cost_svr_parameters, kernel_matrices), "Train predict cycle");
            const auto mae = std::get<0>(cycle_returns);
            const auto mape = std::get<1>(cycle_returns);
            const auto lin_mape = std::get<4>(cycle_returns);
            std::scoped_lock printout_guard(printout_mutex);
            if (best_mae > mae) {
                best_mae = mae;
                best_parameters = cost_svr_parameters;
                best_mape = mape;
                best_lin_mape = lin_mape;
                LOG4_INFO("Best cost found MAE: " << best_mae << " Parameters: " << best_parameters.to_sql_string() << ", MAPE " << mape << ", Lin MAPE " << lin_mape);
            }
        }
    }
#endif
 //   svr::business::ModelService::final_cycle(best_parameters, best_parameters->get_svr_decremental_distance(), p_labels, p_features);

    return {};
}

TEST(path_tune_train_predict2, basic_integration)
{
    svr::IKernel<double>::IKernelInit();
    std::mutex printout_mutex;
    std::atomic<size_t> running_ct = 0;
    std::vector<std::vector<double>> predicted_mx(file_name_pairs.size());
    std::vector<double> predicted_prices;
#if 0 // All levels?
    for (size_t level_ix = 0; level_ix < file_name_pairs.size(); ++level_ix) {
        const auto ff = file_name_pairs[level_ix];
        while (running_ct.load() >= RUN_LIMIT) sleep(1);
        //cilk_spawn
        predicted_mx[level_ix] = run_file(ff, printout_mutex, running_ct);
    }
#else
    // run_file("../SVRRoot/SVRBusiness-tests/test/test_data/input_queue_eurusd_4h_avg.csv", printout_mutex, running_ct);
#endif
}
