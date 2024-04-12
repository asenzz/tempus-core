//
// Created by zarko on 21/05/19.
//
#include <gtest/gtest.h>
#include <ModelService.hpp>
#include <optimizer.hpp>
#include <appcontext.hpp>
#include "kernel_basic_integration_test_smo.hpp"

#define DO_TEST_PSO


#define C_test_validation_window 500
#define DEFAULT_LAG 650
#define DEFAULT_SVR_DECREMENT 20000
#define DEFAULT_ADJACENT 1
#define DEFAULT_KERNEL datamodel::kernel_type_e::PATH
#define INPUT_LIMIT (DEFAULT_SVR_DECREMENT + DEFAULT_LAG + TEST_VALIDATION_WINDOW)

using namespace svr;

#if 0

constexpr size_t lag = DEFAULT_LAG;

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

static const svr::mat_ptr load_file(const std::string &file_name)
{
    svr::mat_ptr res_arma;
    res_arma->load(file_name);
    return res_arma;
}

static std::vector<double>
run_file(
        const std::string &level_file,
        std::mutex &printout_mutex, std::atomic<size_t> &running_ct)
{
    const svr::mat_ptr p_features = load_file(svr::common::formatter() << "/mnt/slowstore/var/tmp/features_" << 0 << "_" << 0 << ".out");
    const svr::mat_ptr p_labels = load_file(svr::common::formatter() << "/mnt/slowstore/var/tmp/labels_" << 0 << "_" << 0 << ".out");
    auto best_parameters = std::make_shared<datamodel::SVRParameters>(
            0, 0, "test path 2", "test path 2", 1, 0, 0, 0, 0, 0, 1, 1, DEFAULT_SVR_DECREMENT, DEFAULT_ADJACENT, DEFAULT_KERNEL,
            DEFAULT_LAG);
    {
#ifdef DO_TEST_PSO
//        svr::OnlineMIMOSVR::tune_kernel_params(best_parameters, *p_features, *p_labels);
#else
        best_parameters.set_svr_kernel_param(0.46733608245812541);
        best_parameters.set_svr_kernel_param2(1.2209334891491921);
#endif
    }

    return {};
}

TEST(path_tune_train_predict2, basic_integration)
{
    //svr::IKernel<double>::IKernelInit();
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

#endif
