//
// Created by zarko on 21/05/19.
// SMO test
#if 0
#include <gtest/gtest.h>
#include "test_harness.hpp"
#include "kernel_basic_integration_test.hpp"

#include "nm.hpp"


#define DEFAULT_SVR_DECREMENT 20000

#define DEFAULT_LAG 600

#define DEFAULT_ADJACENT 1 //0.640625

double nm_wrapper3(const std::vector<double> &params, const arma::mat &learning_data, const arma::mat &reference_data,
                  const int n_total_samples, const int num_values_trained_batch)
{
#if 0
    SVRParameters model_svr_parameters(
            0, 0, "", "", 0, 152168000, 0, params[0], 1, 20000, 0.640625, static_cast<const e_kernel_type>(7), 400);
    double mae = 0.;
    PROFILE_EXEC_TIME(mae = train_predict_cycle(
            model_svr_parameters, labels_mx, features_mx, n_total_samples, num_values_trained_batch,
            reference_data, learning_data), "Train predict cycle");
    return mae;
#else
    double cur_variance_diff;
    SVRParameters model_svr_parameters(
            0, 0, "", "", 0, 0, 0, params[0], 1, DEFAULT_SVR_DECREMENT, DEFAULT_ADJACENT, static_cast<const e_kernel_type>(7),
            DEFAULT_LAG);
    // PROFILE_EXEC_TIME(cur_variance_diff = svr::OnlineSVR::produce_kernel_matrices_variance(model_svr_parameters, learning_data, reference_data, false),
//                      "produce_kernel_matrices_variance");
    return cur_variance_diff;
#endif
}

// TODO Test unusable, update if needed or remove
void run_file3(const std::pair<std::string /* labels full path */, std::string /* features full path */> &level_file, std::mutex &printout_mutex, std::atomic<size_t> &running_ct)
{
    ++running_ct;
    arma::mat all_labels_mx;
    all_labels_mx.load(level_file.first);
    all_labels_mx = all_labels_mx.rows(all_labels_mx.n_rows - all_labels_mx.n_rows / 5, all_labels_mx.n_rows - 1);
    arma::mat all_features_mx;
    all_features_mx.load(level_file.second);
    all_features_mx = all_features_mx.rows(all_labels_mx.n_rows - all_features_mx.n_rows / 5, all_labels_mx.n_rows - 1);

    const int n_total_samples = all_features_mx.n_rows;
    LOG4_DEBUG("Features " << level_file.second << " size " << all_features_mx.n_rows << " x " << all_features_mx.n_cols <<
                           " labels " << level_file.first << " size " << all_labels_mx.n_rows << " x " << all_labels_mx.n_cols);
    const auto num_values_trained_batch = n_total_samples - TEST_FUTURE_PREDICT_COUNT - ONLINE_LUFTA;
    double err = 0;
    SVRParameters best_parameters(
            0, 0, "", "", 0, 2521, 1e-9, 1, 1, DEFAULT_SVR_DECREMENT, DEFAULT_ADJACENT, static_cast<const e_kernel_type>(7), DEFAULT_LAG);
    svr::nm::NM_parameters nm_param;
    nm_param.max_iteration_number_ = 30;
    nm_param.tolerance_ = 1E-5;
    svr::nm::loss_callback_t f = std::bind(
            nm_wrapper3,
            std::placeholders::_1,
            std::ref(all_features_mx),
            std::ref(all_labels_mx),
            n_total_samples,
            num_values_trained_batch);
    const auto err_args = svr::nm::nm(f, {1}, nm_param);
    err = err_args.first;
    LOG4_INFO("Final kernel error " << err << " file " << level_file.first);
    const auto opt_args = err_args.second;
    best_parameters.set_svr_kernel_param(std::pow(opt_args[0], 7));
//    best_parameters.set_svr_C(opt_args[1]);
    best_parameters.set_svr_kernel_param2(1);

    std::atomic<double> best_mae = err;//std::numeric_limits<double>::max();
    //const auto best_kernel_param = best_parameters.get_svr_kernel_param();
    //const auto best_kernel_param2 = best_parameters.get_svr_kernel_param2();
#if 0
    for (size_t cost_exp = 1; cost_exp < 3; ++cost_exp) {
        const double cost = std::pow(10, cost_exp);
        for (size_t kernel_param_exp = 1; kernel_param_exp < 2; ++kernel_param_exp) { // 10
            SVRParameters model_svr_parameters(best_parameters);
            model_svr_parameters.set_svr_C(cost);
            model_svr_parameters.set_svr_kernel_param(std::pow(model_svr_parameters.get_svr_kernel_param(), kernel_param_exp));
            double mae = 0.;
            PROFILE_EXEC_TIME(mae = train_predict_cycle_smo(
                    model_svr_parameters, n_total_samples, num_values_trained_batch, all_labels_mx, all_features_mx), "Train predict cycle SMO");
            {
                std::scoped_lock printout_guard(printout_mutex);
                if (best_mae > mae) {
                    best_mae = mae;
                    best_parameters = model_svr_parameters;
                    LOG4_INFO("Best cost: " << level_file.second << " MAE: " << best_mae << " Parameters: "
                                            << best_parameters.to_sql_string());
                }
            }
        }
    }
#endif

    {
        LOG4_INFO("Finished optimizing: " << level_file.second << " MAE: " << best_mae << " Parameters: " << best_parameters.to_sql_string());
        train_predict_cycle_smo(best_parameters, n_total_samples, num_values_trained_batch, all_labels_mx, all_features_mx);
    }
    --running_ct;
}

TEST(path_tune_train_predict3, basic_integration)
{
    svr::IKernel<double>::IKernelInit();
    std::mutex printout_mutex;
    std::atomic<size_t> running_ct = 0;
#if 0 // All levels?
    for (const auto &ff: file_name_pairs) {
        while (running_ct.load() >= RUN_LIMIT) sleep(1);
        //cilk_spawn
        run_file(ff, printout_mutex, running_ct);
    }
    //cilk_sync;
#else
    run_file3(file_name_pairs[10], printout_mutex, running_ct);
#endif
}
#endif // 0