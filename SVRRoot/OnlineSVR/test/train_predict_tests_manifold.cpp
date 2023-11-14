//
// Created by zarko on 21/05/19.
//
#include <gtest/gtest.h>
#include <ModelService.hpp>
#include <optimizer.hpp>
#include <appcontext.hpp>
#include "test_harness.hpp"
#include "kernel_basic_integration_test.hpp"
#include "common/compatibility.hpp"

#define DO_TEST_PSO


#define VALIDATION_WINDOW 100
#define DEFAULT_LAG 650
#define DEFAULT_SVR_DECREMENT 200
#define DEFAULT_ADJACENT 1
#define DEFAULT_KERNEL kernel_type_e::DEEP_PATH
#define INPUT_LIMIT (DEFAULT_SVR_DECREMENT + DEFAULT_LAG + VALIDATION_WINDOW)

static const size_t lag = DEFAULT_LAG;

#if 0
static const std::pair<arma::mat, arma::mat>
generate_features_labels(const arma::mat &data)
{
    arma::mat features(data.n_rows - lag, lag);
    arma::mat labels(data.n_rows - lag, 1);
    __omp_pfor_i(lag, data.n_rows,
        const arma::rowvec row_features = data.rows(i - lag, i - 1).t();
        features.row(i - lag) = row_features;
        labels.row(i - lag) = data.row(i);
    )
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


static std::vector<double> run_file()
{
    const svr::matrix_ptr p_features = load_file(svr::common::formatter() << "/mnt/slowstore/var/tmp/features_" << svr::C_logged_level << "_" << 0 << ".out");
    const svr::matrix_ptr p_labels = load_file(svr::common::formatter() << "/mnt/slowstore/var/tmp/labels_" << svr::C_logged_level << "_" << 0 << ".out");
    arma::mat last_knowns(arma::size(*p_labels));
    auto p_best_parameters = std::make_shared<SVRParameters>(
            0, 0, "test manifold", "test manifold", 0, 0, 0, 0, 0, 0, 0, DEFAULT_SVR_DECREMENT, DEFAULT_ADJACENT, DEFAULT_KERNEL,
            DEFAULT_LAG);

    LOG4_DEBUG("Train predict cycle, multistep len " << PROPS.get_multistep_len() << ", future predict count " << PROPS.get_future_predict_count());
    {
        svr::OnlineMIMOSVR p_svr_model(
                p_best_parameters,
                std::make_shared<arma::mat>(p_features->rows(0, p_best_parameters->get_svr_decremental_distance())),
                std::make_shared<arma::mat>(p_labels->rows(0, p_best_parameters->get_svr_decremental_distance())),
                false, svr::matrices_ptr{}, false, svr::MimoType::single, p_labels->n_cols);

        std::vector<bpt::ptime> times(p_features->n_rows, bpt::special_values::not_a_date_time); // TODO Buggy change to increment
        svr::OnlineMIMOSVR::future_validate(p_best_parameters->get_svr_decremental_distance(), p_svr_model, *p_features, *p_labels, last_knowns, times, false, 1, 0);
    }

    {
        p_best_parameters->set_kernel_type(kernel_type_e::PATH);
        p_best_parameters->set_svr_kernel_param(0);
        p_best_parameters->set_svr_kernel_param2(0);
        svr::OnlineMIMOSVR p_svr_model(
                p_best_parameters,
                std::make_shared<arma::mat>(p_features->rows(0, p_best_parameters->get_svr_decremental_distance())),
                std::make_shared<arma::mat>(p_labels->rows(0, p_best_parameters->get_svr_decremental_distance())),
                false, svr::matrices_ptr{}, false, svr::MimoType::single, p_labels->n_cols);
        std::vector<bpt::ptime> times(p_features->n_rows, bpt::special_values::not_a_date_time); // TODO Buggy change to increment
        svr::OnlineMIMOSVR::future_validate(p_best_parameters->get_svr_decremental_distance(), p_svr_model, *p_features, *p_labels, last_knowns, times, false, 1, 0);
    }

    return {};
}

TEST(manifold_tune_train_predict, basic_integration)
{
    svr::IKernel<double>::IKernelInit();
    run_file();
}
