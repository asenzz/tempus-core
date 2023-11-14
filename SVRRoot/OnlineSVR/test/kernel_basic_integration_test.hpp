//
// Created by jarko on 02/11/18.
//

#ifndef SVR_KERNEL_BASIC_INTEGRATION_TEST_HPP
#define SVR_KERNEL_BASIC_INTEGRATION_TEST_HPP

#define TRAIN_BATCH_TEST_FACTOR         .9
#define INPLACE_VALIDATION_TEST_FACTOR  .01
#define TEST_FUTURE_PREDICT_COUNT       200
#define ONLINE_LUFTA                    5
#define RUN_LIMIT                       4


#include "test_harness.hpp"
#include "onlinesvr.hpp"
#include "train_files.hpp"
#include "_deprecated/OnlineSMOSVR.hpp"

#if 0

void
kernel_basic_integration_test(
        const std::string &model_file,
        const std::string &dump_file,
        const std::string &saved_model_file);

void
kernel_single_run_test(
        const SVRParameters &model_file,
        const std::string &dump_file,
        const std::string &saved_model_file,
        const std::string &features_data_file,
        const std::string &labels_data_file);

double
inplace_validate(
        const int start_inplace_validate_idx,
        const int current_index,
        const svr::OnlineMIMOSVR &online_svr,
        const svr::datamodel::vmatrix<double> &features_data,
        const svr::datamodel::vmatrix<double> &labels_data);

double
future_validate(
        const int n_total_samples,
        const int current_index,
        const svr::OnlineSVR &online_svr,
        const arma::mat &features_data,
        const arma::mat &labels_data);

std::tuple<double, double, std::vector<double>, std::vector<double>, double, std::vector<double>>
future_validate(
        const size_t from_idx,
        const svr::OnlineMIMOSVR &online_svr,
        const arma::mat &features,
        const arma::mat &labels,
        const bool single_pred = false);

std::tuple<double, double, std::vector<double>, std::vector<double>, double, std::vector<double>>
train_predict_cycle_online(
        const int validate_start_pos,
        const arma::mat &labels,
        const arma::mat &features,
        svr::OnlineMIMOSVR &svr_model);

double
train_predict_cycle_smo(
        const SVRParameters &model_svr_parameters,
        const int n_total_samples,
        const int num_values_trained_batch,
        const arma::mat &all_labels_mx,
        const arma::mat &all_features_mx);

#endif

#endif //SVR_KERNEL_BASIC_INTEGRATION_TEST_HPP

