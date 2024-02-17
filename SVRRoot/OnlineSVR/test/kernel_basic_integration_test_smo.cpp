//
// Created by jarko on 02/11/18.
//
#if 0
#include <business.hpp>
#include "kernel_basic_integration_test.hpp"
#include <matplotlibcpp.h>

using namespace arma;

// Inplace MAE
double
inplace_validate(
        const int start_inplace_validate_idx,
        const int current_index,
        const svr::OnlineMIMOSVR &online_svr,
        const svr::datamodel::vmatrix<double> &features_data,
        const svr::datamodel::vmatrix<double> &labels_data)
{
    LOG4_DEBUG("Inplace predict start index " << start_inplace_validate_idx << " and end index is " << current_index);
    double counter = 0;
    double mae =0;
    const auto predicted_values = online_svr.predict(features_data);
#pragma omp parallel for reduction(+:mae,counter) default(shared) num_threads(adj_threads(current_index - start_inplace_validate_idx))
    for (int i_inplace = start_inplace_validate_idx; i_inplace <= current_index; ++i_inplace){
        for(int mimo_ix = 0; mimo_ix < labels_data.get_length_cols(); ++mimo_ix) {
            mae += std::abs(labels_data.get_value(i_inplace, mimo_ix) -
                                         predicted_values.get_value(i_inplace, mimo_ix));
            counter += 1;
        }
    }
    mae /= counter;
    LOG4_DEBUG("Inplace predict between sample " << start_inplace_validate_idx << " and " <<
               current_index << " MAE " << mae);
    return mae;
}

#define ADDED_ERROR 1000.
#define VALIDATION_BAD_RESULTS {ADDED_ERROR, ADDED_ERROR, arma::conv_to<std::vector<double>>::from(ADDED_ERROR + arma::vectorise(labels_data.rows(current_index, to_row_idx))), arma::conv_to<std::vector<double>>::from(arma::vectorise(labels_data.rows(current_index, to_row_idx))), ADDED_ERROR}

// Works only for one-step predictions
std::tuple<double, double, std::vector<double>, std::vector<double>, double, std::vector<double>>
future_validate(
        const size_t from_idx,
        const svr::OnlineMIMOSVR &online_svr,
        const arma::mat &features,
        const arma::mat &labels,
        const bool single_pred)
{
    if (labels.n_rows == from_idx) {
        LOG4_WARN("Calling future validate at the end of p_labels_data array. MAE = 1000");
        return {ADDED_ERROR, ADDED_ERROR, {}, {}, 0, {}};
    }

    const size_t to_row_idx = std::min((size_t)labels.n_rows - 1, single_pred ? from_idx : (size_t)labels.n_rows - 1);
    LOG4_DEBUG("Future predict start index " << from_idx << " and end index is " << to_row_idx);
    const size_t num_preds = 1 + to_row_idx - from_idx;
    arma::mat predicted_values;
    try {
        PROFILE_EXEC_TIME(predicted_values = online_svr.predict(features.rows(from_idx, to_row_idx)), "Chunk predict");
    } catch (const std::exception &ex) {
        LOG4_ERROR("Error predicting values: " << ex.what());
        predicted_values.set_size(num_preds, 1);
        predicted_values.fill(ADDED_ERROR);
    }
    if (predicted_values.n_rows != num_preds)
        LOG4_ERROR("predicted_values.n_rows " << predicted_values.n_rows << " != num_preds " << num_preds);
    std::vector<double> ret_predicted_values(num_preds), actual_values(num_preds), lin_pred_values(num_preds);
    double red_mae{0};
    for (size_t i_future = from_idx; i_future <= to_row_idx; ++i_future) {
        const auto ix = i_future - from_idx;
        const auto predicted_val = predicted_values(ix, 0);
        const auto actual_val = labels.at(i_future, 0);
        ret_predicted_values[ix] = predicted_val;
        actual_values[ix] = actual_val;
        lin_pred_values[ix] = labels.at(i_future - 1, 0);
        LOG4_TRACE("Predicted " << predicted_val << " actual " << actual_val << " row " << ix << " col " << 0);
        red_mae += std::abs(actual_val - predicted_val);
    };
    const double mae = red_mae / double(actual_values.size());
    const double avg_label = arma::mean(arma::abs(arma::vectorise(labels)));
    const double mape = 100. * mae / avg_label;
    const double lin_mape = 100. * arma::mean(arma::abs(arma::rowvec(actual_values) - arma::rowvec(lin_pred_values))) / avg_label;
    LOG4_DEBUG("Future predict from row " << from_idx << " until " << to_row_idx << " MAE " << mae << " MAPE " << mape << " Lin MAPE " << lin_mape);

    return {mae, mape, ret_predicted_values, actual_values, lin_mape, lin_pred_values};
}


#include "OnlineSMOSVR.hpp"

double
future_validate(
        const int n_total_samples,
        const int current_index,
        const svr::OnlineSVR &online_svr,
        const arma::mat &features_data,
        const arma::mat &labels_data)
{
    const auto to_row_idx = std::min(n_total_samples, current_index + TEST_FUTURE_PREDICT_COUNT);
    LOG4_DEBUG("Future predict start index " << current_index << " and end index is " << to_row_idx);
    if (n_total_samples == current_index) {
        LOG4_WARN("Calling future validate at the end of labels_data array. MAE = 0");
        return 0;
    }
    const auto predicted_values = online_svr.predict((arma::mat)features_data.rows(current_index, to_row_idx));

    size_t counter = 0;
    double mae = 0.;
    for (int i_future = current_index; i_future < to_row_idx; ++i_future) {
        mae += std::abs(labels_data(i_future, 0) - predicted_values[i_future - current_index]);
        counter += 1;
    }
    mae = mae / double(counter);
    LOG4_DEBUG("Future predict between sample " << current_index << " and " << to_row_idx << " MAE " << mae);
    return mae;
}


void
kernel_basic_integration_test(
        const std::string &model_file,
        const std::string &dump_file,
        const std::string &saved_model_file)
{
    LOG4_BEGIN();
    svr::IKernel<double>::IKernelInit();
#if 0
    OnlineMIMOSVR_ptr mimo_model;
    try {
        mimo_model = svr::OnlineMIMOSVR::load_online_mimosvr(model_file.c_str());
    } catch (const std::exception &e) {
        LOG4_ERROR("Failed loading model for testing. " << e.what());
        throw e;
    }

    // Throw away all data from the online_svr object, except the parameters themselves.
    svr::OnlineMIMOSVR online_svr_(mimo_model->get_svr_parameters(), mimo_model->get_multistep_len());

    const arma::mat x_train = mimo_model->get_learning_matrix();
    const arma::mat y_train = mimo_model->get_reference_matrix();
    mimo_model.reset();

    LOG4_DEBUG("Model from file " << model_file << " loaded.");
    const vmatrix<double> features_data(x_train);
    const vmatrix<double> labels_data(y_train);

    SVRParameters param = online_svr_.get_svr_parameters();
    LOG4_FILE(dump_file.c_str(), "SVR parameters:\n");
    LOG4_FILE(dump_file.c_str(), "Cost " << param.get_svr_C() << "\n");
    LOG4_FILE(dump_file.c_str(), "Kernel Type " << (int) param.get_kernel_type() << "\n");
    LOG4_FILE(dump_file.c_str(), "SVR Epsilon " << param.get_svr_epsilon() << "\n");
    LOG4_FILE(dump_file.c_str(), "Kernel Param 1 " << param.get_svr_kernel_param() << "\n");
    LOG4_FILE(dump_file.c_str(), "Kernel Param 2 " << param.get_svr_kernel_param2() << "\n");
    LOG4_FILE(dump_file.c_str(), "Lag count " << param.get_lag_count() << "\n");
    LOG4_FILE(dump_file.c_str(), "Decremental distance " << param.get_svr_decremental_distance() << "\n");
    const int n_total_samples = features_data.get_length_rows();
    const auto num_values_trained_batch = (int) std::round(n_total_samples * TRAIN_BATCH_TEST_FACTOR);
    const int num_values_inplace_validation = num_values_trained_batch * INPLACE_VALIDATION_TEST_FACTOR;

    // Get the first n training samples, to train with the SMO.
    {
        auto learning_data_batch = features_data.extract_rows(0, num_values_trained_batch - 1);
        auto reference_data_batch = labels_data.extract_rows(0, num_values_trained_batch - 1);
        const auto reference_data = vmatrix_to_admat(reference_data_batch);
        const auto learning_data = vmatrix_to_admat(learning_data_batch);
        online_svr_.batch_train(learning_data, reference_data, false);
    }

    {
        const int i = num_values_trained_batch - 1;
        const int start_inplace_validate_idx = i - num_values_inplace_validation;
        inplace_validate(start_inplace_validate_idx, i, online_svr_, features_data, labels_data);
        future_validate(n_total_samples, i, online_svr_, features_data, labels_data);
    }

    // Train-Forget online on the remaining samples.
    for (auto i = num_values_trained_batch; i < n_total_samples; ++i) {
            if (num_values_trained_batch == i) LOG4_FILE(dump_file, "Start learning.. SAMPLE INDEX " << i);
        LOG4_DEBUG("Extracting new row for online train.. idx: " << i);
        const int start_inplace_validate_idx = i - num_values_inplace_validation;

        // TRAIN ON THE NEXT SAMPLE: IDX = i
        {
            svr::datamodel::vmatrix<double> X = features_data.extract_rows(i, i);
            svr::datamodel::vmatrix<double> Y = labels_data.extract_rows(i, i);

            online_svr_.train(X, Y, true);
        }
        inplace_validate(start_inplace_validate_idx, i, online_svr_, features_data, labels_data);
        future_validate(n_total_samples, i, online_svr_, features_data, labels_data);

        // FORGET THE LAST SAMPLE: IDX = const int start_inplace_validate_idx = i - num_values_trained_batch;
        LOG4_FILE(dump_file.c_str(), "Start forget.. SAMPLE INDEX " << start_inplace_validate_idx);
        LOG4_DEBUG("Extracting new row for online forget.. idx: " << start_inplace_validate_idx);

        // Forget
        online_svr_.best_forget(true);
        LOG4_FILE(dump_file.c_str(), "End forget.. SAMPLE INDEX " << start_inplace_validate_idx);
        inplace_validate(start_inplace_validate_idx, i, online_svr_, features_data, labels_data);
        future_validate(n_total_samples, i, online_svr_, features_data, labels_data);
    } // Learn-Forget with window
    LOG4_DEBUG("Finish SIMULATION Learn-Forget");
    LOG4_FILE(dump_file.c_str(), "Finish SIMULATION Learn-Forget");

//    online_svr.save_onlinesvr(saved_model_file.c_str());
    LOG4_END();
#endif
}


vmatrix<double>
comb_matrix(
        const vmatrix<double> &v,
        const size_t step_in,
        const size_t step_out)
{
    vmatrix<double> res;
    size_t j = 0;
    for (decltype(v.get_length_rows()) i = 0; i < v.get_length_rows(); ++i) {
        if (j < step_in) res.add_row_copy(v.get_row_ref(i));
        ++j;
        j %= (step_in + step_out);
    }
    return res;
}

#define TEST_MULTISTEP_LEN (5)

void
kernel_single_run_test(
        const SVRParameters &model_svr_parameters,
        const std::string &dump_file,
        const std::string &saved_model_file,
        const std::string &features_data_file,
        const std::string &labels_data_file)
{
    LOG4_BEGIN();

 datamodel::SVRParameters_ptr param = std::make_shared<SVRParameters>(model_svr_parameters);
    svr::OnlineMIMOSVR online_svr_(param, svr::MimoType::single, TEST_MULTISTEP_LEN);

    const vmatrix<double> features_data = svr::datamodel::vmatrix<double>::load(features_data_file);
    const vmatrix<double> labels_data = svr::datamodel::vmatrix<double>::load(labels_data_file);

    param->set_svr_decremental_distance(features_data.get_length_rows());
    LOG4_FILE(dump_file.c_str(), "SVR parameters:\n");
    LOG4_FILE(dump_file.c_str(), "Cost " << param->get_svr_C() << "\n");
    LOG4_FILE(dump_file.c_str(), "Kernel Type " << (int) param->get_kernel_type() << "\n");
    LOG4_FILE(dump_file.c_str(), "SVR Epsilon " << param->get_svr_epsilon() << "\n");
    LOG4_FILE(dump_file.c_str(), "Kernel Param 1 " << param->get_svr_kernel_param() << "\n");
    LOG4_FILE(dump_file.c_str(), "Kernel Param 2 " << param->get_svr_kernel_param2() << "\n");
    LOG4_FILE(dump_file.c_str(), "Lag count " << param->get_lag_count() << "\n");
    LOG4_FILE(dump_file.c_str(), "Decremental distance " << param->get_svr_decremental_distance() << "\n");

    const int n_total_samples = features_data.get_length_rows();
    const int num_values_inplace_validation = n_total_samples * 0.5;

    {
        const vmatrix<double> combed_features = comb_matrix(features_data, 45, 15);
        const vmatrix<double> combed_labels = comb_matrix(labels_data, 45, 15);
        const auto combed_features_admat = vmatrix_to_admat(combed_features);
        const auto combed_labels_admat = vmatrix_to_admat(combed_labels);
        online_svr_.batch_train(std::make_shared<arma::mat>(combed_features_admat), std::make_shared<arma::mat>(combed_labels_admat), false);
    }

    {
        const int i = n_total_samples;
        const int start_inplace_validate_idx = i - num_values_inplace_validation;
        inplace_validate(start_inplace_validate_idx, n_total_samples, online_svr_, features_data, labels_data);
    }
}


std::tuple<double, double, std::vector<double>, std::vector<double>, double, std::vector<double>>
train_predict_cycle_online(
        const int validate_start_pos,
        const arma::mat &labels,
        const arma::mat &features,
        svr::OnlineMIMOSVR &svr_model)
{
    const auto training_features = features.row(validate_start_pos - 1);
    const auto training_labels = labels.row(validate_start_pos - 1);
    PROFILE_EXEC_TIME(svr_model.learn(training_features, training_labels, true), "Online train");
    return future_validate(validate_start_pos, svr_model, features, labels, true);
}


double train_predict_cycle_smo(
        const SVRParameters &model_svr_parameters,
        const int n_total_samples,
        const int num_values_trained_batch,
        const arma::mat &all_labels_mx,
        const arma::mat &all_features_mx)
{
    svr::OnlineSVR svr_model(model_svr_parameters);
    const auto learning_data_batch = all_features_mx.rows(0, num_values_trained_batch - 1);
    auto reference_data_batch = all_labels_mx.rows(0, num_values_trained_batch - 1);
    LOG4_DEBUG("Cycle parameters " << model_svr_parameters.to_sql_string());
    try {
        PROFILE_EXEC_TIME(svr_model.train_batch(learning_data_batch, reference_data_batch.col(0)), "Batch train");
    } catch (...) {
        return 100;
    }
    return future_validate(n_total_samples, num_values_trained_batch - 1, svr_model, all_features_mx, all_labels_mx);
}

#endif