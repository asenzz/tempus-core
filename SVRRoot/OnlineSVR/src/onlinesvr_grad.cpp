#include "onlinesvr.hpp"

#ifdef GRAD_BOOST

void
OnlineMIMOSVR::batch_train(
        const matrix_ptr &p_xtrain,
        const matrix_ptr &p_ytrain,
        const bool update_r_matrix,
        const matrices_ptr &kernel_matrices,
        const bool pseudo_online)
{
    LOG4_DEBUG(
            "Training on X " << common::present(*p_xtrain) << ", Y " << common::present(*p_ytrain) << ", update R matrix " << update_r_matrix << ", pre-calculated kernel matrices " << (kernel_matrices ? kernel_matrices->size() : 0) <<
                             ", pseudo online " << pseudo_online << ", parameters " << p_param_set->to_string());
    const std::scoped_lock learn_lock(learn_mx);

#ifdef INTEGRATION_TEST_DEBUG
    if (!p_param_set->get_decon_level() || p_param_set->get_decon_level() == 28 || p_param_set->get_decon_level() == 2) {
        p_xtrain->save(common::formatter() << "/tmp/xtrain_" << p_param_set->get_decon_level() << ".csv", arma::csv_ascii);
        p_ytrain->save(common::formatter() << "/tmp/ytrain_" << p_param_set->get_decon_level() << ".csv", arma::csv_ascii);
#ifdef EXPERIMENTAL_FEATURES
        matplotlibcpp::plot(std::vector<double>{(double *)p_ytrain->memptr(), (double *)p_ytrain->memptr() + p_ytrain->n_elem}, {{"label", "Level 0 labels"}});
        matplotlibcpp::legend();
        matplotlibcpp::show();
        matplotlibcpp::cla();
        matplotlibcpp::clf();
        matplotlibcpp::close();
        matplotlibcpp::plot(std::vector<double>{(double *)p_xtrain->memptr(), (double *)p_xtrain->memptr() + p_ytrain->n_cols}, {{"label", "Level 0 first features"}});
        matplotlibcpp::legend();
        matplotlibcpp::show();
        matplotlibcpp::cla();
        matplotlibcpp::clf();
        matplotlibcpp::close();
        matplotlibcpp::plot(std::vector<double>{(double *)p_xtrain->memptr() + p_xtrain->n_elem - p_ytrain->n_cols, (double *)p_xtrain->memptr() + p_xtrain->n_elem}, {{"label", "Level 0 last features"}});
        matplotlibcpp::legend();
        matplotlibcpp::show();
#endif
    }
#endif

    reset_model(pseudo_online);

    if (!p_param_set->get_svr_kernel_param()) {
        if (kernel_matrices) {
            LOG4_ERROR("Kernel matrices provided will be cleared because SVR kernel parameters are not initialized!");
            kernel_matrices->clear();
        }
//        PROFILE_EXEC_TIME(tune_kernel_params(p_param_set, *p_xtrain, *p_ytrain), "Tune kernel params for model " << p_param_set->get_decon_level());
    }
    if (p_xtrain->n_rows > p_param_set->get_svr_decremental_distance())
        p_xtrain->shed_rows(0, p_xtrain->n_rows - p_param_set->get_svr_decremental_distance() - 1);
    if (p_ytrain->n_rows > p_param_set->get_svr_decremental_distance())
        p_ytrain->shed_rows(0, p_ytrain->n_rows - p_param_set->get_svr_decremental_distance() - 1);

    if (not pseudo_online) {
        p_features = p_xtrain;
        p_labels = p_ytrain;
        if (p_xtrain->n_rows != p_ytrain->n_rows || p_xtrain->empty() || p_ytrain->empty() || !p_xtrain->n_cols || !p_ytrain->n_cols)
            LOG4_THROW("Invalid data dimensions X train " << arma::size(*p_xtrain) << ", Y train " << arma::size(*p_ytrain) << ", level " << p_param_set->get_decon_level());
    }

    if (!p_manifold) init_manifold(*p_xtrain, *p_ytrain);

    ixs = generate_indexes();
    if (kernel_matrices && kernel_matrices->size() == ixs.size()) {
        p_kernel_matrices = kernel_matrices;
        LOG4_DEBUG("Using " << ixs.size() << " pre-calculated matrices.");
    } else if (kernel_matrices && !kernel_matrices->empty()) {
        LOG4_ERROR("External kernel matrices do not match needed chunks count!");
    } else {
        LOG4_DEBUG("Initializing kernel matrices from scratch.");
    }

    for (auto &kv : main_components) {
        auto &component = kv.second;
        // the regression parameters
        //component.total_weights = arma::zeros(n_m, n_k);
        if (component.chunk_weights.size() != ixs.size()) component.chunk_weights.resize(ixs.size());
        if (component.mae_chunk_values.size() != ixs.size()) component.mae_chunk_values.resize(ixs.size(), 0.);
    }

    if (p_kernel_matrices->size() != ixs.size()) p_kernel_matrices->resize(ixs.size());
    if (auto_gu.size() != ixs.size()) auto_gu.resize(ixs.size(), p_param_set->get_svr_epsco());

#pragma omp parallel for num_threads(ixs.size()) schedule(static, 1)
    for (uint32_t i = 0; i < ixs.size(); ++i) {
        do_over_train_zero_epsilon(i);
        if (!p_param_set[i].get_grad_level()) continue;
        auto p_grad_parameters = ptr<SVRParameters>(p_param_set[i]);
        p_grad_parameters->decrement_gradient()

        const arma::uvec other_ixs = get_other_ixs(i);
        grad_svr[i] = OnlineMIMOSVR(
                p_grad_parameters,
                p_features,
                ptr<arma::mat>(single_chunk_predict(p_features->rows(other_ixs)) - p_labels->rows(other_ixs)));
    }
    update_total_weights();
    samples_trained = p_features->n_rows;

    if (update_r_matrix) PROFILE_EXEC_TIME(init_r_matrix(), "init_r_matrix");
}

arma::mat OnlineMIMOSVR::grad_predict(const arma::mat &x_predict, const bpt::ptime &pred_time)
{
    if (ixs.size() == 1 && main_components.size() == 1)
        return init_predict_kernel_matrix(*p_param_set, *p_features, x_predict, p_manifold, pred_time) * main_components.begin()->second.total_weights;

    std::vector<size_t> min_ixs;
    std::vector<size_t> max_ixs;
    std::vector<arma::mat> predicted;
    arma::mat start_mat = arma::zeros(x_predict.n_rows, multistep_len);
    std::vector<arma::mat> chunk_predicted;

    std::mutex mx;
    std::vector<double> mae_total(ixs.size(), 0.);
    for (auto &kv: main_components) tbb_pfor_i__(0, mae_total.size(), mae_total[i] += kv.second.mae_chunk_values[i])
    auto sorted_ix = sort_indexes(mae_total);
    std::unordered_set<size_t> best_ixs(sorted_ix.begin(), sorted_ix.size() < 2 ? sorted_ix.end() : sorted_ix.begin() + std::round(sorted_ix.size() / BEST_PREDICT_CHUNKS_DIVISOR));
    for (auto &kv: main_components) {
#pragma omp parallel for schedule(static, 1) num_threads(ixs.size())
        for (size_t i = 0; i < ixs.size(); ++i) {
            if (best_ixs.find(i) == best_ixs.end()) continue;
            arma::mat multiplicated(arma::size(p_labels->rows(ixs[i])));
            if (p_param_set->is_manifold()) {
                const arma::mat chunk_kernel_matrix = init_predict_kernel_matrix(*p_param_set, x_predict, p_features->rows(ixs[i]), p_manifold, pred_time);
                multiplicated = arma::mean(chunk_kernel_matrix + p_labels->rows(ixs[i]));
            } else {
                const arma::mat chunk_kernel_matrix = init_predict_kernel_matrix(*p_param_set, p_features->rows(ixs[i]), x_predict, p_manifold, pred_time);
#pragma omp parallel for
                for (size_t col_ix = 0; col_ix < kv.second.chunk_weights[i].n_cols; ++col_ix)
                    multiplicated.col(col_ix) = kv.second.epsilon + chunk_kernel_matrix * kv.second.chunk_weights[i].col(col_ix);
            }
#pragma omp critical(add_chunk_predicted)
            chunk_predicted.emplace_back(multiplicated + (grad_svr[i] ? grad_svr[i]->predict(x_predict, predict_time) : 0);
        }

        arma::mat sum_steps = arma::zeros(x_predict.n_rows, chunk_predicted.size());
        tbb_pfor_i__(0, chunk_predicted.size(),
            // last_step.col(i) = chunk_predicted[i].col(multistep_len - 1);
            // Replacing last column check only for sum of all columns
            //arma::colvec sum_of_multistep_predictions = arma::sum(chunk_predicted[i], 1);
            sum_steps.col(i) = arma::colvec(arma::sum(chunk_predicted[i], 1)); // TODO Is this basically transpose?
        )
        std::vector<unsigned> outlier_result = outlier_test(sum_steps, OUTLIER_ALPHA, min_ixs, max_ixs);
        arma::mat accumulated = accumulate_chunks(outlier_result, min_ixs, max_ixs, chunk_predicted, *p_param_set);

//#pragma omp critical(predict_add)
        predicted.emplace_back(accumulated);
    }
    arma::mat pred = (std::accumulate(predicted.begin(), predicted.end(), start_mat) / double(predicted.size())) * labels_scaling_factor;
#ifdef DEBUG_PREDICTIONS // Debug log to file
    const auto prev_pred_find = prev_pred.find({p_param_set.get_decon_level(), p_param_set.get_input_queue_column_name()});
    if (prev_pred_find != prev_pred.end()) {
        LOG4_DEBUG("Level " << p_param_set.get_decon_level() << " column " << p_param_set.get_input_queue_column_name() << " prev_pred_find " << prev_pred_find->second(0, 0) << " pred " << pred(0, 0) << " prev label " << p_labels->at(p_labels->n_rows - 1, 0) << " prev prev label " << p_labels->at(p_labels->n_rows - 2, 0));
        const auto diff1 = p_labels->at(p_labels->n_rows - 1, 0) - prev_pred_find->second(0, 0);
        const auto diff2 = p_labels->at(p_labels->n_rows - 1, 0) - p_labels->at(p_labels->n_rows - 2, 0);
        LOG4_DEBUG("Level " << p_param_set.get_decon_level() << " column " << p_param_set.get_input_queue_column_name() << " prev label - prev_pred_find " << diff1 << " prev label - prev prev label" << diff2 << " prev pred is wronger " << (std::abs(diff1) > std::abs(diff2)));
    }
    prev_pred[{p_param_set.get_decon_level(), p_param_set.get_input_queue_column_name()}] = pred;
    if (get_param_set().get_decon_level() == C_logged_level) {
        static size_t call_ct = 0;
        std::stringstream ss_file_name;
        ss_file_name << "/mnt/faststore/level_" << p_param_set.get_decon_level() << "_" << p_param_set.get_input_queue_column_name() << "_chunk_predict_output_" << call_ct++ << ".csv";
        pred.save(ss_file_name.str(), arma::csv_ascii);
    }
#endif
    LOG4_TRACE("Level " << p_param_set->get_decon_level() << " predicted " << pred);
    return pred;
}

#endif