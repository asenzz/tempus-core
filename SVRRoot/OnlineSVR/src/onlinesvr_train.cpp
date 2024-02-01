#include "onlinesvr.hpp"

#include "common/parallelism.hpp"
#include "model/Model.hpp"


namespace svr {

#ifdef PSEUDO_ONLINE
constexpr bool __always_retrain = true;
#else
constexpr bool __always_retrain = false;
#endif

void
OnlineMIMOSVR::batch_train(
        const matrix_ptr &p_xtrain,
        const matrix_ptr &p_ytrain,
        const matrices_ptr &kernel_matrices,
        const bool pseudo_online)
{
    LOG4_DEBUG(
            "Training on X " << common::present(*p_xtrain) << ", Y " << common::present(*p_ytrain) << ", pre-calculated kernel matrices " << (kernel_matrices ? kernel_matrices->size() : 0) <<
                             ", pseudo online " << pseudo_online << ", parameters " << get_params());
    const std::scoped_lock lk(learn_mx);

    reset_model(pseudo_online);
    const auto &params = get_params();
    if (params.get_svr_kernel_param() == 0) {
        if (kernel_matrices) {
            LOG4_ERROR("Kernel matrices provided will be cleared because SVR kernel parameters are not initialized!");
            kernel_matrices->clear();
        }
//        PROFILE_EXEC_TIME(tune_kernel_params(p_param_set, *p_xtrain, *p_ytrain), "Tune kernel params for model " << p_param_set->get_decon_level());
    }
    if (p_xtrain->n_rows > params.get_svr_decremental_distance())
        p_xtrain->shed_rows(0, p_xtrain->n_rows - params.get_svr_decremental_distance() - 1);
    if (p_ytrain->n_rows > params.get_svr_decremental_distance())
        p_ytrain->shed_rows(0, p_ytrain->n_rows - params.get_svr_decremental_distance() - 1);

    if (!pseudo_online) {
        p_features = p_xtrain;
        p_labels = p_ytrain;
        if (p_xtrain->n_rows != p_ytrain->n_rows || p_xtrain->empty() || p_ytrain->empty() || !p_xtrain->n_cols || !p_ytrain->n_cols)
            LOG4_THROW("Invalid data dimensions X train " << arma::size(*p_xtrain) << ", Y train " << arma::size(*p_ytrain) << ", level " << params.get_decon_level());
    }

    datamodel::SVRParameters_ptr p;
    if ((p = is_manifold())) {
        init_manifold(p);
        samples_trained = p_xtrain->n_rows;
        return;
    }

    if (ixs.empty()) ixs = get_indexes();
    if (kernel_matrices && kernel_matrices->size() == ixs.size()) {
        p_kernel_matrices = kernel_matrices;
        LOG4_DEBUG("Using " << ixs.size() << " pre-calculated matrices.");
    } else if (kernel_matrices && !kernel_matrices->empty()) {
        LOG4_ERROR("External kernel matrices do not match needed chunks count!");
    } else {
        LOG4_DEBUG("Initializing kernel matrices from scratch.");
    }

    for (auto &component: main_components) {
        if (component->chunk_weights.size() != ixs.size()) component->chunk_weights.resize(ixs.size());
        if (component->chunks_weight.size() != ixs.size()) component->chunks_weight.resize(ixs.size());
        if (component->mae_chunk_values.size() != ixs.size()) component->mae_chunk_values.resize(ixs.size(), 0.);
    }

    if (p_kernel_matrices->size() != ixs.size()) p_kernel_matrices->resize(ixs.size());
    const arma::mat eye_K = arma::eye(p_xtrain->n_rows, p_xtrain->n_rows);
#pragma omp parallel for
    for (size_t i = 0; i < get_num_chunks(); ++i)
        do_over_train_zero_epsilon(*p_features, *p_labels, eye_K, i);
    update_total_weights();
    samples_trained = p_features->n_rows;

#ifdef PSEUDO_ONLINE // Clear kernel matrices once weight is calculated
    p_kernel_matrices = std::make_shared<std::deque<arma::mat>>(ixs.size());
#endif
}

size_t OnlineMIMOSVR::learn(const arma::mat &new_x, const arma::mat &new_y, const bool temp_learn, const std::deque<size_t> forget_ixs, const bpt::ptime &label_time)
{
    const std::scoped_lock learn_lock(learn_mx);

    if (new_x.empty() or new_y.empty()) {
        LOG4_WARN("Features are empty " << new_x.empty() << " or labels are empty " << new_y.empty());
        return 0;
    }

    if (new_x.n_cols != p_features->n_cols || new_y.n_cols != p_labels->n_cols)
        LOG4_THROW("New data dimensions " << arma::size(new_y) << ", " << arma::size(new_x) << " do not match model data dimensions " <<
                    arma::size(*p_labels) << ", " << arma::size(*p_features));
    if (get_params().is_manifold()) {
        if (!p_manifold) LOG4_THROW("Manifold not initialized.");
        const auto new_learning_rows = new_x.n_rows * p_features->n_rows;
        arma::mat new_x_manifold(new_learning_rows, new_x.n_cols + p_features->n_cols);
        arma::mat new_y_manifold(new_learning_rows, new_y.n_cols);
#pragma omp parallel for collapse(2)
        for (size_t i = 0; i < p_features->n_rows; ++i)
            for (size_t j = 0; j < new_x.n_rows; ++j)
                MANIFOLD_SET(new_x_manifold.row(i * new_x.n_rows + j),
                             new_y_manifold.row(i * new_x.n_rows + j),
                             new_x.row(j), new_y.row(j),
                             p_features->row(i), p_labels->row(i));
        p_manifold->learn(new_x_manifold, new_y_manifold);
    }

    const auto new_rows = new_x.n_rows;
    if (!forget_ixs.empty() && new_rows != forget_ixs.size())
        LOG4_THROW("Forget index size " << forget_ixs.size() << " does not equal train samples count " << new_rows);

    if (new_rows > p_features->n_rows / 2 || __always_retrain) {
        p_features->insert_rows(p_features->n_rows, new_x);
        p_labels->insert_rows(p_labels->n_rows, new_y);
        p_features->shed_rows(0, new_rows - 1);
        p_labels->shed_rows(0, new_rows - 1);
        batch_train(p_features, p_labels, {}, true);
        return samples_trained;
    }

    arma::uvec shed_ixs;
    {
        arma::mat abs_total_weights = arma::zeros(arma::size(main_components.front()->total_weights));
        for (auto &kv: main_components) abs_total_weights += arma::abs(kv->total_weights);
        const auto active_ixs = get_active_ixs();
        if (forget_ixs.empty()) {
            if (!tmp_ixs.empty()) {
                const auto temp_ct = std::min<size_t>(tmp_ixs.size(), new_rows) - 1;
                for (size_t i = 0; i < temp_ct; ++i)
                    shed_ixs.insert_rows(shed_ixs.empty() ? 0 : shed_ixs.n_rows, tmp_ixs^i);
                tmp_ixs.unsafe_erase(std::next(tmp_ixs.begin(), std::min(tmp_ixs.size() - 1, temp_ct)), tmp_ixs.end());
                LOG4_DEBUG("Forgetting temporary row at " << shed_ixs);
            }
#ifdef UPDATE_FEATURES // Enable to forget most similar features
            for (size_t r = 0; r < p_features->n_rows; ++r) {
                if (arma::approx_equal(p_features->row(r), new_x, "absdiff", TOL_ROWS)) {
                    shed_ixs = r;
                    LOG4_DEBUG("Found similar features row at " << r);
                }
            }
#endif
            while (shed_ixs.n_rows < new_rows) {
#ifdef FORGET_MIN_WEIGHT // Forgetting min weight works the best
                const size_t ix_to_shed = abs_total_weights.rows(active_ixs).index_min();
                shed_ixs.insert_rows(shed_ixs.empty() ? 0 : shed_ixs.n_rows - 1, active_ixs(ix_to_shed, 0));
#else // Forget oldest active index
                shed_ixs = active_ixs.min();
#endif
                LOG4_DEBUG("Forgetting least significant row at " << ix_to_shed);
            }
        } else {
            for (const auto f: forget_ixs) {
                arma::uvec contained_ixs = arma::find(abs_total_weights.rows(active_ixs) == f);
                if (contained_ixs.empty()) {
                    LOG4_WARN("Suggested forget index " << f << " not found in active indexes, aborting online train.");
                    return 0;
                } else {
                    LOG4_DEBUG("Using suggested index " << f);
                    shed_ixs.insert_rows(shed_ixs.n_rows, f);
                }
            }
        }
    }

    // Do the shedding
    p_features->shed_rows(shed_ixs);
    p_labels->shed_rows(shed_ixs);
    tbb::concurrent_map<size_t /* chunk */, size_t /* affected rows */> affected_chunks;
    tbb::concurrent_map<std::pair<size_t /* chunk */, size_t /* component */>, tbb::concurrent_map<size_t /* shedix */, arma::rowvec /* weights */>> chunk_shedded_weights;
#pragma omp parallel for
    for (size_t chunk_ix = 0; chunk_ix < ixs.size(); ++chunk_ix) {
        auto &chunk_ixs = ixs[chunk_ix];
        arma::uvec find_res = arma::sort(arma::unique(common::find(chunk_ixs, shed_ixs)));
        if (find_res.empty()) {
            LOG4_DEBUG("Shed index " << shed_ixs << " not found in chunk " << chunk_ix);
            size_t ct = 0;
            for (auto shed_ix: shed_ixs) {
                shed_ix -= ct++;
                chunk_ixs.rows(arma::find(chunk_ixs >= shed_ix)) -= 1;
            }
            continue;
        }

        size_t ct = 0;
        for (auto &shed_ix: find_res) {
            shed_ix -= ct++;

            chunk_ixs.shed_row(shed_ix);
            chunk_ixs.rows(arma::find(chunk_ixs >= shed_ix)) -= 1;
#pragma omp parallel for
            for (size_t com = 0; com < main_components.size(); ++com) {
                chunk_shedded_weights[{chunk_ix, com}][shed_ix] = main_components[com]->chunk_weights[chunk_ix].row(shed_ix);
                main_components[com]->chunk_weights[chunk_ix].shed_row(shed_ix);
            }

            if (p_kernel_matrices->at(chunk_ix).n_rows > shed_ix and p_kernel_matrices->at(chunk_ix).n_cols > shed_ix) {
                p_kernel_matrices->at(chunk_ix).shed_row(shed_ix);
                p_kernel_matrices->at(chunk_ix).shed_col(shed_ix);
            } else
                LOG4_ERROR("Failed removing kernel distances for index " << shed_ix << " from kernel matrix for chunk " << chunk_ix << " of size "
                                                                         << arma::size(p_kernel_matrices->at(chunk_ix)));
        }

        affected_chunks.emplace(chunk_ix, find_res.n_rows);
    }

    // Append new distances
    LOG4_DEBUG("Affected chunks " << affected_chunks.size());
#pragma omp parallel for
    for (size_t i = 0; i < affected_chunks.size(); ++i) {
        const auto chunk_ix = affected_chunks % i;
        const auto chunk_new_rows = affected_chunks ^ i;
        arma::mat &K_chunk = p_kernel_matrices->at(chunk_ix);
        arma::mat newnew_K, newold_K,
            chunk_x(new_x.rows(new_x.n_rows - chunk_new_rows, new_x.n_rows - 1)),
            chunk_y(new_y.rows(new_y.n_rows - chunk_new_rows, new_y.n_rows - 1));
#pragma omp parallel
        {
            PROFILE_EXEC_TIME(newnew_K = init_kernel_matrix(get_params(chunk_ix), new_x, new_y),
                              "Init kernel matrix for chunk " << chunk_ix);
            PROFILE_EXEC_TIME(newold_K = init_predict_kernel_matrix(
                    get_params(chunk_ix), p_features->rows(ixs[chunk_ix]), new_x, label_time),
                              "Init predict kernel matrix for chunk " << chunk_ix);
#pragma omp single
            K_chunk.resize(K_chunk.n_rows + chunk_new_rows, K_chunk.n_cols + chunk_new_rows);
#pragma omp barrier
            K_chunk.submat(0, K_chunk.n_cols - chunk_new_rows,
                           K_chunk.n_rows - chunk_new_rows - 1, K_chunk.n_cols - 1) = newold_K.t();
            K_chunk.submat(K_chunk.n_rows - chunk_new_rows, 0,
                           K_chunk.n_rows - 1, K_chunk.n_cols - chunk_new_rows - 1) = newold_K;
            K_chunk.submat(K_chunk.n_rows - chunk_new_rows, K_chunk.n_cols - chunk_new_rows,
                           K_chunk.n_rows - 1, K_chunk.n_cols - 1) = newnew_K;
        }

        ixs[chunk_ix].insert_rows(ixs[chunk_ix].n_rows, arma::linspace<arma::uvec>(ixs[chunk_ix].n_rows, ixs[chunk_ix].n_rows + chunk_new_rows - 1));
#ifdef KEEP_PREV_WEIGHTS
#pragma omp parallel for
        for (size_t com = 0; com < main_components.size(); ++com)
            for (const auto &w: chunk_shedded_weights[{chunk_ix, com}])
                main_components[com].chunk_weights[chunk_ix].insert_rows(main_components[i].chunk_weights[chunk_ix].n_rows, w.second);
#else
#pragma omp parallel for
        for (size_t com = 0; com < main_components.size(); ++com) {
            auto &w = main_components[com]->chunk_weights[chunk_ix];
            const double mean_weight = arma::mean(arma::vectorise(w));
            w.insert_rows(w.n_rows, arma::mat(chunk_new_rows, w.n_cols, arma::fill::value(mean_weight)));
        }
#endif
    }
    if (temp_learn)
        for (size_t row_i = p_labels->n_rows; row_i < p_labels->n_rows + new_rows; ++row_i)
            tmp_ixs.emplace(row_i);
    p_features->insert_rows(p_features->n_rows, new_x);
    p_labels->insert_rows(p_labels->n_rows, new_y);

    const arma::mat k_eye = arma::eye(arma::size(p_kernel_matrices->front()));
#pragma omp parallel for collapse(2)
    for (size_t i = 0; i < affected_chunks.size(); ++i)
        for (size_t j = 0; j < main_components.size(); ++j) {
            const auto chunk_ix = affected_chunks % i;
            auto &com = main_components[j];
            calc_weights(chunk_ix, p_labels->rows(ixs[chunk_ix]), k_eye / (2. * get_params_ptr(chunk_ix)->get_svr_C()), *com, IRWLS_ITER_ONLINE);
        }
    update_total_weights();
    samples_trained += new_rows;
    return new_rows;
}


void OnlineMIMOSVR::do_over_train_zero_epsilon(const arma::mat &xx_train, const arma::mat &yy_train, const arma::mat &k_eye, const size_t chunk_idx)
{
    const auto params = get_params(chunk_idx);
    const arma::mat x_train = xx_train.rows(ixs[chunk_idx]);
    const arma::mat y_train = yy_train.rows(ixs[chunk_idx]);
    if (p_kernel_matrices->at(chunk_idx).empty()) {
        PROFILE_EXEC_TIME(p_kernel_matrices->at(chunk_idx) = init_kernel_matrix(params, x_train, y_train), "Init kernel matrix for chunk " << chunk_idx);
    } else
        LOG4_DEBUG("Using pre-calculated kernel matrix for " << chunk_idx);

    const arma::mat k_eye_epsco = k_eye / (2. * params.get_svr_C());
    for (auto &component: main_components) {
        // The regression parameters
        component->chunk_weights[chunk_idx] = arma::ones(y_train.n_rows, y_train.n_cols);
        // Prediction error
        calc_weights(chunk_idx, y_train, k_eye_epsco, *component, IRWLS_ITER);
    }
}


void OnlineMIMOSVR::calc_weights(const size_t chunk_ix, const arma::mat &y_train, const arma::mat &k_eye_epsco, OnlineMIMOSVR::MimoBase &component, const size_t iters)
{
#ifndef MEASURE_INIT_WEIGHTS
    component.mae_chunk_values[chunk_ix] = ixs.size() < 2 ? 1. :
                                           common::meanabs<double>(y_train + component.epsilon - p_kernel_matrices->at(chunk_ix) * component.chunk_weights[chunk_ix]);
    LOG4_DEBUG("Initial MAE " << component.mae_chunk_values[chunk_ix] << " for chunk " << chunk_ix);
#endif

    solve_dispatch(k_eye_epsco, p_kernel_matrices->at(chunk_ix), y_train, component.chunk_weights[chunk_ix], iters, false /* DPOSV always fails! // p_kernel_matrices->at(chunk_ix).is_symmetric() && p_kernel_matrices->at(chunk_ix).min() >= 0 */);

    component.mae_chunk_values[chunk_ix] = ixs.size() < 2 ? 1. :
                                           common::meanabs<double>(y_train + component.epsilon - p_kernel_matrices->at(chunk_ix) * component.chunk_weights[chunk_ix]);
    LOG4_DEBUG("Final MAE " << component.mae_chunk_values[chunk_ix] << " for chunk " << chunk_ix);
}

void OnlineMIMOSVR::update_total_weights()
{
#pragma omp parallel for
    for (auto &comp: main_components) {
        comp->total_weights = comp->chunk_weights.front();
        arma::mat divisors = arma::ones(arma::size(*p_labels));
        for (size_t i = 1; i < ixs.size(); ++i) {
            divisors.rows(ixs[i]) += 1;
            comp->total_weights.rows(ixs[i]) += comp->chunk_weights[i];
        }
        comp->total_weights /= divisors;
        LOG4_TRACE("Weights for level " << get_params().get_decon_level() << " " << comp->total_weights);
    }

    tbb::concurrent_map<size_t /* comp */, double> component_avg_mae;
#pragma omp parallel for collapse(2)
    for (size_t i = 0; i < ixs.size(); ++i)
        for (size_t c = 0; c < main_components.size(); ++c)
            component_avg_mae[i] += main_components[c]->mae_chunk_values[i];
#pragma omp parallel for
    for (size_t c = 0; c < main_components.size(); ++c)
        component_avg_mae[c] /= double(ixs.size());

#pragma omp parallel for collapse(2)
    for (size_t i = 0; i < ixs.size(); ++i)
        for (size_t c = 0; c < main_components.size(); ++c)
            main_components[c]->chunks_weight[i] = component_avg_mae[c] / main_components[c]->mae_chunk_values[i];
}

} // svr
