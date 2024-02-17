#include "onlinesvr.hpp"

#include "common/parallelism.hpp"
#include "model/Model.hpp"
#include "SVRParametersService.hpp"
#include "DQScalingFactorService.hpp"
#include "appcontext.hpp"

namespace svr {

void
OnlineMIMOSVR::batch_train(
        const matrix_ptr &p_xtrain,
        const matrix_ptr &p_ytrain,
        const matrices_ptr &kernel_matrices)
{
    LOG4_DEBUG(
            "Training on X " << common::present(*p_xtrain) << ", Y " << common::present(*p_ytrain) << ", pre-calculated kernel matrices " <<
                  (kernel_matrices ? kernel_matrices->size() : 0) << ", parameters " << get_params());
    const std::scoped_lock lk(learn_mx);

    reset_model();

    if (needs_tuning() && !is_manifold()) {
        if (kernel_matrices) {
            LOG4_ERROR("Kernel matrices provided will be cleared because SVR kernel parameters are not initialized!");
            kernel_matrices->clear();
        }
        t_gradient_tuned_parameters params_predictions;
        PROFILE_EXEC_TIME(tune(params_predictions, *p_param_set, *p_xtrain, *p_ytrain, *p_ytrain, max_chunk_size),
                          "Tune kernel p_head_params for model " << decon_level);
        p_param_set = business::SVRParametersService::get_best_params(params_predictions);
    }
    p_features = p_xtrain;
    p_labels = p_ytrain;
    if (p_xtrain->n_rows != p_ytrain->n_rows || p_xtrain->empty() || p_ytrain->empty() || !p_xtrain->n_cols || !p_ytrain->n_cols)
        LOG4_THROW("Invalid data dimensions X train " << arma::size(*p_xtrain) << ", Y train " << arma::size(*p_ytrain) << ", level " << decon_level);

    datamodel::SVRParameters_ptr p;
    if (is_manifold(p)) {
        init_manifold(p);
        samples_trained = p_xtrain->n_rows;
        return;
    }
    const auto decrement = p_param_set->begin()->get()->get_svr_decremental_distance();
    if (p_xtrain->n_rows > decrement) p_xtrain->shed_rows(0, p_xtrain->n_rows - decrement - 1);
    if (p_ytrain->n_rows > decrement) p_ytrain->shed_rows(0, p_ytrain->n_rows - decrement - 1);

    if (is_gradient() && labels_scaling_factor != 1)
        labels_scaling_factor = business::DQScalingFactorService::calc_scaling_factor(*p_labels);

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
        if (component->weights_mask.size() != ixs.size()) component->weights_mask.resize(ixs.size());
        if (component->mae_chunk_values.size() != ixs.size()) component->mae_chunk_values.resize(ixs.size(), 0.);
    }

    if (p_kernel_matrices->size() != ixs.size()) p_kernel_matrices->resize(ixs.size());
    const arma::mat eye_K = arma::eye(ixs.front().n_rows, ixs.front().n_rows);
#pragma omp parallel for schedule(static, 1) num_threads(adj_threads(get_num_chunks()))
    for (size_t i = 0; i < get_num_chunks(); ++i)
        do_over_train_zero_epsilon(*p_features, *p_labels, eye_K, i);
    update_total_weights();
    samples_trained = p_features->n_rows;
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
    if (is_manifold()) {
        if (!p_manifold) LOG4_THROW("Manifold not initialized.");
        const auto new_learning_rows = new_x.n_rows * p_features->n_rows / C_interlace_manifold_factor;
        arma::mat new_x_manifold(new_learning_rows, new_x.n_cols + p_features->n_cols);
        arma::mat new_y_manifold(new_learning_rows, new_y.n_cols);
#pragma omp parallel for collapse(2) num_threads(adj_threads(p_features->n_rows * new_x.n_rows))
        for (size_t i = 0; i < p_features->n_rows; ++i)
            for (size_t j = 0; j < new_x.n_rows; ++j) {
                const auto r = i * new_x.n_rows + j;
                if (r % C_interlace_manifold_factor) continue;
                const auto rand_add = std::rand() % C_interlace_manifold_factor;
                const size_t r_i = std::min<size_t>(new_x.n_rows - 1, i + rand_add);
                const size_t r_j = std::min<size_t>(p_features->n_rows - 1, j + rand_add);
                MANIFOLD_SET(new_x_manifold.row(r / C_interlace_manifold_factor),
                             new_y_manifold.row(r / C_interlace_manifold_factor),
                             new_x.row(r_j), new_y.row(r_j),
                             p_features->row(r_i), p_labels->row(r_i));
            }
        samples_trained += new_learning_rows;
        const auto res = p_manifold->learn(new_x_manifold, new_y_manifold);
        p_features->insert_rows(p_features->n_rows, new_x);
        p_labels->insert_rows(p_labels->n_rows, new_x);
        return res;
    }

    const auto new_rows = new_x.n_rows;
    if (!forget_ixs.empty() && new_rows != forget_ixs.size())
        LOG4_WARN("Forget index size " << forget_ixs.size() << " does not equal train samples count " << new_rows);

    if (new_rows > ixs.front().size()) {
        p_features->insert_rows(p_features->n_rows, new_x);
        p_labels->insert_rows(p_labels->n_rows, new_y);
        p_features->shed_rows(0, new_rows - 1);
        p_labels->shed_rows(0, new_rows - 1);
        const auto backup_samples_trained = samples_trained + new_rows;
        batch_train(matrix_ptr{p_features}, matrix_ptr{p_labels});
        samples_trained = backup_samples_trained;
        return samples_trained;
    }

    arma::uvec shed_ixs; // Active indexes to be shedded for the purpose of making space for new learning data
    auto active_ixs = get_active_ixs();
    if (!forget_ixs.empty()) {
#pragma omp parallel for num_threads(adj_threads(forget_ixs.size())) schedule(static, 1)
        for (const auto f: forget_ixs) {
            arma::uvec contained_ixs = arma::find(active_ixs == f);
            if (contained_ixs.empty()) {
                const auto oldest_ix = active_ixs.index_min();
                LOG4_ERROR("Suggested forget index " << f << " not found in active indexes, using oldest active index " << oldest_ix << " " << active_ixs[oldest_ix] << " instead.");
#pragma omp critical
                shed_ixs.insert_rows(shed_ixs.n_rows, active_ixs[oldest_ix]);
                active_ixs.shed_row(oldest_ix);
            } else {
                LOG4_DEBUG("Using suggested index " << f);
#pragma omp critical
                shed_ixs.insert_rows(shed_ixs.n_rows, f);
            }
        }
    }
    if (!tmp_ixs.empty()) {
        size_t i = 0;
        while (i < tmp_ixs.size() && shed_ixs.n_rows < new_rows)
            shed_ixs.insert_rows(shed_ixs.n_rows, tmp_ixs ^ i++);
        tmp_ixs.unsafe_erase(std::next(tmp_ixs.begin(), i), tmp_ixs.end());
        LOG4_DEBUG("Forgetting temporary rows " << shed_ixs.rows(shed_ixs.n_rows - i, shed_ixs.n_rows - 1));
    }
    if (shed_ixs.n_rows < new_rows) {
        arma::mat abs_total_weights = arma::zeros(arma::size(main_components.front()->total_weights));
        for (auto &kv: main_components) abs_total_weights += arma::abs(kv->total_weights);
        abs_total_weights = abs_total_weights.rows(active_ixs);
        const auto mean_abs_total_weights = arma::mean(abs_total_weights);
        while (shed_ixs.n_rows < new_rows) {
#ifdef FORGET_MIN_WEIGHT // Forgetting min weight works the best
            const auto ix_to_shed = abs_total_weights.index_min();
#else // Forget oldest active index
            const auto ix_to_shed = active_ixs.min_index();
#endif
            LOG4_DEBUG("Forgetting least significant row at " << active_ixs[ix_to_shed] << " weighting "
                                                              << 100. * abs_total_weights[ix_to_shed] / mean_abs_total_weights << " pct.");
            abs_total_weights.shed_row(ix_to_shed);
            shed_ixs.insert_rows(shed_ixs.n_rows, active_ixs[ix_to_shed]);
            active_ixs.shed_row(ix_to_shed);
        }
    }

    // Do the shedding
    p_features->shed_rows(shed_ixs);
    p_labels->shed_rows(shed_ixs);
    tbb::concurrent_map<size_t /* chunk */, size_t /* affected rows */> affected_chunks;
#ifdef KEEP_PREV_WEIGHTS
    tbb::concurrent_map<std::pair<size_t /* chunk */, size_t /* component */>, tbb::concurrent_map<size_t /* shedix */, arma::rowvec /* weights */>> chunk_shedded_weights;
#endif
#pragma omp parallel for schedule(static, 1) num_threads(adj_threads(ixs.size()))
    for (size_t chunk_ix = 0; chunk_ix < ixs.size(); ++chunk_ix) {
        auto &chunk_ixs = ixs[chunk_ix];
        arma::uvec find_res = arma::sort(arma::unique(common::find(chunk_ixs, shed_ixs)));
        if (find_res.empty()) {
            LOG4_DEBUG("Shed indexes " << shed_ixs << " not found in chunk " << chunk_ix);
            size_t ct = 0;
            for (auto shed_ix: shed_ixs) { // Bump indexes down for rows above the shedded
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

#pragma omp parallel for num_threads(adj_threads(main_components.size())) schedule(static, 1)
            for (size_t com = 0; com < main_components.size(); ++com) {
#ifdef KEEP_PREV_WEIGHTS
                chunk_shedded_weights[{chunk_ix, com}][shed_ix] = main_components[com]->chunk_weights[chunk_ix].row(shed_ix);
#endif
                main_components[com]->chunk_weights[chunk_ix].shed_row(shed_ix);
            }

            if (p_kernel_matrices->at(chunk_ix).n_rows > shed_ix and p_kernel_matrices->at(chunk_ix).n_cols > shed_ix) {
                p_kernel_matrices->at(chunk_ix).shed_row(shed_ix);
                p_kernel_matrices->at(chunk_ix).shed_col(shed_ix);
            } else
                LOG4_ERROR("Kernel matrices corrupt, failed removing kernel distances for index " << shed_ix << " from kernel matrix for chunk "
                            << chunk_ix << " of size " << arma::size(p_kernel_matrices->at(chunk_ix)));
        }

        affected_chunks.emplace(chunk_ix, find_res.n_rows);
    }

    // Append new distances
    LOG4_DEBUG("Affected chunks " << affected_chunks.size());
#pragma omp parallel for num_threads(adj_threads(affected_chunks.size())) schedule(static, 1)
    for (size_t i = 0; i < affected_chunks.size(); ++i) {
        const auto chunk_ix = affected_chunks % i;
        const auto chunk_new_rows = affected_chunks ^ i;
        arma::mat &K_chunk = p_kernel_matrices->at(chunk_ix);
        arma::mat newnew_K, newold_K,
            chunk_x(new_x.rows(new_x.n_rows - chunk_new_rows, new_x.n_rows - 1)),
            chunk_y(new_y.rows(new_y.n_rows - chunk_new_rows, new_y.n_rows - 1));
#pragma omp parallel num_threads(adj_threads(2))
#pragma omp single
        {
#pragma omp task
            PROFILE_EXEC_TIME(newnew_K = init_kernel_matrix(get_params(chunk_ix), chunk_x),
                              "Init kernel matrix for chunk " << chunk_ix);
#pragma omp task
            PROFILE_EXEC_TIME(newold_K = init_predict_kernel_matrix(
                    get_params(chunk_ix), p_features->rows(ixs[chunk_ix]), chunk_x, label_time),
                              "Init predict kernel matrix for chunk " << chunk_ix);

            K_chunk.resize(K_chunk.n_rows + chunk_new_rows, K_chunk.n_cols + chunk_new_rows);
#pragma omp taskwait
#pragma omp task
            K_chunk.submat(0, K_chunk.n_cols - chunk_new_rows, K_chunk.n_rows - chunk_new_rows - 1, K_chunk.n_cols - 1) = newold_K.t();
#pragma omp task
            K_chunk.submat(K_chunk.n_rows - chunk_new_rows, 0, K_chunk.n_rows - 1, K_chunk.n_cols - chunk_new_rows - 1) = newold_K;

            K_chunk.submat(K_chunk.n_rows - chunk_new_rows, K_chunk.n_cols - chunk_new_rows, K_chunk.n_rows - 1, K_chunk.n_cols - 1) = newnew_K;
        }
        auto chunk_p = get_params(chunk_ix);
        chunk_p.set_svr_C(1. / calc_epsco(K_chunk, std::min<size_t>(K_chunk.n_rows, chunk_p.get_svr_decremental_distance())));

        ixs[chunk_ix].insert_rows(ixs[chunk_ix].n_rows, arma::regspace<arma::uvec>(ixs[chunk_ix].n_rows, ixs[chunk_ix].n_rows + chunk_new_rows - 1));
#ifdef KEEP_PREV_WEIGHTS
#pragma omp parallel for schedule(static, 1) num_threads(adj_threads(main_components.size()))
        for (size_t com = 0; com < main_components.size(); ++com)
            for (const auto &w: chunk_shedded_weights[{chunk_ix, com}])
                main_components[com].chunk_weights[chunk_ix].insert_rows(main_components[i].chunk_weights[chunk_ix].n_rows, w.second);
#else
#pragma omp parallel for schedule(static, 1) num_threads(adj_threads(main_components.size()))
        for (auto &com: main_components) {
            auto &w = com->chunk_weights[chunk_ix];
            w.insert_rows(w.n_rows, arma::mat(chunk_new_rows, w.n_cols, arma::fill::value(arma::mean(arma::vectorise(w)))));
        }
#endif
    }
    if (temp_learn)
        for (size_t row_i = p_labels->n_rows; row_i < p_labels->n_rows + new_rows; ++row_i)
            tmp_ixs.emplace(row_i);
    p_features->insert_rows(p_features->n_rows, new_x);
    p_labels->insert_rows(p_labels->n_rows, new_y);

    const arma::mat k_eye = arma::eye(arma::size(p_kernel_matrices->front()));
#pragma omp parallel for collapse(2) schedule(static, 1) num_threads(adj_threads(affected_chunks.size() * main_components.size()))
    for (size_t i = 0; i < affected_chunks.size(); ++i)
        for (const auto &com: main_components) {
            const auto chunk_ix = affected_chunks % i;
            calc_weights(chunk_ix, p_labels->rows(ixs[chunk_ix]), k_eye / (2. * get_params_ptr(chunk_ix)->get_svr_C()), *com, PROPS.get_online_learn_iter_limit());
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
        PROFILE_EXEC_TIME(p_kernel_matrices->at(chunk_idx) = init_kernel_matrix(params, x_train), "Init kernel matrix for chunk " << chunk_idx);
    } else
        LOG4_DEBUG("Using pre-calculated kernel matrix for " << chunk_idx);

    const arma::mat k_eye_epsco = k_eye / (2. * params.get_svr_C());
#pragma omp parallel for num_threads(adj_threads(main_components.size())) schedule(static, 1)
    for (auto &component: main_components) {
        component->chunk_weights[chunk_idx] = y_train;
        calc_weights(chunk_idx, y_train, k_eye_epsco, *component, PROPS.get_stabilize_iterations_count());
    }
}


void OnlineMIMOSVR::calc_weights(const size_t chunk_ix, const arma::mat &y_train, const arma::mat &k_eye_epsco, MimoBase &component, const size_t iters)
{
    auto &chunk_weights = component.chunk_weights[chunk_ix];
    solve_dispatch(k_eye_epsco, p_kernel_matrices->at(chunk_ix), y_train, chunk_weights, iters, false /* cuz DPOSV always fails! */); // p_kernel_matrices->at(chunk_ix).is_symmetric() && p_kernel_matrices->at(chunk_ix).min() >= 0
    component.mae_chunk_values[chunk_ix] = ixs.size() < 2 ? 1. : common::meanabs<double>(y_train + component.epsilon - p_kernel_matrices->at(chunk_ix) * chunk_weights);
    LOG4_DEBUG("Final MAE " << component.mae_chunk_values[chunk_ix] << " for chunk " << chunk_ix);
    component.weights_mask[chunk_ix] = arma::mat(arma::size(chunk_weights), arma::fill::ones);
    component.weights_mask[chunk_ix].elem(arma::find(arma::abs(chunk_weights) < C_mean_weight_threshold * arma::mean(arma::abs(arma::vectorise(chunk_weights))))).zeros();
}

void OnlineMIMOSVR::update_total_weights()
{
#pragma omp parallel for num_threads(adj_threads(main_components.size()))
    for (auto &comp: main_components) {
        comp->total_weights.set_size(arma::size(*p_labels));
        comp->total_weights.fill(0);
        comp->total_weights.rows(ixs.front()) += comp->chunk_weights.front();
        arma::mat divisors(arma::size(*p_labels), arma::fill::ones);
        for (size_t i = 1; i < ixs.size(); ++i) {
            comp->total_weights.rows(ixs[i]) += comp->chunk_weights[i];
            divisors.rows(ixs[i]) += 1;
        }
        comp->total_weights /= divisors;
        LOG4_TRACE("Total weights for level " << decon_level << " " << arma::size(comp->total_weights));
    }

    tbb::concurrent_map<size_t /* comp */, double> component_avg_mae;
#pragma omp parallel for collapse(2) num_threads(adj_threads(ixs.size() * main_components.size()))
    for (size_t i = 0; i < ixs.size(); ++i)
        for (size_t c = 0; c < main_components.size(); ++c)
            component_avg_mae[i] += main_components[c]->mae_chunk_values[i];
#pragma omp parallel for num_threads(adj_threads(main_components.size()))
    for (size_t c = 0; c < main_components.size(); ++c)
        component_avg_mae[c] /= double(ixs.size());
#pragma omp parallel for collapse(2) num_threads(adj_threads(ixs.size() * main_components.size()))
    for (size_t i = 0; i < ixs.size(); ++i)
        for (size_t c = 0; c < main_components.size(); ++c)
            main_components[c]->chunks_weight[i] = component_avg_mae[c] / main_components[c]->mae_chunk_values[i];
}

} // svr
