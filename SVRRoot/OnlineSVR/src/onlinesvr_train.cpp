#include "onlinesvr.hpp"
#include "common/parallelism.hpp"
#include "model/Model.hpp"
#include "DQScalingFactorService.hpp"
#include "appcontext.hpp"
#include "util/math_utils.hpp"
#include "pprune.hpp"
#include <cstddef>
#include <magma_auxiliary.h>

namespace svr {
namespace datamodel {

void
OnlineMIMOSVR::batch_train(
        const mat_ptr &p_xtrain,
        const mat_ptr &p_ytrain,
        const vec_ptr &p_ylastknown,
        const bpt::ptime &last_value_time,
        const matrices_ptr &precalc_kernel_matrices)
{
    if (p_xtrain->n_rows != p_ytrain->n_rows || p_xtrain->empty() || p_ytrain->empty() || !p_xtrain->n_cols || !p_ytrain->n_cols || p_ytrain->n_rows != p_ylastknown->n_rows)
        LOG4_THROW("Invalid data dimensions, features " << arma::size(*p_xtrain) << ", labels " << arma::size(*p_ytrain) <<
                                                        ", last knowns " << arma::size(*p_ylastknown) << ", level " << level);

    LOG4_DEBUG(
            "Training on features " << common::present(*p_xtrain) << ", labels " << common::present(*p_ytrain) << ", last-knowns " << common::present(*p_ylastknown)
            << ", pre-calculated kernel matrices " << (precalc_kernel_matrices ? precalc_kernel_matrices->size() : 0) << ", parameters " << **param_set.cbegin() <<
            ", last value time " << last_value_time);
    reset();

    p_features = p_xtrain;
    p_labels = p_ytrain;
    p_last_knowns = p_ylastknown;

    {
        datamodel::SVRParameters_ptr p;
        if ((p = is_manifold())) {
            init_manifold(p, last_value_time);
            samples_trained = std::min<size_t>(p->get_svr_decremental_distance(), p_xtrain->n_rows);
            return;
        }
    }

    bool tuned = false;
    if (needs_tuning() && !is_manifold()) {
        if (precalc_kernel_matrices && precalc_kernel_matrices->size())
            LOG4_WARN("Provided kernel matrices will be ignored because SVR parameters are not initialized.");
        PROFILE_EXEC_TIME(tune_fast(), "Tune kernel parameters for level " << level << ", step " << step << ", gradient " << (**param_set.cbegin()).get_grad_level());
#ifdef SAVE_PARAMETERS
        if (model_id) {
            for (const auto &p: param_set) {
                if (APP.svr_parameters_service.exists(p))
                    APP.svr_parameters_service.remove(p);
                APP.svr_parameters_service.save(p);
            }
            for (const auto &sf: scaling_factors) {
                if (APP.dq_scaling_factor_service.exists(sf))
                    APP.dq_scaling_factor_service.remove(sf);
                APP.dq_scaling_factor_service.save(sf);
            }
        }
#endif
        tuned = true;
    }

    const unsigned num_chunks = ixs.size();
    if (!tuned && precalc_kernel_matrices && precalc_kernel_matrices->size() == num_chunks) {
        p_kernel_matrices = precalc_kernel_matrices;
        LOG4_DEBUG("Using " << num_chunks << " precalculated matrices.");
    } else if (precalc_kernel_matrices && !precalc_kernel_matrices->empty()) {
        LOG4_ERROR("Precalculated kernel matrices do not match needed chunks count!");
    } else {
        LOG4_DEBUG("Initializing kernel matrices from scratch.");
    }

    if (!tuned) {
        if (weight_chunks.size() != num_chunks) weight_chunks.resize(num_chunks);
        if (p_kernel_matrices->size() != num_chunks) p_kernel_matrices->resize(num_chunks);
#pragma omp parallel for schedule(static, 1) num_threads(adj_threads(num_chunks))
        for (unsigned i = 0; i < num_chunks; ++i) {
            const auto p_params = get_params_ptr(i);

            const auto chunk_sf = business::DQScalingFactorService::slice(scaling_factors, i, gradient, step);
            business::DQScalingFactorService::scale_features(i, gradient, step, p_params->get_lag_count(), chunk_sf, train_feature_chunks_t[i]);
            const auto p_sf = business::DQScalingFactorService::find(chunk_sf, model_id, i, gradient, step, level, false, true);
            business::DQScalingFactorService::scale_labels(*p_sf, train_label_chunks[i]);

            if (p_kernel_matrices->at(i).empty()) p_kernel_matrices->at(i) = *prepare_K(ccache(), *get_params_ptr(i), train_label_chunks[i], last_value_time);
            else LOG4_DEBUG("Using pre-calculated kernel matrix for " << i);

            calc_weights(i, PROPS.get_stabilize_iterations_count());
        }
    }
    update_all_weights();
    samples_trained = p_features->n_rows;
    last_trained_time = last_value_time;
}


void OnlineMIMOSVR::learn(
    const arma::mat &new_x, const arma::mat &new_y, const arma::vec &new_ylk, const bpt::ptime &last_value_time,
    const bool temp_learn, const std::deque<size_t> &forget_ixs)
{
    if (new_x.empty() || new_y.empty() || new_x.n_cols != p_features->n_cols || new_y.n_cols != p_labels->n_cols || new_ylk.n_rows != new_y.n_rows || new_x.n_rows != new_y.n_rows)
        LOG4_THROW("New data dimensions labels " << arma::size(new_y) << ", last-knowns " << arma::size(new_ylk) << ", features " << arma::size(new_x) <<
            " not sane or do not match model data dimensions labels " << arma::size(*p_labels) << ", features " << arma::size(*p_features) << ", last-knowns " << arma::size(*p_last_knowns));
    if (p_features->n_rows == samples_trained) { // First call to online learn copy batch data
        p_features = otr(*p_features);
        p_labels = otr(*p_labels);
        p_last_knowns = otr(*p_last_knowns);
    }
    const auto new_rows_ct = new_x.n_rows;

    if (is_manifold()) {
        if (!p_manifold) LOG4_THROW("Manifold not initialized.");
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<size_t> rand_int(0, C_interlace_manifold_factor);
        const auto new_manifold_rows_ct = new_x.n_rows * p_features->n_rows / C_interlace_manifold_factor;
        arma::mat new_x_manifold(new_manifold_rows_ct, new_x.n_cols + p_features->n_cols);
        arma::mat new_y_manifold(new_manifold_rows_ct, new_y.n_cols);
        arma::mat new_ylk_manifold(new_manifold_rows_ct, 1);
#pragma omp parallel for num_threads(adj_threads(p_features->n_rows * new_x.n_rows)) collapse(2)
        for (size_t i = 0; i < p_features->n_rows; ++i)
            for (size_t j = 0; j < new_x.n_rows; ++j) {
                const auto r = i * new_x.n_rows + j;
                if (r % C_interlace_manifold_factor == 0) {
                    const auto r_i = common::bounce<size_t>(new_x.n_rows - 1, i + rand_int(gen));
                    const auto r_j = common::bounce<size_t>(p_features->n_rows - 1, j + rand_int(gen));
                    MANIFOLD_SET(new_x_manifold.row(r / C_interlace_manifold_factor), new_y_manifold.row(r / C_interlace_manifold_factor),
                                 new_x.row(r_j), new_y.row(r_j), p_features->row(r_i), p_labels->row(r_i));
                }
            }
        samples_trained += new_rows_ct;
        p_manifold->learn(new_x_manifold, new_y_manifold, new_ylk_manifold, last_value_time);
        p_features->shed_rows(0, new_rows_ct - 1);
        p_labels->shed_rows(0, new_rows_ct - 1);
        p_last_knowns->shed_rows(0, new_rows_ct - 1);
        p_features->insert_rows(p_features->n_rows, new_x);
        p_labels->insert_rows(p_labels->n_rows, new_y);
        p_last_knowns->insert_rows(p_last_knowns->n_rows, new_ylk);
        return;
    }

    if (!forget_ixs.empty() && new_rows_ct != forget_ixs.size())
        LOG4_WARN("Forget index size " << forget_ixs.size() << " does not equal new train samples count " << new_rows_ct);

    if (new_rows_ct > ixs.front().size() / 2) {
        p_features->shed_rows(0, new_rows_ct - 1);
        p_labels->shed_rows(0, new_rows_ct - 1);
        p_last_knowns->shed_rows(0, new_rows_ct - 1);
        p_features->insert_rows(p_features->n_rows, new_x);
        p_labels->insert_rows(p_labels->n_rows, new_y);
        p_last_knowns->insert_rows(p_last_knowns->n_rows, new_ylk);
        const auto backup_samples_trained = samples_trained + new_rows_ct;
        batch_train(p_features, p_labels, p_last_knowns, last_value_time);
        samples_trained = backup_samples_trained;
        return;
    }

    arma::uvec shed_ixs; // Active indexes to be shedded to make space for new learning data
    auto active_rows = get_active_ixs();
    if (!forget_ixs.empty()) {
        t_omp_lock learn_lk;
#pragma omp parallel for num_threads(adj_threads(forget_ixs.size())) schedule(static, 1)
        for (const auto f: forget_ixs) {
            arma::uvec contained_ixs = arma::find(active_rows == f);
            if (contained_ixs.empty()) {
                const auto oldest_ix = active_rows.index_min();
                LOG4_ERROR("Suggested forget index " << f << " not found in active indexes, using oldest active index " << oldest_ix << ", " << active_rows[oldest_ix] << " instead.");
                learn_lk.set();
                shed_ixs.insert_rows(shed_ixs.n_rows, active_rows[oldest_ix]);
                learn_lk.unset();
                active_rows.shed_row(oldest_ix);
            } else {
                LOG4_DEBUG("Using suggested index " << f);
                learn_lk.set();
                shed_ixs.insert_rows(shed_ixs.n_rows, f);
                learn_lk.unset();
            }
        }
    }
    if (!tmp_ixs.empty()) {
        size_t i = 0;
        while (i < tmp_ixs.size() && shed_ixs.n_rows < new_rows_ct)
            shed_ixs.insert_rows(shed_ixs.n_rows, tmp_ixs ^ i++);
        tmp_ixs.erase(std::next(tmp_ixs.begin(), i), tmp_ixs.end());
        LOG4_DEBUG("Forgetting temporary rows " << shed_ixs.rows(shed_ixs.n_rows - i, shed_ixs.n_rows - 1));
    }
    if (shed_ixs.n_rows < new_rows_ct) {
        arma::vec abs_total_weights = arma::sum(arma::abs(all_weights.rows(active_rows)), 1);
        const auto mean_abs_total_weights = arma::mean(abs_total_weights);
        while (shed_ixs.n_rows < new_rows_ct) {
#ifdef FORGET_MIN_WEIGHT // Forgetting min weight works the best
            const auto ix_to_shed = abs_total_weights.index_min();
#else // Forget oldest active index
            const auto ix_to_shed = active_ixs.min_index();
#endif
            LOG4_DEBUG("Forgetting least significant row at " << active_rows[ix_to_shed] << " weighting "
                                                              << 100. * abs_total_weights[active_rows[ix_to_shed]] / mean_abs_total_weights << "pc");
            abs_total_weights.shed_row(active_rows[ix_to_shed]);
            shed_ixs.insert_rows(shed_ixs.n_rows, active_rows[ix_to_shed]);
            active_rows.shed_row(ix_to_shed);
        }
    }

    // Do the shedding
    p_features->shed_rows(shed_ixs);
    p_labels->shed_rows(shed_ixs);
    p_last_knowns->shed_rows(shed_ixs);
    tbb::concurrent_map<size_t /* chunk */, size_t /* affected rows */> affected_chunks;
#ifdef KEEP_PREV_WEIGHTS
    std::deque<tbb::concurrent_map<size_t /* shedix */, arma::rowvec /* weights */>> chunk_shedded_weights(ixs.size());
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
                const auto find_res2 = arma::find(chunk_ixs >= shed_ix);
                if (find_res.empty()) continue;
                chunk_ixs.rows(find_res2) -= 1;
            }
            continue;
        }

        size_t ct = 0;
        for (auto shed_ix: find_res) {
            shed_ix -= ct++;
            chunk_ixs.shed_row(shed_ix);
            chunk_ixs.rows(arma::find(chunk_ixs >= shed_ix)) -= 1;

#ifdef KEEP_PREV_WEIGHTS
            chunk_shedded_weights[chunk_ix][shed_ix] = chunk_weights[chunk_ix].row(shed_ix);
#endif
            weight_chunks[chunk_ix].shed_row(shed_ix);
            train_label_chunks[chunk_ix].shed_row(shed_ix);
            train_feature_chunks_t[chunk_ix].shed_col(shed_ix);

            if (p_kernel_matrices->at(chunk_ix).n_rows > shed_ix && p_kernel_matrices->at(chunk_ix).n_cols > shed_ix) {
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
    std::atomic<size_t> add_new_row_ix{0};
#pragma omp parallel for num_threads(adj_threads(affected_chunks.size())) schedule(static, 1)
    for (size_t i = 0; i < affected_chunks.size(); ++i) {
        const auto chunk_ix = affected_chunks % i;
        const auto chunk_new_rows = affected_chunks ^ i;
        auto &params = *get_params_ptr(chunk_ix);
        auto &K_chunk = p_kernel_matrices->at(chunk_ix);
        arma::uvec new_ixs(chunk_new_rows);
        for (size_t r = 0; r < new_ixs.n_rows; ++r)
            new_ixs[r] = add_new_row_ix++ % new_x.n_rows;
        arma::mat new_chunk_features_t = new_x.rows(new_ixs).t();
        business::DQScalingFactorService::scale_features(chunk_ix, *this, new_chunk_features_t);
        K_chunk.resize(K_chunk.n_rows + chunk_new_rows, K_chunk.n_cols + chunk_new_rows);
#pragma omp parallel num_threads(adj_threads(2))
#pragma omp single
        {
#pragma omp task
            K_chunk.submat(K_chunk.n_rows - chunk_new_rows, K_chunk.n_cols - chunk_new_rows, K_chunk.n_rows - 1, K_chunk.n_cols - 1) = *prepare_K(params, new_chunk_features_t);
#pragma omp task
            {
                arma::mat newold_K;
                PROFILE_EXEC_TIME(newold_K = *prepare_Ky(params, train_feature_chunks_t[chunk_ix], new_chunk_features_t), "Init predict kernel matrix for chunk " << chunk_ix);
                K_chunk.submat(0, K_chunk.n_cols - chunk_new_rows, K_chunk.n_rows - chunk_new_rows - 1, K_chunk.n_cols - 1) = newold_K.t();
                K_chunk.submat(K_chunk.n_rows - chunk_new_rows, 0, K_chunk.n_rows - 1, K_chunk.n_cols - chunk_new_rows - 1) = newold_K;
            }
        }
        const auto epsco = calc_epsco(K_chunk);
        params.set_svr_C(1 / epsco);
        K_chunk.diag() += epsco;

        ixs[chunk_ix].insert_rows(ixs[chunk_ix].n_rows, p_labels->n_rows + new_ixs);
        train_feature_chunks_t[chunk_ix].insert_cols(train_feature_chunks_t[chunk_ix].n_cols, new_chunk_features_t);
        arma::mat new_chunk_labels = new_y.rows(new_ixs);
        business::DQScalingFactorService::scale_labels(chunk_ix, *this, new_chunk_labels);
        train_label_chunks[chunk_ix].insert_rows(train_label_chunks[chunk_ix].n_rows, new_chunk_labels);
#ifdef KEEP_PREV_WEIGHTS
#pragma omp parallel for schedule(static, 1) num_threads(adj_threads(main_components.size()))
            for (const auto &w: chunk_shedded_weights[chunk_ix])
                chunk_weights[chunk_ix].insert_rows(chunk_weights[chunk_ix].n_rows, w.second);
#else
        auto &w = weight_chunks[chunk_ix];
        w.insert_rows(w.n_rows, arma::mat(chunk_new_rows, w.n_cols, arma::fill::value(arma::mean(arma::vectorise(w)))));
#endif
    }
    if (temp_learn)
        for (size_t row_i = p_labels->n_rows; row_i < p_labels->n_rows + new_rows_ct; ++row_i)
            tmp_ixs.emplace(row_i);
    p_features->insert_rows(p_features->n_rows, new_x);
    p_labels->insert_rows(p_labels->n_rows, new_y);
    p_last_knowns->insert_rows(p_last_knowns->n_rows, new_ylk);

    OMP_FOR_i(affected_chunks.size()) calc_weights(affected_chunks % i, PROPS.get_online_learn_iter_limit());
    update_all_weights();
    samples_trained += new_rows_ct;
    last_trained_time = last_value_time;
}


void OnlineMIMOSVR::calc_weights(const size_t chunk_ix, const size_t iters)
{
    const auto p_params = get_params_ptr(chunk_ix);
    arma::mat K_epsco = p_kernel_matrices->at(chunk_ix);
    K_epsco.diag().fill(1. / p_params->get_svr_C());
    PROFILE_EXEC_TIME(solve_irwls(K_epsco, p_kernel_matrices->at(chunk_ix), train_label_chunks[chunk_ix], weight_chunks[chunk_ix], iters),
                        "Solve weights for chunk " << chunk_ix << ", level " << level << ", iters " << iters);
    PROFILE_EXEC_TIME(solve_opt(K_epsco, p_kernel_matrices->at(chunk_ix), train_label_chunks[chunk_ix], weight_chunks[chunk_ix], iters),
                      "Solve weights for chunk " << chunk_ix << ", level " << level);
    LOG4_TRACE("Chunk " << chunk_ix << ", level " << level << ", gradient " << gradient << ", parameters " << *p_params << ", K epsco " << common::present(K_epsco)
                        << ", w " << common::present(weight_chunks[chunk_ix]));
}


void OnlineMIMOSVR::update_all_weights()
{
    all_weights.zeros(arma::size(*p_labels));
    all_weights.rows(ixs.front()) = weight_chunks.front();
    arma::mat divisors(arma::size(*p_labels), arma::fill::ones);
    for (size_t i = 1; i < ixs.size(); ++i) {
        all_weights.rows(ixs[i]) += weight_chunks[i];
        divisors.rows(ixs[i]) += 1;
    }
    all_weights /= divisors;
    LOG4_TRACE("Total weights for level " << level << " " << arma::size(all_weights));
}

} // datamodel
} // svr
