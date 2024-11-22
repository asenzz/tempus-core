#include "onlinesvr.hpp"
#include "common/parallelism.hpp"
#include "model/Model.hpp"
#include "DQScalingFactorService.hpp"
#include "appcontext.hpp"
#include "util/math_utils.hpp"
#include "pprune.hpp"
#include <cstddef>
#include <magma_auxiliary.h>
#include <xoshiro.h>

namespace svr {
namespace datamodel {

void
OnlineMIMOSVR::batch_train(const mat_ptr &p_xtrain, const mat_ptr &p_ytrain, const vec_ptr &p_ylastknown, const mat_ptr &p_input_weights_, const bpt::ptime &last_value_time,
                           const matrices_ptr &precalc_kernel_matrices)
{
    if (p_xtrain->n_rows != p_ytrain->n_rows || p_ytrain->n_rows != p_ylastknown->n_rows || p_input_weights_->n_rows != p_ytrain->n_rows || p_xtrain->empty() || p_ytrain->empty())
        LOG4_THROW("Invalid data dimensions, features " << arma::size(*p_xtrain) << ", labels " << arma::size(*p_ytrain) << ", last knowns " << arma::size(*p_ylastknown) <<
                                                        ", instance weights " << arma::size(*p_input_weights_) << ", level " << level);

    LOG4_DEBUG(
            "Training on features " << common::present(*p_xtrain) << ", labels " << common::present(*p_ytrain) << ", last-knowns " << common::present(*p_ylastknown)
                                    << ", pre-calculated kernel matrices " << (precalc_kernel_matrices ? precalc_kernel_matrices->size() : 0) << ", parameters "
                                    << **param_set.cbegin() <<
                                    ", last value time " << last_value_time);
    reset();

    p_features = p_xtrain;
    p_labels = p_ytrain;
    p_last_knowns = p_ylastknown;
    p_input_weights = p_input_weights_;

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
        OMP_FOR_i(num_chunks) {
            const auto p_params = get_params_ptr(i);

            const auto chunk_sf = business::DQScalingFactorService::slice(scaling_factors, i, gradient, step);
            business::DQScalingFactorService::scale_features(i, gradient, step, p_params->get_lag_count(), chunk_sf, train_feature_chunks_t[i]);
            const auto p_sf = business::DQScalingFactorService::find(chunk_sf, model_id, i, gradient, step, level, false, true);
            business::DQScalingFactorService::scale_labels(*p_sf, train_label_chunks[i]);

            if (p_kernel_matrices->at(i).empty())
                p_kernel_matrices->at(i) = *prepare_K(ccache(), *p_params, train_feature_chunks_t[i], last_value_time);
            else
                LOG4_DEBUG("Using pre-calculated kernel matrix for " << i);

            calc_weights(i, PROPS.get_stabilize_iterations_count(), ixs[i].n_rows / C_solve_opt_div);
        }
    }
    update_all_weights();
    samples_trained = p_features->n_rows;
    last_trained_time = last_value_time;
}


void OnlineMIMOSVR::learn(
        const arma::mat &new_x, const arma::mat &new_y, const arma::vec &new_ylk, const arma::mat &new_w, const bpt::ptime &last_value_time,
        const bool temp_learn, const std::deque<uint32_t> &forget_ixs)
{
    if (new_x.empty() || new_y.empty() || new_x.n_cols != p_features->n_cols || new_y.n_cols != p_labels->n_cols || new_ylk.n_rows != new_y.n_rows || new_x.n_rows != new_y.n_rows)
        LOG4_THROW("New data dimensions labels " << arma::size(new_y) << ", last-knowns " << arma::size(new_ylk) << ", features " << arma::size(new_x) <<
                                                 " not sane or do not match model data dimensions labels " << arma::size(*p_labels) << ", features " << arma::size(*p_features)
                                                 << ", last-knowns " << arma::size(*p_last_knowns));
    if (p_features->n_rows == samples_trained) { // First call to online learn copy batch data, TODO maybe move to end of batch_train
        p_features = ptr(*p_features);
        p_labels = ptr(*p_labels);
        p_last_knowns = ptr(*p_last_knowns);
        p_input_weights = ptr(*p_input_weights);
    }
    const auto new_rows_ct = new_x.n_rows;

    if (is_manifold()) {
        if (!p_manifold) LOG4_THROW("Manifold not initialized.");
        auto gen = common::reproducibly_seeded_64<xso::rng64>();
        std::uniform_int_distribution<unsigned> rand_int(0, C_interlace_manifold_factor);
        const auto new_manifold_rows_ct = new_x.n_rows * p_features->n_rows / C_interlace_manifold_factor;
        arma::mat new_x_manifold(new_manifold_rows_ct, new_x.n_cols + p_features->n_cols);
        arma::mat new_y_manifold(new_manifold_rows_ct, new_y.n_cols);
        arma::mat new_ylk_manifold(new_manifold_rows_ct, 1);
#pragma omp parallel for num_threads(adj_threads(p_features->n_rows * new_x.n_rows)) collapse(2)
        for (unsigned i = 0; i < p_features->n_rows; ++i)
            for (unsigned j = 0; j < new_x.n_rows; ++j) {
                const auto r = i * new_x.n_rows + j;
                if (r % C_interlace_manifold_factor == 0) {
                    const auto r_i = common::bounce<unsigned>(new_x.n_rows - 1, i + rand_int(gen));
                    const auto r_j = common::bounce<unsigned>(p_features->n_rows - 1, j + rand_int(gen));
                    MANIFOLD_SET(new_x_manifold.row(r / C_interlace_manifold_factor), new_y_manifold.row(r / C_interlace_manifold_factor),
                                 new_x.row(r_j), new_y.row(r_j), p_features->row(r_i), p_labels->row(r_i));
                }
            }
        samples_trained += new_rows_ct;
        p_manifold->learn(new_x_manifold, new_y_manifold, new_ylk_manifold, new_w, last_value_time);
        p_features->shed_rows(0, new_rows_ct - 1);
        p_labels->shed_rows(0, new_rows_ct - 1);
        p_last_knowns->shed_rows(0, new_rows_ct - 1);
        p_features->insert_rows(p_features->n_rows, new_x);
        p_labels->insert_rows(p_labels->n_rows, new_y);
        p_last_knowns->insert_rows(p_last_knowns->n_rows, new_ylk);
        return;
    }

    if (new_rows_ct > ixs.front().size() / 2) {
        p_features->shed_rows(0, new_rows_ct - 1);
        p_labels->shed_rows(0, new_rows_ct - 1);
        p_last_knowns->shed_rows(0, new_rows_ct - 1);
        p_features->insert_rows(p_features->n_rows, new_x);
        p_labels->insert_rows(p_labels->n_rows, new_y);
        p_last_knowns->insert_rows(p_last_knowns->n_rows, new_ylk);
        const auto backup_samples_trained = samples_trained + new_rows_ct;
        batch_train(p_features, p_labels, p_last_knowns, p_input_weights, last_value_time);
        samples_trained = backup_samples_trained;
        return;
    }

    arma::uvec shed_ixs; // Active indexes to be shedded to make space for new learning data
    auto active_rows = get_active_ixs();
    if (forget_ixs.size()) {
        if (new_rows_ct != forget_ixs.size()) LOG4_WARN("Forget index size " << forget_ixs.size() << " does not equal new train samples count " << new_rows_ct);
        t_omp_lock learn_lk;
        OMP_FOR(forget_ixs.size())
        for (const auto f: forget_ixs) {
            const arma::uvec found = arma::find(active_rows == f);
            if (found.n_elem) {
                LOG4_DEBUG("Using suggested index " << f << " found among active indexes at " << found);
                learn_lk.set();
                shed_ixs.insert_rows(shed_ixs.n_rows, f);
                active_rows.shed_row(found.front());
                learn_lk.unset();
            } else {
                const auto oldest_ix = active_rows.index_min();
                LOG4_ERROR(
                        "Suggested forget index " << f << " not found in active indexes, using oldest active index " << oldest_ix << ", " << active_rows[oldest_ix] << " instead.");
                learn_lk.set();
                shed_ixs.insert_rows(shed_ixs.n_rows, active_rows[oldest_ix]);
                active_rows.shed_row(oldest_ix);
                learn_lk.unset();
            }
        }
    }

    if (tmp_ixs.size())
        while (tmp_ixs.size() && shed_ixs.n_rows < new_rows_ct) {
            const auto tmp_i = front(tmp_ixs);
            LOG4_DEBUG("Forgetting temporary row " << tmp_i);
            shed_ixs.insert_rows(shed_ixs.n_rows, tmp_i);
            const arma::uvec found_active = arma::find(active_rows == tmp_i);
            if (found_active.n_elem) active_rows.shed_row(found_active.front());
            tmp_ixs.erase(tmp_ixs.begin());
        }

    if (shed_ixs.n_rows < new_rows_ct) {
        arma::vec active_total_weights = arma::sum(arma::abs(all_weights.rows(active_rows)), 1);
        while (shed_ixs.n_rows < new_rows_ct) {
#ifdef FORGET_MIN_WEIGHT // Forgetting min weight works the best
            const auto ix_to_shed = active_total_weights.index_min();
#else // Forget oldest active index // TODO Wrong fix
            const auto ix_to_shed = active_ixs.min_index();
#endif
            const auto active_row_ix = active_rows[ix_to_shed];
            LOG4_DEBUG("Forgetting least significant row at " << active_row_ix << " weighting " << active_total_weights[ix_to_shed]);
            active_total_weights.shed_row(ix_to_shed);
            shed_ixs.insert_rows(shed_ixs.n_rows, active_row_ix);
            active_rows.shed_row(ix_to_shed);
        }
    }

    // Do the shedding
    p_features->shed_rows(shed_ixs);
    p_labels->shed_rows(shed_ixs);
    p_last_knowns->shed_rows(shed_ixs);
    tbb::concurrent_map<uint32_t /* chunk */, std::pair<uint32_t /* affected rows */, arma::mat /* forgotten weights */>> affected_chunks;
    OMP_FOR_i(ixs.size()) { // TODO Review
        auto &chunk_ixs = ixs[i];
        const auto found_ixs = common::find(chunk_ixs, shed_ixs);
        uint32_t ct = 0;
        for (auto f_ix: found_ixs) {
            f_ix -= ct++;
            if (f_ix >= chunk_ixs.n_rows) continue;
            if (f_ix < chunk_ixs.n_rows - 1) chunk_ixs.rows(f_ix + 1, chunk_ixs.n_rows - 1) -= 1;
            chunk_ixs.shed_row(f_ix);
        }
        ixs[i].shed_rows(found_ixs);
        weight_chunks[i].shed_rows(found_ixs);
        train_label_chunks[i].shed_rows(found_ixs);
        train_feature_chunks_t[i].shed_cols(found_ixs);
        p_kernel_matrices->at(i).shed_rows(found_ixs);
        p_kernel_matrices->at(i).shed_cols(found_ixs);
        affected_chunks.emplace(i, std::pair{found_ixs.n_rows, weight_chunks[i].rows(found_ixs)});
    }

    // Append new distances
    LOG4_DEBUG("Affected chunks " << affected_chunks.size());
    std::atomic<uint32_t> add_new_row_ix = 0;
    OMP_FOR_i(affected_chunks.size()) {
        const auto chunk_ix = affected_chunks % i;
        const auto &af = affected_chunks ^ i;
        const auto chunk_shed_rows = af.first;
        auto &K_chunk = p_kernel_matrices->at(chunk_ix);
        arma::uvec new_ixs(chunk_shed_rows);
        for (uint32_t r = 0; r < new_ixs.n_rows; ++r)
            new_ixs[r] = add_new_row_ix++ % new_x.n_rows;
        arma::mat new_chunk_features_t = new_x.rows(new_ixs).t();
        arma::mat new_chunk_labels = new_y.rows(new_ixs);
        business::DQScalingFactorService::scale_labels(chunk_ix, *this, new_chunk_labels);
        business::DQScalingFactorService::scale_features(chunk_ix, *this, new_chunk_features_t);
        K_chunk.resize(K_chunk.n_rows + chunk_shed_rows, K_chunk.n_cols + chunk_shed_rows);
        const auto &params = get_params(i);
#pragma omp parallel ADJ_THREADS(2)
#pragma omp single
        {
#pragma omp task mergeable
            K_chunk.submat(K_chunk.n_rows - chunk_shed_rows, K_chunk.n_cols - chunk_shed_rows, K_chunk.n_rows - 1, K_chunk.n_cols - 1) = *prepare_K(params, new_chunk_features_t);
#pragma omp task mergeable
            {
                arma::mat newold_K;
                PROFILE_EXEC_TIME(newold_K = *prepare_Ky(params, train_feature_chunks_t[chunk_ix], new_chunk_features_t), "Init predict kernel matrix for chunk " << chunk_ix);
                K_chunk.submat(0, K_chunk.n_cols - chunk_shed_rows, K_chunk.n_rows - chunk_shed_rows - 1, K_chunk.n_cols - 1) = newold_K.t();
                K_chunk.submat(K_chunk.n_rows - chunk_shed_rows, 0, K_chunk.n_rows - 1, K_chunk.n_cols - chunk_shed_rows - 1) = newold_K;
            }
        }
        ixs[chunk_ix].insert_rows(ixs[chunk_ix].n_rows, ixs[chunk_ix].n_rows + new_ixs);
        train_feature_chunks_t[chunk_ix].insert_cols(train_feature_chunks_t[chunk_ix].n_cols, new_chunk_features_t);
        train_label_chunks[chunk_ix].insert_rows(train_label_chunks[chunk_ix].n_rows, new_chunk_labels);
        const auto epsco = calc_epsco(K_chunk, train_label_chunks[chunk_ix]);
        K_chunk.diag().fill(epsco);
        weight_chunks[chunk_ix].insert_rows(weight_chunks[chunk_ix].n_rows, af.second.rows(new_ixs));
    }
    if (temp_learn)
        for (unsigned row_i = p_labels->n_rows; row_i < p_labels->n_rows + new_rows_ct; ++row_i)
            tmp_ixs.emplace(row_i);
    p_features->insert_rows(p_features->n_rows, new_x);
    p_labels->insert_rows(p_labels->n_rows, new_y);
    p_last_knowns->insert_rows(p_last_knowns->n_rows, new_ylk);

    OMP_FOR_i(affected_chunks.size()) calc_weights(affected_chunks % i, PROPS.get_online_learn_iter_limit(), ixs[i].n_rows / (C_solve_opt_div * C_solve_opt_div));
    update_all_weights();
    samples_trained += new_rows_ct;
    last_trained_time = last_value_time;
}


arma::mat OnlineMIMOSVR::weight_matrix(const arma::uvec &ixs, const arma::mat &weights) {
    arma::mat w(ixs.n_rows, ixs.n_rows, arma::fill::none);
    const arma::vec Wv = arma::mean(weights.rows(ixs), 1);
    OMP_FOR_i(ixs.n_rows) w.col(i) = Wv * Wv[i];
    return arma::sqrt(w);
}

arma::mat OnlineMIMOSVR::calc_weights(
        arma::mat K, const arma::mat &labels, const arma::mat &instance_weights_matrix, const double epsco, const uint16_t iters_irwls, const uint16_t iters_opt)
{
    arma::mat K_epsco = K;
    K_epsco *= instance_weights_matrix;
    K *= instance_weights_matrix;
    K_epsco.diag().fill(epsco);
    arma::mat w;
    if (iters_irwls) PROFILE_EXEC_TIME(solve_irwls(K_epsco, K, labels * C_diff_coef, w, iters_irwls),
                                       "Solve weights IRWLS for K " << arma::size(K) << ", iterations " << iters_irwls);
    if (iters_opt) PROFILE_EXEC_TIME(solve_opt(K, labels * C_diff_coef, w, iters_opt),
                                     "Solve weights PPrune for K " << arma::size(K) << ", iterations " << iters_opt);
    return w;
}

arma::mat OnlineMIMOSVR::calc_weights(const arma::mat &K, const arma::mat &labels, const double epsco, const uint16_t iters_irwls, const uint16_t iters_opt)
{
    arma::mat K_epsco = K;
    K_epsco.diag().fill(epsco);
    arma::mat w;
    if (iters_irwls) PROFILE_EXEC_TIME(solve_irwls(K_epsco, K, labels * C_diff_coef, w, iters_irwls),
                                       "Solve weights IRWLS for K " << arma::size(K) << ", iterations " << iters_irwls);
    if (iters_opt) PROFILE_EXEC_TIME(solve_opt(K, labels * C_diff_coef, w, iters_opt), "Solve weights PPrune for K " << arma::size(K) << ", iterations " << iters_opt);
    return w;
}

void OnlineMIMOSVR::calc_weights(const uint16_t chunk_ix, const uint16_t iters_irwls, const uint16_t iters_opt) // TODO Remove
{
    const auto &params = get_params(chunk_ix);
    PROFILE_EXEC_TIME(weight_chunks[chunk_ix] = calc_weights(p_kernel_matrices->at(chunk_ix), train_label_chunks[chunk_ix], weight_matrix(ixs[chunk_ix], *p_input_weights),
                                                             1. / params.get_svr_C(), iters_irwls, ixs[chunk_ix].n_rows / C_solve_opt_div),
                      "Chunk " << chunk_ix << ", level " << level << ", gradient " << gradient << ", parameters " << params << ", w " << common::present(weight_chunks[chunk_ix]));
}


void OnlineMIMOSVR::update_all_weights()
{
    if (arma::size(all_weights) != arma::size(*p_labels)) all_weights.set_size(arma::size(*p_labels));
    all_weights.zeros();
    all_weights.rows(ixs.front()) = weight_chunks.front();
    if (ixs.size() > 1) {
        arma::mat divisors(arma::size(*p_labels), arma::fill::ones);
        for (uint16_t i = 1; i < ixs.size(); ++i) {
            all_weights.rows(ixs[i]) += weight_chunks[i];
            divisors.rows(ixs[i]) += 1;
        }
        all_weights /= divisors;
    }
    LOG4_TRACE("Total weights for level " << level << " " << arma::size(all_weights));
}

} // datamodel
} // svr
