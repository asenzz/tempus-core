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
#include "kernel_factory.hpp"
#include "matrix_solver.hpp"
#include "common/barrier.hpp"

namespace svr {
namespace datamodel {
void
OnlineSVR::batch_train(const mat_ptr &p_xtrain, const mat_ptr &p_ytrain, const mat_ptr &p_input_weights_, const bpt::ptime &time, const matrices_ptr &precalc_kernel_matrices)
{
#ifdef INSTANCE_WEIGHTS
    if (p_xtrain->n_rows != p_ytrain->n_rows || p_ytrain->n_rows != p_ylastknown->n_rows || p_input_weights_->n_rows != p_ytrain->n_rows || p_xtrain->empty() || p_ytrain->empty())
        LOG4_THROW("Invalid data dimensions, features " << arma::size(*p_xtrain) << ", labels " << arma::size(*p_ytrain) << ", last knowns " << arma::size(*p_ylastknown) <<
                                                        ", instance weights " << arma::size(*p_input_weights_) << ", level " << level);
#else
    if (p_xtrain->n_rows != p_ytrain->n_rows || p_xtrain->empty() || p_ytrain->empty())
        LOG4_THROW("Invalid data dimensions, features " << arma::size(*p_xtrain) << ", labels " << arma::size(*p_ytrain) << ", level " << level);
#endif

    p_features = p_xtrain;
    p_labels = p_ytrain;
    p_input_weights = p_input_weights_;
    ixs = generate_indexes();
    last_trained_time = time;
    const uint32_t num_chunks = ixs.size();
    if (train_feature_chunks_t.size() != num_chunks) train_feature_chunks_t.resize(num_chunks);
    if (train_label_chunks.size() != num_chunks) train_label_chunks.resize(num_chunks);
    if (chunks_score.size() != num_chunks) chunks_score.resize(num_chunks);
    if (precalc_kernel_matrices && precalc_kernel_matrices->size() == num_chunks) {
        p_kernel_matrices = precalc_kernel_matrices;
        LOG4_DEBUG("Using " << num_chunks << " precalculated matrices.");
    } else if (precalc_kernel_matrices && !precalc_kernel_matrices->empty())
        LOG4_ERROR("Precalculated kernel matrices do not match needed chunks count!");

    LOG4_DEBUG("Initializing kernel matrices from scratch.");
    if (!p_kernel_matrices) p_kernel_matrices = ptr<DTYPE(*p_kernel_matrices) >(num_chunks);
    else if (p_kernel_matrices->size() != num_chunks) p_kernel_matrices->resize(num_chunks);

    if (weight_chunks.size() != num_chunks) weight_chunks.resize(num_chunks);
    tbb::mutex param_set_mx;
    OMP_FOR_i(num_chunks) {
        tbb::mutex::scoped_lock param_set_l(param_set_mx);
        SVRParameters_ptr p = get_params_ptr(i);
        if (!p) {
            p = otr(get_params());
            p->set_chunk_index(i);
            p->set_svr_kernel_param(0);
            param_set.emplace(p);
        }
        param_set_l.release();
        prepare_chunk(p);
    }

    LOG4_DEBUG("Training on features " << common::present(*p_xtrain) << ", labels " << common::present(*p_ytrain) <<
        ", pre-calculated kernel matrices " << (precalc_kernel_matrices ? precalc_kernel_matrices->size() : 0) << ", parameters " << **param_set.cbegin() <<
        ", last value time " << time);

    if (needs_tuning()) {
        if (precalc_kernel_matrices && precalc_kernel_matrices->size())
            LOG4_WARN("Provided kernel matrices will be ignored because SVR parameters are not initialized.");
        PROFILE_MSG(tune(), "Tune kernel parameters for level " << level << ", step " << step << ", gradient " << (**param_set.cbegin()).get_grad_level());
    }

    OMP_FOR_i(num_chunks) {
        auto p_params = get_params_ptr(i);
        if (p_kernel_matrices->at(i).empty()) {
            p_kernel_matrices->at(i) = false /* p_params->get_kernel_type() == e_kernel_type::GBM || p_params->get_kernel_type() == e_kernel_type::TFT */
                                           // Enable if inplace predict of the model is precise enough (without overfitting)
                                           ? kernel::get_reference_Z<double>(train_label_chunks[i])
                                           : p_kernel_matrices->at(i) = kernel::IKernel<double>::get(*p_params)->kernel(ccache(), train_feature_chunks_t[i], time);
        } else
            LOG4_DEBUG("Using pre-calculated kernel " << arma::size(p_kernel_matrices->at(i)) << " for chunk " << i);
        LOG4_TRACE("Difference from reference kernel " << common::present<double>(p_kernel_matrices->at(i) - kernel::get_reference_Z<double>(train_label_chunks[i])));
        calc_weights(i, ixs[i].n_rows * PROPS.get_solve_iterations_coefficient(), PROPS.get_stabilize_iterations_count());
    }

    const auto chunks_score_max = common::max(chunks_score);
    const auto chunks_score_min = common::min(chunks_score);
    chunks_score = chunks_score_max - chunks_score + chunks_score_min;
    const auto chunks_score_mean = common::mean(chunks_score);
    if (PROPS.get_weight_inertia() != 0) {
        const auto mean_inertia = chunks_score_mean * PROPS.get_weight_inertia();
        chunks_score = (chunks_score + mean_inertia) / (chunks_score_mean + mean_inertia);
    } else chunks_score /= chunks_score_mean;
    samples_trained = p_features->n_rows;
}

void OnlineSVR::prepare_chunk(const uint32_t i)
{
    const auto p = get_params_ptr(i);
    prepare_chunk(p);
}

void OnlineSVR::prepare_chunk(const SVRParameters_ptr &p)
{
    const auto i = p->get_chunk_index();
    train_feature_chunks_t[i] = feature_chunk_t(ixs[i]);
    train_label_chunks[i] = p_labels->rows(ixs[i]);
    LOG4_TRACE("Before scaling chunk " << i << ", train labels " << common::present(train_label_chunks[i]) << ", train features " << common::present(train_feature_chunks_t[i]));
    DQScalingFactor_ptr p_labels_sf;
    DTYPE(scaling_factors) features_sf;
#pragma omp critical
    {
        p_labels_sf = business::DQScalingFactorService::find(scaling_factors, model_id, i, gradient, step, level, false, true);
        features_sf = business::DQScalingFactorService::slice(scaling_factors, i, gradient, step);
    }
    const auto lag = p->get_lag_count();
    if (!p_labels_sf || features_sf.size() != train_feature_chunks_t[i].n_rows / lag) {
        features_sf = business::DQScalingFactorService::calculate(model_id, *p, train_feature_chunks_t[i], train_label_chunks[i]);
#pragma omp critical
        set_scaling_factors(features_sf);
        if (model_id)
            for (const auto &sf: features_sf) {
                if (APP.dq_scaling_factor_service.exists(sf)) (void) APP.dq_scaling_factor_service.remove(sf);
                (void) APP.dq_scaling_factor_service.save(sf);
            }
        p_labels_sf = business::DQScalingFactorService::find(features_sf, model_id, i, gradient, step, level, false, true);
        assert(p_labels_sf);
    }
    business::DQScalingFactorService::scale_features_I(i, gradient, step, lag, features_sf, train_feature_chunks_t[i]);
    business::DQScalingFactorService::scale_labels_I(*p_labels_sf, train_label_chunks[i]);
    LOG4_TRACE("After scaling chunk " << i << ", train labels " << common::present(train_label_chunks[i]) << ", train features " <<
        common::present(train_feature_chunks_t[i]) << ", labels scaling factor " << *p_labels_sf << ", features scaling factors " << features_sf);
}


void OnlineSVR::learn(
    const arma::mat &new_x, const arma::mat &new_y, const arma::mat &new_w, const bpt::ptime &last_value_time,
    const bool temp_learn, const std::deque<uint32_t> &forget_ixs)
{
    // TODO Review and fix this method
    last_trained_time = last_value_time;
    return;

    if (new_x.empty() || new_y.empty() || new_x.n_cols != p_features->n_cols || new_y.n_cols != p_labels->n_cols || new_x.n_rows != new_y.n_rows)
        LOG4_THROW("New data dimensions labels " << arma::size(new_y) << ", features " << arma::size(new_x) <<
        " not sane or do not match model data dimensions labels " << arma::size(*p_labels) << ", features " << arma::size(*p_features));
    if (p_features->n_rows == samples_trained) {
        // First call to online learn copy batch data, TODO maybe move to end of batch_train
        p_features = ptr(*p_features);
        p_labels = ptr(*p_labels);
#ifdef INSTANCE_WEIGHTS
        p_input_weights = ptr(*p_input_weights);
#endif
    }
    const auto new_rows_ct = new_x.n_rows;

    if (is_manifold()) {
        auto gen = common::reproducibly_seeded_64<xso::rng64>();
        common::threadsafe_uniform_int_distribution<uint32_t> rand_int(0, PROPS.get_interleave());
        const auto new_manifold_rows_ct = new_x.n_rows * p_features->n_rows / PROPS.get_interleave();
        arma::mat new_x_manifold(new_manifold_rows_ct, new_x.n_cols + p_features->n_cols);
        arma::mat new_y_manifold(new_manifold_rows_ct, new_y.n_cols);
        arma::mat new_ylk_manifold(new_manifold_rows_ct, 1);
        const auto manifold_interleave = PROPS.get_interleave();
#pragma omp parallel for num_threads(adj_threads(p_features->n_rows * new_x.n_rows)) collapse(2)
        for (unsigned i = 0; i < p_features->n_rows; ++i)
            for (unsigned j = 0; j < new_x.n_rows; ++j) {
                const auto r = i * new_x.n_rows + j;
                if (r % manifold_interleave == 0) {
                    const auto rand_int_g = rand_int(gen);
                    const auto r_i = common::bounce<unsigned>(new_x.n_rows - 1, i + rand_int_g);
                    const auto r_j = common::bounce<unsigned>(p_features->n_rows - 1, j + rand_int_g);
                    MANIFOLD_SET(new_x_manifold.row(r / manifold_interleave), new_y_manifold.row(r / manifold_interleave),
                                 new_x.row(r_j), new_y.row(r_j), p_features->row(r_i), p_labels->row(r_i));
                }
            }
        samples_trained += new_rows_ct;
        // p_manifold->learn(new_x_manifold, new_y_manifold, new_ylk_manifold, new_w, last_value_time);
        p_features->shed_rows(0, new_rows_ct - 1);
        p_labels->shed_rows(0, new_rows_ct - 1);
        p_features->insert_rows(p_features->n_rows, new_x);
        p_labels->insert_rows(p_labels->n_rows, new_y);
        return;
    }

    if (new_rows_ct > ixs.front().size() / 2) {
        p_features->shed_rows(0, new_rows_ct - 1);
        p_labels->shed_rows(0, new_rows_ct - 1);
        p_features->insert_rows(p_features->n_rows, new_x);
        p_labels->insert_rows(p_labels->n_rows, new_y);
        const auto backup_samples_trained = samples_trained + new_rows_ct;
        batch_train(p_features, p_labels, p_input_weights, last_value_time);
        samples_trained = backup_samples_trained;
        return;
    }

    arma::uvec shed_ixs; // Active indexes to be shedded to make space for new learning data
    auto active_rows = get_active_ixs();
    if (forget_ixs.size()) {
        if (new_rows_ct != forget_ixs.size())
            LOG4_WARN("Forget index size " << forget_ixs.size() << " does not equal new train samples count " << new_rows_ct);
        tbb::mutex learn_lk;
        OMP_FOR(forget_ixs.size())
        for (const auto f: forget_ixs) {
            const arma::uvec found = arma::find(active_rows == f);
            if (found.n_elem) {
                LOG4_DEBUG("Using suggested index " << f << " found among active indexes at " << found);
                const tbb::mutex::scoped_lock lk(learn_lk);
                shed_ixs.insert_rows(shed_ixs.n_rows, f);
                active_rows.shed_row(found.front());
            } else {
                const auto oldest_ix = active_rows.index_min();
                LOG4_ERROR("Suggested forget index " << f << " not found in active indexes, using oldest active index " << oldest_ix << ", " << active_rows[oldest_ix] << " instead.");
                const tbb::mutex::scoped_lock lk(learn_lk);
                shed_ixs.insert_rows(shed_ixs.n_rows, active_rows[oldest_ix]);
                active_rows.shed_row(oldest_ix);
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
    tbb::concurrent_map<uint32_t /* chunk */, std::pair<uint32_t /* affected rows */, arma::mat /* forgotten weights */> > affected_chunks;
    OMP_FOR_i(ixs.size()) {
        // TODO Review
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
#pragma omp parallel ADJ_THREADS(C_n_cpu)
#pragma omp single
    {
#pragma omp taskloop mergeable grainsize(1) default(shared)
        for (uint16_t i = 0; i < CAST2(i) affected_chunks.size(); ++i) {
            const auto chunk_ix = affected_chunks % i;
            const auto &af = affected_chunks ^ i;
            const auto chunk_shed_rows = af.first;
            auto &K_chunk = p_kernel_matrices->at(chunk_ix);
            arma::uvec new_ixs(chunk_shed_rows);
            for (uint32_t r = 0; r < new_ixs.n_rows; ++r)
                new_ixs[r] = add_new_row_ix++ % new_x.n_rows;
            arma::mat new_chunk_features_t = new_x.rows(new_ixs).t();
            arma::mat new_chunk_labels = new_y.rows(new_ixs);
            business::DQScalingFactorService::scale_labels_I(chunk_ix, *this, new_chunk_labels);
            business::DQScalingFactorService::scale_features_I(chunk_ix, *this, new_chunk_features_t);
            K_chunk.resize(K_chunk.n_rows + chunk_shed_rows, K_chunk.n_cols + chunk_shed_rows);
            auto &params = get_params(i);
#pragma omp task mergeable
            K_chunk.submat(K_chunk.n_rows - chunk_shed_rows, K_chunk.n_cols - chunk_shed_rows, K_chunk.n_rows - 1, K_chunk.n_cols - 1) =
                    kernel::IKernel<double>::get(params)->kernel(new_chunk_features_t, new_chunk_features_t);
#pragma omp task mergeable
            {
                arma::mat newold_K;
                PROFILE_MSG(newold_K = kernel::IKernel<double>::get(params)->kernel(train_feature_chunks_t[chunk_ix], new_chunk_features_t),
                            "Init predict kernel matrix for chunk " << chunk_ix);
                K_chunk.submat(0, K_chunk.n_cols - chunk_shed_rows, K_chunk.n_rows - chunk_shed_rows - 1, K_chunk.n_cols - 1) = newold_K;
                K_chunk.submat(K_chunk.n_rows - chunk_shed_rows, 0, K_chunk.n_rows - 1, K_chunk.n_cols - chunk_shed_rows - 1) = newold_K.t();
            }
            ixs[chunk_ix].insert_rows(ixs[chunk_ix].n_rows, ixs[chunk_ix].n_rows + new_ixs);
            train_feature_chunks_t[chunk_ix].insert_cols(train_feature_chunks_t[chunk_ix].n_cols, new_chunk_features_t);
            train_label_chunks[chunk_ix].insert_rows(train_label_chunks[chunk_ix].n_rows, new_chunk_labels);
            //            const auto epsco = calc_epsco(K_chunk, train_label_chunks[chunk_ix]);
            //            K_chunk.diag().fill(epsco);
            weight_chunks[chunk_ix].insert_rows(weight_chunks[chunk_ix].n_rows, af.second.rows(new_ixs));
        }
    }
    if (temp_learn)
        for (unsigned row_i = p_labels->n_rows; row_i < p_labels->n_rows + new_rows_ct; ++row_i)
            tmp_ixs.emplace(row_i);
    p_features->insert_rows(p_features->n_rows, new_x);
    p_labels->insert_rows(p_labels->n_rows, new_y);

    OMP_FOR_i(affected_chunks.size()) calc_weights(affected_chunks % i, ixs[i].n_rows * PROPS.get_solve_iterations_coefficient(), PROPS.get_online_learn_iter_limit());
    update_all_weights();
    samples_trained += new_rows_ct;
    last_trained_time = last_value_time;
}


arma::mat OnlineSVR::instance_weight_matrix(const arma::uvec &ixs, const arma::mat &weights)
{
    arma::mat w(ixs.n_rows, ixs.n_rows, ARMA_DEFAULT_FILL);
    const arma::vec Wv = arma::mean(weights.rows(ixs), 1);
    OMP_FOR_i(ixs.n_rows) w.col(i) = Wv * Wv[i];
    return arma::sqrt(w);
}


arma::mat get_bounds(const arma::mat &A, const arma::mat &b, const float lim_coef, const uint16_t layers)
{
#if 0 // Old way
    constexpr auto qalpha = 1;
    const auto limhi = common::meanabs_hiquant(b.mem, b.n_elem, qalpha) / common::meanabs_loquant(A.mem, A.n_elem, qalpha) / b.n_rows;
    const auto limlo = -limhi; // common::mean_loquant(b.mem, b.n_elem, qalpha) / common::mean_hiquant(A.mem, A.n_elem, qalpha) / b.n_rows;
    LOG4_TRACE("Bounds " << limlo << " to " << limhi << ", quantile " << qalpha);
#else
    const auto limhi = lim_coef * common::meanabs(b) / common::meanabs(A) / b.n_elem;
    assert(limhi > 0);
    const auto limlo = -limhi;
    LOG4_TRACE("Bounds " << limlo << " to " << limhi);
#endif
    assert(std::isnormal(limhi) && std::isnormal(limlo) && limlo != limhi);
    arma::mat r(b.n_elem * layers, 2, ARMA_DEFAULT_FILL);
    if (limhi > limlo) {
        r.col(0).fill(limlo);
        r.col(1).fill(limhi);
    } else {
        r.col(0).fill(limhi);
        r.col(1).fill(limlo);
    }
    return r;
}


double OnlineSVR::calc_weights(const arma::mat &K, const arma::mat &L, const arma::uvec &chunk_ixs, arma::mat &weights, const uint32_t iter_opt, const uint16_t iter_irwls, const double limes)
{
#ifdef INSTANCE_WEIGHTS // TODO Rewrite
    K %= instance_weight_matrix(chunk_ixs, *p_input_weights);
    L *= p_input_weights->rows(ixs[chunk_ix]);
#endif
    const uint32_t n_rows = K.n_rows;
    assert(n_rows % 2 == 0);
    const uint32_t n_cols = L.n_cols;
    const auto L_n = L.n_elem;
    const auto layers = PROPS.get_weight_layers();
    const auto W_n = L_n * layers;
    const auto W_cols = n_cols * layers;
    const auto L_size = L_n * sizeof(double);
    const auto p_K = K.mem;
    solvers::score_weights *sw;
    constexpr uint32_t C_gpu_threshold = 1750; // Threshold for GPU usage, 1750 is a good value for most GPUs
    const bool use_gpu = n_rows >= C_gpu_threshold;
    const uint32_t pop_opt = (use_gpu ? 750 : 1500) / std::max(1., .5 * layers);
    const auto L_mean_mask = common::mean_mask(L, PROPS.get_solve_radius() * (L.n_rows - 1));
    const auto L_mm_ptr = L_mean_mask.mem;
    /* Hybrid scoring both on CPU and GPU degrades tuning quality because of the precision offset introduced by difference in GPU precision, so do either but not both. */
#define CO_ (optimizer::t_pprune_cost_fun)
    const auto loss_fun = n_rows < 1024 // On stack
                              ? CO_ [p_K, L_mm_ptr, n_cols, n_rows, layers, L_size](CPTRd x, RPTR(double) f) {
                                  const auto tmp = ALIGN_ALLOCA(double, L_size, MEM_ALIGN);
                                  *f = score_weights(n_rows, n_cols, layers, L_mm_ptr, p_K, x, tmp);
                              }
                              : n_rows < C_gpu_threshold // Heap
                                    ? CO_ [p_K, L_mm_ptr, n_cols, n_rows, layers, L_size](CPTRd x, RPTR(double) f) {
                                        const auto tmp = (double *const) ALIGNED_ALLOC_(MEM_ALIGN, L_size);
                                        *f = score_weights(n_rows, n_cols, layers, L_mm_ptr, p_K, x, tmp);
                                        ALIGNED_FREE_(tmp);
                                    }
                                    : // GPU
                                    CO_ [&sw](CPTRd x, RPTR(double) f) { *f = (*sw)(x); };

    bool fresh_start;
    if (weights.n_rows != n_rows || weights.n_cols != W_cols) {
        weights.set_size(n_rows, W_cols);
        fresh_start = true;
    } else
        fresh_start = false;

    arma::mat x0(W_n, 1, ARMA_DEFAULT_FILL); // Starting points using IRWLS and/or linsolver, columns 1 or 2
    if (use_gpu) sw = new solvers::score_weights(K, L_mean_mask, n_rows, n_cols, L_n, layers);
    const auto bounds = get_bounds(K, L_mean_mask, limes, layers);
    if (fresh_start) {
#if 1 // Dual iterative reweighted random butterfly transform solver used as preconditioner
        arma::mat precond;
        solvers::solve_irwls(K, L_mean_mask, precond, iter_irwls, layers);
        if (precond.n_rows != n_rows || precond.n_cols != W_cols || precond.has_nonfinite()) {
            LOG4_ERROR("Preconditioned matrix contains non-finite values.");
            x0.clear();
        } else
            x0.col(0) = arma::vectorise(precond);
#else
        x0.clear();
#endif
    } else x0.col(0) = arma::vectorise(weights);

#if 0 // PETSc linear solver seems unstable (DGMRES with Jacobi preconditioner), TODO Update for layers
    if (x0.n_cols < 2 || x0.n_rows != W_n) {
        x0.resize(W_n, 2);
        x0.col(1).zeros();
    }
    const solvers::antisymmetric_solver solver(n_rows, n_cols, iter_opt, x0.mem, K.mem, residuals_ptr, false, iter_irwls);
    const auto err = solver(x0.colptr(1));
    LOG4_DEBUG("Linear solver error " << err << ", weight layer " << wcol);
    if (x0.col(1).has_nonfinite()) x0.col(1).zeros();
#endif

    static const auto max_weight_calc = CDIVI(C_n_cpu, PROPS.get_solve_particles());
    // Pruned BiteOpt
    static std::counting_semaphore<> sem(max_weight_calc);
    sem.acquire();
    common::AppConfig::set_global_log_level(boost::log::trivial::info);
    const optimizer::t_pprune_res res = optimizer::pprune(optimizer::pprune::C_default_algo, PROPS.get_solve_particles(), bounds, loss_fun, iter_opt, 0, 0,
                                                    x0, {}, std::min<DTYPE(iter_opt) >(iter_opt, PROPS.get_opt_depth()), false, pop_opt);
    memcpy(weights.memptr(), res.best_parameters.mem, layers * L_size);
    common::AppConfig::set_global_log_level(PROPS.get_log_level());
    sem.release();

    LOG4_TRACE("MAE " << res.best_score << ", layers " << layers << ", iterations " << iter_opt << ", IRWLS " << iter_irwls << ", fresh start " <<
        fresh_start << ", labels " << common::present(L) << ", weights " << common::present(weights) << ", max weight calculations " << max_weight_calc);
    if (use_gpu) delete sw;

    return res.best_score;
}

void OnlineSVR::calc_weights(const uint16_t chunk_ix, const uint32_t iter_opt, const uint16_t iter_irwls)
{
    LOG4_BEGIN();

    assert(chunks_score.size() > chunk_ix);
    auto &params = get_params(chunk_ix);
    const auto n_rows = train_label_chunks[chunk_ix].n_rows;
    arma::mat K(arma::size(p_kernel_matrices->at(chunk_ix)), ARMA_DEFAULT_FILL); {
        const arma::mat L_t = train_label_chunks[chunk_ix].t();
        OMP_FOR_i(n_rows) K.row(i) = p_kernel_matrices->at(chunk_ix).row(i) + L_t;
    }
    const arma::mat L = train_label_chunks[chunk_ix] * n_rows;
    PROFILE_MSG(chunks_score[chunk_ix] = calc_weights(K, L, ixs[chunk_ix], weight_chunks[chunk_ix], iter_opt, iter_irwls, PROPS.get_limes()),
                "Calculate weights for " << params << ", chunk score " << chunks_score[chunk_ix] << ", chunk " << chunk_ix << ", iter_opt " << iter_opt <<
                ", iter_irwls " << iter_irwls);
}


void OnlineSVR::update_all_weights()
{
    /*
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
     */
    LOG4_THROW("Deprecated.");
}
} // datamodel
} // svr
