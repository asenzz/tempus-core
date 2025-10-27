#include <magma_auxiliary.h>
#include "onlinesvr.hpp"
#include "common/parallelism.hpp"
#include "DQScalingFactorService.hpp"
#include "appcontext.hpp"
#include "util/math_utils.hpp"
#include "pprune.hpp"
#include "kernel_factory.hpp"
#include "matrix_solver.hpp"

namespace svr {
namespace datamodel {
void OnlineSVR::batch_train(
        const mat_ptr &p_xtrain, const mat_ptr &p_ytrain, const mat_ptr &p_input_weights_, const bpt::ptime &time, const matrices_ptr &precalc_kernel_matrices, const uint32_t temp_learn)
{
    if (p_xtrain->n_rows != p_ytrain->n_rows || p_input_weights_->n_rows != p_ytrain->n_rows || p_xtrain->empty() || p_ytrain->empty())
        LOG4_THROW("Invalid data dimensions, features " << arma::size(*p_xtrain) << ", labels " << arma::size(*p_ytrain) << ", instance weights " << arma::size(*p_input_weights_) <<
                                                        ", level " << level);
    p_features = p_xtrain;
    p_labels = p_ytrain;
    p_input_weights = p_input_weights_;
    ixs = generate_indexes();
    last_trained_time = time;
    const uint32_t num_chunks = ixs.size();
    for (auto i = temp_learn; i < p_labels->n_rows; ++i) tmp_ixs.insert(i);
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
    if (instance_weights.size() != num_chunks) instance_weights.resize(num_chunks);
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
    active_rows = get_active_ixs();

    LOG4_DEBUG("Training on features " << common::present(*p_xtrain) << ", labels " << common::present(*p_ytrain) << ", pre-calculated kernel matrices " <<
                                       (precalc_kernel_matrices ? precalc_kernel_matrices->size() : 0) << ", parameters " << **param_set.cbegin() << ", last value time " << time);

    if (needs_tuning()) {
        if (precalc_kernel_matrices && precalc_kernel_matrices->size())
            LOG4_WARN("Provided kernel matrices will be ignored because SVR parameters are not initialized.");
        PROFILE_INFO(tune(), "Tune kernel parameters for level " << level << ", step " << step << ", gradient " << (**param_set.cbegin()).get_grad_level());
    }

#pragma omp parallel for schedule(static, 1) ADJ_THREADS(std::min<uint32_t>(num_chunks, CDIVI(PROPS.get_gpu_chunk(), ixs.front().n_elem))) default(shared) firstprivate(num_chunks)
    for (DTYPE(num_chunks) i = 0; i < num_chunks; ++i) {
        auto p_params = get_params_ptr(i);
        if (p_kernel_matrices->at(i).empty()) {
            p_kernel_matrices->at(i) = false /* p_params->get_kernel_type() == e_kernel_type::GBM || p_params->get_kernel_type() == e_kernel_type::TFT */
                                           // Enable if inplace predict of the kernel model is precise enough (without overfitting)
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

    active_total_weights = total_weights.rows(active_rows);
}

void OnlineSVR::learn(
        const arma::mat &new_x, const arma::mat &new_y, const arma::mat &new_w, const bpt::ptime &last_time, const uint32_t temp_learn, const std::deque<uint32_t> &forget_ixs)
{
    // TODO Review and test this method
    last_trained_time = last_time;

    for (auto i = temp_learn; i < new_y.n_rows; ++i) tmp_ixs.insert(i);

    if (new_x.empty() || new_y.empty() || new_x.n_cols != p_features->n_cols || new_y.n_cols != p_labels->n_cols || new_x.n_rows != new_y.n_rows)
        LOG4_THROW("New data dimensions labels " << arma::size(new_y) << ", features " << arma::size(new_x) <<
        " not sane or do not match model data dimensions labels " << arma::size(*p_labels) << ", features " << arma::size(*p_features));
    if (p_features->n_rows == samples_trained) {
        // First call to online learn copy batch data, TODO maybe move to end of batch_train
        p_features = ptr(*p_features);
        p_labels = ptr(*p_labels);
        p_input_weights = ptr(*p_input_weights);
    }
    const auto new_rows_ct = new_x.n_rows;

    if (new_rows_ct > ixs.front().size() / 2) {
        p_features->shed_rows(0, new_rows_ct - 1);
        p_labels->shed_rows(0, new_rows_ct - 1);
        p_features->insert_rows(p_features->n_rows, new_x);
        p_labels->insert_rows(p_labels->n_rows, new_y);
        const auto backup_samples_trained = samples_trained + new_rows_ct;
        batch_train(p_features, p_labels, p_input_weights, last_time);
        samples_trained = backup_samples_trained;
        return;
    }

    arma::uvec replace_ixs; // Active indexes to be shedded to make space for new learning data

    if (forget_ixs.size() > 0 && new_rows_ct != forget_ixs.size())
            LOG4_WARN("Forget index size " << forget_ixs.size() << " does not equal new train samples count " << new_rows_ct);
    for (uint32_t forget_i = 0; forget_i < forget_ixs.size() && replace_ixs.n_rows < new_rows_ct; ++forget_i)
        replace_ixs.insert_rows(replace_ixs.n_rows, forget_ixs[forget_i]);

    // Indexes of temporary rows to be shed
    while (tmp_ixs.size() && replace_ixs.n_rows < new_rows_ct) {
        const auto tmp_i = front(tmp_ixs);
        LOG4_DEBUG("Forgetting temporary row " << tmp_i);
        replace_ixs.insert_rows(replace_ixs.n_rows, tmp_i);
        tmp_ixs.erase(tmp_ixs.begin());
    }

    while (replace_ixs.n_rows < new_rows_ct) {
#ifdef FORGET_MIN_WEIGHT // Forgetting min weight works the best
        const auto ix_to_shed = active_rows[active_total_weights.index_min()];
#else // Forget oldest active index
        const auto ix_to_shed = active_rows.min();
#endif
        LOG4_DEBUG("Forgetting least significant row at " << ix_to_shed << " weighting " << total_weights[ix_to_shed]);
        replace_ixs.insert_rows(replace_ixs.n_rows, ix_to_shed);
        }

        // Replace shed with new instances
        uint32_t new_ct = 0;
        for (const auto six: replace_ixs) {
            p_features->row(six) = new_x.row(new_ct);
            p_labels->row(six) = new_y.row(new_ct);
            ++new_ct;
            if (new_ct >= new_rows_ct) {
                LOG4_DEBUG("Shed " << replace_ixs.n_rows << " rows, new rows count " << new_rows_ct << ", new_ct " << new_ct);
                break;
        }
    }
    new_ct = 0;
    total_weights.zeros();
    OMP_PAR(std::max<uint32_t>(replace_ixs.size(), ixs.size()))
    {
        for (uint32_t i = 0; i < ixs.size() && new_ct < new_rows_ct; ++i) {
            // TODO Review
            auto &chunk_ixs = ixs[i];
            const auto found_ixs = std::get<0>(common::find(chunk_ixs, replace_ixs));
            if (found_ixs.empty()) {
                LOG4_DEBUG("No indexes to shed in chunk " << i << ", active rows " << active_rows.n_rows << ", chunk rows " << chunk_ixs.n_rows);
                continue;
            }

            const auto start_ct = new_ct;
            const auto end_ct = start_ct + found_ixs.n_rows - 1;
            new_ct = end_ct + 1;
#pragma omp task
            {
                const auto chunk_new_w = new_w.rows(start_ct, end_ct);
                instance_weights[i].rows(found_ixs) = chunk_new_w;
                const auto chunk_sf = business::DQScalingFactorService::slice(scaling_factors, i, gradient, step);
                const auto p_labels_sf = business::DQScalingFactorService::find(chunk_sf, model_id, i, gradient, step, level, false, true);
                train_label_chunks[i].rows(found_ixs) = business::DQScalingFactorService::scale_labels(*p_labels_sf, new_y.rows(start_ct, end_ct) % chunk_new_w);
                auto &param = get_params(i);
                train_feature_chunks_t[i].cols(found_ixs) = business::DQScalingFactorService::scale_features(
                        i, gradient, step, param.get_lag_count(), chunk_sf, sst(*p_features, param.get_feature_mechanics(), ixs[i](found_ixs)));
                OMP_FOR(found_ixs.n_rows)
                for (const auto j: found_ixs) {
                    p_kernel_matrices->at(i).row(j) = kernel::IKernel<double>::get(param)->kernel(train_feature_chunks_t[i].col(j), train_feature_chunks_t[i]);
                    p_kernel_matrices->at(i).col(j) = p_kernel_matrices->at(i).row(j).t();
                }
                OnlineSVR::calc_weights(i, PROPS.get_online_learn_iter_limit(), PROPS.get_online_irwls());

                // replace_ixs.shed_rows(std::get<0>(common::find(replace_ixs, found_ixs)));
            }
        }
    }
    LOG4_DEBUG("Replaced " << replace_ixs.n_rows << " rows with new " << new_rows_ct << " rows, new total weights " << common::present(total_weights));
    samples_trained += new_rows_ct;
    last_trained_time = last_time;
}


arma::mat OnlineSVR::instance_weight_matrix(const arma::uvec &ixs, const arma::mat &weights)
{
    arma::mat w(ixs.n_rows, ixs.n_rows, ARMA_DEFAULT_FILL);
    const arma::vec Wv = arma::mean(weights.rows(ixs), 1);
    OMP_FOR_i(ixs.n_rows) w.col(i) = Wv * Wv[i];
    return arma::sqrt(w);
}


arma::mat get_weight_bounds(const arma::mat &A, const arma::mat &b, const float lim_coef, const uint16_t layers, const uint32_t W_elem)
{
    const auto limhi = lim_coef * common::meanabs(b) / common::meanabs(A) / W_elem;
    assert(limhi > 0);
    const auto limlo = -limhi;
    LOG4_TRACE("Bounds " << limlo << " to " << limhi);
    assert(std::isnormal(limhi) && std::isnormal(limlo) && limlo != limhi);
    arma::mat r(W_elem * layers, 2, ARMA_DEFAULT_FILL);
    if (limhi > limlo) {
        r.col(0).fill(limlo);
        r.col(1).fill(limhi);
    } else {
        r.col(0).fill(limhi);
        r.col(1).fill(limlo);
    }
    return r;
}


uint32_t get_population_size(const uint32_t n_rows, const uint16_t layers)
{
    uint32_t pop_opt;
    if (n_rows > 3500) pop_opt = 400;
    else if (n_rows > 1750) pop_opt = 750;
    else pop_opt = 1500;
    pop_opt /= std::max(1., .5 * layers);
    return std::max(pop_opt, 100u); // Minimum population size
}

double OnlineSVR::calc_weights(arma::mat &weights, const arma::mat &K, const arma::mat &L, const uint32_t iter_opt, const uint16_t iter_irwls)
{
    const uint32_t m_ = K.n_rows;
    const uint32_t n_ = L.n_cols;
    const uint32_t k_ = K.n_cols;
    const auto limes = PROPS.get_limes();
    assert(m_ % 2 == 0);
    assert(m_ == L.n_rows);
    assert(k_ == weights.n_rows);
    const auto L_n = L.n_elem;
    const auto layers = PROPS.get_weight_layers();
    const auto W_elem = k_ * n_;
    const auto W_n = W_elem * layers;
    const auto W_cols = n_ * layers;
    const auto L_size = L_n * sizeof(double);
    const auto p_K = K.mem;
    solvers::score_weights *sw;
    constexpr uint32_t C_gpu_threshold = 1750; // Threshold for GPU usage, 1750 is a good value for most GPUs
    const bool use_gpu = m_ >= C_gpu_threshold;
    const auto pop_opt = get_population_size(k_ * n_, layers);
    const auto L_mean_mask = common::mean_mask(L, PROPS.get_solve_radius() * (L.n_rows - 1));
    const auto L_mm_ptr = L_mean_mask.mem;
    /* Hybrid scoring both on CPU and GPU degrades tuning quality because of the precision offset introduced by difference in GPU precision, so do either but not both. */
#define CO_ (optimizer::t_pprune_cost_fun)
    const auto loss_fun = m_ < 1024 // On stack
                          ? CO_ [p_K, L_mm_ptr, m_, n_, k_, layers, L_size](CRPTRd x, RPTR(double) f) {
                const auto tmp = ALIGN_ALLOCA(double, L_size, MEM_ALIGN);
                *f = score_weights(m_, n_, k_, layers, L_mm_ptr, p_K, x, tmp);
            }
                          : m_ < C_gpu_threshold // Heap
                            ? CO_ [p_K, L_mm_ptr, m_, n_, k_, layers, L_size](CRPTRd x, RPTR(double) f) {
                        const auto tmp = (double *const) ALIGNED_ALLOC_(MEM_ALIGN, L_size);
                        *f = score_weights(m_, n_, k_, layers, L_mm_ptr, p_K, x, tmp);
                                        ALIGNED_FREE_(tmp);
                }
                                    : // GPU
                            CO_ [&sw](CRPTRd x, RPTR(double) f) { *f = (*sw)(x); };

    arma::mat x0(W_n, 3, ARMA_DEFAULT_FILL); // Starting points using IRWLS and/or linsolver, columns 1 or 2
    uint16_t x0_col_i = 0;
    if (weights.n_rows != k_ || weights.n_cols != W_cols)
        weights.set_size(k_, W_cols);
    else
        x0.col(x0_col_i++) = arma::vectorise(weights);

    if (use_gpu) sw = new solvers::score_weights(K, L_mean_mask, m_, n_, k_, L_n, layers);
    const auto bounds = get_weight_bounds(K, L_mean_mask, limes, layers, W_elem);
    if (K.n_rows == K.n_cols) {
        // Dual iterative reweighted random butterfly transform solver (limited to square matrices) used as preconditioner
        arma::mat precond;
        solvers::solve_irwls(K, L_mean_mask, precond, iter_irwls, layers);
        if (precond.n_rows != m_ || precond.n_cols != W_cols || precond.has_nonfinite()) {
            LOG4_ERROR("Preconditioned matrix contains non-finite values.");
        } else
            x0.col(x0_col_i++) = arma::vectorise(precond);
    }

#if 0 // PETSc linear solver seems unstable (DGMRES with Jacobi preconditioner), TODO Update for layers
    const solvers::antisymmetric_solver solver(m_, n_, k_, iter_opt, x0_col_i ? x0.mem : nullptr, K.mem, residuals_ptr, false, iter_irwls);
    const auto err = solver(x0.colptr(x0_col_i));
    LOG4_DEBUG("Linear solver error " << err << ", weight layer " << wcol);
    if (x0.col(x0_col_i).has_nonfinite()) x0.col(x0_col_i).zeros();
    else ++x0_col_i;
#endif

    if (x0_col_i < x0.n_cols) x0.shed_cols(x0_col_i, x0.n_cols - 1); // Remove empty columns
    static const auto max_weight_calc = CDIVI(C_n_cpu, PROPS.get_solve_particles());

    // Pruned BiteOpt, takes previous solutions in x0 per-column and optimizes upon them
    static std::counting_semaphore<> sem(max_weight_calc);
    sem.acquire();
    common::AppConfig::set_global_log_level(boost::log::trivial::info);
    const optimizer::t_pprune_res res = optimizer::pprune(
            optimizer::pprune::C_default_algo, PROPS.get_solve_particles(), bounds, loss_fun, iter_opt, 0, 0, x0, {}, common::iter_depth(iter_opt), false, pop_opt);
    memcpy(weights.memptr(), res.best_parameters.mem, layers * W_elem * sizeof(double));
    common::AppConfig::set_global_log_level(PROPS.get_log_level());
    sem.release();

    LOG4_TRACE("MAE " << res.best_score << ", layers " << layers << ", iterations " << iter_opt << ", IRWLS " << iter_irwls << ", labels " << common::present(L) << ", weights "
                      << common::present(weights) << ", max weight calculations " << max_weight_calc);
    if (use_gpu) delete sw;

    return res.best_score;
}

void OnlineSVR::calc_weights(const uint16_t chunk_ix, const uint32_t iter_opt, const uint16_t iter_irwls)
{
    LOG4_BEGIN();
    assert(train_label_chunks[chunk_ix].n_cols == 1);
    assert(chunks_score.size() > chunk_ix);
    auto &param = get_params(chunk_ix);
    const uint32_t n = train_feature_chunks_t[chunk_ix].n_cols;
    const auto il = PROPS.get_weight_ileave();
    const auto ni = std::max<uint32_t>(1, n / il);
    arma::mat L;
    arma::mat K;
    if (ni == 1) {
        K = p_kernel_matrices->at(chunk_ix);
        L = train_label_chunks[chunk_ix];
    } else {
        arma::mat F_interlaced(train_feature_chunks_t[chunk_ix].n_rows, n * ni);
        L.set_size(n * ni, train_label_chunks[chunk_ix].n_cols);
        OMP_FOR_(F_interlaced.n_cols, SSIMD collapse(2))
        for (DTYPE(n) i = 0; i < n; ++i)
            for (DTYPE(ni) j = 0; j < ni; ++j) {
                const auto out_pos = i * ni + j;
                const auto j_ = j * il;
                if (i == j_ || (j == 0 && i % il)) {
                    F_interlaced.col(out_pos) = train_feature_chunks_t[chunk_ix].col(i);
                    L.row(out_pos) = train_label_chunks[chunk_ix].row(i);
                } else {
#define INTERLACE_OP(X, Y) (X + Y) * .5
// #define INTERLACE_OP(X, Y) X + Y
                    F_interlaced.col(out_pos) = INTERLACE_OP(train_feature_chunks_t[chunk_ix].col(i), train_feature_chunks_t[chunk_ix].col(j_));
                    L.row(out_pos) = INTERLACE_OP(train_label_chunks[chunk_ix].row(i), train_label_chunks[chunk_ix].row(j_));
                }
            }
        K = kernel::IKernel<double>::get(param)->kernel(F_interlaced, train_feature_chunks_t[chunk_ix]);
    }
    {
        const arma::rowvec L_t = train_label_chunks[chunk_ix].t();
        OMP_FOR_i(n) K.row(i) += L_t;
    }
    L *= n;

    PROFILE_INFO(chunks_score[chunk_ix] = calc_weights(weight_chunks[chunk_ix], K, L, iter_opt, iter_irwls),
                 "Calculate weights for " << param << ", chunk score " << chunks_score[chunk_ix] << ", chunk " << chunk_ix << ", iterations " << iter_opt << ", iter IRWLS " << iter_irwls);
    // TODO Test if this chunk scoring technique is better
    // chunks_score[chunk_ix] = common::meanabs<double>(kernel::get_reference_Z(train_label_chunks[chunk_ix]) - p_kernel_matrices->at(chunk_ix));
    const tbb::mutex::scoped_lock wl(weight_chunks_mx);
    if (total_weights.empty()) {
        total_weights.set_size(p_labels->n_rows, 1);
        total_weights.zeros();
    }
    total_weights.rows(ixs[chunk_ix]) += arma::sum(arma::abs(weight_chunks[chunk_ix]), 1);
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
    instance_weights[i] = p_input_weights->rows(ixs[i]);
    train_label_chunks[i] = p_labels->rows(ixs[i]); // % instance_weights[i];
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
    assert(train_label_chunks[i].n_rows == train_feature_chunks_t[i].n_rows);
    business::DQScalingFactorService::scale_features_I(i, gradient, step, lag, features_sf, train_feature_chunks_t[i]);
    business::DQScalingFactorService::scale_labels_I(*p_labels_sf, train_label_chunks[i]);
    LOG4_TRACE("After scaling chunk " << i << ", train labels " << common::present(train_label_chunks[i]) << ", train features " <<
                common::present(train_feature_chunks_t[i]) << ", labels scaling factor " << *p_labels_sf << ", features scaling factors " << features_sf);
}

} // datamodel
} // svr
