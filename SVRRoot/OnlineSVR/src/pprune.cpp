//
// Created by zarko on 3/24/24.
//

#include <oneapi/tbb/mutex.h>
#include <deque>
#include <prima/prima.h>
#include <xoshiro.h>
#include <biteopt/biteopt.h>

#ifdef USE_KNITRO
#include <knitro.h>
#endif

#include "model/DBTable.hpp"
#include "pprune.hpp"
#include "common/compatibility.hpp"
#include "common/defines.h"
#include "util/math_utils.hpp"
#include "util/string_utils.hpp"
#include "sobol.hpp"

namespace svr {
namespace optimizer {
constexpr std::array<double, 1> C_maxfun_drop_coefs{}; // .5};
constexpr double C_default_rhoend = 5e-10;
constexpr double C_default_rhobeg = .25;
constexpr uint32_t C_elect_threshold = 10;
constexpr int32_t C_rand_disperse = 2;

struct t_calfun_data
{
    bool no_elect = true;
    std::shared_ptr<std::deque<t_calfun_data *> > p_particles;
    const t_pprune_cost_fun cost_fun;
    const uint32_t particle_index = 0;
    const pprune *p_pprune;
    double best_f = std::numeric_limits<double>::max();
    uint32_t nf = 0;
    bool zombie = false;

    bool drop(const uint32_t keep_particles);
};

#ifdef USE_KNITRO

#define kn_errchk(cmd) { const auto __err = (cmd); \
    if (__err) LOG4_THROW("KNitro call " #cmd " failed with error " << __err); }

int kn_callback(KN_context_ptr kc,
                CB_context_ptr cb,
                KN_eval_request_ptr const eval_request,
                KN_eval_result_ptr const eval_result,
                void *const user_params)
{
    if (eval_request->type != KN_RC_EVALFC) {
        LOG4_ERROR("Incorrectly called with eval type " << eval_request->type);
        return -1;
    }
    auto p_cost_cb = (t_pprune_cost_fun_ptr)
            user_params;
    int num_iters, D;
    kn_errchk(KN_get_number_iters(kc, &num_iters));
    kn_errchk(KN_get_number_vars(kc, &D));
    PROFILE_MSG((*p_cost_cb)(eval_request->x, eval_result->obj),
                "Cost, score " << *eval_result->obj << ", index " << eval_request->threadID << ", iterations " << num_iters << ", parameters " <<
                common::to_string(eval_request->x, std::min<uint32_t>(4, D)));
    return 0;
}

#endif

class t_biteopt_cost : public CBiteOpt
{
    t_calfun_data_ptr const calfun_data;
    const arma::mat bounds;

public:
    t_biteopt_cost(t_calfun_data_ptr const calfun_data, const arma::mat &bounds, const uint32_t max_population) : calfun_data(calfun_data), bounds(bounds)
    {
        const auto D = bounds.n_rows;
        const auto depth = calfun_data->p_pprune->depth;
        updateDims(D, D > max_population / 3 / depth ? max_population / depth : 20 + D * 3);
    }

    void getMinValues(double *const p) const override
    {
        OMP_FOR_i(bounds.n_rows) p[i] = bounds(i, 0);
    }

    void getMaxValues(double *const p) const override
    {
        OMP_FOR_i(bounds.n_rows) p[i] = bounds(i, 1);
    }

    double optcost(CPTRd x) override
    {
        double f;
        pprune::calfun(x, &f, calfun_data);
        return f;
    }
};


using t_biteopt_cost_ptr = std::shared_ptr<t_biteopt_cost>;
using CBiteRnd_ptr = std::shared_ptr<CBiteRnd>;

void pprune::calfun(CPTRd x, double *const f, t_calfun_data_ptr const calfun_data)
{
    ++(calfun_data->nf);
    if (calfun_data->zombie) {
        *f = calfun_data->best_f;
        return;
    } else {
        calfun_data->cost_fun(x, f);
        if (*f < calfun_data->best_f) {
            LOG4_DEBUG("New best score " << *f << ", previous " << calfun_data->best_f << ", improvement " << common::imprv(*f, calfun_data->best_f) <<
                "pc, at particle " << calfun_data->particle_index << ", calls " << calfun_data->nf);
            calfun_data->best_f = *f;
        }
    }
    if (calfun_data->no_elect) return;
    for (const auto drop_coef: C_maxfun_drop_coefs) {
        if (calfun_data->nf != uint32_t(drop_coef * calfun_data->p_pprune->iter)) continue;
        calfun_data->zombie = calfun_data->drop(calfun_data->p_pprune->depth * uint32_t((1. - drop_coef) * calfun_data->p_pprune->n));
        if (calfun_data->zombie) {
            LOG4_DEBUG("Particle " << calfun_data->particle_index << " is a zombie, dropping it");
            calfun_data->nf = calfun_data->p_pprune->maxfun;
        }
        break;
    }
}

bool t_calfun_data::drop(const uint32_t keep_particles)
{
    std::deque<uint32_t> ixs(p_particles->size());
    std::iota(ixs.begin(), ixs.end(), 0);
#ifdef NDEBUG
    LOG4_DEBUG("Particle index " << particle_index << ", nf " << nf << ", maxfun " << p_pprune->maxfun << ", keep particles " << keep_particles);
#endif
__do_wait:
    const auto nf_d = nf - p_pprune->depth;
    const uint32_t ct = std::count_if(C_default_exec_policy, p_particles->cbegin(), p_particles->cend(), [nf_d](const auto v) { return v && v->nf >= nf_d; });
    if (ct != p_particles->size()) {
        thread_yield_wait__;
        goto __do_wait;
    }
    std::stable_sort(ixs.begin(), ixs.end(), [&](const auto lhs, const auto rhs) { return p_particles->at(lhs)->best_f < p_particles->at(rhs)->best_f; });
    return std::find(C_default_exec_policy, ixs.cbegin(), ixs.cend(), particle_index) - ixs.cbegin() > keep_particles;
}


void prima_progress_callback(
    const int n, const double x[], const double f, const int nf, const int tr, const double cstrv, const int m_nlcon, const double nlconstr[], bool *const terminate)
{
    (void) n;
    (void) m_nlcon;
    (void) nlconstr;
    LOG4_DEBUG("Prima best point parameters " << common::to_string(x, std::min(3, n)) << ", score " << f << ", cstrv " << cstrv <<
        ", iterations " << nf << ", of which in trust regions " << tr);
    *terminate = false;
};


arma::vec pprune::ensure_bounds(CPTRd x, const arma::mat &bounds)
{
    arma::vec xx(bounds.n_rows);
    for (uint32_t i = 0; i < xx.size(); ++i)
        if (x[i] < bounds(i, 0)) xx[i] = bounds(i, 0);
        else if (x[i] > bounds(i, 1)) xx[i] = bounds(i, 1);
        else xx[i] = x[i];
    return xx;
}


pprune::pprune(const e_algo_type algo_type, const uint32_t n_particles, const arma::mat &bounds, const t_pprune_cost_fun &cost_f, const uint32_t maxfun, double rhobeg,
               double rhoend, arma::mat x0, const arma::vec &pows, const uint16_t depth, const bool force_no_elect, const uint32_t max_population) : n(n_particles), D(bounds.n_rows),
    maxfun(maxfun), depth(depth), iter(maxfun / depth), max_population(max_population), bounds(bounds),
    pows(pows), ranges(bounds.col(1) - bounds.col(0)), no_elect(n < C_elect_threshold || maxfun < C_elect_threshold || force_no_elect)
{
    LOG4_BEGIN();

    assert(maxfun >= depth && depth >= 1);
    assert(D > 0);
    if (n < 1) {
        LOG4_WARN("Number of particles is zero, not optimizing.");
        return;
    }
    if (x0.n_rows != D || x0.n_cols != n) {
        arma::mat x0_backup;
        if (x0.n_rows == D && x0.n_cols) x0_backup = x0;
        const auto x0_n_cols = x0.n_cols;
        x0.set_size(D, n - x0_n_cols);
        common::equispaced(x0, bounds, pows);
        if (x0_backup.n_elem) {
            x0 = arma::join_horiz(x0, x0_backup);
            result.best_parameters = x0_backup.col(0);
        } else {
            result.best_parameters = x0.col(0);
        }
    } else {
        result.best_parameters.set_size(D);
        result.best_parameters.zeros();
    }

    if (!std::isnormal(rhobeg)) rhobeg = ranges.front() * C_default_rhobeg;
    if (!std::isnormal(rhoend)) rhoend = ranges.front() * C_default_rhoend;

    switch (algo_type) {
        case e_algo_type::e_biteopt:
            PROFILE_MSG(pprune_biteopt(n_particles, cost_f, rhobeg, rhoend, x0), "PPrune BiteOpt " << n_particles << " particles, " << maxfun << " maxfun, " << depth << " depth");
            break;

        case e_algo_type::e_knitro:
            PROFILE_MSG(pprune_knitro(n_particles, cost_f, rhobeg, rhoend, x0), "PPrune KNitro " << n_particles << " particles, " << maxfun << " maxfun");
            break;

        case e_algo_type::e_prima:
            PROFILE_MSG(pprune_prima(n_particles, cost_f, rhobeg, rhoend, x0), "PPrune Prima " << n_particles << " particles, " << maxfun << " maxfun");
            break;

        case e_algo_type::e_petsc:
            PROFILE_MSG(pprune_petsc(n_particles, cost_f, rhobeg, rhoend, x0), "PPrune PETSc " << n_particles << " particles, " << maxfun << " maxfun");
            break;

        default:
            LOG4_THROW("Unknown optimization algorithm " << algo_type);
    }

    LOG4_END();
}

#ifdef USE_KNITRO

typedef struct KN_init_userparams
{
    const arma::mat &x0;
    const uint32_t D;
} t_KN_init_userparams, *t_KN_init_userparams_ptr;

int KN_ms_initpt_callback(KN_context_ptr kc, const KNINT nSolveNumber, double *const x, double *const lambda, void *const userParams)
{
    const auto p = (t_KN_init_userparams_ptr) userParams;
    if (nSolveNumber >= p->x0.n_cols) {
        LOG4_ERROR("Invalid particle index " << nSolveNumber << ", max " << p->x0.n_cols);
        return 1;
    }
    memcpy(x, p->x0.colptr(nSolveNumber), p->D * sizeof(double));
    memset(lambda, 0, p->D * sizeof(double));
    return 0;
}

#endif

void pprune::pprune_knitro(const uint32_t n_particles, const t_pprune_cost_fun &cost_f, double rhobeg, double rhoend, const arma::mat &x0)
{
#ifdef USE_KNITRO

    /** Create a new Knitro solver instance. */
    KN_context *kc;
    kn_errchk(KN_new(&kc));
    if (kc == NULL)
        LOG4_THROW("Failed to find a valid license.");

    /** Illustrate how to override default options by reading from
     *  the knitro.opt file. */
    kn_errchk(KN_load_param_file(kc, "../config/knitro.opt"));

    /** Initialize Knitro with the problem definition. */

    /** Add the variables and set their bounds.
     *  Note: any unset lower bounds are assumed to be
     *  unbounded below and any unset upper bounds are
     *  assumed to be unbounded above. */
    kn_errchk(KN_add_vars(kc, D, nullptr));
    kn_errchk(KN_set_var_lobnds_all(kc, bounds.colptr(0)));
    kn_errchk(KN_set_var_upbnds_all(kc, bounds.colptr(1)));
    kn_errchk(KN_set_var_primal_init_values_all(kc, x0.colptr(x0.n_cols - 1)));

    const KN_init_userparams init_user_params{x0, D};
    kn_errchk(KN_set_ms_initpt_callback(kc, KN_ms_initpt_callback, (void *const) &init_user_params));

    /** Add a callback function "kn_callback" to evaluate the nonlinear
     *  (non-quadratic) objective.  Note that the linear and
     *  quadratic terms in the objective could be loaded separately
     *  via "KN_add_obj_linear_struct()" / "KN_add_obj_quadratic_struct()".
     *  However, for simplicity, we evaluate the whole objective
     *  function through the callback. */
    /** Pointer to structure holding information for callback */
    CB_context *cb;
    kn_errchk(KN_add_eval_callback(kc, KNTRUE, 0, nullptr, kn_callback, &cb));
    // auto calfun_data = new t_calfun_data{no_elect, elect_ready, elect_mx, f_score, cost_f, i, maxfun};
    kn_errchk(KN_set_cb_user_params(kc, cb, (void *) &cost_f));

    /** Set the non-default SQP algorithm, which typically converges in the
     *  fewest number of function evaluations.  This algorithm (or the
     *  active-set algorithm ("KN_ALG_ACT_CG") may be preferable for
     *  derivative-free optimization models with expensive function
     *  evaluations. */
    kn_errchk(KN_set_int_param(kc, KN_PARAM_ALGORITHM, KN_ALG_ACT_SQP));
    /** Enable multi-start */
    kn_errchk(KN_set_int_param(kc, KN_PARAM_MULTISTART, KN_MS_ENABLE_YES));

    kn_errchk(KN_set_int_param(kc, KN_PARAM_MS_MAXSOLVES, n_particles));
    kn_errchk(KN_set_int_param(kc, KN_PARAM_MAXIT, maxfun));

    auto gen = common::reproducibly_seeded_64<xso::rng64>();
    std::uniform_real_distribution<double> dis(0., 1.);
    /** Perform multistart in parallel using max number of available threads */
    LOG4_TRACE("Running KNitro multistart in parallel with " << C_n_cpu << " threads.");
    kn_errchk(KN_set_int_param(kc, KN_PARAM_CONCURRENT_EVALS, KN_CONCURRENT_EVALS_YES));
    kn_errchk(KN_set_int_param(kc, KN_PARAM_MS_SEED, dis(gen)));
    kn_errchk(KN_set_int_param(kc, KN_PARAM_HONORBNDS, KN_HONORBNDS_ALWAYS));

    /** Solve the problem.
     *
     *  Return status codes are defined in "knitro.h" and described
     *  in the Knitro manual.
     */
    auto n_status = KN_solve(kc);
    /*              0: the final solution is optimal to specified tolerances;
     *   -100 to -109: a feasible solution was found (but not verified optimal);
     *   -200 to -209: Knitro terminated at an infeasible point;
     *   -300 to -301: the problem was determined to be unbounded;
     *   -400 to -409: Knitro terminated because it reached a pre-defined limit
     *                (a feasible point was found before reaching the limit);
     *   -410 to -419: Knitro terminated because it reached a pre-defined limit
     *                (no feasible point was found before reaching the limit);
     *   -500 to -599: Knitro terminated with an input error or some non-standard error.
    */
    LOG4_DEBUG("KNitro solve status " << n_status);

    /** Allocate arrays to hold solution.
     *  Notice lambda has multipliers for both constraints and bounds
     *  (a common mistake is to allocate its size as m). */
    auto lambda = (double *) malloc(2 * D * sizeof(double));
    result.best_parameters.set_size(D);
    int total_iterations;
    KN_get_number_iters(kc, &total_iterations);
    result.total_iterations = total_iterations ? total_iterations : maxfun * n_particles;

    /** An example of obtaining solution information. */
    kn_errchk(KN_get_solution(kc, &n_status, &result.best_score, result.best_parameters.memptr(), lambda));
    LOG4_DEBUG("Optimal objective value " << result.best_score << ", status " << n_status <<
        ", optimal solution (with corresponding multiplier): " << common::present(result.best_parameters) <<
        ", lambda " << common::to_string(lambda + D, std::min<uint32_t>(D, 4)));
    double feas_error, opt_error;
    kn_errchk(KN_get_abs_feas_error(kc, &feas_error));
    LOG4_DEBUG("Feasibility violation " << feas_error);
    kn_errchk(KN_get_abs_opt_error(kc, &opt_error));
    LOG4_DEBUG("Optimality violation " << opt_error);

    /** Delete the Knitro solver instance. */
    KN_free(&kc);
    free(lambda);
    LOG4_DEBUG("PPrune type KNitro, score " << result.best_score << ", total iterations " << result.total_iterations << ", particles " << n << ", parameters " << D <<
        ", max iterations per particle " << maxfun << ", var start " << rhobeg << ", var end " << rhoend << ", bounds " << common::present(bounds) << ", ranges " <<
        common::present(ranges));
#else

    LOG4_THROW("KNitro is not available.");

#endif
}


struct t_biteopt_prune
{
    std::shared_ptr<CBiteRnd> rnd;
    std::shared_ptr<t_biteopt_cost> biteopt;
    t_calfun_data_ptr calfun_data = nullptr;
};

void pprune::pprune_biteopt(const uint32_t n_particles, const t_pprune_cost_fun &cost_f, double rhobeg, double rhoend, const arma::mat &x0)
{
    LOG4_BEGIN();

    auto p_particles = ptr<std::deque<t_calfun_data_ptr> >(n_particles);
    std::deque<std::deque<t_biteopt_prune> > biteopt_particle(n_particles, std::deque<t_biteopt_prune>(depth));
    OMP_FOR_(n_particles * depth, SSIMD firstprivate(n_particles, maxfun, no_elect, C_rand_disperse) collapse(2))
    for (uint32_t i = 0; i < n_particles; ++i) {
        for (uint32_t d = 0; d < depth; ++d) {
            auto &rnd = biteopt_particle[i][d].rnd;
            auto &biteopt = biteopt_particle[i][d].biteopt;
            auto &calfun_data = biteopt_particle[i][d].calfun_data;
            const auto particle_index = i * depth + d;
            calfun_data = p_particles->at(i) = new t_calfun_data{no_elect, p_particles, cost_f, particle_index, this};
            rnd = ptr<CBiteRnd>(std::pow(C_rand_disperse * particle_index, C_rand_disperse));
            biteopt = otr<t_biteopt_cost>(calfun_data, bounds, max_population);
            biteopt->init(*rnd, x0.colptr(i));
        }
    }

    tbb::task_arena tta(adj_threads(n_particles));
    for (DTYPE(iter) j = 0; j < iter; ++j) {
        tta.execute([&] {
            tbb_zpfor_i__(n_particles,
                for (DTYPE(depth) d = 0; d < depth; ++d) {
                    if (biteopt_particle[i][d].calfun_data->zombie) continue;
                    const auto &biteopt = biteopt_particle[i][d].biteopt;
                    auto &rnd = biteopt_particle[i][d].rnd;
                    biteopt->optimize(*rnd, d + 1 >= depth ? nullptr : biteopt_particle[i][d + 1].biteopt.get());
                }
            )
        });
    }
    const auto D_size = D * sizeof(double);
    for (uint16_t i = 0; i < n_particles; ++i) {
        for (DTYPE(depth) d = 0; d < depth; ++d) {
            auto &rnd = biteopt_particle[i][d].rnd;
            auto &biteopt = biteopt_particle[i][d].biteopt;
            const auto this_best_score = biteopt->getBestCost();
            if (this_best_score < result.best_score) {
                result.best_score = this_best_score;
                memcpy(result.best_parameters.memptr(), biteopt->getBestParams(), D_size);
                LOG4_DEBUG("Best score " << result.best_score << " at particle " << i << ", parameters " << common::present(result.best_parameters));
            }
            rnd.reset();
            biteopt.reset();
            delete biteopt_particle[i][d].calfun_data;
        }
    }

    result.total_iterations = maxfun;
    if (result.best_parameters.has_nonfinite())
        LOG4_THROW("Best parameters contain non-finite values " << common::present(result.best_parameters));
    LOG4_DEBUG(
        "PPrune BiteOpt, final score " << result.best_score << ", total iterations " << result.total_iterations << ", particles " << n << ", parameters " << D <<
        ", max iterations per particle " << maxfun << ", var start " << rhobeg << ", var end " << rhoend << ", best parameters " << common::present(result.best_parameters));
}


void pprune::pprune_prima(const uint32_t n_particles, const t_pprune_cost_fun &cost_f, double rhobeg, double rhoend, const arma::mat &x0)
{
    prima_problem_t all_problem;
    prima_init_problem(&all_problem, D);
    all_problem.calfun = (prima_obj_t) calfun;
    all_problem.xl = (double *) bounds.colptr(0);
    all_problem.xu = (double *) bounds.colptr(1);
    prima_options_t all_prima_options;
    prima_init_options(&all_prima_options);
    all_prima_options.npt = std::min<DTYPE(all_prima_options.npt) >(maxfun - 1, (D + 2. + (D + 1.) * (D + 2.) / 2.) / 2.);
    all_prima_options.iprint = PRIMA_MSG_EXIT;
    all_prima_options.rhobeg = rhobeg;
    all_prima_options.rhoend = rhoend;
    all_prima_options.maxfun = maxfun;
    all_prima_options.callback = prima_progress_callback;

    auto p_particles = ptr<std::deque<t_calfun_data_ptr> >(n_particles);
    tbb::mutex res_l;
#pragma omp parallel for SSIMD num_threads(n_particles) schedule(static, 1) firstprivate(n_particles, maxfun, no_elect) default(shared)
    for (uint32_t i = 0; i < n_particles; ++i) {
        all_prima_options.data = p_particles->at(i) = new t_calfun_data{no_elect, p_particles, cost_f, i, this};

        auto problem = all_problem;
        problem.x0 = (double *) x0.colptr(i);

        auto prima_options = all_prima_options;

        prima_result_t prima_result;
        const auto rc = prima_minimize(PRIMA_LINCOA, problem, prima_options, &prima_result);
        LOG4_DEBUG("Prima particle " << i <<
            " final parameters " << common::to_string(prima_result.x, _MIN(3, D)) <<
            ", score " << prima_result.f <<
            ", cstrv " << prima_result.cstrv <<
            ", return code " << rc <<
            /* ", " << prima_get_rc_string(static_cast<const prima_rc_t>(rc))  << */ ", message '" << prima_result.message << "', iterations "
            << prima_result.nf);

        tbb::mutex::scoped_lock lk(res_l);
        if (prima_result.f < result.best_score) {
            result.best_score = prima_result.f;
            result.best_parameters = arma::vec(prima_result.x, D, true, true);
            LOG4_TRACE("New best particle " << result.best_score << " at particle " << i << ", parameters " << common::present(result.best_parameters));
        }
        result.total_iterations += prima_result.nf;
        lk.release();
        prima_free_result(&prima_result);
    }

    for (auto &f: *p_particles) delete f; // Keep this out of the for loop above
    if (result.best_parameters.has_nonfinite())
        LOG4_THROW("Best parameters contain non-finite values " << common::present(result.best_parameters));
    LOG4_DEBUG(
        "PPrune Prima, final score " << result.best_score << ", total iterations " << result.total_iterations << ", particles " << n << ", parameters " << D <<
        ", max iterations per particle " << maxfun << ", var start " << rhobeg << ", var end " << rhoend << ", best parameters " << common::present(result.best_parameters));
}


void pprune::pprune_petsc(const uint32_t n_particles, const t_pprune_cost_fun &cost_f, double rhobeg, double rhoend, const arma::mat &x0)
{
#if 0
    auto p_particles = ptr<std::deque<t_calfun_data_ptr>>(n_particles);

    tbb::mutex res_l;
#pragma omp parallel ADJ_THREADS(n_particles)
#pragma omp single
    {
#pragma omp taskloop simd mergeable default(shared) grainsize(1) firstprivate(maxfun, no_elect) untied // Untied task is a must if election is used
        for (uint32_t i = 0; i < n_particles; ++i) {
            auto const calfun_data = p_particles->at(i) = new t_calfun_data{no_elect, p_particles, cost_f, i, maxfun, D};


            ierr = VecCreate(PETSC_COMM_SELF, &x);
            CHKERRQ(ierr);
            ierr = VecSetSizes(x, PETSC_DECIDE, D);
            CHKERRQ(ierr);
            ierr = VecSetFromOptions(x);
            CHKERRQ(ierr);

            CBiteRnd rnd(std::pow(C_rand_disperse * i, C_rand_disperse));
            std::deque<std::shared_ptr<t_biteopt_cost>> biteopt(depth);
            UNROLL()
            for (auto &o: biteopt) {
                o = std::make_shared<t_biteopt_cost>(calfun_data, bounds, max_population);
                o->init(rnd, x0.colptr(i));
            }
            UNROLL()
            for (uint32_t j = 0; j < maxfun / biteopt.size(); ++j) {
                for (uint32_t d = 0; d < biteopt.size(); ++d)
                    biteopt[d]->optimize(rnd, d + 1 >= biteopt.size() ? nullptr : biteopt[d + 1].get());
                if (calfun_data->zombie) {
                    calfun_data->nf = maxfun;
                    break;
                }
            }

            const tbb::mutex::scoped_lock lk(res_l));
            if (biteopt.back()->getBestCost() < result.best_score) {
                result.best_score = biteopt.back()->getBestCost();
                result.best_parameters = arma::vec((double *) biteopt.back()->getBestParams(), D, true, true);
                LOG4_TRACE("New best score " << result.best_score << " at particle " << i << ", parameters " << common::present(result.best_parameters));
            }
            result.total_iterations += maxfun;
        }
    }

    for (auto &f: *p_particles) delete f; // Keep this out of the for loop above
    if (result.best_parameters.has_nonfinite()) LOG4_THROW("Best parameters contain non-finite values " << common::present(result.best_parameters));
    LOG4_DEBUG(
            "PPrune PETSC, score " << result.best_score << ", total iterations " << result.total_iterations << ", particles " << n << ", parameters " << D <<
                                     ", max iterations per particle " << maxfun << ", var start " << rhobeg << ", var end " << rhoend << ", best parameters "
                                     << common::present(result.best_parameters));
#endif
}


pprune::operator t_pprune_res() const noexcept
{
    if (!std::isnormal(result.best_score) || result.best_parameters.empty() || !common::isnormalz(result.total_iterations))
        LOG4_THROW("No valid solution found.");
    return result;
}
}
}
