//
// Created by zarko on 3/24/24.
//

// #undef USE_PRIMA

#ifdef USE_BITEOPT

constexpr unsigned C_max_population = 3333;

#include <biteopt/biteopt.h>

#endif

#include <oneapi/tbb/mutex.h>
#include <deque>
#include <prima/prima.h>
#include "model/DBTable.hpp"
#include "pprune.hpp"
#include "common/compatibility.hpp"
#include "common/defines.h"
#include "util/math_utils.hpp"
#include "sobol.hpp"
#include "util/string_utils.hpp"

constexpr std::array<double, 1> C_maxfun_drop_coefs{.5};

#ifdef USE_KNITRO

#include "knitro.h"

#else

constexpr double C_default_rhoend = 5e-10;
constexpr double C_default_rhobeg = .25;

#endif

namespace svr {
namespace optimizer {

struct t_calfun_data {
    bool no_elect = true;
    std::shared_ptr<std::deque<t_calfun_data *>> p_particles;
    const t_pprune_cost_fun cost_fun;
    const unsigned particle_index = 0;
    const unsigned maxfun = 0;
    const unsigned D = 0;
    double best_f = std::numeric_limits<double>::max();
    unsigned nf = 0;
    bool zombie = false;

    bool drop(const unsigned keep_particles);
};

#ifdef USE_KNITRO

#define kn_errchk(cmd) { const auto __err = (cmd); \
    if (__err) LOG4_THROW("KNitro call " #cmd " failed with error " << __err); }

int callbackEvalF(KN_context_ptr kc,
                  CB_context_ptr cb,
                  KN_eval_request_ptr const eval_request,
                  KN_eval_result_ptr const eval_result,
                  void *const user_params)
{
    if (eval_request->type != KN_RC_EVALFC) {
        LOG4_ERROR("Incorrectly called with eval type " << eval_request->type);
        return -1;
    }
    auto p_cost_cb = (t_pprune_cost_fun_ptr) user_params;
    int num_iters, D;
    kn_errchk(KN_get_number_iters(kc, &num_iters));
    kn_errchk(KN_get_number_vars(kc, &D));
    PROFILE_EXEC_TIME( (*p_cost_cb)(eval_request->x, eval_result->obj),
       "Cost, score " << *eval_result->obj << ", index " << eval_request->threadID << ", iterations " << num_iters << ", parameters " <<
       common::to_string(eval_request->x, std::min<unsigned>(4, D)));
    return 0;
}

#else

constexpr unsigned C_elect_threshold = 10;

#endif

#ifdef USE_BITEOPT

constexpr int C_rand_disperse = 2;

class t_biteopt_cost : public CBiteOpt {
    t_calfun_data_ptr const calfun_data;
    const arma::mat bounds;
public:
    t_biteopt_cost(t_calfun_data_ptr const calfun_data, const arma::mat &bounds) : calfun_data(calfun_data), bounds(bounds)
    {
        constexpr auto max_population_3 = C_max_population / 3;
        const auto D = bounds.n_rows;
        updateDims(D, D > max_population_3 ? C_max_population : 20 + D * 3);
    }

    void getMinValues(double *const p) const override
    {
        OMP_FOR(bounds.n_rows)
        for (unsigned i = 0; i < bounds.n_rows; ++i) p[i] = bounds(i, 0);
    }

    void getMaxValues(double *const p) const override
    {
        OMP_FOR(bounds.n_rows)
        for (unsigned i = 0; i < bounds.n_rows; ++i) p[i] = bounds(i, 1);
    }

    double optcost(CPTR(double) x) override
    {
        double f;
        pprune::calfun(x, &f, calfun_data);
        return f;
    }
};

using t_biteopt_cost_ptr = std::shared_ptr<t_biteopt_cost>;
using CBiteRnd_ptr = std::shared_ptr<CBiteRnd>;

#endif


void pprune::calfun(CPTR(double) x, double *const f, t_calfun_data_ptr calfun_data)
{
    ++calfun_data->nf;
    if (!calfun_data->zombie) {
        PROFILE_EXEC_TIME(calfun_data->cost_fun(x, f), "Cost, score " << *f << ", call count " << calfun_data->nf << ", index " << calfun_data->particle_index
            << ", parameters " << common::to_string(x, std::min<unsigned>(4, calfun_data->D)));
        if (*f < calfun_data->best_f) {
            LOG4_DEBUG("New best score " << *f << ", previous " << calfun_data->best_f << ", improvement " << common::imprv(*f, calfun_data->best_f) <<
                "pc, at particle " << calfun_data->particle_index << ", calls " << calfun_data->nf);
            calfun_data->best_f = *f;
        }
    }
    if (calfun_data->no_elect || calfun_data->zombie) return;
    UNROLL()
    for (const auto drop_coef: C_maxfun_drop_coefs) {
        if (calfun_data->nf != unsigned(drop_coef * calfun_data->maxfun)) continue;
        calfun_data->zombie = calfun_data->drop((1. - drop_coef) * calfun_data->p_particles->size());
        break;
    }
}


bool t_calfun_data::drop(const unsigned keep_particles)
{
    std::deque<unsigned> ixs(p_particles->size());
    std::iota(ixs.begin(), ixs.end(), 0);

    __do_wait:
    const unsigned ct = std::count_if(C_default_exec_policy, p_particles->cbegin(), p_particles->cend(), [this](const auto v) { return v && v->nf >= nf; });
    if (ct != p_particles->size()) {
        task_yield_wait__;
        goto __do_wait;
    }
    std::stable_sort(C_default_exec_policy, ixs.begin(), ixs.end(),
                     [&](const auto lhs, const auto rhs) { return p_particles->at(lhs)->best_f < p_particles->at(rhs)->best_f; });

    return std::find(C_default_exec_policy, ixs.cbegin(), ixs.cend(), particle_index) - ixs.cbegin() > keep_particles;
}

#ifdef USE_PRIMA

void prima_progress_callback(
        const int n, const double x[], const double f, const int nf,
        const int tr, const double cstrv, const int m_nlcon, const double nlconstr[], bool *const terminate)
{
    (void) n;
    (void) m_nlcon;
    (void) nlconstr;
    LOG4_DEBUG("Prima best point parameters " << common::to_string(x, std::min(3, n)) << ", score " << f << ", cstrv " << cstrv <<
                                              ", iterations " << nf << ", of which in trust regions " << tr);
    *terminate = false;
};

#endif

arma::vec pprune::ensure_bounds(CPTR(double) x, const arma::mat &bounds)
{
    arma::vec xx(bounds.n_rows);
    for (unsigned i = 0; i < xx.size(); ++i)
        if (x[i] < bounds(i, 0)) xx[i] = bounds(i, 0);
        else if (x[i] > bounds(i, 1)) xx[i] = bounds(i, 1);
        else xx[i] = x[i];
    return xx;
}

#ifdef USE_KNITRO // TODO Test this

pprune::pprune(const unsigned algo_type, const unsigned n_particles, const arma::mat &bounds,
               const t_pprune_cost_fun &cost_f,
               const unsigned maxfun, double rhobeg, double rhoend,
               arma::mat x0, const arma::vec &pows, const unsigned depth) :
        n(n_particles), D(bounds.n_rows), bounds(bounds), pows(pows), ranges(bounds.col(1) - bounds.col(0))
{
    if (x0.n_rows != D || x0.n_cols != n) {
        x0.set_size(D, n);
        common::equispaced(x0, bounds, pows);
    }
    /** Create a new Knitro solver instance. */
    KN_context *kc;
    kn_errchk(KN_new(&kc));
    if (kc == NULL) LOG4_THROW("Failed to find a valid license.");

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
    kn_errchk(KN_set_var_primal_init_values_all(kc, x0.memptr()));

    /** Add a callback function "callbackEvalF" to evaluate the nonlinear
     *  (non-quadratic) objective.  Note that the linear and
     *  quadratic terms in the objective could be loaded separately
     *  via "KN_add_obj_linear_struct()" / "KN_add_obj_quadratic_struct()".
     *  However, for simplicity, we evaluate the whole objective
     *  function through the callback. */
    /** Pointer to structure holding information for callback */
    CB_context *cb;
    kn_errchk(KN_add_eval_callback(kc, KNTRUE, 0, nullptr, callbackEvalF, &cb));
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

    /** Perform multistart in parallel using max number of available threads */
    const auto n_threads = C_n_cpu;
    if (n_threads > 1) {
        LOG4_TRACE("Running KNitro multistart in parallel with " << n_threads << " threads.");
        kn_errchk(KN_set_int_param(kc, KN_PARAM_PAR_MSNUMTHREADS, n_threads));
    }

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
    result.best_parameters = arma::vec(D);
    int total_iterations;
    KN_get_number_iters(kc, &total_iterations);
    result.total_iterations = total_iterations ? total_iterations : maxfun * n_particles;

    /** An example of obtaining solution information. */
    kn_errchk(KN_get_solution(kc, &n_status, &result.best_score, result.best_parameters.memptr(), lambda));
    LOG4_DEBUG("Optimal objective value " << result.best_score << ", status " << n_status);
    LOG4_DEBUG("Optimal x (with corresponding multiplier)");
    for (unsigned i = 0; i < std::min<unsigned>(4, D); ++i) LOG4_TRACE("  x[" << i << "] " << result.best_parameters[i] << " (lambda " << lambda[D + i]);
    double feas_error, opt_error;
    kn_errchk(KN_get_abs_feas_error(kc, &feas_error));
    LOG4_DEBUG("Feasibility violation " << feas_error);
    kn_errchk(KN_get_abs_opt_error(kc, &opt_error));
    LOG4_DEBUG("Optimality violation " << opt_error);

    /** Delete the Knitro solver instance. */
    KN_free(&kc);
    free(lambda);
    LOG4_DEBUG(
            "PPrune type KNitro, score " << result.best_score << ", total iterations " << result.total_iterations << ", particles " << n << ", parameters " << D
           << ", max iterations per particle " << maxfun << ", var start " << rhobeg << ", var end " << rhoend << ", bounds " << bounds << ", ranges " << ranges);
}

#else

pprune::pprune(const unsigned algo_type, const unsigned n_particles, const arma::mat &bounds,
               const t_pprune_cost_fun &cost_f,
               const unsigned maxfun, double rhobeg, double rhoend,
               arma::mat x0, const arma::vec &pows, const unsigned depth) : // TODO Implement depth
        n(n_particles), D(bounds.n_rows), bounds(bounds), pows(pows), ranges(bounds.col(1) - bounds.col(0))
{
    if (!std::isnormal(rhobeg)) rhobeg = ranges.front() * C_default_rhobeg;
    if (!std::isnormal(rhoend)) rhoend = ranges.front() * C_default_rhoend;
    if (x0.n_rows != D || x0.n_cols != n) {
        arma::mat x0_backup;
        if (x0.n_rows == D && x0.n_cols) x0_backup = x0;
        x0.set_size(D, n);
        common::equispaced(x0, bounds, pows);
        if (x0_backup.n_elem) x0.cols(x0.n_cols - x0_backup.n_cols, x0.n_cols - 1) = x0_backup;
    }

#ifdef USE_PRIMA
    prima_problem_t all_problem;
    prima_init_problem(&all_problem, D);
    all_problem.calfun = calfun;
    all_problem.xl = (double *)bounds.colptr(0);
    all_problem.xu = (double *)bounds.colptr(1);
    prima_options_t all_prima_options;
    prima_init_options(&all_prima_options);
    all_prima_options.npt = std::min<dtype(all_prima_options.npt)>(maxfun - 1, (D + 2. + (D + 1.) * (D + 2.) / 2.) / 2.);
    all_prima_options.iprint = PRIMA_MSG_EXIT;
    all_prima_options.rhobeg = rhobeg;
    all_prima_options.rhoend = rhoend;
    all_prima_options.maxfun = maxfun;
    all_prima_options.callback = prima_progress_callback;
#endif

    const auto no_elect = n < C_elect_threshold || maxfun < C_elect_threshold;
    auto p_particles = ptr<std::deque<t_calfun_data_ptr>>(n_particles);

#ifdef USE_BITEOPT

    t_omp_lock res_l;
#pragma omp parallel ADJ_THREADS(n_particles)
#pragma omp single
    {
#pragma omp taskloop simd mergeable default(shared) grainsize(1) firstprivate(maxfun, no_elect, C_rand_disperse, C_biteopt_depth) untied // Untied task is a must
        for (unsigned i = 0; i < n_particles; ++i) {
            auto const calfun_data = p_particles->at(i) = new t_calfun_data{no_elect, p_particles, cost_f, i, maxfun, D};
#endif
#ifdef USE_BITEOPT
            CBiteRnd rnd(std::pow(C_rand_disperse * i, C_rand_disperse));
            std::deque<std::shared_ptr<t_biteopt_cost>> biteopt(depth);
UNROLL()
            for (auto &o: biteopt) {
                o = std::make_shared<t_biteopt_cost>(calfun_data, bounds);
                o->init(rnd, x0.colptr(i));
            }
UNROLL()
            for (unsigned j = 0; j < maxfun / biteopt.size(); ++j) {
                for (unsigned d = 0; d < biteopt.size(); ++d)
                    biteopt[d]->optimize(rnd, d + 1 >= biteopt.size() ? nullptr : biteopt[d + 1].get());
                if (calfun_data->zombie) {
                    calfun_data->nf = maxfun;
                    break;
                }
            }

            res_l.set();
            if (biteopt.back()->getBestCost() < result.best_score) {
                result.best_score = biteopt.back()->getBestCost();
                result.best_parameters = arma::vec((double *) biteopt.back()->getBestParams(), D, true, true);
                LOG4_TRACE("New best score " << result.best_score << " at particle " << i << ", parameters " << common::present(result.best_parameters));
            }
            result.total_iterations += maxfun;
            res_l.unset();
        }
    }

#elif defined(USE_PRIMA)

#pragma omp parallel num_threads(n_particles) // Start as many threads as particles
#pragma omp single
    {
#pragma omp taskloop simd mergeable default(shared) grainsize(1) untied firstprivate(n_particles, maxfun, no_elect, C_rand_disperse)
        for (unsigned i = 0; i < n_particles; ++i) {
            auto const calfun_data = f_score->at(i) = new t_calfun_data{no_elect, elect_ready, elect_mx, f_score, cost_f, i, maxfun};

            auto problem = all_problem;
            problem.x0 = (double *) x0.colptr(i);

            auto prima_options = all_prima_options;
            prima_options.data = f_score->at(i) = &calfun_data;

            prima_result_t prima_result;
            const auto rc = prima_minimize(algo_type, problem, prima_options, &prima_result);
            LOG4_DEBUG("pprune particle " << i << " final parameters " << common::to_string(prima_result.x, std::min(3, D)) << ", score " << prima_result.f << ", cstrv " <<
                                          prima_result.cstrv << ", return code " << rc
                                          << /* ", " << prima_get_rc_string(static_cast<const prima_rc_t>(rc))  << */ ", message '"
                                          << prima_result.message << "', iterations " << prima_result.nf);

            const tbb::mutex::scoped_lock l(mx);
            if (prima_result.f < result.best_score) {
                result.best_score = prima_result.f;
                result.best_parameters = arma::vec(prima_result.x, D, true, true);
                LOG4_TRACE("New best score " << result.best_score << " at particle " << i << ", parameters " <<
                    common::present(result.best_parameters) << ", from " << x[0] << ", " << x[1]);
            }
            result.total_iterations += prima_result.nf;
            prima_free_result(&prima_result);
    }
}

#endif

    for (auto &f: *p_particles) delete f; // Keep this out of the for loop above
    if (result.best_parameters.has_nonfinite()) LOG4_THROW("Best parameters contain non-finite values " << common::present(result.best_parameters));
    LOG4_DEBUG(
            "PPrune type " << algo_type << ", score " << result.best_score << ", total iterations " << result.total_iterations << ", particles " << n << ", parameters " << D <<
            ", max iterations per particle " << maxfun << ", var start " << rhobeg << ", var end " << rhoend << ", best parameters " << common::present(result.best_parameters));
}

#endif

pprune::operator t_pprune_res()
{
    if (!std::isnormal(result.best_score) || result.best_parameters.empty() || !common::isnormalz(result.total_iterations)) LOG4_THROW("No valid solution found.");
    return result;
}

}
}
