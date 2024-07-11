//
// Created by zarko on 3/24/24.
//

// #define USE_PRIMA
#define USE_BITEOPT
#undef USE_KNITRO

#ifdef USE_BITEOPT
#include <biteopt/biteopt.h>
#endif

#include "pprune.hpp"
#include "common/compatibility.hpp"
#include "common/defines.h"
#include "util/math_utils.hpp"
#include "sobolvec.h"
#include "util/string_utils.hpp"

#ifdef USE_KNITRO

#include "knitro.h"

#endif

namespace svr {
namespace optimizer {

#ifdef USE_KNITRO

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
    (*p_cost_cb)(eval_request->x, eval_result->obj);
    return 0;
}

#endif

#ifdef USE_BITEOPT

constexpr int C_rand_disperse = 2;
constexpr unsigned C_biteopt_depth = 1;

class t_biteopt_cost : public CBiteOpt
{
    t_calfun_data_ptr const calfun_data;
    const arma::mat bounds;
    const unsigned D;
public:
    t_biteopt_cost(t_calfun_data_ptr const calfun_data, const arma::mat &bounds) :
            calfun_data(calfun_data), bounds(bounds), D(bounds.n_rows)
    {
        updateDims(D);
    }

    void getMinValues(double *const p) const override
    {
        for (unsigned i = 0; i < D; ++i) p[i] = bounds(i, 0);
    }

    void getMaxValues(double *const p) const override
    {
        for (unsigned i = 0; i < D; ++i) p[i] = bounds(i, 1);
    }

    double optcost(const double *const x) override
    {
        double f;
        pprune::calfun(x, &f, calfun_data);
        return f;
    }
};

#endif


void pprune::calfun(const double x[], double *const f, t_calfun_data_ptr calfun_data)
{
    constexpr std::array<double, 1> maxfun_drop_coefs {.5};
    ++calfun_data->nf;
    if (!calfun_data->zombie) {
        PROFILE_EXEC_TIME(calfun_data->cost_fun(x, f), "Cost, score " << *f << ", call count " << calfun_data->nf << ", index " << calfun_data->particle_index);
        if (*f < calfun_data->best_f) calfun_data->best_f = *f;
    }
    if (calfun_data->no_elect) return;
    calfun_data->elect_ready->notify_all(); // Should be before cv wait
    if (calfun_data->zombie) return;
    for (const auto drop_coef: maxfun_drop_coefs) {
        if (calfun_data->nf != size_t(drop_coef * calfun_data->maxfun)) continue;
        calfun_data->zombie = calfun_data->drop((1. - drop_coef) * calfun_data->f_score->size());
        break;
    }
}

bool t_calfun_data::drop(const size_t keep_particles)
{
    std::deque<size_t> ixs(f_score->size());
    std::iota(ixs.begin(), ixs.end(), 0);
    std::unique_lock lk(*elect_mx);
    elect_ready->wait(lk, [this] {
        const auto ct = std::count_if(
                f_score->cbegin(), f_score->cend(), [this](const auto v) {
                    return v->best_f != std::numeric_limits<double>::quiet_NaN() && v->nf >= nf;
                });
        LOG4_TRACE("Visited " << ct << ", calls " << nf << ", particle " << particle_index);
        return ct == f_score->size();
    });
    std::stable_sort(std::execution::par_unseq, ixs.begin(), ixs.end(),
                     [&](const auto lhs, const auto rhs) { return f_score->at(lhs)->best_f < f_score->at(rhs)->best_f; });
    lk.unlock();
    return std::find(ixs.cbegin(), ixs.cend(), particle_index) - ixs.cbegin() > keep_particles;
}


void equispaced(arma::mat &x0, const arma::mat &bounds, const arma::vec &pows, unsigned long sobol_ctr = 0)
{
    static const auto default_sobol_ctr = common::init_sobol_ctr();
    if (!sobol_ctr) sobol_ctr = default_sobol_ctr;
    const auto n = x0.n_cols;
    const auto D = x0.n_rows;
    if (bounds.n_cols != 2 || bounds.n_rows != D) THROW_EX_FS(std::invalid_argument, "n " << n << ", D " << D << ", bounds " << arma::size(bounds));
    const arma::vec range = bounds.col(1) /* ub */ - bounds.col(0); /* lb */
    const auto init_type = true; // D <= n; // Produce equispaced grid in case parameters less than particles
    if (init_type) do_sobol_init();
#pragma omp parallel for num_threads(adj_threads(n)) schedule(static, 1)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < D; ++j) {
            const auto pow_j = pows.empty() ? 1 : pows[j];
            if (init_type) { // Produce equispaced grid
                const double dim_range = double(n) / std::pow<double>(2, j);
                x0(j, i) = std::pow(std::fmod<double>(i + 1, dim_range) / dim_range, pow_j);
                // LOG4_TRACE("Equispaced particle " << i << ", parameter " << j << ", value " << x0(j, i) << ", ub " << bounds(j, 1) << ", lb " << bounds(j, 0)
                //                                  << ", dim range " << dim_range);
            } else { // Use Sobol pseudo-random number
                x0(j, i) = std::pow(sobolnum(j, sobol_ctr + i), pow_j);
                LOG4_TRACE("Random particle " << i << ", parameter " << j << ", value " << x0(j, i) << ", ub " << bounds(j, 1) << ", lb " << bounds(j, 0) << ", pow "
                                              << pow_j);
            }
        }
        x0.col(i) = x0.col(i) % range + bounds.col(0);
    }
    LOG4_TRACE("Scaled particles - parameters matrix " << x0);
}


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


arma::vec pprune::ensure_bounds(const double *x, const arma::mat &bounds)
{
    arma::vec xx(bounds.n_rows);
    for (size_t i = 0; i < xx.size(); ++i)
        if (x[i] < bounds(i, 0)) xx[i] = bounds(i, 0);
        else if (x[i] > bounds(i, 1)) xx[i] = bounds(i, 1);
        else xx[i] = x[i];
    return xx;
}

#define kn_errchk(cmd) { const auto __err = (cmd); \
    if (__err) LOG4_THROW("KNitro call " #cmd " failed with error " << __err); }

pprune::pprune(const prima_algorithm_t type, const size_t n_particles, const arma::mat &bounds,
               const t_pprune_cost_fun &cost_f,
               const size_t maxfun, const double rhobeg, const double rhoend,
               const arma::mat &x0_, const arma::vec &pows) :
        n(n_particles), D(bounds.n_rows), bounds(bounds), pows(pows), ranges(bounds.col(1) - bounds.col(0))
{
    auto x0 = x0_;
    if (x0.n_rows != D || x0.n_cols != n) {
        x0.set_size(D, n);
        equispaced(x0, bounds, pows, common::init_sobol_ctr());
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
#elif defined(USE_KNITRO)
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
    // kn_errchk(KN_set_var_primal_init_values_all(kc, x0.colptr(i)));

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
    kn_errchk(KN_set_int_param(kc, KN_PARAM_MULTISTART, KN_MULTISTART_YES));

    /** Perform multistart in parallel using max number of available threads */
    const auto n_threads = omp_get_max_threads();
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
    auto x = (double *) malloc(D * sizeof(double));
    auto lambda = (double *) malloc(2 * D * sizeof(double));

    /** An example of obtaining solution information. */
    kn_errchk(KN_get_solution(kc, &n_status, &result.best_score, x, lambda));
    LOG4_DEBUG("Optimal objective value " << result.best_score << ", status " << n_status);
    LOG4_DEBUG("Optimal x (with corresponding multiplier)");
    result.best_parameters = arma::vec(x, D, true, true);
    for (unsigned i = 0; i < D; ++i) LOG4_DEBUG("  x[" << i << "] " << x[i] << " (lambda " << lambda[D + i]);
    double feas_error, opt_error;
    kn_errchk(KN_get_abs_feas_error(kc, &feas_error));
    LOG4_DEBUG("Feasibility violation " << feas_error);
    kn_errchk(KN_get_abs_opt_error(kc, &opt_error));
    LOG4_DEBUG("Optimality violation " << opt_error);

    /** Delete the Knitro solver instance. */
    KN_free(&kc);
    free(x);
    free(lambda);
#else
    auto elect_ready = std::make_shared<std::condition_variable>();
    auto elect_mx = std::make_shared<std::mutex>();
    const auto no_elect = n < C_elect_threshold || maxfun < C_elect_threshold;
    auto f_score = std::make_shared<std::deque<t_calfun_data_ptr>>(n_particles);
    t_omp_lock res_l;
#pragma omp parallel for num_threads(n_particles) schedule(static, 1)
    for (size_t i = 0; i < n_particles; ++i) {
        auto const calfun_data = f_score->at(i) = new t_calfun_data{no_elect, elect_ready, elect_mx, f_score, cost_f, i, maxfun};
#endif
#ifdef USE_BITEOPT
        CBiteRnd rnd(std::pow(C_rand_disperse * i, C_rand_disperse));
        std::vector<std::shared_ptr<t_biteopt_cost>> biteopt(C_biteopt_depth);
        for (auto &o: biteopt) {
            o = std::make_shared<t_biteopt_cost>(calfun_data, bounds);
            o->init(rnd, x0.colptr(i));
        }
#pragma unroll
        for (unsigned j = 0; j < maxfun / biteopt.size(); ++j)
            for (unsigned d = 0; d < biteopt.size(); ++d)
                biteopt[d]->optimize(rnd, d + 1 >= biteopt.size() ? nullptr : biteopt[d + 1].get());

        res_l.set();
        if (biteopt.back()->getBestCost() < result.best_score) {
            result.best_score = biteopt.back()->getBestCost();
            result.best_parameters = arma::vec((double *) biteopt.back()->getBestParams(), D, true, true);
        }
        result.total_iterations += maxfun;
        res_l.unset();

#elif defined(USE_PRIMA)
        auto problem = all_problem;
        problem.x0 = (double *) x0.colptr(i);

        auto prima_options = all_prima_options;
        prima_options.data = f_score->at(i) = &calfun_data;

        prima_result_t prima_result;
        const auto rc = prima_minimize(type, problem, prima_options, &prima_result);
        LOG4_DEBUG("pprune particle " << i << " final parameters " << common::to_string(prima_result.x, std::min(3, D)) << ", score " << prima_result.f << ", cstrv " <<
                                     prima_result.cstrv << ", return code " << rc
                                     << /* ", " << prima_get_rc_string(static_cast<const prima_rc_t>(rc))  << */ ", message '"
                                     << prima_result.message << "', iterations " << prima_result.nf);

        res_l.set();
        if (prima_result.f < result.best_score) {
            result.best_score = prima_result.f;
            result.best_parameters = arma::vec(prima_result.x, D, true, true);
        }
        result.total_iterations += prima_result.nf;
        res_l.unset();
        prima_free_result(&prima_result);
#endif
#ifndef USE_KNITRO
    }
#pragma omp unroll
    for (auto &f: *f_score) delete f;
#endif
    LOG4_DEBUG("PPrune type " << type << ", score " << result.best_score << ", total iterations " << result.total_iterations << ", particles " << n << ", parameters " << D
        << ", max iterations per particle " << maxfun << ", var start " << rhobeg << ", var end " << rhoend << ", bounds " << bounds << ", ranges " << ranges <<
        ", starting parameters " << x0.head_rows(std::min<unsigned>(3, x0.n_rows)));
}

pprune::operator t_pprune_res()
{
    return result;
}

}
}
