//
// Created by zarko on 3/24/24.
//

#include "pprima.hpp"
#include "common/compatibility.hpp"
#include "util/math_utils.hpp"
#include "sobolvec.h"
#include "util/string_utils.hpp"
#include <omp.h>

namespace svr {
namespace optimizer {

unsigned long init_sobol_ctr()
{
    return 78786876896ULL + (long) floor(svr::common::get_uniform_random_value() * (double) 4503599627370496ULL);
}

void equispaced(arma::mat &x0, const arma::mat &bounds, const arma::vec &pows, unsigned long sobol_ctr = 0)
{
    static const auto default_sobol_ctr = init_sobol_ctr();
    if (!sobol_ctr) sobol_ctr = default_sobol_ctr;
    const auto n = x0.n_cols;
    const auto D = x0.n_rows;
    if (bounds.n_cols != 2 || bounds.n_rows != D) THROW_EX_FS(std::invalid_argument, "n " << n << ", D " << D << ", bounds " << arma::size(bounds));
    const arma::vec range = bounds.col(1) /* ub */ - bounds.col(0); /* lb */
    const auto init_type = D <= n; // Produce equispaced grid in case parameters less than half of particles
    if (init_type) do_sobol_init();
#pragma omp parallel for num_threads(adj_threads(n)) schedule(static, 1)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < D; ++j) {
            const auto pow_j = pows.empty() ? 1 : pows[j];
            if (init_type) { // Produce equispaced grid
                const double dim_range = double(n) / std::pow<double>(2, j);
                x0(j, i) = std::pow(std::fmod<double>(i + 1, dim_range) / dim_range, pow_j);
                LOG4_TRACE("Equispaced particle " << i << ", parameter " << j << ", value " << x0(j, i) << ", ub " << bounds(j, 1) << ", lb " << bounds(j, 0)
                                                  << ", dim range " << dim_range);
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

void pprima::prima_progress_callback(
        const int n, const double x[], const double f, const int nf,
        const int tr, const double cstrv, const int m_nlcon, const double nlconstr[], bool *const terminate)
{
    (void) n;
    (void) m_nlcon;
    (void) nlconstr;
    LOG4_DEBUG("Prima best point parameters " << common::to_string(x, n) << ", score " << f << ", cstrv " << cstrv <<
                                              ", iterations " << nf << ", of which in trust regions " << tr);
    *terminate = false;
};

arma::vec pprima::ensure_bounds(const double *x, const arma::mat &bounds)
{
    arma::vec xx(bounds.n_rows);
    for (size_t i = 0; i < xx.size(); ++i)
        if (x[i] < bounds(i, 0)) xx[i] = bounds(i, 0);
        else if (x[i] > bounds(i, 1)) xx[i] = bounds(i, 1);
        else xx[i] = x[i];
    return xx;
}


pprima::pprima(const prima_algorithm_t type, const size_t n_particles, const arma::mat &bounds,
               const t_prima_cost_fun &cost_f,
               const size_t maxfun, const double rhobeg, const double rhoend,
               const arma::mat &x0_, const arma::vec &pows) :
        n(n_particles), D(bounds.n_rows), bounds(bounds), pows(pows), ranges(bounds.col(1) - bounds.col(0))
{
    arma::mat x0 = x0_;
    if (x0.n_elem) return;

    const auto sobol_ctr = init_sobol_ctr();
    if (x0.n_rows != D || x0.n_cols != n) x0.set_size(D, n);
    equispaced(x0, bounds, pows, sobol_ctr);

    OMP_LOCK(res_l);
#pragma omp parallel for num_threads(adj_threads(n)) schedule(static, 1)
    for (size_t i = 0; i < n; ++i) {
        prima_problem_t problem;
        prima_options_t prima_options;
        prima_result_t prima_result;
        int rc;
        prima_init_problem(&problem, D);
        problem.calfun = [](const double x[], double *const f, const void *data) {
            PROFILE_EXEC_TIME((*(dtype(cost_f) * )data)(x, f), "Cost, score " << *f);
        };
        problem.x0 = const_cast<double *>(x0.col(i).colmem);
        problem.xl = const_cast<double *>(bounds.col(0).colmem);
        problem.xu = const_cast<double *>(bounds.col(1).colmem);

        // Set up the options
        prima_init_options(&prima_options);
        prima_options.npt = (D + 2. + .5 * (D + 1.) * (D + 2.)) / 2.;
        prima_options.iprint = PRIMA_MSG_EXIT;
        prima_options.rhobeg = rhobeg;
        prima_options.rhoend = rhoend;
        prima_options.maxfun = maxfun;
        prima_options.data = (void *) &cost_f;
        prima_options.callback = prima_progress_callback;

        // Call the solver
        rc = prima_minimize(type, problem, prima_options, &prima_result);
        // Print the result
        LOG4_DEBUG("Prima particle " << i << " final parameters " << common::to_string(prima_result.x, D) << ", score " << prima_result.f << ", cstrv " <<
                                     prima_result.cstrv << ", return code " << rc << ", " << prima_get_rc_string(static_cast<const prima_rc_t>(rc)) << ", message '"
                                     << prima_result.message << "', iterations " << prima_result.nf);

        omp_set_lock(&res_l);
        if (prima_result.f < result.best_score) {
            result.best_score = prima_result.f;
            result.best_parameters = arma::vec(prima_result.x, D, false, true);
        }
        result.total_iterations += prima_result.nf;
        omp_unset_lock(&res_l);
        // Check the result
        // int success = (fabs(result.x[0] - 4.5) > 2e-2 || fabs(result.x[1] - 3.5) > 2e-2);
        // Free the result
        prima_free_result(&prima_result);
    }
    LOG4_DEBUG(
            "Prima type " << type << ", score " << result.best_score << ", total iterations " << result.total_iterations << ", particles " << n <<
                          ", parameters " << D << ", max iterations per particle " << maxfun << ", var start " << rhobeg << ", var end " << rhoend << ", bounds "
                          << bounds <<
                          ", ranges " << ranges << ", starting parameters " << x0);
}

pprima::operator t_pprima_res()
{
    return result;
}

}
}
