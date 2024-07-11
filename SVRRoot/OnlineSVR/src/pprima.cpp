//
// Created by zarko on 3/24/24.
//

#include "pprima.hpp"
#include "common/compatibility.hpp"
#include "common/defines.h"
#include "util/math_utils.hpp"
#include "sobolvec.h"
#include "util/string_utils.hpp"
#include <omp.h>

namespace svr {
namespace optimizer {

unsigned long long init_sobol_ctr()
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
    const auto init_type = true; // D <= n; // Produce equispaced grid in case parameters less than particles
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

struct calfun_data
{
    std::condition_variable &elect_ready;
    std::mutex &elect_mx;
    std::deque<double> &f_score;
    const t_pprima_cost_fun cost_fun;
    const size_t particle_index = 0;
    const size_t maxfun = 0;
    double best_f = std::numeric_limits<double>::infinity();
    size_t nf = 0;
    bool zombie = false;

    bool drop(const size_t keep_particles)
    {
        std::unique_lock lk(elect_mx);
        elect_ready.wait(lk, [this] { return std::count(f_score.cbegin(), f_score.cend(), std::numeric_limits<double>::quiet_NaN()) == 0; });
        lk.unlock();
        std::deque<size_t> ixs(f_score.size());
        std::iota(ixs.begin(), ixs.end(), 0);
        std::stable_sort(std::execution::par_unseq, ixs.begin(), ixs.end(),
                         [&](const auto lhs, const auto rhs) { return f_score[lhs] < f_score[rhs]; });
        return std::find(ixs.cbegin(), ixs.cend(), particle_index) - ixs.cbegin() > keep_particles;
    }
};


void pprima::calfun(const double x[], double *const f, const void *data)
{
    static const std::deque<double> maxfun_drop_coefs{.4, .6, .8};

    auto p_calfun_data = (calfun_data *) data;
    if (p_calfun_data->zombie) {
        *f = common::C_bad_validation;
        return;
    }
    for (const auto drop_coef: maxfun_drop_coefs) {
        if (p_calfun_data->nf != size_t(drop_coef * p_calfun_data->maxfun)) continue;
        p_calfun_data->f_score[p_calfun_data->particle_index] = p_calfun_data->best_f;
        p_calfun_data->elect_ready.notify_all();
        if (p_calfun_data->drop((1. - drop_coef) * p_calfun_data->f_score.size())) p_calfun_data->zombie = true;
        break;
    }
    ++p_calfun_data->nf;
    PROFILE_EXEC_TIME(p_calfun_data->cost_fun(x, f), "Cost, score " << *f);
    if (*f < p_calfun_data->best_f) p_calfun_data->best_f = *f;
}

pprima::pprima(const prima_algorithm_t type, const size_t n_particles, const arma::mat &bounds,
               const t_pprima_cost_fun &cost_f,
               const size_t maxfun, const double rhobeg, const double rhoend,
               const arma::mat &x0_, const arma::vec &pows, const size_t n_threads) :
        n(n_particles), D(bounds.n_rows), bounds(bounds), pows(pows), ranges(bounds.col(1) - bounds.col(0))
{
    auto x0 = x0_;
    if (x0.n_rows != D || x0.n_cols != n) {
        x0.set_size(D, n);
        equispaced(x0, bounds, pows, init_sobol_ctr());
    }
    std::deque<double> f_score(n_particles, std::numeric_limits<double>::quiet_NaN());
    std::condition_variable elect_ready;
    std::mutex elect_mx;
    OMP_LOCK(res_l);
#pragma omp parallel for num_threads(adj_threads(n_threads ? n_threads : n_particles)) schedule(static, 1)
    for (size_t i = 0; i < n; ++i) {
        prima_problem_t problem;
        prima_init_problem(&problem, D);
        problem.calfun = calfun;

        auto cd = new calfun_data{elect_ready, elect_mx, f_score, cost_f, i, maxfun};

        problem.x0 = (double *)x0.col(i).colmem;
        problem.xl = (double *)bounds.col(0).colmem;
        problem.xu = (double *)bounds.col(1).colmem;

        // Set up the options
        prima_options_t prima_options;
        prima_init_options(&prima_options);
        prima_options.npt = std::min<dtype(prima_options.npt)>(maxfun - 1, (D + 2. + (D + 1.) * (D + 2.) / 2.) / 2.);
        prima_options.iprint = PRIMA_MSG_EXIT;
        prima_options.rhobeg = rhobeg;
        prima_options.rhoend = rhoend;
        prima_options.maxfun = maxfun;
        prima_options.data = cd;
        prima_options.callback = prima_progress_callback;

        prima_result_t prima_result;
        const auto rc = prima_minimize(type, problem, prima_options, &prima_result);
        LOG4_DEBUG("Prima particle " << i << " final parameters " << common::to_string(prima_result.x, D) << ", score " << prima_result.f << ", cstrv " <<
                                     prima_result.cstrv << ", return code " << rc << /* ", " << prima_get_rc_string(static_cast<const prima_rc_t>(rc))  << */ ", message '"
                                     << prima_result.message << "', iterations " << prima_result.nf);

        delete cd;
        omp_set_lock(&res_l);
        if (prima_result.f < result.best_score) {
            result.best_score = prima_result.f;
            result.best_parameters = arma::vec(prima_result.x, D, true, true);
        }
        result.total_iterations += prima_result.nf;
        omp_unset_lock(&res_l);
        prima_free_result(&prima_result);
    }
    LOG4_DEBUG(
            "Prima type " << type << ", score " << result.best_score << ", total iterations " << result.total_iterations << ", particles " << n << ", parameters " << D
                          << ", max iterations per particle " << maxfun << ", var start " << rhobeg << ", var end " << rhoend << ", bounds " << bounds << ", ranges "
                          << ranges << ", starting parameters " << x0);
}

pprima::operator t_pprima_res()
{
    return result;
}

}
}
