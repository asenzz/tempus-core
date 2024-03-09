//
// Created by zarko on 12/4/22.
//

//============================================================================
// Name        : Firefly.cpp
// Authors     : Dr. Iztok Fister and Iztok Fister Jr.
// Version     : v1.0
// Created on  : Jan 23, 2012
//============================================================================

/* Classic Firefly algorithm coded using C/C++ programming language */

/* Reference Paper*/

/*I. Fister Jr.,  X.-S. Yang,  I. Fister, J. Brest, Memetic firefly algorithm for combinatorial optimization,
in Bioinspired Optimization Methods and their Applications (BIOMA 2012), B. Filipic and J.Silc, Eds.
Jozef Stefan Institute, Ljubljana, Slovenia, 2012 */

/*Contact:
Iztok Fister Jr. (iztok.fister1@um.si)
*/

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <armadillo>
#include <algorithm>
#include <execution>

#include "firefly.hpp"
#include "common/compatibility.hpp"
#include "sobolvec.h"
#include "common/Logging.hpp"
#include "common/parallelism.hpp"
#include "util/math_utils.hpp"
#include "common/gpu_handler.hpp"
#ifdef FIREFLY_CUDA
#include "firefly.cuh"
#endif

namespace svr {
namespace optimizer {


firefly::firefly(
        const size_t D,
        const size_t n,
        const size_t MaxGeneration,
        const double alpha,
        const double betamin,
        const double gamma,
        const arma::vec &lb,
        const arma::vec &ub,
        const arma::vec &pows,
        const loss_callback_t function) :
        levy_random(common::levy(D)), D(D), n(n), MaxGeneration(MaxGeneration),
        sobol_ctr(78786876896ULL + (long) floor(svr::common::get_uniform_random_value() * (double) 4503599627370496ULL)),
        range(ub - lb), lb(lb), ub(ub), alpha(alpha), betamin(betamin), gamma(gamma), ffa(D, n), particles(n), best_parameters(D), function(function)
{
    LOG4_DEBUG(
            "Init, particles " << n << ", iterations " << MaxGeneration << ", dimensions " << D << ", alpha " << alpha << ", betamin " << betamin << ", gamma " << gamma); // <<

    // Firefly algorithm optimization loop
    // determine the starting point of random generator
    do_sobol_init();
#pragma omp parallel for num_threads(adj_threads(n)) schedule(static, 1)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < D; ++j) {
            if (D < n / 2) { // Produce equispaced grid
                const double dim_range = double(n) / std::pow<double>(2, j);
                ffa(j, i) = std::pow(std::fmod<double>(i + 1, dim_range) / dim_range, pows[j]);
                LOG4_TRACE("Particle " << i << ", argument " << j << ", parameter " << ffa(j, i) << ", ub " << ub[j] << ", lb " << lb[j] << ", dim range " << dim_range);
            } else // Use Sobol pseudo-random number
                ffa(j, i) = std::pow(sobolnum(j, sobol_ctr + i), pows[j]);
        }
        ffa.col(i) = ffa.col(i) % range + lb;
        particles[i].I = particles[i].f = 1;            // initialize attractiveness
        if (D < 5) LOG4_DEBUG("Particle " << i << ", parameters " << ffa.col(i));
    }

    PROFILE_EXEC_TIME(
          ffa_main(), "FFA with particles " << n << ", iterations " << MaxGeneration << ", dimensions " << D << ", alpha " << alpha <<
                        ", betamin " << betamin << ", gamma " << gamma << ", final score " << best_score);
}

// optionally recalculate the new alpha value
double firefly::alpha_new(const double old_alpha, const double NGen)
{
    delta = 1. - std::pow(firefly::delta_base, 1. / NGen);            // delta parameter
    return (1. - delta) * old_alpha;
}

// implementation of bubble sort
void firefly::sort_ffa()
{
    // initialization of indexes
    for (size_t i = 0; i < particles.size(); ++i) particles[i].Index = i;
    std::sort(std::execution::par_unseq, particles.begin(), particles.end(), [](const auto &lhs, const auto &rhs) { return lhs.I < rhs.I; });
}

// replace the old population according the new Index values
void firefly::replace_ffa()
{
    arma::uvec row_ixs(n);
    const arma::mat ffa_tmp = ffa;
    // generational selection in sense of EA
#pragma omp parallel for num_threads(adj_threads(n)) schedule(static, 1)
    for (size_t i = 0; i < n; ++i) ffa.col(i) = ffa_tmp.col(particles[i].Index);
}

void firefly::findlimits(const size_t k)
{
#ifdef FFA_HARD_WALL
    __omp_pfor_i(0, D,
        if (ffa[k][i] < lb[i])
            ffa[k][i] = lb[i];
        else if (ffa[k][i] > ub[i])
            ffa[k][i] = ub[i];
    )
#else // Bounce
#pragma omp parallel for num_threads(adj_threads(D)) schedule(static, 1)
    for (size_t i = 0; i < D; ++i) {
        if (ffa(i, k) < lb[i]) ffa(i, k) = lb[i] + std::fmod(lb[i] - ffa(i, k), range[i]);
        if (ffa(i, k) > ub[i]) ffa(i, k) = ub[i] - std::fmod(ffa(i, k) - ub[i], range[i]);
    }
#endif
}

// Better, adaptive Firefly algorithm as described in
// An improved firefly algorithm for global continuous optimization problems by Jinran Wu et al. 2020
void firefly::move_ffa_adaptive(const double rate)
{
#ifdef FIREFLY_CUDA
    // TODO Port to CUDA
#else
    const auto ffa_prev = ffa;
    const auto beta0_betamin = beta0 - betamin;
#pragma omp parallel for num_threads(adj_threads(n)) collapse(2)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            if (i == j || particles[i].I <= particles[j].I) continue;    // brighter and more attractive
            const double r = arma::accu(arma::abs(ffa.col(i) - ffa_prev.col(j)));
            const double beta = beta0_betamin * std::exp(-gamma * r * r) + betamin;
            const double l = -1. + 2. * drand48();
            if (drand48() > rate)
                ffa.col(i) = ffa.col(i) * (1. - beta) + ffa_prev.col(j) * beta + alpha * std::copysign(1., (drand48() - .5)) * levy_random * range;
            else
                ffa.col(i) = (ffa_prev.col(j) - ffa.col(i)) * beta * std::exp(l * b_1) * std::cos(2. * M_PI * -1. + 2. * drand48());
        }
    }
#pragma omp parallel for num_threads(adj_threads(n)) schedule(static, 1)
    for (size_t i = 0; i < n; ++i) findlimits(i);
#endif
}

void firefly::run_iteration()
{
#pragma omp parallel for num_threads(adj_threads(n)) schedule(static, 1)
    for (size_t i = 0; i < n; ++i)
        particles[i].I = particles[i].f = function(common::wrap_vector(const_cast<double *>(ffa.col(i).colmem), ffa.n_rows));
}

/* display syntax messages */
void firefly::ffa_main()
{
    LOG4_BEGIN();

#ifdef DUMP
    dump_ffa(t);
#endif
    size_t t = 1;        // generation  counter
    while (true) {
        // this line of reducing alpha is optional
        alpha = alpha_new(alpha, MaxGeneration);

        // evaluate new solutions
        PROFILE_EXEC_TIME(run_iteration(), "Iteration " << t << " of " << MaxGeneration << " on " << n << " particles, " << D << " dimensions");

        // ranking fireflies by their light intensity
        sort_ffa();
        // replace old population
        replace_ffa();

        // find the current best
        arma::vec nbest = ffa.col(0);
        const auto this_fbest = particles.front().I;

        // move all fireflies to the better locations
        double rate;
        if (t == 1) {
            rate = .5;
        } else {
            double a, b;
            if (std::floor(std::log10(std::fabs(best_score))) == std::floor(std::log10(std::fabs(this_fbest)))) {
                const double theta = std::floor(std::log10(std::fabs(this_fbest - std::trunc(this_fbest))));
                a = best_score - std::pow<double>(10, theta) * std::floor(best_score / std::pow<double>(10, theta + 1.));
                b = this_fbest - std::pow<double>(10, theta) * std::floor(this_fbest / std::pow<double>(10, theta + 1.));
            } else {
                a = best_score;
                b = this_fbest;
            }
            rate = 1. / (1. + std::exp(-a / b));
        }
        if (t > 1 && this_fbest < best_score) {
            LOG4_DEBUG("Firefly iteration " << t << ", alpha " << alpha << ", new best score " << this_fbest << ", previous best score " << best_score << ", improvement "
                                            << 100. * (1. - this_fbest / best_score) << " pct."); // << ", parameters " << common::deep_to_string(nbest));
            best_parameters = nbest;
            best_score = this_fbest;
        }
        move_ffa_adaptive(rate);
        ++t;
        if (t > MaxGeneration) break;
    }
    LOG4_END();
}

} // namespace optimizer
} // namespace svr
