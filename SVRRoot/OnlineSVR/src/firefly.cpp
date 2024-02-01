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

#include "firefly.hpp"
#include "sobolvec.h"

#include "common/Logging.hpp"
#include "common/parallelism.hpp"
#include "util/math_utils.hpp"
#include "common/gpu_handler.hpp"

using namespace std;


namespace svr {
namespace optimizer {

firefly::firefly(
        const size_t D,
        const size_t n,
        const size_t MaxGeneration,
        const double alpha,
        const double betamin,
        const double gamma,
        const std::vector<double> &lb,
        const std::vector<double> &ub,
        const std::vector<double> &pows,
        const loss_callback_t function) :
        D(D), n(n), MaxGeneration(MaxGeneration), lb(lb), ub(ub), alpha(alpha), betamin(betamin), gamma(gamma), function(function)
{
    LOG4_DEBUG(
            "Init, particles " << n << ", iterations " << MaxGeneration << ", dimensions " << D << ", alpha " << alpha << ", betamin " << betamin << ", gamma " << gamma); // <<
    // ", mins " << common::deep_to_string(lb) << ", maxes " << common::deep_to_string(ub));
    this->fbest = std::numeric_limits<double>::max();
    this->levy_random = common::levy(D);

    Index.resize(n, 0);
    f.resize(n, 0.);
    I.resize(n, 0.);
    nbest.resize(D, 0.);
    allround_best.resize(D, 0.);
    ffa.resize(n);	    // firefly agents
    ffa_tmp.resize(n);  // intermediate population

    // firefly algorithm optimization loop
    // determine the starting point of random generator
    do_sobol_init();
    sobol_ctr = 78786876896ULL + (long) floor(svr::common::get_uniform_random_value() * (double) 4503599627370496ULL);

#pragma omp parallel for
    for(size_t p = 0; p < n; ++p) {
        ffa[p].resize(D);
        ffa_tmp[p].resize(D);
        for (size_t j = 0; j < D; ++j) {
            if (D < n / 2) {
                const double dim_range = double(n) / std::pow(2, j);
                ffa[p][j] = std::pow(std::fmod<double>(p + 1, dim_range) / dim_range, pows[j]) * (ub[j] - lb[j]) + lb[j];
                LOG4_TRACE("Particle " << p << ", argument " << j << ", parameter " << ffa[p][j] << ", ub " << ub[j] << ", lb " << lb[j] << ", dim range " << dim_range);
            } else {
                ffa[p][j] = std::pow(sobolnum(j, sobol_ctr + p), pows[j]) * (ub[j] - lb[j]) + lb[j];
            }
        }
        I[p] = f[p] = 1;            // initialize attractiveness
        if (D < 5) LOG4_DEBUG("Particle " << p << ", parameters " << common::to_string(ffa[p]));
    }

    PROFILE_EXEC_TIME(ffa_main(), "FFA with particles " << n << ", iterations " << MaxGeneration << ", dimensions " << D << ", alpha " << alpha << ", betamin " << betamin << ", gamma " << gamma << ", final score " << fbest);
}

// optionally recalculate the new alpha value
double firefly::alpha_new(const double old_alpha, const double NGen)
{
    delta = 1. - pow(firefly::delta_base, 1. / NGen);            // delta parameter
    return (1. - delta) * old_alpha;
}

// implementation of bubble sort
void firefly::sort_ffa()
{
    // initialization of indexes
    for (size_t i = 0; i < n; ++i) Index[i] = i;

    // Bubble sort TODO parallelize
    for (size_t i = 0; i < n - 1; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            if (I[i] <= I[j]) continue;
            auto z = I[i];    // exchange attractiveness
            I[i] = I[j];
            I[j] = z;
            z = f[i];            // exchange fitness
            f[i] = f[j];
            f[j] = z;
            const auto k = Index[i];    // exchange indexes
            Index[i] = Index[j];
            Index[j] = k;
        }
    }
}

// replace the old population according the new Index values
void firefly::replace_ffa()
{
    // copy original population to temporary area
    ffa_tmp = ffa;

    // generational selection in sense of EA
    __tbb_pfor_i(0, n, for (size_t j = 0; j < D; ++j) ffa[i][j] = ffa_tmp[Index[i]][j]; )
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
    __tbb_pfor(l, 0, D,
               if (ffa[k][l] < lb[l])
                   ffa[k][l] = lb[l] + std::fmod(lb[l] - ffa[k][l], ub[l] - lb[l]);
                       if (ffa[k][l] > ub[l])
                           ffa[k][l] = ub[l] - std::fmod(ffa[k][l] - ub[l], ub[l] - lb[l]);
    )
#endif
}

// Original Firefly algorithm, slow, not used
void firefly::move_ffa()
{
    for (size_t i = 0; i < n; ++i) {
        const double scale = abs(ub[i] - lb[i]);
        for (size_t j = 0; j < n; ++j) {
            if (i == j || I[i] <= I[j]) continue;    // brighter and more attractive
            double r = 0;
            for (size_t k = 0; k < D; ++k)
                r += (ffa[i][k] - ffa[j][k]) * (ffa[i][k] - ffa[j][k]);
            r = sqrt(r);
            const double beta = (beta0 - betamin) * exp(-gamma * pow(r, 2.0)) + betamin;
            for (size_t k = 0; k < D; ++k) {
                const double tmpf = alpha * (sobolnum(j, sobol_ctr + i) - .5) * scale;
                ffa[i][k] = ffa[i][k] * (1. - beta) + ffa_tmp[j][k] * beta + tmpf;
            }
        }
        findlimits(i);
    }
}

// Better, adaptive Firefly algorithm as described in
// An improved firefly algorithm for global continuous optimization problems by Jinran Wu et al. 2020
void firefly::move_ffa_adaptive(const double rate)
{
    arma::mat scale(D, 1);
    __tbb_pfor_i(0, D, scale(i, 0) = abs(ub[i] - lb[i]))

    const auto ffa_prev = ffa;
    __tbb_pfor_i(0, n,
                 for (size_t j = 0; j < n; ++j) {
                     if (i == j || I[i] <= I[j]) continue;    // brighter and more attractive
                     double r = 0;
                     for (size_t k = 0; k < D; ++k)
                         r += std::abs<double>(ffa[i][k] - ffa_prev[j][k]);
                     const double beta = (beta0 - betamin) * exp(-gamma * pow(r, 2)) + betamin;
                     if (drand48() > rate) {
                         for (size_t k = 0; k < D; ++k)
                             ffa[i][k] = ffa[i][k] * (1. - beta) + ffa_prev[j][k] * beta + alpha * copysign(1., (drand48() - .5)) * levy_random[k] * scale(k, 0);
                     } else {
                         const double l = -1. + 2. * drand48();
                         for (size_t k = 0; k < D; ++k)
                             ffa[i][k] = ffa[i][k] + (-ffa[i][k] + ffa_prev[j][k]) * (beta * exp(l * b_1) * cos(2. * M_PI * l));
                     }
                 }
                 findlimits(i);
    )
}

void firefly::run_iteration()
{
    __tbb_pfor(ptcl, 0, n, I[ptcl] = f[ptcl] = function(ffa[ptcl]))
}

/* display syntax messages */
void firefly::ffa_main()
{
    LOG4_BEGIN();

#ifdef DUMP
    dump_ffa(t);
#endif
    size_t t = 1;        // generation  counter
    while (1) {
        // this line of reducing alpha is optional
#ifdef FFA_RANDOMLESNESS
        alpha = alpha_new(alpha, double(MaxGeneration) - double(t) / FFA_RANDOMLESNESS);
#else
        alpha = alpha_new(alpha, MaxGeneration);
#endif

        // evaluate new solutions
        PROFILE_EXEC_TIME(run_iteration(), "Iteration " << t << " of " << MaxGeneration << " on " << n << " particles, " << D << " dimensions");

        // ranking fireflies by their light intensity
        sort_ffa();
        // replace old population
        replace_ffa();

        // find the current best
#pragma omp parallel for
        for (size_t i = 0; i < D; ++i) nbest[i] = ffa[0][i];
        auto this_fbest = I[0];

        // move all fireflies to the better locations
#ifdef ADAPTIVE_FFA
        double rate;
        if (t == 1) {
            rate = .5;
        } else {
            double a, b;
            if (floor(log10(fabs(fbest))) == floor(log10(fabs(this_fbest)))) {
                const double theta = floor(log10(fabs(this_fbest - trunc(this_fbest))));
                a = fbest - pow(10, theta) * floor(fbest / pow(10, theta + 1.));
                b = this_fbest - pow(10, theta) * floor(this_fbest / pow(10, theta + 1.));
            } else {
                a = fbest;
                b = this_fbest;
            }
            rate = 1. / (1. + exp(-a / b));
        }
#endif
        if (t > 1 && this_fbest < fbest) {
            LOG4_DEBUG("Firefly iteration " << t << ", alpha " << alpha << ", new best score " << this_fbest << ", previous best score " << fbest << ", improvement "
                                            << 100. * (1. - this_fbest / fbest) << " pct."); // << ", parameters " << common::deep_to_string(nbest));
            allround_best = nbest;
            fbest = this_fbest;
        }
#ifdef ADAPTIVE_FFA
        move_ffa_adaptive(rate);
#else
        move_ffa();
#endif
        ++t;
        if (t > MaxGeneration) break;
    }
    LOG4_END();
}

} // namespace optimizer
} // namespace svr
