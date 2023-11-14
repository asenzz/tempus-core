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

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <memory.h>
#include <vector>
#include <algorithm>
#include <functional>

#define ARMA_NO_DEBUG
#define ARMA_ALLOW_FAKE_GCC
#include <armadillo>


#include "firefly.hpp"





firefly::firefly(
        const size_t D,
        const size_t n,
        const size_t MaxGeneration,
        const double alpha,
        const double betamin,
        const double gamma,
        const std::vector<double> &lb,
        const std::vector<double> &ub,
        const loss_callback_t function)
{
    //LOG4_DEBUG("Init, particles " << n << ", iterations " << MaxGeneration << ", dimensions " << D << ", alpha " << alpha << ", betamin " << betamin << ", gamma " << gamma);
    this->D = D;
    this->n = n;
    this->MaxGeneration = MaxGeneration;
    this->function = function;
    this->alpha = alpha;
    this->betamin = betamin;
    this->gamma = gamma;
    this->fbest.resize(MaxGeneration, std::numeric_limits<double>::max());

    Index.resize(n, 0);
    f.resize(n, 0.);
    I.resize(n, 0.);
    nbest.resize(D, 0.);
    ffa.resize(n);	    // firefly agents
    ffa_tmp.resize(n);  // intermediate population
    for(int i=0;i<n;i++){
     ffa[i].resize(D); ffa_tmp[i].resize(D); 
    }

    // firefly algorithm optimization loop
    // determine the starting point of random generator

    // initialize upper and lower bounds
    this->lb = lb;
    this->ub = ub;
    static bool run_once = [](){
	srand48(9287987);
        return true;
    } ();
    for(int i=0;i<n;i++){
        for (size_t j = 0; j < D; ++j){
            ffa[i][j] = SUB_LIMIT * drand48() * (ub[j] - lb[j]) + lb[j];
	}
        I[i] = f[i] = 1;            // initialize attractiveness
    }

    ffa_main();
}

// optionally recalculate the new alpha value
double firefly::alpha_new(const double old_alpha, const size_t NGen)
{
    const double delta = 1. - pow((1e-3 / .9), 1. / double(NGen));            // delta parameter
    return (1. - delta) * old_alpha;
}

// implementation of bubble sort
void firefly::sort_ffa()
{
    // initialization of indexes
    for (size_t i = 0; i < n; i++) Index[i] = i;

    // Bubble sort
    for (size_t i = 0; i < n - 1; i++) {
        for (size_t j = i + 1; j < n; j++) {
            if (I[i] > I[j]) {
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
}

// replace the old population according the new Index values
void firefly::replace_ffa()
{
    // copy original population to temporary area
    for(int i=0;i< n; i++){ 
	for (size_t j = 0; j < D; ++j) ffa_tmp[i][j] = ffa[i][j];
    }

    // generational selection in sense of EA
    for(int i=0;i<n;i++){
		 for (size_t j = 0; j < D; ++j) ffa[i][j] = ffa_tmp[Index[i]][j];
    }
}

//#pragma warning(push, 0)
//#pragma GCC diagnostic ignored "-Wembedded-directive"
//#pragma warning(disable:4820 4619 4668)
void firefly::findlimits(const int k)
{
#ifdef FFA_HARD_WALL
    for(int i=0;i<D;i++){
        if (ffa[k][i] < lb[i])
            ffa[k][i] = lb[i];
        else if (ffa[k][i] > ub[i])
            ffa[k][i] = ub[i];
    }
#else // Bounce
    for(int i=0;i<D;i++){
        if (ffa[k][i] < lb[i])
            ffa[k][i] = lb[i] + std::fmod(lb[i] - ffa[k][i], ub[i] - lb[i]);
        if (ffa[k][i] > ub[i])
            ffa[k][i] = ub[i] - std::fmod(ffa[k][i] - ub[i], ub[i] - lb[i]);
    }
#endif
}

//#pragma warning(pop)

void firefly::move_ffa()
{
    constexpr double beta0 = 1;

    for (size_t i = 0; i < n; ++i) {
        const double scale = abs(ub[i] - lb[i]);
        for (size_t j = 0; j < n; ++j) {
            double r = 0;
            for (size_t k = 0; k < D; ++k) {
                r += (ffa[i][k] - ffa[j][k]) * (ffa[i][k] - ffa[j][k]);
            }
            r = sqrt(r);
            if (I[i] > I[j])    // brighter and more attractive
            {
                const double beta = (beta0 - betamin) * exp(-gamma * pow(r, 2.0)) + betamin;
                for(int k=0;k<D;k++){
                    const double tmpf = alpha * drand48() * scale;
                    ffa[i][k] = ffa[i][k] * (1. - beta) + ffa_tmp[j][k] * beta + tmpf;
                }
            }
        }
        findlimits(i);
    }
}


const arma::mat levy(const int d) // Return colvec of columns d
{
    constexpr double beta = 3. / 2.;
    static const double sigma = std::pow(tgamma(1. + beta) * sin(M_PI * beta / 2.) / (tgamma((1. + beta) / 2.) * beta * pow(2, ((beta - 1.) / 2.))), 1. / beta);

    arma::mat u(1, d, arma::fill::randn);
    u = u * sigma;
    arma::mat v(1, d, arma::fill::randn);
    arma::mat step = u / arma::pow(arma::abs(v), 1. / beta);
    return step;
}


void firefly::move_ffa_adaptive(const double rate)
{
    constexpr double b = 1;
    constexpr double beta0 = 1;
    const arma::mat levy_random = levy(D);
    const auto ffa_prev = ffa;
    arma::mat scale(D, 1);
    for (size_t k = 0; k < D; k++) {
        scale(k, 0) = abs(ub[k] - lb[k]);
    }
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double r = 0;
            for (size_t k = 0; k < D; k++) {
                r += (ffa[i][k] - ffa[j][k]) * (ffa[i][k] - ffa[j][k]);
            }
            r = sqrt(r);
            if (I[i] > I[j])    // brighter and more attractive
            {
                const double beta = (beta0 - betamin) * exp(-gamma * pow(r, 2.0)) + betamin;
                if (drand48() > rate) {
                    for(int k= 0;k< D;k++){
                        const double tmpf = alpha * copysign(1., (drand48() - 0.5)) * levy_random(0, k) * scale(k, 0);
                        ffa[i][k] = ffa[i][k] * (1. - beta) + ffa_prev[j][k] * beta + tmpf;
                    }
                } else {
                    const double l = -1. + 2. * drand48();
                    for(int k=0;k<D;k++){
                        ffa[i][k] = ffa[i][k] + (-ffa[i][k] + ffa_prev[j][k]) * (beta * exp(l * b) * cos(2 * M_PI * l));
		    }
                }
            }
        }
        findlimits(i);
    }
}


/* display syntax messages */
void firefly::ffa_main()
{
    //LOG4_BEGIN();

#ifdef DUMP
    dump_ffa(t);
#endif
    size_t t = 1;        // generation  counter
    while (t <= MaxGeneration) {
        // this line of reducing alpha is optional
        alpha = alpha_new(alpha, MaxGeneration);

        // evaluate new solutions
        for(int i=0;i<n;i++){
              I[i] = f[i] = function(ffa[i]);        // obtain fitness of solution
        }

        // ranking fireflies by their light intensity
        sort_ffa();
        // replace old population
        replace_ffa();

        // find the current best
        for (size_t i = 0; i < D; ++i) nbest[i] = ffa[0][i];
        fbest[t - 1] = I[0];

        // move all fireflies to the better locations
        double rate;
        if (t == 1) {
            rate = 0.5;
        } else {
            double a, b;
            if (floor(log10(fabs(fbest[t - 2]))) == floor(log10(fabs(fbest[t - 1])))) {
                const double theta = floor(log10(fabs(fbest[t - 1] - trunc(fbest[t - 1]))));
                a = fbest[t - 2] - pow(10, theta) * floor(fbest[t - 2] / pow(10, theta + 1.));
                b = fbest[t - 1] - pow(10, theta) * floor(fbest[t - 1] / pow(10, theta + 1.));
            } else {
                a = fbest[t - 2];
                b = fbest[t - 1];
            }
            rate = 1. / (1. + exp(-a / b));
        }
        move_ffa_adaptive(rate);
        //LOG4_DEBUG("Firefly iteration " << t << ", best score " << fbest[t - 1] << ", parameters " << common::deep_to_string(nbest));
        ++t;
    }
    //LOG4_END();
}

