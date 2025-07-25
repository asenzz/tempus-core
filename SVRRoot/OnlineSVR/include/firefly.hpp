//
// Created by zarko on 12/4/22.
//
#include <deque>
#include <set>
#include <thread>
#include "common/defines.h"
#include "optimizer.hpp"
#include "common/compatibility.hpp"
#include "common/constants.hpp"


#ifndef SVR_FIREFLY_HPP
#define SVR_FIREFLY_HPP

// #define FFA_HARD_WALL
#define ADAPTIVE_FFA
// #define DUMP	1
// #define FIREFLY_CUDA

namespace svr {
namespace optimizer {

struct ffa_particle {
    int Index = 0;
    double f = 0;
    double I = 0;
};

constexpr double C_FFA_alpha = .5;   // Starting randomness
constexpr double C_FFA_betamin = .5;   // Attraction
constexpr double C_FFA_gamma = 1.;   // Visibility

class firefly {
    static constexpr double delta_base = 1e-3 / .9;
    static constexpr double b_1 = 1;
    static constexpr double beta0 = 1;

    arma::vec levy_random;

    const unsigned D = 2;	    		        // dimension of the problem
    const unsigned n = OPT_PARTICLES;			// number of fireflies
    const unsigned MaxGeneration = MAX_ITERATIONS_OPT;  // number of iterations
    const size_t sobol_ctr = 0;

    const arma::vec range;
    const arma::vec lb;	        // upper bound
    const arma::vec ub;         // lower bound

    double alpha = C_FFA_alpha;	    // alpha parameter
    const double betamin = C_FFA_betamin;  // beta parameter
    const double gamma = C_FFA_gamma;	   // gamma parameter

    double delta = 0; // delta parameter for calculating alpha_new
    arma::mat ffa;	    // firefly agents
    std::deque<ffa_particle> particles;
    arma::vec best_parameters;          // the best solution found so far

    double best_score = std::numeric_limits<double>::max();   // the best objective function

    const loss_callback_t function = [&](CRPTRd args, const uint32_t n) -> double
    {
        return std::numeric_limits<double>::quiet_NaN();
    };

    double alpha_new(const double alpha, const double NGen);
    void sort_ffa();
    void replace_ffa();
    void move_ffa_adaptive(const double rate);
    void ffa_main();
    void run_iteration();

public:
    operator std::pair<double, std::vector<double>>() { return {best_score, arma::conv_to<std::vector<double>>::from(best_parameters)}; }

    firefly(const unsigned D, const unsigned n, const unsigned MaxGeneration, const double alpha, const double betamin, const double gamma, const arma::mat &bounds,
            const arma::vec &pows, const loss_callback_t function);
};


}
}

#endif //SVR_FIREFLY_HPP
