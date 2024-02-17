//
// Created by zarko on 12/4/22.
//
#include <deque>
#include <set>
#include <thread>
#include "common/defines.h"
#include "optimizer.hpp"


#ifndef SVR_FIREFLY_HPP
#define SVR_FIREFLY_HPP

// #define FFA_HARD_WALL
#define ADAPTIVE_FFA
// #define DUMP	1
// #define FFA_RANDOMLESNESS 20.  // Decrease of randomness as iterations pass, range 0 - MAX_GENERATIONS, higher is more randomness at the last iteration, slower decrease of randomness.


namespace svr {
namespace optimizer {

struct ffa_particle {
    int Index;
    double f;
    double I;
};

class firefly {
    static constexpr double delta_base = 1e-3 / .9;
    static constexpr double b_1 = 1;
    static constexpr double beta0 = 1;
    double delta; // delta parameter for calculating alpha_new

    std::vector<double> levy_random;

    size_t D = 2;	    		        // dimension of the problem
    size_t n = OPT_PARTICLES;			// number of fireflies
    size_t MaxGeneration = MAX_ITERATIONS_OPT;  // number of iterations
    size_t sobol_ctr = 0;

    std::deque<std::vector<double>> ffa;	    // firefly agents
    std::deque<std::vector<double>> ffa_tmp; // intermediate population
    std::deque<ffa_particle> particles;
    std::vector<double> nbest, allround_best;          // the best solution found so far
    std::vector<double> lb;	        // upper bound
    std::vector<double> ub;         // lower bound

    double alpha = FFA_ALPHA;	    // alpha parameter
    double betamin = FFA_BETAMIN;  // beta parameter
    double gamma = FFA_GAMMA;	   // gamma parameter

    double fbest;			        // the best objective function

    loss_callback_t function = [&](const std::vector<double> &args) -> double
    {
        double res = 0;
        for (const auto a: args) res += std::abs(a);
        return res;
    };

    double alpha_new(const double alpha, const double NGen);
    void sort_ffa();
    void replace_ffa();
    void move_ffa(); // Old not used
    void move_ffa_adaptive(const double rate);
    void findlimits(const size_t k);
    void ffa_main();
    void run_iteration();

public:
    operator std::pair<double, std::vector<double>>() { return {fbest, allround_best}; }

    firefly(const size_t D, const size_t n, const size_t MaxGeneration, const double alpha, const double betamin, const double gamma,
            const std::vector<double> &lb, const std::vector<double> &ub, const std::vector<double> &pows, const loss_callback_t function);
};


}
}

#endif //SVR_FIREFLY_HPP
