//
// Created by zarko on 12/4/22.
//
#include <set>
#include <thread>
#include "common/defines.h"
#include "optimizer.hpp"


#ifndef SVR_FIREFLY_HPP
#define SVR_FIREFLY_HPP

#define FFA_ALPHA       .2   // Starting randomness
#define FFA_BETAMIN     .5   // Attraction
#define FFA_GAMMA       1.   // Visibility
// #define FFA_HARD_WALL
#define ADAPTIVE_FFA
// #define DUMP	1
// #define FFA_RANDOMLESNESS 1.  // Decrease of randomness as iterations pass, range 0 - MAX_GENERATIONS, higher is more randomness at the last iteration, slower decrease of randomness.


namespace svr {
namespace optimizer {

struct f_I_Index {
    double f;
    double I;
    int Index;
};

bool comp_f_I_Index(const f_I_Index &lhs, const f_I_Index &rhs);

class firefly {
    static constexpr double delta_base = 1e-3 / .9;
    static constexpr double b_1 = 1;
    static constexpr double beta0 = 1;
    double delta; // delta parameter for calculating alpha_new

    std::vector<double> levy_random;

    size_t D = 2;	    		        // dimension of the problem
    size_t n = OPT_PARTICLES;			// number of fireflies
    size_t MaxGeneration = MAX_ITERATIONS_OPT;  // number of iterations
    std::vector<f_I_Index> fireflies;		        // sort of fireflies according to fitness values
    size_t sobol_ctr = 0;

    std::vector<std::vector<double>> ffa;	    // firefly agents
    std::vector<std::vector<double>> ffa_tmp; // intermediate population
    std::vector<double> nbest;          // the best solution found so far
    std::vector<double> lb;	        // upper bound
    std::vector<double> ub;         // lower bound

    double alpha = FFA_ALPHA;	    // alpha parameter
    double betamin = FFA_BETAMIN;  // beta parameter
    double gamma = FFA_GAMMA;	   // gamma parameter

    std::vector<double> fbest;			        // the best objective function

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
    operator std::tuple<double, std::vector<double>>() { return {fbest[MaxGeneration], nbest}; }

    firefly(const size_t D, const size_t n, const size_t MaxGeneration, const double alpha, const double betamin, const double gamma,
            const std::vector<double> &lb, const std::vector<double> &ub, const std::vector<double> &pows, const loss_callback_t function);
};


}
}

#endif //SVR_FIREFLY_HPP
