//
// Created by zarko on 12/4/22.
//

#ifndef SVR_FIREFLY_HPP
#define SVR_FIREFLY_HPP

#define ADAPTIVE_FFA
// #define DUMP	1
#define MAX_FFA	100
#define MAX_D	100
#define SUB_LIMIT 0.7


//Emo
#define OPT_PARTICLES 10
#define MAX_ITERATIONS_OPT 100
#define FFA_ALPHA 0.2
#define FFA_BETAMIN 1.
#define FFA_GAMMA 1.




typedef std::function<double (const std::vector<double>&)> loss_callback_t;
typedef std::vector<std::pair<double, std::vector<double>>> pso_returns_t;


// TODO Replace arrays with vectors
class firefly {
    size_t D = 2;	    		        // dimension of the problem
    size_t n = OPT_PARTICLES;			// number of fireflies
    size_t MaxGeneration = MAX_ITERATIONS_OPT;         // number of iterations
    std::vector<int> Index;		        // sort of fireflies according to fitness values
    size_t sobol_ctr = 0;

    std::vector<std::vector<double>> ffa;	    // firefly agents
    std::vector<std::vector<double>> ffa_tmp; // intermediate population
    std::vector<double> f;		        // fitness values
    std::vector<double> I;		        // light intensity
    std::vector<double> nbest;            // the best solution found so far
    std::vector<double> lb;	        // upper bound
    std::vector<double> ub;         // lower bound

    double alpha = FFA_ALPHA;	    // alpha parameter
    double betamin = FFA_BETAMIN;  // beta parameter
    double gamma = FFA_GAMMA;	   // gamma parameter

    std::vector<double> fbest;			        // the best objective function

/*Write your own objective function */
    loss_callback_t function;

    double alpha_new(const double alpha, const size_t NGen);
    void sort_ffa();
    void replace_ffa();
    void move_ffa();
    void move_ffa_adaptive(const double rate);
    void findlimits(const int k);
    void ffa_main();

public:
    operator std::vector<double> () { return nbest; }

    firefly(const size_t D, const size_t n, const size_t MaxGeneration, const double alpha, const double betamin, const double gamma,
            const std::vector<double> &lb, const std::vector<double> &ub, const loss_callback_t function);
};



#endif //SVR_FIREFLY_HPP
