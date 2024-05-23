//
// Created by zarko on 3/24/24.
//

#ifndef SVR_PPRIMA_HPP
#define SVR_PPRIMA_HPP

#include <prima/prima.h>
#include <armadillo>
#include <deque>


namespace svr {
namespace optimizer {

struct t_pprima_res {
    double best_score = std::numeric_limits<double>::infinity();
    arma::vec best_parameters;
    size_t total_iterations = 0;
};

typedef std::function<void(const double x[], double *const f)> t_pprima_cost_fun, *t_prima_cost_fun_ptr;

class pprima {
    const size_t n, D;
    const arma::mat bounds;
    const arma::vec pows, ranges;
    t_pprima_res result;

    static void prima_progress_callback(
            const int n, const double x[], const double f, const int nf,
            const int tr, const double cstrv, const int m_nlcon, const double nlconstr[], bool *const terminate);

    static void calfun(const double x[], double *const f, const void *data);

public:
    pprima(const prima_algorithm_t type, const size_t n_particles, const arma::mat &bounds,
           const t_pprima_cost_fun &cost_f,
           const size_t maxfun = 50,
           const double rhobeg = std::numeric_limits<double>::quiet_NaN(),
           const double rhoend = std::numeric_limits<double>::quiet_NaN(),
           const arma::mat &x0_ = {}, const arma::vec &pows = {}, const size_t n_threads = 0);

    static arma::vec ensure_bounds(const double *x, const arma::mat &bounds);

    operator t_pprima_res();
};

}
}

#endif //SVR_PPRIMA_HPP
