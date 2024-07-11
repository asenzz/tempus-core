//
// Created by zarko on 3/24/24.
//

#ifndef SVR_PPRUNE_HPP
#define SVR_PPRUNE_HPP

#include <condition_variable>
#include <deque>
#include <memory>
#include <armadillo>
#include <prima/prima.h>


namespace svr {
namespace optimizer {

constexpr unsigned C_elect_threshold = 10;

struct t_pprune_res {
    double best_score = std::numeric_limits<double>::infinity();
    arma::vec best_parameters;
    size_t total_iterations = 0;
};

typedef std::function<void(const double x[], double *const f)> t_pprune_cost_fun, *t_pprune_cost_fun_ptr;

struct t_calfun_data
{
    bool no_elect = true;
    std::shared_ptr<std::condition_variable> elect_ready;
    std::shared_ptr<std::mutex> elect_mx;
    std::shared_ptr<std::deque<t_calfun_data *>> f_score;
    const t_pprune_cost_fun cost_fun;
    const size_t particle_index = 0;
    const size_t maxfun = 0;
    double best_f = std::numeric_limits<double>::infinity();
    size_t nf = 0;
    bool zombie = false;
    bool drop(const size_t keep_particles);
};

using t_calfun_data_ptr = t_calfun_data *;


class pprune {
    const size_t n, D;
    const arma::mat bounds;
    const arma::vec pows, ranges;
    t_pprune_res result;

public:
    pprune(const prima_algorithm_t type, const size_t n_particles, const arma::mat &bounds,
           const t_pprune_cost_fun &cost_f,
           const size_t maxfun = 50,
           const double rhobeg = std::numeric_limits<double>::quiet_NaN(),
           const double rhoend = std::numeric_limits<double>::quiet_NaN(),
           const arma::mat &x0_ = {}, const arma::vec &pows = {});

    static void calfun(const double x[], double *const f, t_calfun_data_ptr const calfun_data);

    static arma::vec ensure_bounds(const double *x, const arma::mat &bounds);

    operator t_pprune_res();
};

}
}

#endif //SVR_PPRUNE_HPP
