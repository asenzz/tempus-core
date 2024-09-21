//
// Created by zarko on 3/24/24.
//

#ifndef SVR_PPRUNE_HPP
#define SVR_PPRUNE_HPP

#include <armadillo>
#include "common/constants.hpp"
#include "common/compatibility.hpp"

namespace svr {
namespace optimizer {

constexpr unsigned C_biteopt_depth = 1;

struct t_pprune_res {
    double best_score = std::numeric_limits<double>::infinity();
    arma::vec best_parameters;
    unsigned total_iterations = 0;
};

typedef std::function<void(CPTR(double) x, double *const f)> t_pprune_cost_fun, *t_pprune_cost_fun_ptr;

struct t_calfun_data;
using t_calfun_data_ptr = t_calfun_data *;

class pprune {
    const unsigned n, D;
    const arma::mat bounds;
    const arma::vec pows, ranges;
    t_pprune_res result;

public:
    pprune(const unsigned algo_type, const unsigned n_particles, const arma::mat &bounds,
           const t_pprune_cost_fun &cost_f,
           const unsigned maxfun = 50,
           double rhobeg = 0,
           double rhoend = 0,
           arma::mat x0 = {}, const arma::vec &pows = {},
           const unsigned depth = C_biteopt_depth);

    static void calfun(CPTR(double) x, double *const f, t_calfun_data_ptr const calfun_data);

    static arma::vec ensure_bounds(CPTR(double) x, const arma::mat &bounds);

    operator t_pprune_res();
};

}
}

#endif //SVR_PPRUNE_HPP
