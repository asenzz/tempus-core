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

constexpr uint16_t C_biteopt_depth = 1;

struct t_pprune_res {
    double best_score = std::numeric_limits<double>::infinity();
    arma::vec best_parameters;
    uint32_t total_iterations = 0;
};

typedef std::function<void(CPTRd x, double *const f)> t_pprune_cost_fun, *t_pprune_cost_fun_ptr;

struct t_calfun_data;
using t_calfun_data_ptr = t_calfun_data *;

class t_biteopt_cost;

class pprune {
    friend t_calfun_data;
    friend t_biteopt_cost;

    static constexpr uint32_t C_max_population = 3000;

    const uint32_t n, D, maxfun, depth, iter, max_population;
    const arma::mat bounds;
    const arma::vec pows, ranges;
    const bool no_elect;
    t_pprune_res result;

public:

    enum e_algo_type : uint8_t {
        e_biteopt = 0,
        e_knitro = 1,
        e_prima = 2,
        e_petsc = 3
    };

    static constexpr e_algo_type C_default_algo = e_algo_type::e_biteopt;

    pprune(const e_algo_type algo_type, const uint32_t n_particles, const arma::mat &bounds,
           const t_pprune_cost_fun &cost_f,
           const uint32_t maxfun = 50,
           double rhobeg = 0,
           double rhoend = 0,
           arma::mat x0 = {}, // x0 should have at most n_particles columns, and rows equal to bounds.n_rows
           const arma::vec &pows = {},
           const uint16_t depth = C_biteopt_depth,
           const bool force_no_elect = false,
           const uint32_t max_population = C_max_population);

    void pprune_biteopt(const uint32_t n_particles, const t_pprune_cost_fun &cost_f, double rhobeg, double rhoend, const arma::mat &x0);

    void pprune_knitro(const uint32_t n_particles, const t_pprune_cost_fun &cost_f, double rhobeg, double rhoend, const arma::mat &x0);

    void pprune_prima(const uint32_t n_particles, const t_pprune_cost_fun &cost_f, double rhobeg, double rhoend, const arma::mat &x0);

    void pprune_petsc(const uint32_t n_particles, const t_pprune_cost_fun &cost_f, double rhobeg, double rhoend, const arma::mat &x0);

    static void calfun(CPTRd x, double *const f, t_calfun_data_ptr const calfun_data);

    static arma::vec ensure_bounds(CPTRd x, const arma::mat &bounds);

    operator t_pprune_res() const noexcept;
};

}
}

#endif //SVR_PPRUNE_HPP
