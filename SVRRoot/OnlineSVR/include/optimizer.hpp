#pragma once

#include <vector>
#include <functional>
#include <string>

namespace svr {
namespace optimizer {


constexpr unsigned MAX_ITERATIONS_OPT = 50;
constexpr unsigned OPT_PARTICLES = 100;


typedef std::function<double (const std::vector<double>&)> loss_callback_t;
typedef std::vector<std::pair<double, std::vector<double>>> pso_returns_t;

enum class PsoTopology : size_t {global = 0, ring = 1, random = 2};

struct PSO_parameters
{
    size_t iteration_number_;
    size_t particles_number_;
    std::string state_file_;
    PsoTopology pso_topology_;
    std::vector<double> lows_;
    std::vector<double> highs_;
};

struct NM_parameters
{
    size_t max_iteration_number_;
    double tolerance_;
};


pso_returns_t
pso(const loss_callback_t &f, const PSO_parameters& pso_parameters, const size_t& level_idx, const std::string& column_name);

} //optimizer
} //svr

