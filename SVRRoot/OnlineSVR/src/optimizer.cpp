#include "optimizer.hpp"
#include "pso.hpp"

namespace svr {
namespace optimizer {

pso_returns_t pso(const loss_callback_t &f, const PSO_parameters& pso_parameters, const size_t& decon_level, const std::string &column_name)
{
    // set the default settings
    pso_settings_t settings;

    pso_set_default_settings(&settings);
    //settings.x_lo_limit = bounds.to_vector(bound_type::min);
    //settings.x_hi_limit = bounds.to_vector(bound_type::max); // Breaks old paramtune TODO enable if ever paramtune is reintroduced
    settings.x_lo_limit = pso_parameters.lows_;
    settings.x_hi_limit = pso_parameters.highs_;
    settings.dim = std::min(settings.x_lo_limit.size(), settings.x_hi_limit.size());

    // set PSO settings manually
    settings.size = pso_parameters.particles_number_;
    settings.steps = pso_parameters.iteration_number_;
    settings.nhood_strategy = static_cast<size_t>(pso_parameters.pso_topology_);
    settings.state_file = pso_parameters.state_file_;

    // run optimization algorithm
    return pso_solve(f, &settings, decon_level, column_name);
}


}
}
