/* An implementation of the Particle Swarm Optimization algorithm

   Copyright 2010 Kyriakos Kentzoglanakis

   This program is free software: you can redistribute it and/or
   modify it under the terms of the GNU General Public License version
   3 as published by the Free Software Foundation.

   This program is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see
   <http://www.gnu.org/licenses/>.
 */


#pragma once

#include <vector>
#include <boost/throw_exception.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <unordered_map>
#include <atomic>
#include <mutex>
#include "common/compatibility.hpp"

namespace svr {
namespace optimizer {

#define PSO_TOPOLOGY            svr::optimizer::PsoTopology::global
#define PSO_SAVE_STATE_PATH     svr::common::formatter() << "/tmp/pso_state_" << PRIMARY_COLUMN << "_" << _p_svr_parameters->get_decon_level() << ".sav"

// CONSTANTS
#define PSO_MAX_SIZE 1000 // max swarm size
#define PSO_INERTIA 0.7298 // default value of w (see clerc02)


// === NEIGHBORHOOD SCHEMES ===

// global best topology
#define PSO_NHOOD_GLOBAL 0

// ring topology
#define PSO_NHOOD_RING 1

// Random neighborhood topology
// **see http://clerc.maurice.free.fr/pso/random_topology.pdf**
#define PSO_NHOOD_RANDOM 2



// === INERTIA WEIGHT UPDATE FUNCTIONS ===
#define PSO_W_CONST 0
#define PSO_W_LIN_DEC 1


// PSO SOLUTION -- Initialized by the user
typedef struct
{
    double error;
    double *gbest; // should contain DIM elements!!
} pso_result_t;


// OBJECTIVE FUNCTION TYPE
typedef double (*pso_obj_fun_t)(double *, int, void *);


// PSO SETTINGS
struct pso_settings_t
{
    std::string state_file;
    int dim; // problem dimensionality
    std::vector<double> x_lo_limit; // lower range limit
    std::vector<double> x_hi_limit; // higher range limit
    double goal; // optimization goal (error threshold)

    int size; // swarm size (number of particles)
    int print_every; // ... N steps (set to 0 for no output)
    int steps; // maximum number of iterations
    int step; // current PSO step
    double c1; // cognitive coefficient
    double c2; // social coefficient
    double w_max; // max inertia weight value
    double w_min; // min inertia weight value

    int clamp_pos; // whether to keep particle position within defined bounds (TRUE)
    // or apply periodic boundary conditions (FALSE)
    int nhood_strategy; // neighborhood strategy (see PSO_NHOOD_*)
    int nhood_size; // neighborhood size 
    int w_strategy; // inertia weight strategy (see PSO_W_*)
};


// return the swarm size based on dimensionality
int pso_calc_swarm_size(const int dim);

// set the default PSO settings
void pso_set_default_settings(pso_settings_t *p_settings);


// minimize the provided obj_fun using PSO with the specified settings
// and store the result in *solution
std::vector<std::pair<double, std::vector<double>>>
pso_solve(
        const std::function<double(std::vector<double> &)> &f,
        pso_settings_t *settings,
        const size_t decon_level,
        const std::string &column_name);

struct pso_state
{
    //std::vector<double> pos_;
    std::vector<std::vector<double>> pos_;
    std::vector<std::vector<double>> vel_;
    std::vector<std::vector<double>> pos_nb_;
    std::vector<std::vector<double>> pos_b_;
    std::vector<double> fit_;
    std::vector<double> fit_b_;
    std::vector<bool> particles_ready_;
    std::vector<double> gbest_;
    int step_;
    double error_;

    friend class boost::serialization::access;
    // When the class Archive corresponds to an output archive, the
    // & operator is defined similar to <<.  Likewise, when the class Archive
    // is a type of input archive the & operator is defined similar to >>.

    pso_state();

    pso_state(
            std::vector<std::vector<double>> pos,
            std::vector<std::vector<double>> vel,
            std::vector<std::vector<double>> pos_nb,
            std::vector<std::vector<double>> pos_b,
            std::vector<double> fit,
            std::vector<double> fit_b,
            std::vector<bool> particles_ready,
            std::vector<double> gbest,
            int step,
            double error);

    std::string save_to_string();

    template<class Archive>
    void serialize(Archive &ar, const unsigned int version)
    {
        ar & pos_;
        ar & vel_;
        ar & pos_nb_;
        ar & pos_b_;
        ar & fit_;
        ar & fit_b_;
        ar & particles_ready_;
        ar & gbest_;
        ar & step_;
        ar & error_;

    }

    static std::vector<std::vector<double>> convert2vector2d(size_t dim1, size_t dim2, const void *input);

    static void convert_from_vector2d(const std::vector<std::vector<double>> &input, size_t dim1, size_t dim2, void *output);

    static bool load_from_string(const std::string &full_message, pso_state &success);
};

class pso_state_io
{
    pso_state_io();

    ~pso_state_io() = default;

    pso_state_io(const pso_state_io &) = delete;

    pso_state_io &operator=(const pso_state_io &) = delete;

    std::unordered_map<std::pair<std::string, std::string>, pso_state> pso_states;
    std::unordered_map<std::pair<std::string, std::string>, bool> finish_states;

    std::atomic<bool> dirty_state;
    std::mutex pso_states_mutex;
    std::mutex finish_state_mutex;

public:
    static pso_state_io &get_instance();

    void read_state_file(const std::string &input_state_file);

    void save_state_to_file(const std::string &filename);

    bool get_state(const std::pair<std::string, std::string> &model_id, pso_state &state);

    void update_state(const std::pair<std::string, std::string> &model_id, const pso_state &state);

    void set_finish_state(const std::pair<std::string, std::string> &model_id, bool status);
};

}
}
