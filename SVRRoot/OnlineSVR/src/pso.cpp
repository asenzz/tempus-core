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

#include <cfloat>
#include "pso.hpp"

#include <util/math_utils.hpp>
#include <util/string_utils.hpp>
#include <common/thread_pool.hpp>

#include "common/logging.hpp"
#include "common/compatibility.hpp"
#include "common/parallelism.hpp"
#include "optimizer.hpp"
#include "sobol.hpp"

namespace svr {
namespace optimizer {

pso_state_io &pso_state_io::get_instance()
{
    static pso_state_io instance;
    return instance;
}

pso_state_io::pso_state_io()
{
    dirty_state.store(false, std::memory_order_relaxed);
};

//==============================================================
// calulate swarm size based on dimensionality
int pso_calc_swarm_size(const int dim)
{
    return std::min<int>(10 + 2 * sqrt(dim), PSO_MAX_SIZE);
}


//==============================================================
//          INERTIA WEIGHT UPDATE STRATEGIES
//==============================================================
// calculate linearly decreasing inertia weight
double calc_inertia_lin_dec(const int step, const pso_settings_t *settings)
{
    return settings->w_min + (settings->w_max - settings->w_min) * double(settings->steps - step) / double(settings->steps);
}


//==============================================================
//          NEIGHBORHOOD (COMM) MATRIX STRATEGIES
//==============================================================
// global neighborhood
void
inform_global(
        int *comm,
        double *pos_nb,
        double *pos_b,
        double *fit_b,
        double *gbest,
        int improved,
        const pso_settings_t *settings)
{
    // all particles have the same attractor (gbest)
    // copy the contents of gbest to pos_nb
    for (int i = 0; i < settings->size; ++i)
        memmove((void *) &pos_nb[i * settings->dim], (void *) gbest, sizeof(double) * settings->dim);
}


// ===============================================================
// general inform function :: according to the connectivity
// matrix COMM, it copies the best position (from pos_b) of the
// informers of each particle to the pos_nb matrix
void inform(
        int *comm,
        double *pos_nb,
        double *pos_b,
        double *fit_b,
        int improved,
        const pso_settings_t *settings)
{
    // for each particle
    for (int j = 0; j < settings->size; j++) {
        int b_n = j; // self is best
        // who is the best informer??
        for (int i = 0; i < settings->size; ++i)
            // the i^th particle informs the j^th particle
            if (comm[i * settings->size + j] && fit_b[i] < fit_b[b_n])
                // found a better informer for j^th particle
                b_n = i;
        // copy pos_b of b_n^th particle to pos_nb[j]
        memmove((void *) &pos_nb[j * settings->dim],
                (void *) &pos_b[b_n * settings->dim],
                sizeof(double) * settings->dim);
    }
}




// =============
// ring topology
// =============

// topology initialization :: this is a static (i.e. fixed) topology
void init_comm_ring(int *comm, const pso_settings_t *settings)
{
    // reset array
    memset((void *) comm, 0, sizeof(int) * settings->size * settings->size);

    // choose informers
    for (int i = 0; i < settings->size; ++i) {
        // set diagonal to 1
        comm[i * settings->size + i] = 1;
        if (i == 0) {
            // look right
            comm[i * settings->size + i + 1] = 1;
            // look left
            comm[(i + 1) * settings->size - 1] = 1;
        } else if (i == settings->size - 1) {
            // look right
            comm[i * settings->size] = 1;
            // look left
            comm[i * settings->size + i - 1] = 1;
        } else {
            // look right
            comm[i * settings->size + i + 1] = 1;
            // look left
            comm[i * settings->size + i - 1] = 1;
        }
    }
}


void inform_ring(
        int *comm, double *pos_nb,
        double *pos_b, double *fit_b,
        double *gbest, int improved,
        const pso_settings_t *settings)
{
    // update pos_nb matrix
    inform(comm, pos_nb, pos_b, fit_b, improved, settings);
}

// ============================
// random neighborhood topology
// ============================
void init_comm_random(int *const comm, const pso_settings_t *settings)
{

    // reset array
    memset(comm, 0, sizeof(int) * settings->size * settings->size);

    // choose informers
    for (int i = 0; i < settings->size; ++i) {
        // each particle informs itself
        comm[i * settings->size + i] = 1;
        // choose kappa (on average) informers for each particle
        for (int k = 0; k < settings->nhood_size; ++k) {
            // generate a random index
            const int j = floor(svr::common::get_uniform_random_value() * settings->size);
            // particle i informs particle j
            comm[i * settings->size + j] = 1;
        }
    }
}


void inform_random(
        int *comm,
        double *pos_nb,
        double *pos_b,
        double *fit_b,
        double *gbest,
        int improved,
        const pso_settings_t *settings)
{
    // regenerate connectivity??
    if (!improved) init_comm_random(comm, settings);
    inform(comm, pos_nb, pos_b, fit_b, improved, settings);
}


//==============================================================
// return default pso settings
void pso_set_default_settings(pso_settings_t *const p_settings)
{
    // set some default values
    p_settings->dim = 10;
    p_settings->x_lo_limit = std::vector<double>(p_settings->dim, -20);
    p_settings->x_hi_limit = std::vector<double>(p_settings->dim, 20);
    p_settings->goal = 1e-5;

    p_settings->size = pso_calc_swarm_size(p_settings->dim);
    p_settings->print_every = 1000;
    p_settings->steps = 100000;
    p_settings->c1 = 1.496;
    p_settings->c2 = 1.496;
    p_settings->w_max = PSO_INERTIA;
    p_settings->w_min = 0.3;

    p_settings->clamp_pos = 0;
    p_settings->nhood_strategy = PSO_NHOOD_RING;
    p_settings->nhood_size = 5;
    p_settings->w_strategy = PSO_W_LIN_DEC;
    p_settings->state_file = "";
}


pso_state::pso_state(
        std::vector<std::vector<double>> pos,
        std::vector<std::vector<double>> vel,
        std::vector<std::vector<double>> pos_nb,
        std::vector<std::vector<double>> pos_b,
        std::vector<double> fit,
        std::vector<double> fit_b,
        std::vector<bool> particles_ready,
        std::vector<double> gbest,
        int step,
        double error) :
        pos_(pos), vel_(vel), pos_nb_(pos_nb), pos_b_(pos_b), fit_(fit), fit_b_(fit_b),
        particles_ready_(particles_ready), gbest_(gbest), step_(step), error_(error)
{}

std::string pso_state::save_to_string()
{
    std::ostringstream ostr;
    boost::archive::text_oarchive oa(ostr);
    oa << *this;
    std::ostringstream full_message;
    full_message << ostr.str().length() << " " << ostr.str();
    return full_message.str();
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wvla"

std::vector<std::vector<double>> pso_state::convert2vector2d(size_t dim1, size_t dim2, const void *input)
{
    const double **arr = (const double **) input; // const double (*arr)[dim2] = static_cast<const double (*)[dim2]>(input);
    std::vector<std::vector<double>> result(dim1);
#pragma omp parallel for num_threads(adj_threads(dim1)) schedule(static, 1)
    for (size_t i = 0; i < dim1; ++i)
        result[i] = common::wrap_vector<double>(const_cast<double *>(arr[i]), dim2);

    return result;
}

void
pso_state::convert_from_vector2d(const std::vector<std::vector<double>> &input, const size_t dim1, const size_t dim2,
                                 void *output)
{
    double **arr = (double **) output; // double (*arr)[dim2] = static_cast<double (*)[dim2]>(output);
#pragma omp parallel for num_threads(adj_threads(dim1)) schedule(static, 1)
    for (size_t i = 0; i < dim1; ++i)
        memcpy(arr[i], input[i].data(), dim2 * sizeof(double));
}

pso_state::pso_state()
{}

bool pso_state::load_from_string(const std::string &full_message, pso_state &saved_state)
{
    LOG4_DEBUG("Enter load_from_string");
    char junk;
    // create and open an archive for input
    std::istringstream iss(full_message);
    size_t state_size = 0;
    iss >> state_size;
    iss >> std::noskipws >> junk;
    std::string rest;
    std::getline(iss, rest);
    if (state_size == rest.length()) {
        iss = std::istringstream(rest);
        boost::archive::text_iarchive ia(iss);
        try {
            ia >> saved_state;
        } catch (const std::exception &ex) {
            LOG4_DEBUG("Caught exception in load_from_string: " << ex.what());
            return false;
        }
        return true;
    } else
        return false;
}

#ifdef PSO_SAVE_STATE
svr::future<void> async_save_file;
#endif
std::once_flag start_async_flag;

void pso_state_io::save_state_to_file(const std::string &filename)
{
    LOG4_DEBUG("Start save_state_to_file function.");
    bool all_threads_finished = false;
    while (!all_threads_finished) {
        std::unique_lock pso_states_lock(pso_states_mutex);
        if (dirty_state.load(std::memory_order_relaxed)) {
            std::ofstream output_file(filename, std::ios::out);
            for (auto state: this->pso_states) {
                //level << column << archive
                output_file << state.first.first << " " << state.first.second << " " << state.second.save_to_string()
                            << std::endl;
            }
            output_file.close();
            dirty_state.store(false, std::memory_order_relaxed);
        }
        pso_states_lock.unlock();
        all_threads_finished = true;
        std::unique_lock finish_states_lock(finish_state_mutex);
        for (auto finish_state: finish_states) {
            if (!finish_state.second) {
                all_threads_finished = false;
                break;
            }
        }
        finish_states_lock.unlock();
        if (all_threads_finished) {
            LOG4_DEBUG("All threads have finished. Exiting save_state_to_file function");
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5000));
    }
}

void pso_state_io::read_state_file(const std::string &input_state_file)
{
    LOG4_BEGIN();

    std::ifstream saved_particles_file(input_state_file, std::ifstream::in);
    if (input_state_file != "" && saved_particles_file.is_open()) {
        LOG4_DEBUG("Successfully opened file " << input_state_file << ". Reading state.");
        std::string line;
        size_t decon_level_from_file;
        std::string column_name_from_file;
        while (std::getline(saved_particles_file, line)) {
            char junk;
            std::istringstream iss(line);
            iss >> decon_level_from_file;
            iss >> column_name_from_file;
            std::pair<std::string, std::string> model_id(std::to_string(decon_level_from_file), column_name_from_file);
            iss >> std::noskipws >> junk;
            std::string rest;
            std::getline(iss, rest);
            pso_state saved_state;
            if (pso_state::load_from_string(rest, saved_state))
                this->pso_states[model_id] = saved_state;
            else
                LOG4_ERROR("Failed loading model " << model_id.first << "_" << model_id.second);
        }
    }
    saved_particles_file.close();

    LOG4_END();
}

bool pso_state_io::get_state(const std::pair<std::string, std::string> &model_id, pso_state &state)
{
    std::unique_lock<std::mutex> add_new_state_lock(pso_states_mutex);
    if (pso_states.count(model_id)) {
        state = pso_states[model_id];
        add_new_state_lock.unlock();
        return true;
    } else {
        add_new_state_lock.unlock();
        return false;
    }
}

void pso_state_io::update_state(const std::pair<std::string, std::string> &model_id, const pso_state &state)
{
    std::unique_lock<std::mutex> add_new_state_lock(pso_states_mutex);
    pso_states[model_id] = state;
    dirty_state.store(true, std::memory_order_relaxed);
    add_new_state_lock.unlock();
}

void pso_state_io::set_finish_state(const std::pair<std::string, std::string> &model_id, bool status)
{
    std::unique_lock<std::mutex> finish_lock(finish_state_mutex);
    LOG4_DEBUG("Set finish state " << status << " for model " << model_id.first << "_" << model_id.second);
    finish_states[model_id] = status;
    finish_lock.unlock();
}


int
update_particle(
        double *const curr_pos,
        double *const curr_vel,
        double *const curr_pos_b,
        double *const curr_pos_nb,
        const pso_settings_t *settings,
        const size_t d,
        const double w)
{
    // calculate stochastic coefficients
    // random numbers (coefficients)
    const auto rho1 = settings->c1 * svr::common::get_uniform_random_value();
    const auto rho2 = settings->c2 * svr::common::get_uniform_random_value();

    LOG4_DEBUG("w " << w << " curvel " << *curr_vel
                    << " rho1 " << rho1 << " cur_posb " << *curr_pos_b
                    << " curr_pos " << *curr_pos
                    << " rho2 " << rho2 << " curr_pos_nb "
                    << *curr_pos_nb);

    // Update velocity!
    *curr_vel = w * (*curr_vel) + rho1 * (*curr_pos_b - *curr_pos) +
                rho2 * (*curr_pos_nb - *curr_pos);
    // update position
    LOG4_DEBUG("Current velocity " << *curr_vel);
    *curr_pos += *curr_vel;
    // clamp position within bounds?
    if (settings->clamp_pos) {
        if (*curr_pos < settings->x_lo_limit[d]) {
            *curr_pos = settings->x_lo_limit[d];
            *curr_vel = 0;
        } else if (*curr_pos > settings->x_hi_limit[d]) {
            *curr_pos = settings->x_hi_limit[d];
            *curr_vel = 0;
        }
    } else {
        // enforce periodic boundary conditions
        if (*curr_pos < settings->x_lo_limit[d]) {
            *curr_pos = settings->x_lo_limit[d] + fmod(settings->x_lo_limit[d] - *curr_pos, settings->x_hi_limit[d] - settings->x_lo_limit[d]);
            *curr_vel /= 2.;
        } else if (*curr_pos > settings->x_hi_limit[d]) {
            *curr_pos = settings->x_hi_limit[d] - fmod(*curr_pos - settings->x_hi_limit[d], settings->x_hi_limit[d] - settings->x_lo_limit[d]);
            *curr_vel /= 2.;
        }
    }

    return d;
}

//==============================================================
//                     PSO ALGORITHM
//==============================================================
svr::optimizer::pso_returns_t
pso_solve(
        const std::function<double(std::vector<double> &)> &f, pso_settings_t *settings,
        const size_t decon_level, const std::string &column_name)
{
    bool is_caller_thread = false;
    std::pair<std::string, std::string> current_level_column(std::to_string(decon_level), column_name);
    // allocate memory for the best position buffer
    pso_result_t solution;
    solution.gbest = (double *) malloc(settings->dim * sizeof(double));

    // Particles
    double pos[settings->size][settings->dim]; // position matrix
    //std::vector<std::vector<double>> pos(settings->size, std::vector<double>(settings->dim));
    double vel[settings->size][settings->dim]; // velocity matrix
    double pos_b[settings->size][settings->dim]; // best position matrix
    double fit[settings->size]; // particle fitness vector
    double fit_b[settings->size]; // best fitness vector
    // Swarm
    double pos_nb[settings->size][settings->dim]; // what is the best informed
    // position for each particle
    int comm[settings->size][settings->size]; // communications:who informs who
    // rows : those who inform
    // cols : those who are informed
    int improved = 0; // whether solution->error was improved during
    // the last iteration
    int step = 0;

    std::vector<bool> particles_ready(settings->size, false);
    pso_state_io &pso_state_io_obj = pso_state_io::get_instance();
    pso_state_io_obj.set_finish_state(current_level_column, false);

    //try to read state and start async function to save state to fill
    std::call_once(start_async_flag, [&is_caller_thread, settings, &pso_state_io_obj]() {
        try {
            pso_state_io_obj.read_state_file(settings->state_file);
        } catch (const std::exception &ex) {
            LOG4_ERROR("Reading of PSO states file failed! PSO starting from default Sobol topology.");
        }
#ifdef PSO_SAVE_STATE
        async_save_file = svr::async(&pso_state_io::save_state_to_file, &pso_state_io_obj,
                                     settings->state_file);
#endif
        is_caller_thread = true;
    });


    double a, b; // for matrix initialization
    double w; // current omega
    void (*inform_fun)(int *, double *, double *, double *, double *, int, const pso_settings_t *); // neighborhood update function

    // SELECT APPROPRIATE NHOOD UPDATE FUNCTION
    switch (settings->nhood_strategy) {
        case PSO_NHOOD_GLOBAL :
            // comm matrix not used
            inform_fun = inform_global;
            break;
        case PSO_NHOOD_RING :
            init_comm_ring((int *) comm, settings);
            inform_fun = inform_ring;
            break;
        case PSO_NHOOD_RANDOM :
        default:
            init_comm_random((int *) comm, settings);
            inform_fun = inform_random;
            break;
    }

    // INITIALIZE SOLUTION
    pso_state saved_state;
    solution.error = DBL_MAX;
    bool loaded_state = false;
    /*
    bool loaded_state = pso_state_io_obj.get_state(current_level_column, saved_state);
    if (loaded_state) {
        pso_state::convert_from_vector2d(saved_state.pos_, settings->size, settings->dim, pos);
        pso_state::convert_from_vector2d(saved_state.vel_, settings->size, settings->dim, vel);
        pso_state::convert_from_vector2d(saved_state.pos_b_, settings->size, settings->dim, pos_b);
        pso_state::convert_from_vector2d(saved_state.pos_nb_, settings->size, settings->dim, pos_nb);
        std::copy(saved_state.fit_.begin(), saved_state.fit_.end(), fit);
        std::copy(saved_state.fit_b_.begin(), saved_state.fit_b_.end(), fit_b);
        particles_ready = saved_state.particles_ready_;
        solution.error = saved_state.error_;
        step = saved_state.step_;
        std::memcpy(solution.gbest, saved_state.gbest_.data(), saved_state.gbest_.size() * sizeof(double));
        LOG4_DEBUG("Successfully reloaded pso state");
        LOG4_DEBUG("Initial loaded best solution error " << solution.error << " for model " << decon_level << "_"
                                                         << column_name);
    }
     */
    // haven't loaded state
    if (!loaded_state) {
        step = 0;
        //TODO - Emanouil - when using MPI, it should be
        //static thread_local unsigned long Sobol_counter = (unsigned long) mpi_rank * 78786876896ULL ;
        //Using different counters for different threads is desirable, but not required.
        unsigned long sobol_ctr = 78786876896ULL + (long) floor(svr::common::get_uniform_random_value() * (double) 4503599627370496ULL);
        for (dtype(settings->size) i = 0; i < settings->size; ++i) {
            // for each dimension
            std::vector<double> sobol_numbers(2 * settings->dim, 0.);
            for (dtype(settings->dim) d = 0; d < settings->dim; ++d) {
                // generate two numbers within the specified range
                // Use Sobol numbers
                sobol_numbers[d] = sobolnum(d, sobol_ctr + i);
                sobol_numbers[d + settings->dim] = sobolnum(d + settings->dim, sobol_ctr + i);
                a = settings->x_lo_limit[d] + (settings->x_hi_limit[d] - settings->x_lo_limit[d]) * sobol_numbers[d];
                b = settings->x_lo_limit[d] + (settings->x_hi_limit[d] - settings->x_lo_limit[d]) * sobol_numbers[d + settings->dim];
                // initialize position
                pos[i][d] = a;
                // best position is the same
                pos_b[i][d] = a;
                // initialize velocity
                vel[i][d] = (a - b) / 2.;
            }
            LOG4_DEBUG("Positions from Sobol " << svr::common::to_string<double>(sobol_numbers));
        }
    }
    if (step == 0) {
        omp_pfor_i__(0, settings->size,
                     if (!particles_ready[i]) {
                         auto vpos = std::vector<double>{pos[i], pos[i] + settings->dim};
                         fit[i] = f(vpos);
                         memcpy(&pos[i], vpos.data(), settings->dim * sizeof(pos[i]));
                         fit_b[i] = fit[i]; // this is also the personal best
                         LOG4_DEBUG("Initial particle fitness " << fit[i]);
                         particles_ready[i] = true;
                         pso_state state_to_save(
                                 pso_state::convert2vector2d(settings->size, settings->dim, pos),
                                 pso_state::convert2vector2d(settings->size, settings->dim, vel),
                                 pso_state::convert2vector2d(settings->size, settings->dim, pos_b),
                                 pso_state::convert2vector2d(settings->size, settings->dim, pos_nb),
                                 std::vector<double>(fit, fit + settings->size),
                                 std::vector<double>(fit_b, fit + settings->size),
                                 particles_ready,
                                 std::vector<double>(solution.gbest, solution.gbest + settings->dim), step,
                                 solution.error);
                         LOG4_DEBUG(
                                 "Updating initial pso_state after fitness particle update for level and column " <<
                                                                                                                  decon_level << " " << column_name);
                         pso_state_io_obj.update_state(current_level_column, state_to_save);
                     }
        )

        for (dtype(settings->size) i = 0; i < settings->size; ++i) {
            // update gbest?
            if (fit[i] < solution.error) {
                // update best fitness
                solution.error = fit[i];
                // copy particle pos to gbest vector
                memmove((void *) solution.gbest, (void *) &pos[i], sizeof(double) * settings->dim);
            }
        }
        LOG4_DEBUG(
                "Initial best solution error " << solution.error << " for model " << decon_level << "_" << column_name);
    }

    // initialize omega using standard value
    w = PSO_INERTIA;
    // RUN ALGORITHM
    for (; step < settings->steps; ++step) {
        // update current step
        settings->step = step;
        // update inertia weight
        // do not bother with calling a calc_w_const function
        if (settings->w_strategy)
            w = calc_inertia_lin_dec(step, settings);
        // check optimization goal
        if (solution.error <= settings->goal) {
            // SOLVED!!
            if (settings->print_every)
                LOG4_INFO("PSO converged for model " << decon_level << "_" << column_name << ", step " << step
                                                     << ", error " << solution.error);
            break;
        }

        // update pos_nb matrix (find best of neighborhood for all particles)
        inform_fun((int *) comm, (double *) pos_nb, (double *) pos_b, fit_b, solution.gbest, improved, settings);
        // the value of improved was just used; reset it
        improved = 0;

        // in next step after load, reinitialize particles_ready
        if ((loaded_state && saved_state.step_ != step) || !loaded_state)
            std::fill(particles_ready.begin(), particles_ready.end(), false);

            // Update all particles
        omp_pfor_i__(0, settings->size,
                     if (!particles_ready[i]) {
                         // for each dimension
                         for (int d = 0; d < settings->dim; ++d) {
                             double *curr_pos = &pos[i][d];
                             double *curr_vel = &vel[i][d];
                             double *curr_pos_b = &pos_b[i][d];
                             double *curr_pos_nb = &pos_nb[i][d];
                             int dd = update_particle(curr_pos, curr_vel, curr_pos_b, curr_pos_nb, settings, d, w);
                             LOG4_DEBUG("PSO iteration " << step << ", particle " << i << ", dim " << dd << " finished. Model " << decon_level << "_" << column_name);
                         }
                         // Update particle fitness
                         auto vpos = std::vector<double>{pos[i], pos[i] + settings->dim};
                         fit[i] = f(vpos);
                         memcpy(&pos[i], vpos.data(), settings->dim * sizeof(pos[i]));

                         LOG4_DEBUG("Step " << step << " particle fitness " << fit[i] << " best fitness " << fit_b[i]);
                         // update personal best position?
                         if (fit[i] < fit_b[i]) {
                             fit_b[i] = fit[i];
                             // copy contents of pos[i] to pos_b[i]
                             memmove((void *) &pos_b[i], (void *) &pos[i], sizeof(double) * settings->dim);
                         }

                         particles_ready[i] = true;
                         pso_state state_to_save(
                                 pso_state::convert2vector2d(settings->size, settings->dim, pos),
                                 pso_state::convert2vector2d(settings->size, settings->dim, vel),
                                 pso_state::convert2vector2d(settings->size, settings->dim, pos_b),
                                 pso_state::convert2vector2d(settings->size, settings->dim, pos_nb),
                                 std::vector<double>(fit, fit + settings->size),
                                 std::vector<double>(fit_b, fit + settings->size),
                                 particles_ready,
                                 std::vector<double>(solution.gbest, solution.gbest + settings->dim),
                                 step, solution.error);

                         LOG4_TRACE("Updating pso_state after fitness particle update!");
                         pso_state_io_obj.update_state(current_level_column, state_to_save);
                     }
        )
        // check all particles
        for (int i = 0; i < settings->size; ++i) {
            // update gbest??
            if (fit[i] < solution.error) {
                LOG4_DEBUG("Update gbest, step " << step << " particle " << i << " fitness " << fit[i]);
                improved = 1;
                // update best fitness
                solution.error = fit[i];
                // copy particle pos to gbest vector
                memmove((void *) solution.gbest, (void *) &pos[i], sizeof(double) * settings->dim);
            }
        }

        if (settings->print_every && step % settings->print_every == 0)
            LOG4_INFO("Step " << step << " err " << solution.error << " for level " << decon_level << " column "
                              << column_name);
    }
    pso_state_io_obj.set_finish_state(current_level_column, true);
    free(solution.gbest);

    const std::vector<size_t> idx = svr::common::argsort(std::vector<double>{fit, fit + settings->size});
    std::vector<std::pair<double, std::vector<double>>> result(settings->size);
    for (int ix = 0; ix < settings->size; ++ix)
        result[ix] = std::make_pair(fit_b[idx[ix]], std::vector<double>(pos_b[idx[ix]], pos_b[idx[ix]] + settings->dim));
#ifdef PSO_SAVE_STATE
    if (is_caller_thread) async_save_file.get();
#endif
    return result;
}

#pragma GCC diagnostic pop

}
}