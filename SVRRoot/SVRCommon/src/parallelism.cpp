//
// Created by zarko on 7/7/24.
//
#include "common/parallelism.hpp"
#include "appcontext.hpp"
#include "common/logging.hpp"

namespace svr {

t_omp_lock::t_omp_lock()
{
    omp_init_lock(&l);
}

t_omp_lock::operator omp_lock_t *()
{
    return &l;
}

void t_omp_lock::set()
{
    omp_set_lock(&l);
}

void t_omp_lock::unset()
{
    omp_unset_lock(&l);
}

t_omp_lock::~t_omp_lock()
{
    omp_destroy_lock(&l);
}

std::pair<uint32_t, uint32_t> get_mpi_bounds(const uint32_t n_threads)
{
    const auto rank = PROPS.get_mpi_rank();
    const auto world_size = PROPS.get_mpi_size();
    assert(n_threads);
    assert(world_size > 0);
    const auto start_thread = rank * n_threads / world_size;
    const auto end_thread = rank == world_size - 1 ? n_threads : (rank + 1) * n_threads / world_size;
    LOG4_DEBUG("Bounds for " << n_threads << " threads, starting " << start_thread << ", end " << end_thread << ", rank " << rank << ", world size " << world_size);
    return {start_thread, end_thread};
}

}
