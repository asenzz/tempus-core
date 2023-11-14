#include <cassert>
#include <benchmark/benchmark.h>

#include <onlinesvr.hpp>

#include "deprecated/r_matrix.h"

static void BM_update_r_matrix_cpu(benchmark::State& state)
{
    const size_t w = state.range(0);

    viennacl::matrix<double> qm(w, w, viennacl::context(viennacl::MAIN_MEMORY));

    for (size_t i = 0; i < w; i++)
        for (size_t j = 0; j < w; j++)
            qm(i, j) = i * w + j;

    while (state.KeepRunning())
    {
        std::vector<int64_t> sv_indexes;
        svr::batch::r_matrix_cpu rm(qm);

        for (size_t i = 0; i < (w >> 4); i++)
        {
            sv_indexes.push_back(i);
            rm.update_r_matrix(sv_indexes, i);
        }

        assert(rm.getR() && "There should be an R matrix.");
    }
}

BENCHMARK(BM_update_r_matrix_cpu)->Range(1<<5, 1<<14);


static void BM_update_r_matrix_gpu(benchmark::State& state)
{
    const size_t w = state.range(0);

    viennacl::matrix<double> qm(w, w, viennacl::context(viennacl::MAIN_MEMORY));

    for (size_t i = 0; i < w; i++)
        for (size_t j = 0; j < w; j++)
            qm(i, j) = i * w + j;

    while (state.KeepRunning())
    {
        std::vector<int64_t> sv_indexes;
        svr::batch::r_matrix rm(qm);

        for (size_t i = 0; i < (w >> 4); i++)
        {
            sv_indexes.push_back(i);
            rm.update_r_matrix(sv_indexes, i);
        }

        assert(rm.getR() && "There should be an R matrix.");
    }
}

BENCHMARK(BM_update_r_matrix_gpu)->Range(1<<5, 1<<14);

