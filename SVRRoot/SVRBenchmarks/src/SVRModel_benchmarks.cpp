#include <armadillo>
#include <benchmark/benchmark.h>

static void BM_produce_scalar_cpu(benchmark::State& state)
{
    arma::vec v1(1, state.range(0)), v2(2, state.range(0));

    while (state.KeepRunning())
        benchmark::DoNotOptimize(
           v1 = arma::prod(v1, v2);
        );

}

BENCHMARK(BM_produce_scalar_cpu)->Range(10, 1e+6);

static void BM_produce_scalar_gpu(benchmark::State& state)
{
    arma::vec v1(1, state.range(0)), v2(2, state.range(0));

    while (state.KeepRunning())
        benchmark::DoNotOptimize(
             v1 = arma::prod(v1, v2);
        );

}

BENCHMARK(BM_produce_scalar_gpu)->Range(10, 1e+6);

BENCHMARK_MAIN()
