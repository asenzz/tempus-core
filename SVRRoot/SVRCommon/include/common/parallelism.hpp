#pragma once

#include <thread>
#include <omp.h>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/parallel_reduce.h>
#pragma GCC diagnostic pop
#include "compatibility.hpp"

#ifdef PRODUCTION_BUILD
#if defined(__INTEL_LLVM_COMPILER)
#define UNROLL_N(x) PRAGMASTR(unroll x)
#define UNROLL_N0 _Pragma("unroll")
#elif defined(__NVCC__)
#define UNROLL_N(x) PRAGMASTR(unroll x)
#define UNROLL_N0 _Pragma("unroll")
#elif defined(__GNUC__)
#define UNROLL_N(x) PRAGMASTR(GCC unroll x)
#define UNROLL_N0
#endif
#else
#define UNROLL_N(x)
#define UNROLL_N0
#endif

#define UNROLL(...) PP_MACRO_OVERLOAD(UNROLL, __VA_ARGS__)
#define UNROLL_0() UNROLL_N0
#define UNROLL_1(X) UNROLL_N(X)

// Smart SIMD keyword since GNU C gets a bit picky about the SIMD pragma
#ifdef __GNUC__
#define SSIMD
#else
#define SSIMD simd
#endif

#define OMP_TASKLOOP(__n) PRAGMASTR(omp taskloop SSIMD NGRAIN(__n) default(shared) mergeable)
#define OMP_TASKLOOP_1(__x) PRAGMASTR(omp taskloop grainsize(1) default(shared) mergeable __x)
#define OMP_TASKLOOP_(__n, __x) PRAGMASTR(omp taskloop NGRAIN(__n) default(shared) mergeable __x)
#define OMP_FOR_ITERS_(_n) CDIV((_n), C_n_cpu)
#define ADJ_ITERS_CTR TOKENPASTE2(__adj_iters_, __LINE__)
#define OMP_FOR_(__n, __x)                               \
    const unsigned ADJ_ITERS_CTR = OMP_FOR_ITERS_(__n);  \
    PRAGMASTR(omp parallel for __x schedule(static, ADJ_ITERS_CTR) num_threads((unsigned) CDIV(__n, ADJ_ITERS_CTR)) default(shared))
#define OMP_FOR_i_(__n, __x) \
    OMP_FOR_(__n, __x)      \
    for (DTYPE(__n) i = 0; i < __n; ++i)
#if defined(__GNUC__)
#define OMP_FOR_i(__n) OMP_FOR_i_(__n, )
#define OMP_FOR(__n) OMP_FOR_(__n, )
#else
#define OMP_FOR_i(__n) OMP_FOR_i_(__n, simd)
#define OMP_FOR(__n) OMP_FOR_(__n, simd)
#endif

#define NGRAIN(N) grainsize((unsigned) cdiv((N), C_n_cpu))

constexpr auto C_yield_usleep = std::chrono::milliseconds(10);
#define thread_yield_wait__ { std::this_thread::yield(); std::this_thread::sleep_for(C_yield_usleep); }
#define task_yield_wait__ { _Pragma("omp taskyield") std::this_thread::sleep_for(C_yield_usleep); }


#define non_stpfor__(TY, IX, FROM, TO, STEP, ...) \
    for(TY IX = FROM; IX < TO; STEP) { __VA_ARGS__; }
#define non_tpfor__(TY, IX, FROM, TO, ...) non_stpfor__(TY, IX, FROM, TO, ++IX, __VA_ARGS__)
#define non_pfor__(IX, FROM, TO, ...) non_stpfor__(DTYPE(TO), IX, FROM, TO, ++IX, __VA_ARGS__)
#define non_pfor_i__(FROM, TO, ...) non_pfor__(i, FROM, TO, __VA_ARGS__)
#define non_tpfor_i__(TY, FROM, TO, ...) non_stpfor__(TY, i, FROM, TO,  ++i, __VA_ARGS__)
#define non_spfor__(IX, FROM, TO, STEP, ...) non_stpfor__(DTYPE(TO), IX, FROM, TO, STEP, __VA_ARGS__)
#define non_spfor_i__(FROM, TO, STEP, ...) non_spfor__(i, FROM, TO, (STEP), __VA_ARGS__)
#define non_stpfor_i__(TY, FROM, TO, STEP, ...) non_stpfor__(TY, i, FROM, TO, (STEP), __VA_ARGS__)


#ifdef NO_PARALLEL
#define pxt_stpfor__(TYPE, IX, FROM, TO, STEP, ...) non_stpfor__(TYPE, IX, FROM, TO, STEP, __VA_ARGS__)
#else
#define pxt_stpfor__(TYPE, IX, FROM, TO, STEP, ...) { \
                std::deque<std::thread> __thr_q_##IX; \
                for (TYPE IX = FROM; IX < TO; STEP) \
                    __thr_q_##IX.emplace_back([&] (const TYPE IX) { \
                        __VA_ARGS__; }, IX); \
                for (auto &__t_##IX: __thr_q_##IX) __t_##IX.join(); \
            };
#endif
#define pxt_spfor__(IX, FROM, TO, STEP, ...) pxt_stpfor__(DTYPE(TO), IX, FROM, TO, STEP, __VA_ARGS__)
#define pxt_tpfor__(TYPE, IX, FROM, TO, ...) pxt_stpfor__(TYPE, IX, FROM, TO, ++IX, __VA_ARGS__)
#define pxt_pfor__(IX, FROM, TO, ...) pxt_tpfor__(DTYPE(TO), IX, FROM, TO, __VA_ARGS__)
#define pxt_pfor_i__(FROM, TO, ...) pxt_pfor__(i, FROM, TO, __VA_ARGS__)
#define pxt_pfor_i0__(TO, ...) pxt_pfor_i__(0, TO, __VA_ARGS__)

#ifdef NO_PARALLEL
#define omp_stpfor__(TY, IX, FROM, TO, STEP, ...)  non_stpfor__(TYPE, IX, FROM, TO, STEP, __VA_ARGS__)
#else
#define omp_stpfor__(TY, IX, FROM, TO, STEP, ...)  _Pragma("omp parallel for default(shared)") \
    for(TY IX = FROM; IX < TO; STEP) { __VA_ARGS__; } // CilkPLUS is replaced with OpenMP // cilk_for(TY IX = FROM; IX < TO; STEP) { __VA_ARGS__; };
#endif
#define omp_tpfor__(TY, IX, FROM, TO, ...) omp_stpfor__(TY, IX, FROM, TO, ++IX, __VA_ARGS__)
#define omp_pfor__(IX, FROM, TO, ...) omp_stpfor__(DTYPE(TO), IX, FROM, TO, ++IX, __VA_ARGS__)
#define omp_pfor_i__(FROM, TO, ...) omp_pfor__(i, FROM, TO, __VA_ARGS__)
#define omp_tpfor_i__(TY, FROM, TO, ...) omp_stpfor__(TY, i, FROM, TO,  ++i, __VA_ARGS__)
#define omp_spfor__(IX, FROM, TO, STEP, ...) omp_stpfor__(DTYPE(TO), IX, FROM, TO, STEP, __VA_ARGS__)
#define omp_spfor_i__(FROM, TO, STEP, ...) omp_spfor__(i, FROM, TO, (STEP), __VA_ARGS__)
#define omp_stpfor_i__(TY, FROM, TO, STEP, ...) omp_stpfor__(TY, i, FROM, TO, (STEP), __VA_ARGS__)

#ifdef NO_PARALLEL
#define tbb_stpfor__(TY, IX, FROM, TO, STEP, ...)  non_stpfor__(TYPE, IX, FROM, TO, STEP, __VA_ARGS__)
#else
#define tbb_stpfor__(TY, IX, FROM, TO, STEP, ...) \
    tbb::task_arena __limited_arena(C_n_cpu); \
    __limited_arena.execute([&]{ \
    tbb::parallel_for(tbb::blocked_range<TY>(FROM, TO, STEP), [&](const tbb::blocked_range<TY> &__brange_##IX) \
        { for (TY IX = __brange_##IX.begin(); IX < __brange_##IX.end(); IX += STEP) \
            { __VA_ARGS__; }} );  }); // Custom stepping is impossible with random starts
#endif
#define tbb_tpfor__(TY, IX, FROM, TO, ...) tbb_stpfor__(TY, IX, FROM, TO, 1, __VA_ARGS__)
#define tbb_pfor__(IX, FROM, TO, ...) tbb_tpfor__(DTYPE(TO), IX, FROM, TO, __VA_ARGS__)
#define tbb_pfor_i__(FROM, TO, ...) tbb_pfor__(i, FROM, TO, __VA_ARGS__)
#define tbb_zpfor_i__(TO, ...) tbb_pfor_i__(DTYPE(TO)(0), TO, __VA_ARGS__)
#define tbb_tpfor_i__(TY, FROM, TO, ...) tbb_stpfor__(TY, i, FROM, TO, 1, __VA_ARGS__)
#define tbb_spfor__(IX, FROM, TO, STEP, ...) tbb_stpfor__(DTYPE(TO), IX, FROM, TO, STEP, __VA_ARGS__)
#define tbb_spfor_i__(FROM, TO, STEP, ...) tbb_stpfor__(DTYPE(TO), i, FROM, TO, STEP, __VA_ARGS__)
#define tbb_stpfor_i__(TY, FROM, TO, STEP, ...) tbb_stpfor__(TY, i, FROM, TO, STEP, __VA_ARGS__)


// TBB Parallel plus reductor
#define tbb_stppred__(RESULT, INIT_RESULT, TYPE, IX, FROM, TO, STEP, ...) \
    RESULT = tbb::parallel_reduce(tbb::blocked_range<TYPE>(FROM, TO, STEP), INIT_RESULT, [&](const tbb::blocked_range<TYPE> &__bprange_##IX, DTYPE(INIT_RESULT) RESULT) { \
        for (auto IX = __bprange_##IX.begin(); IX < __bprange_##IX.end(); IX += STEP) { __VA_ARGS__; } return RESULT; }, std::plus<DTYPE(INIT_RESULT)>() );

#define tbb_sppred__(RESULT, INIT_RESULT, IX, FROM, TO, STEP, ...) tbb_stppred__(RESULT, INIT_RESULT, DTYPE(TO), IX, FROM, TO, STEP, __VA_ARGS__)
#define tbb_ppred__(RESULT, INIT_RESULT, IX, FROM, TO, ...) tbb_sppred__(RESULT, INIT_RESULT, IX, FROM, TO, 1, __VA_ARGS__)
#define tbb_ppred_i__(RESULT, INIT_RESULT, FROM, TO, ...) tbb_ppred__(RESULT, INIT_RESULT, i, FROM, TO, __VA_ARGS__)
#define tbb_rpred_i__(RESULT, FROM, TO, ...) tbb_ppred__(RESULT, RESULT, i, FROM, TO, __VA_ARGS__)


// TODO Rewrite with TBB parallel_for_each
#define GRAIN_SIZE 10

#define __F(...) [&](const auto _IX) { __VA_ARGS__; }
#define __par_iter(_C_START_ITER, _C_COUNT, _C_ACC_FUNCT) _C_CILK_ITER(_C_START_ITER, _C_COUNT, __F(_C_ACC_FUNCT))
#define _C_CILK_ITER(_C_START_ITER, _C_COUNT, _C_ACC_FUNCT) \
{\
    const size_t __max_cores = _C_COUNT / GRAIN_SIZE;\
    if (__max_cores < 2) {\
        auto _ITER = _C_START_ITER;\
        for (size_t t = 0; t < _C_COUNT; ++t, ++_ITER) {\
            _C_ACC_FUNCT(t);\
        } \
    } else { \
        const size_t __num_cores = C_n_cpu; \
        const size_t __cores_used = __max_cores > __num_cores ? __num_cores : __max_cores; \
        const size_t __chunk_len = _C_COUNT / __cores_used; \
        tbb_tpfor__(size_t, t, 0, __cores_used, \
            const size_t __pos_start = t * __chunk_len; \
            const size_t cur___chunk_len = t == __cores_used - 1 ? _C_COUNT - __pos_start : __chunk_len; \
            auto _ITER = _C_START_ITER + __pos_start; \
            for (size_t sub_t = 0; sub_t < cur___chunk_len; ++sub_t, ++_ITER) \
                _C_ACC_FUNCT(sub_t + __pos_start); \
        ) \
    } \
}

namespace svr {

template<typename T> inline unsigned adj_threads(const std::is_signed<T> iterations)
{
    return (unsigned) std::min<T>(iterations < 0 ? 0 : iterations, C_n_cpu);
}

template<typename T> inline unsigned adj_threads(const T iterations)
{
    return std::min<T>(iterations, C_n_cpu);
}

#define ADJ_THREADS(T) num_threads(adj_threads(T))
#define ADJ_THREADS_MIN(T1, T2) num_threads(adj_threads(std::min<DTYPE(T1)>(T1, T2)))

class t_omp_lock {
    omp_lock_t l;
public:
    t_omp_lock();
    operator omp_lock_t *();
    void set();
    void unset();
    ~t_omp_lock();
};


class thread_calc {
    std::deque<unsigned> level_threads;
public:
    void add_level_desired_threads(const unsigned t) { level_threads.emplace_back(adj_threads(t)); };
    void add_level_desired_threads_min(const unsigned t, const unsigned u) { add_level_desired_threads(std::min(t, u)); };
    unsigned get_remaining_threads() {
        unsigned threads = C_n_cpu;
        for (const auto t: level_threads) threads /= t;
        return threads;
    }
    unsigned operator [] (const unsigned l) { return l < level_threads.size() ? level_threads[l] : get_remaining_threads(); }
    unsigned sched(const unsigned l, const unsigned n) { return std::ceil(float(n) / float(operator[](l))); }
};

}