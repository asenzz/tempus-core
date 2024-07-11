#pragma once

#ifdef __INTEL_LLVM_COMPILER

#include </opt/intel/oneapi/compiler/latest/opt/compiler/include/omp.h>

#else
#include <omp.h>
#endif

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"

#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

#pragma GCC diagnostic pop

#include "compatibility.hpp"

#define _OMP_ADJFOR_ITERS(_n) std::max<unsigned>(1, _CEILDIV((_n), std::thread::hardware_concurrency()))
#define ADJ_ITERS_CTR TOKENPASTE2(__adj_iters_, __LINE__)
#define OMP_FOR_(__n, __x) \
    const auto ADJ_ITERS_CTR = _OMP_ADJFOR_ITERS(__n);                    \
    PRAGMASTR(omp parallel for __x schedule(static, ADJ_ITERS_CTR) num_threads(std::max<unsigned>(1, _CEILDIV(__n, ADJ_ITERS_CTR))))
#define OMP_FOR(__n) \
    const auto ADJ_ITERS_CTR = _OMP_ADJFOR_ITERS(__n);                    \
    PRAGMASTR(omp parallel for simd schedule(static, ADJ_ITERS_CTR) num_threads(std::max<unsigned>(1, _CEILDIV(__n, ADJ_ITERS_CTR))))


#define __non_stpfor(TY, IX, FROM, TO, STEP, ...) \
    for(TY IX = FROM; IX < TO; STEP) { __VA_ARGS__; }
#define __non_tpfor(TY, IX, FROM, TO, ...) __non_stpfor(TY, IX, FROM, TO, ++IX, __VA_ARGS__)
#define __non_pfor(IX, FROM, TO, ...) __non_stpfor(dtype(TO), IX, FROM, TO, ++IX, __VA_ARGS__)
#define __non_pfor_i(FROM, TO, ...) __non_pfor(i, FROM, TO, __VA_ARGS__)
#define __non_tpfor_i(TY, FROM, TO, ...) __non_stpfor(TY, i, FROM, TO,  ++i, __VA_ARGS__)
#define __non_spfor(IX, FROM, TO, STEP, ...) __non_stpfor(dtype(TO), IX, FROM, TO, STEP, __VA_ARGS__)
#define __non_spfor_i(FROM, TO, STEP, ...) __non_spfor(i, FROM, TO, (STEP), __VA_ARGS__)
#define __non_stpfor_i(TY, FROM, TO, STEP, ...) __non_stpfor(TY, i, FROM, TO, (STEP), __VA_ARGS__)


#ifdef NO_PARALLEL
#define __pxt_stpfor(TYPE, IX, FROM, TO, STEP, ...) __non_stpfor(TYPE, IX, FROM, TO, STEP, __VA_ARGS__)
#else
#define __pxt_stpfor(TYPE, IX, FROM, TO, STEP, ...) { \
                std::deque<std::thread> __thr_q_##IX; \
                for (TYPE IX = FROM; IX < TO; STEP) \
                    __thr_q_##IX.emplace_back([&] (const TYPE IX) { \
                        __VA_ARGS__; }, IX); \
                for (auto &__t_##IX: __thr_q_##IX) __t_##IX.join(); \
            };
#endif
#define __pxt_spfor(IX, FROM, TO, STEP, ...) __pxt_stpfor(dtype(TO), IX, FROM, TO, STEP, __VA_ARGS__)
#define __pxt_tpfor(TYPE, IX, FROM, TO, ...) __pxt_stpfor(TYPE, IX, FROM, TO, ++IX, __VA_ARGS__)
#define __pxt_pfor(IX, FROM, TO, ...) __pxt_tpfor(dtype(TO), IX, FROM, TO, __VA_ARGS__)
#define __pxt_pfor_i(FROM, TO, ...) __pxt_pfor(i, FROM, TO, __VA_ARGS__)
#define __pxt_pfor_i0(TO, ...) __pxt_pfor_i(0, TO, __VA_ARGS__)

#ifdef NO_PARALLEL
#define __omp_stpfor(TY, IX, FROM, TO, STEP, ...)  __non_stpfor(TYPE, IX, FROM, TO, STEP, __VA_ARGS__)
#else
#define __omp_stpfor(TY, IX, FROM, TO, STEP, ...)  _Pragma("omp parallel for default(shared)") \
    for(TY IX = FROM; IX < TO; STEP) { __VA_ARGS__; } // CilkPLUS is replaced with OpenMP // cilk_for(TY IX = FROM; IX < TO; STEP) { __VA_ARGS__; };
#endif
#define __omp_tpfor(TY, IX, FROM, TO, ...) __omp_stpfor(TY, IX, FROM, TO, ++IX, __VA_ARGS__)
#define __omp_pfor(IX, FROM, TO, ...) __omp_stpfor(dtype(TO), IX, FROM, TO, ++IX, __VA_ARGS__)
#define __omp_pfor_i(FROM, TO, ...) __omp_pfor(i, FROM, TO, __VA_ARGS__)
#define __omp_tpfor_i(TY, FROM, TO, ...) __omp_stpfor(TY, i, FROM, TO,  ++i, __VA_ARGS__)
#define __omp_spfor(IX, FROM, TO, STEP, ...) __omp_stpfor(dtype(TO), IX, FROM, TO, STEP, __VA_ARGS__)
#define __omp_spfor_i(FROM, TO, STEP, ...) __omp_spfor(i, FROM, TO, (STEP), __VA_ARGS__)
#define __omp_stpfor_i(TY, FROM, TO, STEP, ...) __omp_stpfor(TY, i, FROM, TO, (STEP), __VA_ARGS__)

#ifdef NO_PARALLEL
#define __tbb_stpfor(TY, IX, FROM, TO, STEP, ...)  __non_stpfor(TYPE, IX, FROM, TO, STEP, __VA_ARGS__)
#else
#define __tbb_stpfor(TY, IX, FROM, TO, STEP, ...) \
    tbb::parallel_for(tbb::blocked_range<TY>(FROM, TO, STEP), [&](const tbb::blocked_range<TY> &__brange_##IX) \
        { for (TY IX = __brange_##IX.begin(); IX < __brange_##IX.end(); IX += STEP) \
            { __VA_ARGS__; }} ); // Custom stepping is impossible with random starts
#endif
#define __tbb_tpfor(TY, IX, FROM, TO, ...) __tbb_stpfor(TY, IX, FROM, TO, 1, __VA_ARGS__)
#define __tbb_pfor(IX, FROM, TO, ...) __tbb_tpfor(dtype(TO), IX, FROM, TO, __VA_ARGS__)
#define __tbb_pfor_i(FROM, TO, ...) __tbb_pfor(i, FROM, TO, __VA_ARGS__)
#define __tbb_tpfor_i(TY, FROM, TO, ...) __tbb_stpfor(TY, i, FROM, TO, 1, __VA_ARGS__)
#define __tbb_spfor(IX, FROM, TO, STEP, ...) __tbb_stpfor(dtype(TO), IX, FROM, TO, STEP, __VA_ARGS__)
#define __tbb_spfor_i(FROM, TO, STEP, ...) __tbb_stpfor(dtype(TO), i, FROM, TO, STEP, __VA_ARGS__)
#define __tbb_stpfor_i(TY, FROM, TO, STEP, ...) __tbb_stpfor(TY, i, FROM, TO, STEP, __VA_ARGS__)


// TBB Parallel plus reductor
#define __tbb_stppred(RESULT, INIT_RESULT, TYPE, IX, FROM, TO, STEP, ...) \
    RESULT = tbb::parallel_reduce(tbb::blocked_range<TYPE>(FROM, TO, STEP), INIT_RESULT, [&](const tbb::blocked_range<TYPE> &__bprange_##IX, dtype(INIT_RESULT) RESULT) { \
        for (auto IX = __bprange_##IX.begin(); IX < __bprange_##IX.end(); IX += STEP) { __VA_ARGS__; } return RESULT; }, std::plus<dtype(INIT_RESULT)>() );

#define __tbb_sppred(RESULT, INIT_RESULT, IX, FROM, TO, STEP, ...) __tbb_stppred(RESULT, INIT_RESULT, dtype(TO), IX, FROM, TO, STEP, __VA_ARGS__)
#define __tbb_ppred(RESULT, INIT_RESULT, IX, FROM, TO, ...) __tbb_sppred(RESULT, INIT_RESULT, IX, FROM, TO, 1, __VA_ARGS__)
#define __tbb_ppred_i(RESULT, INIT_RESULT, FROM, TO, ...) __tbb_ppred(RESULT, INIT_RESULT, i, FROM, TO, __VA_ARGS__)
#define __tbb_rpred_i(RESULT, FROM, TO, ...) __tbb_ppred(RESULT, RESULT, i, FROM, TO, __VA_ARGS__)


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
        const size_t __num_cores = std::thread::hardware_concurrency(); \
        const size_t __cores_used = __max_cores > __num_cores ? __num_cores : __max_cores; \
        const size_t __chunk_len = _C_COUNT / __cores_used; \
        __tbb_tpfor(size_t, t, 0, __cores_used, \
            const size_t __pos_start = t * __chunk_len; \
            const size_t cur___chunk_len = t == __cores_used - 1 ? _C_COUNT - __pos_start : __chunk_len; \
            auto _ITER = _C_START_ITER + __pos_start; \
            for (size_t sub_t = 0; sub_t < cur___chunk_len; ++sub_t, ++_ITER) \
                _C_ACC_FUNCT(sub_t + __pos_start); \
        ) \
    } \
}

namespace svr {

class t_omp_lock {
    omp_lock_t l;
public:
    t_omp_lock();
    operator omp_lock_t *();
    void set();
    void unset();
    ~t_omp_lock();
};

}