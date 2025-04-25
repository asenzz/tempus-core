//
// Created by zarko on 7/11/24.
//

#include "common/barrier.hpp"
#include "common/logging.hpp"

namespace svr::common {

barrier::barrier(const unsigned num) : num_threads(num), wait_count(0), instance(0), mut(), cv()
{
    if (num == 0) THROW_EX_FS(std::invalid_argument, "Barrier thread count cannot be 0");
}

void barrier::wait() noexcept
{
    std::unique_lock<std::mutex> lock(mut); // acquire lock
    const auto inst = instance; // store current instance for comparison
    // in predicate

    if (++wait_count == num_threads) { // all threads reached barrier
        wait_count = 0; // reset wait_count
        ++instance; // increment instance for next use of barrier and to
        // pass condition variable predicate
        cv.notify_all();
    } else { // not all threads have reached barrier
        cv.wait(lock, [this, inst]() { return instance != inst; });
        // NOTE: The predicate lambda here protects against spurious
        //       wakeups of the thread. As long as this->instance is
        //       equal to inst, the thread will not wake.
        //       this->instance will only increment when all threads
        //       have reached the barrier and are ready to be unblocked.
    }
}

omp_task_barrier::omp_task_barrier(const unsigned num_threads) : num_threads(num_threads)
{
    if (num_threads == 0) THROW_EX_FS(std::invalid_argument, "Barrier thread count cannot be 0");
}

void omp_task_barrier::wait() noexcept
{
    tbb::mutex::scoped_lock lk(wait_l);
    const auto current_instance = instance;
    ++wait_count;
    lk.release();

    __do_wait:
    if (instance <= current_instance && wait_count < num_threads) { // all threads haven't reached barrier
        task_yield_wait__;
        goto __do_wait;
    }
    // all threads reached barrier
    lk.acquire(wait_l);
    wait_count = 0; // reset wait_count
    instance = current_instance + 1; // increment instance for next use of barrier
}

void omp_task_barrier::reset() noexcept
{
    const tbb::mutex::scoped_lock lk(wait_l);
    wait_count = 0; // reset wait_count
}

} // namespace svr::common

