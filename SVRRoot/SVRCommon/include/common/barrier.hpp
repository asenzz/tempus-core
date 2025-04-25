//
// Created by zarko on 1/17/23.
//

#ifndef SVR_BARRIER_HPP
#define SVR_BARRIER_HPP


#include <condition_variable>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <tbb/mutex.h>
#include "parallelism.hpp"


namespace svr::common {


class barrier
{
public:
    // Construct barrier for use with num threads.
    barrier(const unsigned num);

    // disable copying of barrier
    barrier(const barrier &) = delete;

    barrier &operator=(const barrier &) = delete;

    // This function blocks the calling thread until
    // all threads (specified by num_threads) have
    // called it. Blocking is achieved using a
    // call to condition_variable.wait().
    void wait() noexcept;

private:
    unsigned num_threads; // number of threads using barrier
    unsigned wait_count; // counter to keep track of waiting threads
    unsigned instance; // counter to keep track of barrier use count
    std::mutex mut; // mutex used to protect resources
    std::condition_variable cv; // condition variable used to block threads
};

class omp_task_barrier
{
    tbb::mutex wait_l; // lock to protect wait_count
    const unsigned num_threads; // number of threads using barrier
    unsigned wait_count = 0; // counter to keep track of waiting threads
    unsigned instance = 0; // counter to keep track of barrier use count
public:
    // Construct barrier for use with num threads.
    omp_task_barrier(const unsigned num_threads);

    // disable copying of barrier
    omp_task_barrier(const omp_task_barrier &) = delete;

    omp_task_barrier &operator=(const omp_task_barrier &) = delete;

    // This function blocks the calling thread until
    // all threads (specified by num_threads) have
    // called it. Blocking is achieved using a
    // call to condition_variable.wait().
    void wait() noexcept;

    // Call this function to reset the barrier.
    void reset() noexcept;
};

}

#endif //SVR_BARRIER_HPP
