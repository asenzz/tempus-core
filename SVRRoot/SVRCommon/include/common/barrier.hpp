//
// Created by zarko on 1/17/23.
//

#ifndef SVR_BARRIER_HPP
#define SVR_BARRIER_HPP


#include <condition_variable>
#include <mutex>
#include <stdexcept>
#include <thread>


namespace svr::common {


class barrier
{
public:
    // Construct barrier for use with num threads.
    barrier(std::size_t num)
            : num_threads(num),
              wait_count(0),
              instance(0),
              mut(),
              cv()
    {
        if (num == 0) {
            throw std::invalid_argument("Barrier thread count cannot be 0");
        }
    }

    // disable copying of barrier
    barrier(const barrier &) = delete;

    barrier &operator=(const barrier &) = delete;

    // This function blocks the calling thread until
    // all threads (specified by num_threads) have
    // called it. Blocking is achieved using a
    // call to condition_variable.wait().
    void wait()
    {
        std::unique_lock<std::mutex> lock(mut); // acquire lock
        std::size_t inst = instance; // store current instance for comparison
        // in predicate

        if (++wait_count == num_threads) { // all threads reached barrier
            wait_count = 0; // reset wait_count
            instance++; // increment instance for next use of barrier and to
            // pass condition variable predicate
            cv.notify_all();
        } else { // not all threads have reached barrier
            cv.wait(lock, [this, &inst]() { return instance != inst; });
            // NOTE: The predicate lambda here protects against spurious
            //       wakeups of the thread. As long as this->instance is
            //       equal to inst, the thread will not wake.
            //       this->instance will only increment when all threads
            //       have reached the barrier and are ready to be unblocked.
        }
    }

private:
    std::size_t num_threads; // number of threads using barrier
    std::size_t wait_count; // counter to keep track of waiting threads
    std::size_t instance; // counter to keep track of barrier use count
    std::mutex mut; // mutex used to protect resources
    std::condition_variable cv; // condition variable used to block threads
};

}

#endif //SVR_BARRIER_HPP
