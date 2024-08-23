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
    barrier(const std::size_t num);

    // disable copying of barrier
    barrier(const barrier &) = delete;

    barrier &operator=(const barrier &) = delete;

    // This function blocks the calling thread until
    // all threads (specified by num_threads) have
    // called it. Blocking is achieved using a
    // call to condition_variable.wait().
    void wait() noexcept;

private:
    std::size_t num_threads; // number of threads using barrier
    std::size_t wait_count; // counter to keep track of waiting threads
    std::size_t instance; // counter to keep track of barrier use count
    std::mutex mut; // mutex used to protect resources
    std::condition_variable cv; // condition variable used to block threads
};

}

#endif //SVR_BARRIER_HPP
