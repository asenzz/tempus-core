#pragma once

#include <unordered_set>
#include "model/InputQueue.hpp"
#include "model/Dataset.hpp"
#include "streaming_messages_protocol.hpp"

namespace svr{
namespace daemon{

class DaemonFacade {
    const std::chrono::milliseconds loop_interval, stream_loop_interval;
    long loop_count;
    const long max_loop_count;
    const bool self_request;
    tbb::mutex update_datasets_mx;
    bool running = false;
    std::unordered_set<datamodel::InputQueue_ptr> modified_queues;
    tbb::mutex save_queues_mx;
    business::DatasetService::UserDatasetPairs datasets;
    uint8_t ctr = 0;
    std::thread shm_io_thread, save_queues_thread;

    bool continue_loop();
    static void do_fork();
    void shm_io_callback();
    void process_streams();
    void save_queues_callback();

public:
    DaemonFacade();
    ~DaemonFacade();
    void start_loop();
};

#ifndef NDEBUG

class diagnostic_interface_zwei
{
    struct diagnostic_interface_impl;
    std::shared_ptr<diagnostic_interface_impl> pimpl;

public:
    diagnostic_interface_zwei();
    void wait();
};

#endif

}
}

