#pragma once

#include "model/User.hpp"
#include "model/InputQueue.hpp"
#include "model/Dataset.hpp"

namespace svr{
namespace daemon{

class DaemonFacade {

private:
    void initialize(const std::string &app_properties_path);
    void uninitialize();
    bool continue_loop();
    static void do_fork();

public:
    DaemonFacade(const std::string &app_properties_path);
    void start_loop();

private:
    bool daemonize{};
    long loop_interval{};
    long loop_count;
    long max_loop_count{};


    static std::string S_MAX_LOOP_COUNT;
    static std::string S_DECONSTRUCTING_FRAME_SIZE;
    static std::string S_FILL_MISSING_QUEUE_VALUES;
    static std::string S_LOOP_INTERVAL;
    static std::string S_DAEMONIZE;
};


class diagnostic_interface_zwei
{
public:
    diagnostic_interface_zwei();
    void wait();
private:
    struct diagnostic_interface_impl;
    std::shared_ptr<diagnostic_interface_impl> pimpl;

};

}
}

