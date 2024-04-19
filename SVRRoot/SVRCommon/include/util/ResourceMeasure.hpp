#pragma once

#include <sys/resource.h>
#include <sys/time.h>
#include "common/logging.hpp"

class ResourceMeasure {
    struct timeval start_time = {0, 0};

private:
    double time_duration(struct timeval &begin_time, struct timeval &end_time);

public:
    std::string time_duration_to_human(double time_duration);
    void set_start_time();
    long get_memory_usage();
    double get_cpu_usage_percent();
    double get_cpu_usage();
    double get_time_duration();
    std::string get_time_duration_human();
    void print_measure_info();
    static size_t get_cpu_count();
};
