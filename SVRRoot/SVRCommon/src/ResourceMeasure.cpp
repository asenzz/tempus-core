#include "util/ResourceMeasure.hpp"
#include <thread>

double ResourceMeasure::time_duration(timeval &begin_time, timeval &end_time)
{
    return (double)end_time.tv_sec + (double)end_time.tv_usec / 1000000.0
            - ((double)begin_time.tv_sec + (double)begin_time.tv_usec / 1000000.0);
}

std::string ResourceMeasure::time_duration_to_human(double time_duration)
{
    int hours = (int)time_duration / 3600;
    int minutes = ((int)time_duration % 3600) / 60;
    double seconds = time_duration - 3600.0 * hours - 60.0 * minutes;
    std::stringstream duration;
    duration << hours << ":" << minutes << ":" << seconds;
    return duration.str();
}

void ResourceMeasure::set_start_time()
{
    gettimeofday(&start_time, NULL);
}

long ResourceMeasure::get_memory_usage()
{
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == -1 ) {
        LOG4_ERROR("Can't get rusage structure");
        return -1;
    }
    return usage.ru_maxrss / 1024; // return Mb
}

double ResourceMeasure::get_cpu_usage_percent()
{
    double cpu_usage = get_cpu_usage();
    if (cpu_usage == -1) {
        return -1;
    }

    return cpu_usage / get_time_duration();
}

double ResourceMeasure::get_cpu_usage()
{
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == -1 ) {
        LOG4_ERROR("Can't get rusage structure");
        return -1;
    }
    return ((double)usage.ru_utime.tv_sec + (double)usage.ru_utime.tv_usec / 1000000.0
            + (double)usage.ru_stime.tv_sec + (double)usage.ru_stime.tv_usec / 1000000.0);
}

double ResourceMeasure::get_time_duration()
{
    struct timeval time_now;
    gettimeofday(&time_now, NULL);
    return time_duration(start_time, time_now);
}

std::string ResourceMeasure::get_time_duration_human()
{
    double time_duration = get_time_duration();
    return time_duration_to_human(time_duration);
}

void ResourceMeasure::print_measure_info()
{
    LOG4_INFO("CPU usage: " << get_cpu_usage_percent() << " %");
    LOG4_INFO("RAM usage: " << get_memory_usage() << " Mb");
    LOG4_INFO("Time taken: " << get_time_duration_human());
}

size_t ResourceMeasure::get_cpu_count()
{
    return std::thread::hardware_concurrency();
}