#include <util/MemoryManager.hpp>
#include "common/gpu_handler.hpp"
#include <common/Logging.hpp>
#include <chrono>
#include <thread>
#include <stdio.h>
#include <unistd.h>
#include <ftw.h>
#include <ios>
#include <string>
#include <memory>
#include <stdexcept>
#include <array>

namespace svr {
namespace common {


std::string exec(const char *cmd)
{
    std::array<char, 128> buffer;
    std::string result;
    std::unique_ptr<FILE, dtype(&pclose)> pipe(popen(cmd, "r"), pclose);
    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    return result;
}


#include <pthread.h>
#include <thread>

memory_manager::memory_manager()
        : mem_available_(false), finished(false), worker(&memory_manager::check_memory, this)
{
/*
 * sched_param sch_params;
    sch_params.sched_priority = 75;
    if (pthread_setschedparam(worker.native_handle(), SCHED_RR, &sch_params))
        LOG4_ERROR("Failed setting worker thread priority!");
*/
}


memory_manager::~memory_manager()
{
    finished = true;
    worker.join();
}


memory_manager &memory_manager::get()
{
    static memory_manager mm;
    return mm;
}


bool memory_manager::threads_available()
{
    // const auto &level_todo = PROPS.get_svr_paramtune_level();
    // if (level_todo != "ALL") return true;

    const auto num_of_threads = read_threads();
    const auto num_of_cores = std::thread::hardware_concurrency();
    LOG4_TRACE("Number of cores " << num_of_cores << ", number of threads " << num_of_threads);
    if (num_of_cores < 1 || num_of_threads < 1)
        throw std::runtime_error("Number of cores or number of threads cannot be zero!");
    return num_of_threads <= (svr::common::gpu_handler::get().get_max_running_gpu_threads_number() +
                              num_of_cores * THREADS_CORES_MULTIPLIER);
}


void memory_manager::check_memory()
{
    static const auto sleep_interval = SLEEP_INTERVAL;
    static const size_t sleep_iterations = SLEEP_ITERATIONS;
    while (!finished) {
        memory_info_t mem_info;
        read_memory(mem_info);
        const auto ram_free = mem_info.mem_free + mem_info.buffers + mem_info.cached;
        const double ram_left = double(ram_free) / double(mem_info.mem_total);
        const auto prev_mem_available = mem_available_;
        mem_available_ = (ram_left >= FREE_RAM_THRESHOLD) && true;//threads_available();
        if (mem_available_ != prev_mem_available) {
            LOG4_DEBUG("Changed mem_available to " << mem_available_ << " RAM left, " << 100. * ram_left << " pct., free ram " << static_cast<double>(ram_free) / GB_RAM_UNIT_DIVIDER <<
                        " GB" << ", total ram " << static_cast<double>(mem_info.mem_total) / GB_RAM_UNIT_DIVIDER << " GB");
            cv_.notify_all();
            trigger = true;
        }
        for (size_t i = 0; i < sleep_iterations; ++i) {
            if (finished) return;
            if (trigger) {
                trigger = false;
                break;
            }
            std::this_thread::sleep_for(sleep_interval);
        }
    }
    mem_available_ = true;
}


void memory_manager::barrier()
{
    std::unique_lock wl(mx_);
    while (!mem_available_) cv_.wait(wl);
    mem_available_ = false;
    trigger = true;
}


void
memory_manager::read_memory(memory_info_t &mem_info)
{
    std::ifstream mem_info_file("/proc/meminfo");
    std::string key;
    long val = 0;
    std::string metric;
    while (mem_info_file >> key >> val >> metric) {
        boost::erase_all(key, ":");
        if (key == "MemTotal")
            mem_info.mem_total = val;
        else if (key == "MemFree")
            mem_info.mem_free = val;
        else if (key == "Cached")
            mem_info.cached = val;
        else if (key == "Buffers")
            mem_info.buffers = val;
        else if (key == "SwapTotal")
            mem_info.swap_total = val;
        else if (key == "SwapFree")
            mem_info.swap_free = val;
    }
}


//////////////////////////////////////////////////////////////////////////////
//
// process_mem_usage(double &, double &) - takes two doubles by reference,
// attempts to read the system-dependent data for a process' virtual memory
// size and resident set size, and return the results in KB.
//
// On failure, returns 0.0, 0.0

void memory_manager::process_mem_usage(double &vm_usage, double &resident_set)
{
    using std::ios_base;
    using std::ifstream;
    using std::string;

    vm_usage = 0.0;
    resident_set = 0.0;

    // 'file' stat seems to give the most reliable results
    //
    ifstream stat_stream("/proc/self/stat", ios_base::in);

    // dummy vars for leading entries in stat that we don't care about
    //
    string pid, comm, state, ppid, pgrp, session, tty_nr;
    string tpgid, flags, minflt, cminflt, majflt, cmajflt;
    string utime, stime, cutime, cstime, priority, nice;
    string O, itrealvalue, starttime;

    // the two fields we want
    //
    unsigned long vsize;
    long rss;

    stat_stream >> pid >> comm >> state >> ppid >> pgrp >> session >> tty_nr
                >> tpgid >> flags >> minflt >> cminflt >> majflt >> cmajflt
                >> utime >> stime >> cutime >> cstime >> priority >> nice
                >> O >> itrealvalue >> starttime >> vsize >> rss; // don't care about the rest

    stat_stream.close();

    const double page_size_mb = double(sysconf(_SC_PAGE_SIZE)) / (1024. * 1024.); // in case x86-64 is configured to use 2MB pages
    vm_usage = vsize / (1024. * 1024.);
    resident_set = rss * page_size_mb;
}


static size_t num_of_threads = 0;

static void
parse_task_stats(const char *status_file_name)
{
    std::ifstream proc_status_file(status_file_name);
    std::string key;
    bool state_found = false;
    while (proc_status_file >> key)
        if (key == "State:") state_found = true;
        else if (key == "(running)" && state_found) ++num_of_threads;
}


static int
read_contents(const char *fpath, const struct stat *sb, int tflag, struct FTW *ftwbuf)
{
    char status_file_name[64];
    snprintf(status_file_name, sizeof(status_file_name), "%s/status", fpath);
    if (tflag == FTW_D) parse_task_stats(&status_file_name[0]);
    return 0;
}


std::mutex mtx;
static bool read_threads_running = false;

size_t
memory_manager::read_threads()
{
    if (read_threads_running) {
        LOG4_ERROR("Memory manager thread hit collision!");
        return 1;// std::numeric_limits<dtype(num_of_threads)>::max();
    }
    read_threads_running = true;
    const auto proc_pid = getpid();
    num_of_threads = 0;
    int ftw_flags = FTW_PHYS;
    char proc_tasks_path[64];
    snprintf(proc_tasks_path, sizeof(proc_tasks_path), "/proc/%u/task/", proc_pid);
    if (nftw(proc_tasks_path, read_contents, 10000, ftw_flags) == -1) {
        perror("nftw");
        read_threads_running = false;
        return errno == ENOENT ? std::numeric_limits<dtype(num_of_threads)>::max() : num_of_threads;
    }
    read_threads_running = false;
    return num_of_threads;
}

}
}
