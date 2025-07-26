#include <chrono>
#include <thread>
#include <cstdio>
#include <ios>
#include <string>
#include <memory>
#include <stdexcept>
#include <array>
#include <unistd.h>
#include <ftw.h>
#include "common/parallelism.hpp"
#include "common/gpu_handler.hpp"
#include "common/logging.hpp"
#include "util/MemoryManager.hpp"

namespace svr {
namespace common {

struct pclose_wrapper {
    void operator()(FILE *const f) const
    {
        pclose(f);
    }
};

std::string exec(CRPTR(char) cmd)
{
    std::array<char, BUFSIZ> buffer;
    std::string result;
    std::unique_ptr<FILE, pclose_wrapper> pipe(popen(cmd, "r"));
    if (!pipe) LOG4_THROW("popen() failed!");
    while (fgets(buffer.data(), buffer.size(), pipe.get())) result += buffer.data();
    return result;
}


#include <pthread.h>
#include <thread>

memory_manager::memory_manager() : mem_available_(false), finished(false), worker(&memory_manager::check_memory, this)
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
    LOG4_TRACE("Number of threads " << C_n_cpu);
    return num_of_threads <= (svr::common::gpu_handler_1::get().get_max_gpu_threads() + C_n_cpu * THREADS_CORES_MULTIPLIER);
}


void memory_manager::check_memory()
{
    constexpr auto sleep_interval = SLEEP_INTERVAL;
    std::ifstream mem_info_file("/proc/meminfo");
    memory_info_t mem_info;
    while (!finished) {
        read_memory(mem_info, mem_info_file);
        const auto ram_free = mem_info.mem_free + mem_info.buffers + mem_info.cached;
        const double ram_left = double(ram_free) / double(mem_info.mem_total);
        const auto prev_mem_available = mem_available_;
        mem_available_ = ram_left >= FREE_RAM_THRESHOLD; // && threads_available();
        constexpr double gb_ram_unit_div = GB_RAM_UNIT_DIVIDER;
        if (mem_available_ != prev_mem_available) {
            LOG4_TRACE(
                    "Changed mem_available to " << mem_available_ << ", RAM left, " << 100. * ram_left << "pc, free ram " << ram_free / gb_ram_unit_div <<
                                                " GB" << ", total ram " << mem_info.mem_total / gb_ram_unit_div << " GB");
            cv_.notify_one();
        }
        std::this_thread::sleep_for(sleep_interval);
    }
    mem_available_ = true;
}


void memory_manager::barrier()
{
    std::unique_lock wl(mx_);
    while (!mem_available_) cv_.wait(wl);
    mem_available_ = false;
}


void
memory_manager::read_memory(memory_info_t &mem_info, std::ifstream &mem_info_file)
{
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
/*      
        else if (key == "SwapTotal")
            mem_info.swap_total = val;
        else if (key == "SwapFree")
            mem_info.swap_free = val; 
*/
    }
    mem_info_file.clear();
    mem_info_file.seekg(0);
}

// In MB
float memory_manager::get_proc_rss()
{
    FILE *self_statm = fopen("/proc/self/statm", "r");
    if (!self_statm) {
        LOG4_ERROR("Failed opening /proc/self/statm");
        return 1;
    }
    while (fgetc(self_statm) != ' ') {}
    // fscanf(self_statm, "%ld", &s);
    constexpr uint8_t bufl = 16;
    char buf[bufl];
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-result"
#endif
    static_cast<void>( fread(buf, sizeof(*buf), bufl, self_statm) );
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
    constexpr float mbd = 1024 * 1024;
    static const float pagesize_mbd = sysconf(_SC_PAGESIZE) / mbd;
    *(char *) memchr(buf, ' ', bufl) = 0;
    return std::atol(buf) * pagesize_mbd;
}

static size_t num_of_threads = 0;

static void parse_task_stats(const char *status_file_name)
{
    std::ifstream proc_status_file(status_file_name);
    std::string key;
    bool state_found = false;
    while (proc_status_file >> key)
        if (key == "State:") state_found = true;
        else if (key == "(running)" && state_found) ++num_of_threads;
}


static int read_contents(const char *fpath, const struct stat *sb, int tflag, struct FTW *ftwbuf)
{
    char status_file_name[64];
    snprintf(status_file_name, sizeof(status_file_name), "%s/status", fpath);
    if (tflag == FTW_D) parse_task_stats(status_file_name);
    return 0;
}

static bool read_threads_running = false;

size_t
memory_manager::read_threads()
{
    if (read_threads_running) {
        LOG4_ERROR("Memory manager thread hit collision!");
        return 1;// std::numeric_limits<DTYPE(num_of_threads)>::max();
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
        return errno == ENOENT ? std::numeric_limits<DTYPE(num_of_threads) >::max() : num_of_threads;
    }
    read_threads_running = false;
    return num_of_threads;
}

}
}
