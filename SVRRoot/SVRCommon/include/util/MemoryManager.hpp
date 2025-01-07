#ifndef MEMORYMANAGER_HPP
#define    MEMORYMANAGER_HPP

#include <thread>
#include <mutex>
#include <condition_variable>
#include <map>
#include <atomic>

namespace svr {
namespace common {

#define FREE_RAM_THRESHOLD .2
#define GB_RAM_UNIT_DIVIDER (1024*1024) // As current meminfo report is in KB
#define THREADS_CORES_MULTIPLIER 1
#define SLEEP_INTERVAL std::chrono::milliseconds(200)

enum class memory_manager_state {
    WORK, SHUTDOWN
};

class memory_manager {
    memory_manager();

    virtual ~memory_manager();

    memory_manager(const memory_manager &) = delete;

    memory_manager &operator=(const memory_manager &) = delete;

    void check_memory();
    bool mem_available_;
    mutable std::mutex mx_;
    mutable std::condition_variable cv_;
    std::atomic<bool> finished{false};
    std::atomic<bool> trigger{false};
    std::thread worker;

public:
    struct memory_info_t {
        long mem_free = 0, buffers = 0, cached = 0, mem_total = 0, swap_free = 0, swap_total = 0;
    };

    void barrier();

    static memory_manager &get();

    static void read_memory(memory_info_t &mem_info, std::ifstream &mem_info_file);

    static size_t read_threads();

    static bool threads_available();

    static float get_proc_rss();
};


} // namespace common
} // namespace svr

#endif    /* MEMORYMANAGER_HPP */