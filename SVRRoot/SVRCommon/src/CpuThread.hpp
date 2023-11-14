#ifndef CPUTHREAD_HPP
#define CPUTHREAD_HPP

#include <thread>
#include <mutex>
#include <list>
#include <condition_variable>
#include <atomic>

#include <common/thread_pool.hpp>
#include <util/MemoryManager.hpp>

namespace svr {
namespace common {
namespace threading {

#define CPU_THREAD_WAIT_CYCLE (std::chrono::milliseconds(10))

enum class thread_state{running = 0, blocked};


class cpu_thread;


class future_impl : public future_base
{
    std::mutex wait_mutex;
    std::condition_variable wait_cv;
    std::atomic<bool> locked;
    task_base_ptr ptask;
    std::exception_ptr exception;
public:
    future_impl(task_base_ptr tsk);
    void wait();
    void notify();

    task_base_ptr get_task() const;
    void set_exception(std::exception_ptr const &);
};


struct task_future
{
    task_base_ptr task;
    future_base_ptr fut;

    task_future();
    task_future(task_base_ptr const & task, future_base_ptr const & future);
};


using cpu_thread_ptr = std::shared_ptr<cpu_thread>;


class cpu_thread_throttle
{
public:
    cpu_thread_throttle();
    cpu_thread_throttle(const size_t hardware_concurrency);
    ~cpu_thread_throttle();
    void thread_running();
    void thread_stopped();
    void finish();
private:
    size_t hardware_concurrency;
    size_t running_threads;

    std::mutex mtx;
    std::condition_variable wait_cv;
};


class task_queue
{
public:
    void push_task(task_future && tf);
    bool pop_task_wait_for(task_future & result, std::chrono::milliseconds const & ms);
    void thread_running();
    void thread_stopped();
    void finish();
private:
    std::list<task_future> tasks;
    std::mutex mtx;
    std::condition_variable wait_cv;
    cpu_thread_throttle throttle;
};

typedef std::shared_ptr<task_queue> task_queue_ptr;


class cpu_thread
{
public:
    static cpu_thread_ptr build(task_queue_ptr const & queue);
    ~cpu_thread();

    thread_state get_state() const;
    void set_state(thread_state);

    std::thread::native_handle_type native_handle();
    std::thread::id get_id();

    void finish();
    bool finished() const;

private:
    std::atomic<thread_state> state;
    std::atomic<bool> finish_, finished_;

    task_queue_ptr queue;

    std::thread thread;

    void work();

    cpu_thread(task_queue_ptr const & queue);
};


}}}

#endif /* CPUTHREAD_HPP */
