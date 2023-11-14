#include <functional>

#include <common/thread_pool.hpp>
#include <common/Logging.hpp>

#include <list>
//#include <cilk/cilk.h>

#include "CpuThread.hpp"

namespace svr {
namespace common {

/******************************************************************************/

struct cpu_thread_pool::Impl
{
    size_t const cpu_number{std::thread::hardware_concurrency()};
    std::mutex mtx;

    std::vector<threading::cpu_thread_ptr> cpu_threads, spare_threads, finished_threads;

    threading::task_queue_ptr queue;

    Impl()
            : cpu_threads(cpu_number, nullptr), queue(threading::task_queue_ptr(new threading::task_queue))
    {
        //bool set_thread_affinity = PROPS.get_set_thread_affinity();

        for (size_t i = 0; i < cpu_number; ++i) {
            cpu_threads[i] = threading::cpu_thread::build(queue);

// Commented out as it was shown that OS does not allocate threads on a cpu used like this
//            cpu_set_t cpuset;
//            CPU_ZERO(&cpuset);
//            CPU_SET(i, &cpuset);
//
//            auto s = pthread_setaffinity_np(cpu_threads[i]->native_handle(), sizeof(cpuset), &cpuset);
//            if (s != 0)
//               LOG4_ERROR("Cannot set thread affinity");
        }
    }

    ~Impl()
    {
        for (auto &th : cpu_threads)
            th->finish();

        for (auto &th : spare_threads)
            th->finish();

        queue->finish();

        std::scoped_lock scoped_lock(mtx);

        std::cout <<
                  "Finished with "
                  << cpu_threads.size() << " cpu threads, "
                  << spare_threads.size() << " spare threads and "
                  << finished_threads.size() << " finished threads" << std::endl;
    }

    future_base_ptr execute(task_base_ptr const &task)
    {
        future_base_ptr result = future_base_ptr(new threading::future_impl(task));
        queue->push_task(threading::task_future(task, result));
        return result;
    }

    void current_thread_blocked(bool blocked)
    {
        set_thread_state(blocked);
        if (blocked)
            queue->thread_stopped();
        else
            queue->thread_running();
    }

    void set_thread_state(bool blocked)
    {
        std::scoped_lock scoped_lock(mtx);

        auto const ctid = std::this_thread::get_id();
        auto const stat = blocked ? svr::common::threading::thread_state::blocked
                                  : svr::common::threading::thread_state::running;

        for (auto thread : cpu_threads)
            if (thread->get_id() == ctid) {
                thread->set_state(stat);
                fix_thread_number(blocked);
                return;
            }


        for (auto thread : spare_threads)
            if (thread->get_id() == ctid) {
                thread->set_state(stat);
                fix_thread_number(blocked);
                return;
            }
    }

    void fix_thread_number(bool increase)
    {
        size_t active_thread_count = 0;

        for (auto thread : cpu_threads)
            if (thread->get_state() == svr::common::threading::thread_state::running)
                ++active_thread_count;

        for (auto thread : spare_threads)
            if (thread->get_state() == svr::common::threading::thread_state::running)
                ++active_thread_count;

        if (increase) {
            for (size_t thn = active_thread_count; thn < cpu_number; ++thn)
                spare_threads.push_back(threading::cpu_thread::build(queue));
        } else {
            int surplus_threads = active_thread_count - cpu_number;
            if (surplus_threads > 0 && spare_threads.size() >= size_t(surplus_threads)) {
                std::copy(spare_threads.end() - surplus_threads, spare_threads.end(),
                          std::back_inserter(finished_threads));
                spare_threads.erase(spare_threads.end() - surplus_threads, spare_threads.end());

                for (auto iter = finished_threads.end() - surplus_threads; iter != finished_threads.end(); ++iter)
                    (*iter)->finish();

                finished_threads.erase(
                        std::remove_if(
                                finished_threads.begin(), finished_threads.end(), [](threading::cpu_thread_ptr thread)
                                { return thread->finished(); }
                        ), finished_threads.end()
                );
            }
        }
    }

};


/******************************************************************************/

cpu_thread_pool &cpu_thread_pool::get_instance()
{
    static cpu_thread_pool inst;
    return inst;
}

cpu_thread_pool::cpu_thread_pool()
        : pImpl(*new Impl())
{
}


cpu_thread_pool::~cpu_thread_pool()
{
    delete &pImpl;
}


future_base_ptr cpu_thread_pool::execute(task_base_ptr const &task)
{
    return pImpl.execute(task);
}

void cpu_thread_pool::current_thread_blocked(bool blocked)
{
    pImpl.current_thread_blocked(blocked);
}

size_t cpu_thread_pool::hardware_concurrency() const
{
    return pImpl.cpu_number;
}


/******************************************************************************/


current_thread_wstate_guard::current_thread_wstate_guard()
{
    cpu_thread_pool::get_instance().current_thread_blocked(true);
}


current_thread_wstate_guard::~current_thread_wstate_guard()
{
    cpu_thread_pool::get_instance().current_thread_blocked(false);
}


}
}
