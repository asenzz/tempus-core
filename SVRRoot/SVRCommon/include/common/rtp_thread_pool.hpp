#ifndef THREAD_POOL_HPP
#define THREAD_POOL_HPP

#define HW_THREAD_OVERFLOW_PCT      25

#include <memory>
#include <mutex>
#include <vector>

#include "rtp_task.hpp"

namespace svr{ namespace common { namespace rtp {

class thread;

class thread_pool
{
friend class task;
public:
    static thread_pool& get_instance();
    static void delete_instance();

    bool get_task(std::shared_ptr<task> &, thread const * calling_thread);

    std::shared_ptr<task> get_current_task();

private:
    thread_safe_queue <std::shared_ptr<task>> tasks;
    size_t const thread_number;
    std::vector<thread*> threads;

    void submit(std::shared_ptr<task> const &);

    thread_pool();
    ~thread_pool();

    /* This method is intended for external threads only*/
    template<class Condition>
    void wait_condition(Condition & cond);
};

/******************************************************************************/

template<class Condition>
void thread_pool::wait_condition(Condition & cond)
{
    thread this_thr(*this);

    while(!cond.check())
    {
        this_thr.run_one_task();
        std::this_thread::sleep_for(std::chrono::nanoseconds(100));
    }
}


}}}

#endif /* THREAD_POOL_HPP */

