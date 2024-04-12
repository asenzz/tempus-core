#ifndef RTP_THREAD_HPP
#define RTP_THREAD_HPP

#include <queue>
#include <memory>
#include <atomic>
#include <mutex>
#include <thread>
#include <stack>

namespace svr{ namespace common { namespace rtp {

template<class T>
class thread_safe_queue_cannonical
{
public:
    void push(T const &);
    bool pop(T &);

    size_t size() const {
        std::scoped_lock wl(mtx);
        return cont.size();
    }
private:
    mutable std::mutex mtx;
    std::queue <T> cont;
};

template<class T>
using thread_safe_queue=thread_safe_queue_cannonical<T>;

class task;
class thread_pool;

class thread
{
public:
    thread(thread_pool&);
    ~thread();
    bool pop_local_task(std::shared_ptr<task> &);
    void push_local_task(std::shared_ptr<task> const &);

    template<class Condition>
    void wait_condition(Condition const & cond);

    void can_run();
    void can_finish();
    bool has_finished() const ;

    std::shared_ptr<task> get_current_task();

    std::thread::id get_id();

private:
    friend class thread_pool;

    thread_safe_queue_cannonical <std::shared_ptr<task>> local_tasks;
    thread_pool & tp;
    std::atomic<bool> working, finished;
    std::thread * thr;
    std::stack<std::shared_ptr<task>> tsk_stack;
    std::mutex tsk_mutex;
    void run();
    void run_one_task();
};

/******************************************************************************/
/******************************************************************************/
/******************************************************************************/

template<class T>
void thread_safe_queue_cannonical<T>::push(T const & t)
{
    std::scoped_lock scoped(mtx);
    cont.push(t);
}

template<class T>
bool thread_safe_queue_cannonical<T>::pop(T & t)
{
    std::scoped_lock scoped(mtx);
    if(cont.empty())
        return false;

    t = cont.front();
    cont.pop();
    return true;
}


/******************************************************************************/


template<class Condition>
void thread::wait_condition(Condition const & cond)
{
    while(!cond.check())
    {
        run_one_task();
        std::this_thread::sleep_for(std::chrono::nanoseconds(100));
    }
}

}}}

#endif /* RTP_THREAD_HPP */

