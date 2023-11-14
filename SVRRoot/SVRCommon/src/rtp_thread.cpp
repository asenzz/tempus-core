#include <common/rtp_thread.hpp>
#include <common/rtp_task.hpp>
#include <common/rtp_thread_pool.hpp>
#include <stack>

namespace svr{ namespace common { namespace rtp {

thread::thread(thread_pool& tp)
: tp(tp), working{true}, finished{false}, thr(nullptr)
{}

thread::~thread()
{
    working = false;
    if(thr)
        thr->join();
}

bool thread::pop_local_task(std::shared_ptr<task> & tsk) 
{
    return local_tasks.pop(tsk);
}

void thread::push_local_task(std::shared_ptr<task> const & tsk)
{
    local_tasks.push(tsk);
}

std::shared_ptr<task> thread::get_current_task()
{
    std::scoped_lock scoped(tsk_mutex);
    if(tsk_stack.empty())
        return nullptr;
    return tsk_stack.top();
}

void thread::run()
{
    while(working)
        run_one_task();
    finished = true;
}

void thread::run_one_task()
{
    std::shared_ptr<task> tsk;
    
    {
        std::scoped_lock scoped(tsk_mutex);

        if(!local_tasks.pop(tsk))
            if(!tp.get_task(tsk, this))
                return;
        
        tsk_stack.push(tsk);
    }
    
    tsk->run(this);
    
    {
        std::scoped_lock scoped(tsk_mutex);
        tsk_stack.pop();
    }
}

void thread::can_run()
{
    thr = new std::thread(&thread::run, this);
}

void thread::can_finish()
{
    working = false;
}

bool thread::has_finished() const 
{
    return finished;
}

std::thread::id thread::get_id()
{
    return thr->get_id();
}

}}}