#include <common/rtp_task.hpp>
#include <common/rtp_thread_pool.hpp>
#include <common/Logging.hpp>

static std::atomic<size_t> tid(0);

namespace svr{ namespace common { namespace rtp {

task::task()
: task(nullptr)
{
    id = tid++;
}


task::task(task* parent)
: thr{nullptr}, done_{false}, subtasks{0}, parent_task{parent}
{
    id = tid++;
}


void task::run(thread * th)
{
    svr::common::memory_manager::get().barrier();
    done_ = false;
    thr.store(th);
    try {
        run_job();
    } catch (...) {
    }

    thr.store(nullptr);
    done_ = true;
    if (parent_task) parent_task->subtasks.fetch_add(-1);
    LOG4_DEBUG("Done task: id = " << id);
}


thread * task::get_thread() const
{
    return thr;
}


void task::submit_subtask(std::shared_ptr<task> const & tsk)
{
    subtasks.fetch_add(1);
    tsk->parent_task = this;
    if(thr)
        thr.load()->push_local_task(tsk);
    else
        thread_pool::get_instance().submit(tsk);
}


namespace 
{
    struct wait_subtasks_condition
    {
        wait_subtasks_condition(task const & tsk): tsk(tsk)
        {}
        
        bool check() const 
        {
            return tsk.subtasks_done();
        }
        
        task const & tsk;
    };
}


void task::wait_subtasks_done()
{
    wait_subtasks_condition cond(*this);
    if(thr.load())
        thr.load()->wait_condition(cond);
    else
        thread_pool::get_instance().wait_condition(cond);
}


namespace
{
    struct wait_task_done_condition
    {
        wait_task_done_condition(task const & tsk): tsk(tsk)
        {}
        
        bool check() const 
        {
//            LOG4_DEBUG("Check task: id = " << tsk.id << " done = " << tsk.done());
            return tsk.done();
        }
        
        task const & tsk;
    };
}


void task::wait_done()
{
    wait_subtasks_done();
    wait_task_done_condition cond(*this);
    if(thr.load())
        thr.load()->wait_condition(cond);
    else
        thread_pool::get_instance().wait_condition(cond);
}


task const * task::parent() const
{
    return parent_task;
}


bool task::subtasks_done() const
{
    return subtasks.load() == 0;
}

bool task::done() const
{
    return done_.load();
}

void task::submit(std::shared_ptr<task> const & ptask)
{
    if(thr.load())
        thr.load()->push_local_task(ptask);
    else
        thread_pool::get_instance().submit(ptask);
}


}}}
