#include <common/rtp_thread_pool.hpp>
#include <common/rtp_thread.hpp>
#include <common/logging.hpp>

namespace svr{ namespace common { namespace rtp {

static thread_pool * inst {nullptr};

thread_pool& thread_pool::get_instance()
{
    if(!inst)
        inst = new thread_pool();

    return *inst;
}


void thread_pool::delete_instance()
{
    delete inst;
}


thread_pool::thread_pool()
: thread_number {C_n_cpu + HW_THREAD_OVERFLOW_PCT * C_n_cpu / 100}
{
    for(size_t i = 0; i < thread_number; ++i)    {
        threads.push_back(new thread(*this));
    }

    for(auto th : threads)
        th->can_run();

//    svr::common::monitoring_manager::instance().subscribe(
//        [this] () {
//            LOG4_DEBUG("RTP: tasks = " << tasks.size() << " threads = " << threads.size());
//            for (size_t i = 0; i < threads.size(); i++) {
//                LOG4_DEBUG("RTP: Thread: " << i << " local_tasks = " << threads[i]->local_tasks.size());
//            }
//    }, 3);
}


thread_pool::~thread_pool()
{
    for(auto th : threads)
        th->can_finish();

    size_t working = 1;
    while(working)
    {
        working = 0;
        for(auto th : threads)
            if(!th->has_finished())
                ++working;
    }

    for(auto th : threads)
        delete th;
}

bool thread_pool::get_task(std::shared_ptr<task> & tsk, thread const * calling_thread)
{
    if(tasks.pop(tsk))
        return true;

    static std::atomic<size_t> last_thread_subtasked {0};

    if(threads[last_thread_subtasked % thread_number] == calling_thread)
        ++last_thread_subtasked;

    return threads[last_thread_subtasked++ % thread_number]->pop_local_task(tsk);
}

void thread_pool::submit(std::shared_ptr<task> const & tsk)
{
    tasks.push(tsk);
}


std::shared_ptr<task> thread_pool::get_current_task()
{
    auto const id = std::this_thread::get_id();
    for(auto th : threads)
        if(th->get_id() == id)
            return th->get_current_task();
    throw std::logic_error("Cannot determine current task");
}

}}}