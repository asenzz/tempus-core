#ifndef TASK_HPP
#define TASK_HPP

#include <memory>
#include <atomic>

#include "rtp_thread.hpp"
#include "util/MemoryManager.hpp"

namespace svr{ namespace common { namespace rtp {

class task
{
public:
    task();
    task(task* parent);

    task const * parent() const;

    void run(thread *);
    thread * get_thread() const;

    bool subtasks_done() const;
    bool done() const;

    size_t id;

private:
    mutable std::atomic<thread*> thr;
    std::atomic<bool> done_;
    mutable std::atomic<size_t> subtasks;
    task const * parent_task;

    virtual void run_job() = 0;

    void submit_subtask(std::shared_ptr<task> const &);
    void submit(std::shared_ptr<task> const &);

    void wait_done();
    void wait_subtasks_done();

    template<class Condition>
    void wait_condition(Condition const & cond);
};


/******************************************************************************/

template<class Condition>
void task::wait_condition(Condition const & cond)
{
    thr.load()->wait_condition(cond);
}

}}}

#endif /* TASK_HPP */

