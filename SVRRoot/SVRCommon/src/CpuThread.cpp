#include "common/Logging.hpp"
#include "common/constants.hpp"
#include "CpuThread.hpp"

namespace svr {
namespace common {
namespace threading {


task_future::task_future()
:task(), fut()
{}


task_future::task_future(task_base_ptr const & task, future_base_ptr const &fut)
:task(task), fut(fut)
{}


/******************************************************************************/


cpu_thread_throttle::cpu_thread_throttle()
: cpu_thread_throttle(std::thread::hardware_concurrency())
{}

cpu_thread_throttle::cpu_thread_throttle(const size_t hardware_concurrency)
: hardware_concurrency(hardware_concurrency)
, running_threads{0}
{
    if (!hardware_concurrency) {
        LOG4_WARN("Hardware concurrency report zero, setting to default of " << DEFAULT_HARDWARE_CONCURRENCY);
        this->hardware_concurrency = DEFAULT_HARDWARE_CONCURRENCY;
    }
}

cpu_thread_throttle::~cpu_thread_throttle()
{
    finish();
}

void cpu_thread_throttle::thread_running()
{
    std::unique_lock<std::mutex> lock(mtx);
    wait_cv.wait(lock, [this] { return running_threads < hardware_concurrency; });
    ++running_threads;
}


void cpu_thread_throttle::thread_stopped()
{
    std::unique_lock<std::mutex> lock(mtx);
    --running_threads;
    wait_cv.notify_all();
}

void cpu_thread_throttle::finish()
{
    //for started in waiting state threads to complete;
    std::unique_lock<std::mutex> lock(mtx);
    hardware_concurrency = std::numeric_limits<size_t>::max();
    wait_cv.notify_all();
}


/******************************************************************************/


void task_queue::push_task(task_future && tf)
{
    std::unique_lock<std::mutex> scoped(mtx);
    tasks.emplace_back(std::move(tf));
    wait_cv.notify_one();
}

bool task_queue::pop_task_wait_for(task_future & result, std::chrono::milliseconds const & ms)
{
    std::unique_lock<std::mutex> scoped(mtx);
    bool ret = tasks.empty();
    if(!ret)
    {
        result = tasks.front();
        tasks.pop_front();
        return true;
    }

    if(wait_cv.wait_for(scoped, ms) == std::cv_status::timeout)
        return false;

    ret = tasks.empty();
    if(!ret)
    {
        result = tasks.front();
        tasks.pop_front();
        return true;
    }
    return false;
}


void task_queue::thread_running()
{
    throttle.thread_running();
}


void task_queue::thread_stopped()
{
    throttle.thread_stopped();
}


void task_queue::finish()
{
    throttle.finish();
}


/******************************************************************************/


cpu_thread_ptr cpu_thread::build(task_queue_ptr const & queue)
{
    cpu_thread_ptr result = cpu_thread_ptr( new cpu_thread(queue) );
    return result;
}


cpu_thread::cpu_thread(task_queue_ptr const & queue)
: state(thread_state::running)
, finish_{false}
, finished_{false}
, queue(queue)
, thread(&cpu_thread::work, this)
{
}


cpu_thread::~cpu_thread()
{
    finish_ = true;
    thread.join();
}


void cpu_thread::work()
{
    queue->thread_running();
    while(!finish_)
    {
        static auto const wt = CPU_THREAD_WAIT_CYCLE;
        task_future tf;
        bool got_task = queue->pop_task_wait_for(tf, wt);
        if (!got_task) continue;
        svr::common::memory_manager::instance().wait();
        try{
            tf.task->execute();
        }
        catch(...)
        {
            static_cast<future_impl*>(tf.fut.get())->set_exception(std::current_exception());
        }
        static_cast<future_impl*>(tf.fut.get())->notify();
    }
    finished_ = true;
    queue->thread_stopped();
}


thread_state cpu_thread::get_state() const
{
    return state.load();
}


void cpu_thread::set_state(thread_state status)
{
    state = status;
}


std::thread::native_handle_type cpu_thread::native_handle()
{
    return thread.native_handle();
}

std::thread::id cpu_thread::get_id()
{
    return thread.get_id();
}


void cpu_thread::finish()
{
    finish_ = true;
}


bool cpu_thread::finished() const
{
    return finished_;
}


/******************************************************************************/
/******************************************************************************/
/******************************************************************************/
/******************************************************************************/
/******************************************************************************/


future_impl::future_impl(task_base_ptr tsk)
:locked{true}, ptask{tsk}, exception(nullptr)
{}


void future_impl::wait()
{
    std::unique_lock<std::mutex> lck(wait_mutex);
    wait_cv.wait(lck, [this] { return !locked; });
    if (exception) std::rethrow_exception(exception);
}


void future_impl::notify()
{
    std::unique_lock<std::mutex> lck(wait_mutex);
    locked = false;
    wait_cv.notify_all();
}


task_base_ptr future_impl::get_task() const
{
    return ptask;
}

void future_impl::set_exception(std::exception_ptr const & ptr)
{
    exception = ptr;
}


}}}
