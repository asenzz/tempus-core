//
// Created by zarko on 5/2/24.
//
#include "common/semaphore.hpp"
#include "common/logging.hpp"

namespace svr {

semaphore::semaphore(const ssize_t count) noexcept
        : m_count(count)
{
    LOG4_TRACE("Count " << m_count);
    // assert(count > -1);
}

void semaphore::post() noexcept 
{
    {
        const std::scoped_lock l(m_mutex);
        ++m_count;
    }
    m_cv.notify_one();
}

ssize_t semaphore::wait() noexcept
{
    std::unique_lock lock(m_mutex);
    m_cv.wait(lock, [&]() { return m_count != 0; });
    return --m_count;
}

bool semaphore::try_wait() noexcept
{
    return false; // TODO Not implemented
}

fast_semaphore::fast_semaphore(const ssize_t count) noexcept : m_count(count), m_semaphore(0)
{
}

void fast_semaphore::post()
{
#if SANITIZE != SANITIZE_THREAD
    std::atomic_thread_fence(std::memory_order_release);
#endif
    if (m_count.fetch_add(1, std::memory_order_relaxed) < 0) m_semaphore.post();
}

ssize_t fast_semaphore::wait()
{
    if (m_count.fetch_sub(1, std::memory_order_relaxed) < 1) m_semaphore.wait();
#if SANITIZE != SANITIZE_THREAD
    std::atomic_thread_fence(std::memory_order_acquire);
#endif
    return m_count.load(std::memory_order_seq_cst);
}

bool fast_semaphore::try_wait()
{
    // TODO Not implemented
    return m_semaphore.try_wait();
}

size_t fast_semaphore::get_count()
{
    return m_count.load(std::memory_order_seq_cst);
}

}
