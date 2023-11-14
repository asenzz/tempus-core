//
// Created by zarko on 1/27/23.
//

#ifndef SVR_SEMAPHORE_HPP
#define SVR_SEMAPHORE_HPP
// MIT License

// Copyright (c) 2021 CyanHill

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <limits>
#include <mutex>
#include <atomic>


namespace svr {


class semaphore
{
public:
    explicit semaphore(const ssize_t count) noexcept
            : m_count(count) { assert(count > -1); }

    void post() noexcept
    {
        {
            std::scoped_lock lock(m_mutex);
            ++m_count;
        }
        m_cv.notify_one();
    }

    void wait() noexcept
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_cv.wait(lock, [&]() { return m_count != 0; });
        --m_count;
    }

private:
    ssize_t m_count;
    std::mutex m_mutex;
    std::condition_variable m_cv;
};

class fast_semaphore
{
public:
    explicit fast_semaphore(const ssize_t count) noexcept
            : m_count(count), m_semaphore(0) {}

    void post()
    {
        std::atomic_thread_fence(std::memory_order_release);
        ssize_t count = m_count.fetch_add(1, std::memory_order_relaxed);
        if (count < 0) m_semaphore.post();
    }

    void wait()
    {
        ssize_t count = m_count.fetch_sub(1, std::memory_order_relaxed);
        if (count < 1) m_semaphore.wait();
        std::atomic_thread_fence(std::memory_order_acquire);
    }

private:
    std::atomic<ssize_t> m_count;
    semaphore m_semaphore;
};


}  // namespace svr

#endif //SVR_SEMAPHORE_HPP
