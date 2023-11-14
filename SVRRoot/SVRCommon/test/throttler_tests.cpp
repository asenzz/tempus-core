#include <gtest/gtest.h>
#include <string>
#include <vector>

#include "../src/CpuThread.hpp"

#include "TestSuite.hpp"

namespace 
{
    size_t Test_number = 100;
}

void thread1(svr::common::threading::cpu_thread_throttle & throttler, bool & some_thread_running)
{
    throttler.thread_running();
    
    ASSERT_FALSE(some_thread_running);
    
    some_thread_running = true;
    
    for(size_t i = 0; i < Test_number; ++i)
    {
        ASSERT_TRUE(some_thread_running);
        std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
    
    some_thread_running = false;
    
    throttler.thread_stopped();    
}


TEST(ThrottlerTests, BasicWorkflow)
{
    svr::common::threading::cpu_thread_throttle throttler(1);
    
    bool some_thread_running = false;
    
    auto future = std::async(thread1, std::ref(throttler), std::ref(some_thread_running));
    
    thread1(throttler, some_thread_running);

    future.get();
}
