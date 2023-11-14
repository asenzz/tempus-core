//#include <cmath>
//#include <chrono>
//#include <thread>
//#include <future>
//#include <vector>
//
//#include <gtest/gtest.h>
//#include <future>
//#include "TestSuite.hpp"
//
//#include <common/rtp_thread_pool.hpp>
//
//namespace
//{
//    size_t const test_size = 1e+2;
//    size_t const task_size = 1e+6;
//    size_t const task_num = 10;
//    size_t const task_max_size = 1e+6;
//
//    std::atomic<size_t> action_counter;
//    std::atomic<size_t> subtask_counter;
//}
//
//
//struct long_task : svr::common::rtp::task
//{
//    void run_job()
//    {
//        size_t const task_sz = task_size / task_num;
//        for(size_t i = 0; i < task_num; ++i)
//        {
//            size_t const task_start = i*task_sz;
//            std::vector<size_t> values; values.reserve(task_sz);
//            for(size_t j = task_start; j < task_start + task_sz; ++j)
//            {
//                auto const value = std::sqrt(j);
//                values[j-task_start] =  (value - std::round(value))*1e+3;
//                action_counter.fetch_add(1);
//            }
//        }
//    }
//};
//
//
//struct long_task_sub : svr::common::rtp::task
//{
//    size_t const task_start, task_sz;
//
//    long_task_sub(size_t task_start, size_t tsk_size)
//    : task_start(task_start), task_sz(tsk_size)
//    {
//        subtask_counter.fetch_add(1);
//    }
//
//    void run_job()
//    {
//        if(task_sz > task_max_size)
//        {
//            for(size_t i = 0; i < task_sz / task_max_size; ++i)
//            {
//                auto lt = std::shared_ptr<svr::common::rtp::task>(new long_task_sub(task_start + i*task_max_size, task_sz / task_max_size));
//                submit_subtask(lt);
//            }
//            wait_subtasks_done();
//        }
//        else
//        {
//            std::vector<size_t> values; values.reserve(task_sz);
//            for(size_t i = task_start; i < task_start + task_sz; ++i)
//            {
//                auto const value = std::sqrt(i);
//                values[i-task_start] =  (value - std::round(value))*1e+3;
//                action_counter.fetch_add(1);
//            }
//        }
//    }
//};
//
//
//struct long_task_rtp : svr::common::rtp::task
//{
//    void run_job()
//    {
//        size_t const task_sz = task_size / task_num;
//
//        for(size_t i = 0; i < task_num; ++i)
//        {
//            auto lt = std::shared_ptr<svr::common::rtp::task>(new long_task_sub(task_sz*i, task_sz));
//            submit_subtask(lt);
//        }
//        wait_subtasks_done();
//    }
//};
//
//
//TEST(NewTpTests, RtpBasicLongTaskTest)
//{
//    svr::common::rtp::thread_pool::get_instance();
//
//    action_counter.store(0);
//    subtask_counter.store(0);
//    std::vector<std::shared_ptr<svr::common::rtp::task>> tasks;
//
//    for(size_t i = 0; i < test_size; ++i)
//    {
//        tasks.push_back(std::shared_ptr<svr::common::rtp::task>(new long_task_rtp()));
//        tasks.back()->submit(tasks.back());
//    }
//
//    for(auto tsk : tasks)
//        tsk->wait_done();
//
//    svr::common::rtp::thread_pool::delete_instance();
//
//    ASSERT_EQ(action_counter.load(), test_size * task_size);
//    std::cout << "Subtask number: " << subtask_counter.load() << std::endl;
//}
//
//
//TEST(AsyncTests, AsyBasicLongTaskTest)
//{
//    std::vector<std::future<void>> futures; futures.reserve(test_size);
//
//    action_counter.store(0);
//
//    for(size_t i = 0; i < test_size; ++i)
//    {
//        long_task * lt = new long_task();
//        futures.push_back(std::async(std::launch::async, &long_task::run, lt, nullptr));
//    }
//
//    for(auto & fut : futures)
//        fut.get();
//
//    ASSERT_EQ(action_counter.load(), test_size * task_size);
//}
//
