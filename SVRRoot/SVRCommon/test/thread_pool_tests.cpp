#include <gtest/gtest.h>
#include <string>
#include <vector>
#include "common/thread_pool.hpp"
#include "common/Logging.hpp"
#include "TestSuite.hpp"

TEST(ThreadPoolTests, BasicWorkflow)
{
    using svr::future;
    std::vector<future<std::string>> vec;
    vec.push_back(svr::async([]() -> std::string { return "202020"; }            )    );
    for (auto &fut: vec) ASSERT_EQ(fut.get(), std::string("202020"));
}


namespace {

void void_func()
{
}

void void_func_string(std::string const &str)
{
    if (str != "test1") LOG4_THROW("Received value does not match");
}

std::string string_func_string(std::string const &str, size_t sz)
{
    if (str != "test2" || sz != 123459)
        throw std::runtime_error("Received value does not match");
    return str + "Done";
}

struct ts_a
{
    static ts_a const *some_func1(ts_a const &param)
    {
        return &param;
    }


    ts_a const *some_func2(ts_a const &param)
    {
        return &param;
    }

    void some_func3(int &i)
    {
        ++i;
    }
};

}


TEST(ThreadPoolTests, CompillabilityTests)
{
    svr::async(void_func).get();

    svr::async(void_func_string, std::string("test1")).get();

    auto fut1 = svr::async(string_func_string, std::string("test2"), 123459);
    ASSERT_TRUE(fut1.get() == "test2Done");


    ts_a tsa;

    auto fut2 = svr::async(ts_a::some_func1, std::ref(tsa));
    ASSERT_EQ(fut2.get(), &tsa);


    auto fut3 = svr::async(&ts_a::some_func2, &tsa, std::ref(tsa));
    ASSERT_TRUE(fut3.get() == &tsa);


    int iii = 0;
    auto fut4 = svr::async(&ts_a::some_func3, &tsa, std::ref(iii));
    fut4.get();
    ASSERT_EQ(iii, 1);
}

TEST(ThreadPoolTests, FutureTests)
{
    {
        int x{0};

        auto fut = svr::async([&x](int value) { x = value; }, 1);

        fut.get();
        ASSERT_EQ(x, 1);

        svr::future<int> fut3 = svr::async([](int value) -> int { return value; }, 3);
        fut3.get();
    }
}

namespace {
template<class Exception, class Param>
void func_exception(Param p)
{
    throw Exception(p);
}
}


TEST(ThreadPoolTests, ExceptionPassing)
{
    {
        auto fut = svr::async(func_exception<std::logic_error, char const *>, "Logic error exception");
        try {
            fut.get();
            FAIL() << "Expected an exception";
        }
        catch (std::logic_error const &le) {
            ASSERT_EQ(std::string(le.what()), "Logic error exception");
        }
        catch (...) {
            FAIL() << "Expected another exception";
        }
    }
    {
        auto fut = svr::async(func_exception<int, int>, 123456);
        try {
            fut.get();
            FAIL() << "Expected an exception";
        }
        catch (int const &le) {
            ASSERT_EQ(le, 123456);
        }
        catch (...) {
            FAIL() << "Expected another exception";
        }
    }
}


TEST(ThreadPoolTests, PerformanceTests)
{
    if (!TestSuite::doPerformanceTests())
        return;


    for (size_t i = 0; i < 1000000; ++i) {
        svr::future<int> fut = svr::async([](int value) -> int { return value; }, i);
        fut.get();
    }
}
