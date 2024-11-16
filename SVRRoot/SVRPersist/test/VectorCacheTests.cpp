#include <string>
#include <gtest/gtest.h>
#include "common/compatibility.hpp"
#include "TestSuite.hpp"
#include "../src/AsyncDAO/VectorMruCache.hpp"

struct User
{
    long user_name;
    std::string email;
    User(): user_name(-1), email("") {}
    User(const long user_name, const std::string &email): user_name(user_name), email(email) {}
    static bool shallow(User const & one, User const & another) { return one.user_name == another.user_name; }
};
const auto deep = [] (User const & one, User const & another)  { return one.user_name == another.user_name && one.email == another.email; };

TEST(VectorCacheTests, BasicTests)
{
    size_t const sz = 10;
    svr::dao::VectorMruCache<User, DTYPE(User::shallow), DTYPE(deep)>
        mru(sz, User::shallow, deep );

    ASSERT_EQ(sz, mru.cont.size());

    User tmp{0, "0@"};

    ASSERT_FALSE(mru.cache(tmp));
    ASSERT_EQ(sz, mru.cont.size());

    ASSERT_TRUE(mru.cache(tmp));
    ASSERT_EQ(sz, mru.cont.size());

    tmp.email="1";
    ASSERT_FALSE(mru.cache(tmp));
    ASSERT_EQ(sz, mru.cont.size());

    for(int i = 0; i < 10; ++i)
    {
        User tmp{i, std::to_string(i) + "@"};
        ASSERT_FALSE(mru.cache(tmp));
        ASSERT_EQ(sz, mru.cont.size());
    }

    for(int i = 9, u=0; i >= 0; --i)
    {
        User tmp{i, std::to_string(i) + "@"};
        ASSERT_TRUE(deep(mru.cont[u++], tmp));
    }

    tmp = User{9, "9@"};
    mru.remove(tmp);
    ASSERT_TRUE(deep(mru.cont[0], User{}));

}

TEST(VectorCacheTests, RandomTests)
{
    size_t const sz = 10;
    svr::dao::VectorMruCache<User, DTYPE(User::shallow), DTYPE(deep)>
        mru(sz, User::shallow, deep);

    for(int i = 0; i < 10; ++i)
    {
        User tmp{i, std::to_string(i) + "@"};
        ASSERT_FALSE(mru.cache(tmp));
        ASSERT_EQ(sz, mru.cont.size());
    }

    for(int i = 0; i < 1e+5; ++i)
    {
        int u = i % (sz / 2);
        User tmp{u, std::to_string(i) + "@"};
        ASSERT_EQ(sz, mru.cont.size());
    }

    for(size_t i = 5, u = 4; i < sz; ++i, --u)
    {
        User tmp{long(u), std::to_string(u) + "@"};
        ASSERT_TRUE(deep(mru.cont[i], tmp));
    }
}

#include <random>
#include <chrono>

TEST(VectorCacheTests, PerformanceTests)
{
    if ( !TestSuite::doPerformanceTests() )
        return;

    const size_t iterations = 1e+5;
    for(auto cacheSize : std::vector<size_t>{10, 50, 100, 500, 1000})
    {
        svr::dao::VectorMruCache<User, DTYPE(User::shallow), DTYPE(deep)>
            mru(cacheSize, User::shallow,deep);

        std::default_random_engine generator;
        std::uniform_int_distribution<int> distribution(1, 1000);

        using namespace std::chrono;

        high_resolution_clock::time_point start = high_resolution_clock::now();

        for(size_t i = 0; i < iterations; ++i)
        {
            int u = distribution(generator);
            User tmp{u, std::to_string(u) + "@"};
            mru.cache(tmp);
        }
        high_resolution_clock::time_point finish = high_resolution_clock::now();
        duration<double> time_span = duration_cast<duration<double>>(start - finish);
        std::cout << "TEST(VectorCacheTests, PerformanceTests):" << " \tcache size: " << cacheSize << " \titerations: "
                << iterations << " took " << time_span.count() << "sec." << std::endl;

    }
}
