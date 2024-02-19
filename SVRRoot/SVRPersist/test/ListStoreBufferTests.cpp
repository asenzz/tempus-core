#include "../src/AsyncDAO/ListStoreBuffer.hpp"
#include "gtest/gtest.h"
#include "model/InputQueue.hpp"

#include <deque>

TEST(ListStoreBufferTests, BasicTests)
{
    svr::dao::ListStoreBuffer<int, std::equal_to<int>> dq(10, std::equal_to<int>());
    std::deque<int> que;

    for(int i = 1; i < 11; ++i)
    {
        ASSERT_TRUE(dq.push(i));
        que.push_back(i);
    }

    int i = 11;
    ASSERT_FALSE(dq.push(i)); ASSERT_TRUE(dq.pop(i)); EXPECT_TRUE(i == 1);

    i = 11;
    ASSERT_TRUE(dq.push(i));
    que.push_back(i);

    for(int i = 12; i < 1000; ++i)
    {
        int what = i;
        ASSERT_FALSE(dq.push(what));  ASSERT_TRUE(dq.pop(what)); EXPECT_TRUE(what == que[que.size() - 10]);

        ASSERT_TRUE(dq.push(i));
        que.push_back(i);
    }

    int what = 999;
    ASSERT_TRUE(dq.push(what));

    what = 998;
    ASSERT_FALSE(dq.push(what)); ASSERT_TRUE(dq.pop(what)); ASSERT_EQ(what, que[que.size() - 10]);

    what = 998;
    ASSERT_TRUE(dq.push(what));

    ASSERT_EQ(998, dq.cont.front()); ASSERT_EQ(999, *(++dq.cont.begin()));
    dq.remove(998);

    ASSERT_EQ(8UL, dq.cont.size()); ASSERT_EQ(999, dq.cont.front()); ASSERT_EQ(997, *(++dq.cont.begin()));
}

namespace svr {
template<>
void store_buffer_push_merge<std::string>(std::string &dest, const std::string &src)
{
    dest += src;
}
}

auto first_letter_equal = [](std::string const & s1, std::string const & s2)
{
    return s1.length() > 0 && s2.length() > 0 && s1[0] == s2[0];
};

TEST(ListStoreBufferTests, MergeWithLastValue)
{
    svr::dao::ListStoreBuffer<std::string, decltype(first_letter_equal)> dq(10, first_letter_equal);

    dq.push("a---");
    dq.push("a===");
    std::string res;
    dq.pop(res);

    ASSERT_EQ(res, "a---a===");
}