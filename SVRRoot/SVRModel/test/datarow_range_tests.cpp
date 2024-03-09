#include <gtest/gtest.h>
#include <model/Entity.hpp>
#include <model/DataRow.hpp>

using namespace svr::datamodel;

TEST(datarow_range_tests, time_decomposition)
{
}

TEST(datarow_range_tests, datarow_range_basic_tests)
{
    DataRow::container cont;
    auto const now = bpt::second_clock::local_time();

    {
        datarow_range dr0(cont);
/*
        EXPECT_TRUE(dr0.begin() == cont.begin());
        EXPECT_TRUE(dr0.end() == cont.end());

        EXPECT_TRUE(dr0.rbegin() == cont.rbegin());
        EXPECT_TRUE(dr0.rend() == cont.rend());

        EXPECT_TRUE(dr0.begin() == dr0.end());
        EXPECT_TRUE(dr0.rbegin() == dr0.rend());
        */
    }

/*----------------------------------------------------------------------------*/

    {
        cont.emplace_back(svr::ptr<DataRow>(now, bpt::second_clock::local_time(), svr::common::C_default_value_tick_volume, 1));

        datarow_range dr(cont);
/*
        EXPECT_TRUE(dr.begin() == cont.begin());
        EXPECT_TRUE(dr.end() == cont.end());

        EXPECT_TRUE(dr.rbegin() == cont.rbegin());
        EXPECT_TRUE(dr.rend() == cont.rend());
        */
    }

/*----------------------------------------------------------------------------*/

    {
        auto now1 = now + bpt::seconds(1);
        cont.push_back(svr::ptr<DataRow>(now1, bpt::second_clock::local_time(), svr::common::C_default_value_tick_volume, 1));

        now1 += bpt::seconds(1);
        cont.push_back(svr::ptr<DataRow>(now1, bpt::second_clock::local_time(), svr::common::C_default_value_tick_volume, 1));

        datarow_range dr2(std::next(cont.begin()), std::prev(cont.end()), cont);

        EXPECT_TRUE(std::distance(dr2.begin(), dr2.end()) == 1);
        ASSERT_EQ(std::distance(dr2.rbegin(), dr2.rend()), 1);
    }
}
