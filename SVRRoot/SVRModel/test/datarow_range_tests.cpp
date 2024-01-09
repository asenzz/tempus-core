#include <gtest/gtest.h>

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
        ASSERT_EQ(dr0.begin(), cont.begin());
        ASSERT_EQ(dr0.end(), cont.end());

        ASSERT_EQ(dr0.rbegin(), cont.rbegin());
        ASSERT_EQ(dr0.rend(), cont.rend());

        ASSERT_EQ(dr0.begin(), dr0.end());
        ASSERT_EQ(dr0.rbegin(), dr0.rend());
        */
    }

/*----------------------------------------------------------------------------*/

    {
        cont.push_back(std::make_shared<DataRow>(now));

        datarow_range dr(cont);
/*
        ASSERT_EQ(dr.begin(), cont.begin());
        ASSERT_EQ(dr.end(), cont.end());

        ASSERT_EQ(dr.rbegin(), cont.rbegin());
        ASSERT_EQ(dr.rend(), cont.rend());
        */
    }

/*----------------------------------------------------------------------------*/

    {
        auto now1 = now + bpt::seconds(1);
        cont.push_back(std::make_shared<DataRow>(now1));

        now1 += bpt::seconds(1);
        cont.push_back(std::make_shared<DataRow>(now1));

        datarow_range dr2(std::next(cont.begin()), std::prev(cont.end()), cont);

        ASSERT_EQ(std::distance(dr2.begin(), dr2.end()), 1);
        ASSERT_EQ(std::distance(dr2.rbegin(), dr2.rend()), 1);
    }
}
