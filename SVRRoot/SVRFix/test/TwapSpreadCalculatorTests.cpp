#include <gtest/gtest.h>

#include "../src/MeanCalculator.hpp"

#define ASSERT_DBL_EQ(x, y) ASSERT_LE(fabs((x) - (y)), std::numeric_limits<double>::epsilon()) << " x: " << x << " y: " << y
#define ASSERT_SPR_EQ(spr, bp, bq, ap, aq, tm) \
    ASSERT_DBL_EQ((spr).bid_px, (bp)); ASSERT_EQ((spr).bid_qty, (bq)); \
    ASSERT_DBL_EQ((spr).ask_px, (ap)); ASSERT_EQ((spr).ask_qty, (aq)); \
    ASSERT_EQ((spr).time, (tm))

using svr::fix::bid_ask_spread;

TEST(TwapSpreadCalcTests, BasicTest)
{
    svr::fix::bid_ask_spread spr;

    svr::fix::twap_spread_calculator twap(bpt::seconds(1));

    twap.add_value(bid_ask_spread(1.0, 100, 2.0, 200, bpt::time_from_string("2002-01-20 23:59:59.000") ) );
    ASSERT_TRUE(twap.calculate( bpt::time_from_string("2002-01-20 23:59:59.000"), spr ));
    ASSERT_SPR_EQ(spr, 1.0, 100UL, 2.0, 200UL, bpt::time_from_string("2002-01-20 23:59:59.000") );

    twap.add_value(bid_ask_spread(1.1, 101, 2.1, 201, bpt::time_from_string("2002-01-20 23:59:59.100") ) );
    ASSERT_TRUE(twap.calculate( bpt::time_from_string("2002-01-20 23:59:59.100"), spr ));
    ASSERT_SPR_EQ(spr, 1.05, 101UL, 2.05, 201UL, bpt::time_from_string("2002-01-20 23:59:59.100") );

    twap.add_value(bid_ask_spread(1.1, 102, 2.1, 202, bpt::time_from_string("2002-01-20 23:59:59.200") ) );
    ASSERT_TRUE(twap.calculate( bpt::time_from_string("2002-01-20 23:59:59.200"), spr ));
    ASSERT_SPR_EQ(spr, 1.075, 102UL, 2.075, 202UL, bpt::time_from_string("2002-01-20 23:59:59.200") );
}

TEST(TwapSpreadCalcTests, EvictionOfFadedValues)
{
    svr::fix::bid_ask_spread spr;

    svr::fix::twap_spread_calculator twap(bpt::seconds(1));

    twap.add_value(bid_ask_spread(1.0, 100, 2.0, 200, bpt::time_from_string("2002-01-20 23:59:59.000") ) );
    ASSERT_TRUE(twap.calculate( bpt::time_from_string("2002-01-20 23:59:59.000"), spr ));
    ASSERT_SPR_EQ(spr, 1.0, 100UL, 2.0, 200UL, bpt::time_from_string("2002-01-20 23:59:59.000") );

    twap.add_value(bid_ask_spread(1.1, 101, 2.1, 201, bpt::time_from_string("2002-01-20 23:59:59.100") ) );
    ASSERT_TRUE(twap.calculate( bpt::time_from_string("2002-01-20 23:59:59.100"), spr ));
    ASSERT_SPR_EQ(spr, 1.05, 101UL, 2.05, 201UL, bpt::time_from_string("2002-01-20 23:59:59.100") );

    twap.add_value(bid_ask_spread(0.9, 102, 1.9, 202, bpt::time_from_string("2002-01-20 23:59:59.300") ) );
    twap.add_value(bid_ask_spread(1.0, 103, 2.0, 203, bpt::time_from_string("2002-01-20 23:59:59.400") ) );
    ASSERT_TRUE(twap.calculate( bpt::time_from_string("2002-01-20 23:59:59.400"), spr ));
    ASSERT_SPR_EQ(spr, 1.0, 103UL, 2.0, 203UL, bpt::time_from_string("2002-01-20 23:59:59.400") );

    twap.add_value(bid_ask_spread(1.1, 105, 2.1, 205, bpt::time_from_string("2002-01-20 23:59:59.500") ) );
    twap.add_value(bid_ask_spread(1.1, 106, 2.1, 206, bpt::time_from_string("2002-01-20 23:59:59.600") ) );
    twap.add_value(bid_ask_spread(1.1, 107, 2.1, 207, bpt::time_from_string("2002-01-20 23:59:59.700") ) );
    twap.add_value(bid_ask_spread(1.1, 108, 2.1, 208, bpt::time_from_string("2002-01-20 23:59:59.800") ) );
    twap.add_value(bid_ask_spread(1.1, 109, 2.1, 209, bpt::time_from_string("2002-01-20 23:59:59.900") ) );
    twap.add_value(bid_ask_spread(1.1, 111, 2.1, 211, bpt::time_from_string("2002-01-21 00:00:00.000") ) );
    twap.add_value(bid_ask_spread(1.1, 112, 2.1, 212, bpt::time_from_string("2002-01-21 00:00:00.100") ) );
    twap.add_value(bid_ask_spread(1.1, 113, 2.1, 213, bpt::time_from_string("2002-01-21 00:00:00.200") ) );
    twap.add_value(bid_ask_spread(1.1, 114, 2.1, 214, bpt::time_from_string("2002-01-21 00:00:00.300") ) );
    twap.add_value(bid_ask_spread(1.1, 110, 2.1, 210, bpt::time_from_string("2002-01-21 00:00:00.400") ) );

    ASSERT_TRUE(twap.calculate( bpt::time_from_string("2002-01-21 00:00:00.400"), spr ));
    ASSERT_SPR_EQ(spr, 1.1, 110UL, 2.1, 210UL, bpt::time_from_string("2002-01-21 00:00:00.400") );
}

TEST(TwapSpreadCalcTests, ReturningOfLastSuccess)
{
    svr::fix::bid_ask_spread spr;

    svr::fix::twap_spread_calculator twap(bpt::seconds(1));

    twap.add_value(bid_ask_spread(1.0, 100, 2.0, 200, bpt::time_from_string("2002-01-21 00:00:00.000") ) );
    ASSERT_TRUE(twap.calculate( bpt::time_from_string("2002-01-21 00:00:00.000"), spr ));
    ASSERT_SPR_EQ(spr, 1.0, 100UL, 2.0, 200UL, bpt::time_from_string("2002-01-21 00:00:00.000") );

    ASSERT_TRUE(twap.calculate( bpt::time_from_string("2002-01-21 00:00:10.000"), spr ));
    ASSERT_SPR_EQ(spr, 1.0, 100UL, 2.0, 200UL, bpt::time_from_string("2002-01-21 00:00:10.000") );

    ASSERT_FALSE(twap.calculate( bpt::time_from_string("2002-01-21 00:00:10.001"), spr ));
    ASSERT_SPR_EQ(spr, 1.0, 100UL, 2.0, 200UL, bpt::time_from_string("2002-01-21 00:00:10.000") );
}

