#include <vector>
#include <gtest/gtest.h>
#include "common/compatibility.hpp"
#include "../src/Scheduler.hpp"

template<class T>
struct test_writer : svr::fix::interprocess_writer_base
{
    T t;
    test_writer(T t): t(t) {}
    virtual void do_write(svr::fix::bid_ask_spread const & spread)
    {
        t(spread);
    }
};

struct call_history
{
    bpt::ptime call_time;
    svr::fix::bid_ask_spread spread;
    call_history(bpt::ptime call_time, svr::fix::bid_ask_spread const & spread): call_time(call_time), spread(spread) {}
};

void run_scheduler(std::vector<call_history> & calls, bpt::time_duration averaging_period, bpt::time_duration test_length)
{
    svr::fix::scheduler_base scheduler;

    auto func = [&calls](svr::fix::bid_ask_spread const & spr){ calls.push_back( call_history (bpt::microsec_clock::local_time(), spr) ); };

    svr::fix::twap_spread_calculator reader( averaging_period );

    test_writer<DTYPE(func)> writer(func);

    bpt::ptime const start = bpt::microsec_clock::local_time();

    scheduler.start(averaging_period, reader, writer);

    double tmp = 0;
    while(bpt::microsec_clock::local_time() - start < test_length)
    {
        reader.add_value(svr::fix::bid_ask_spread(tmp, tmp, tmp, tmp, scheduler.to_utc_time(bpt::microsec_clock::local_time())));
        tmp += 1;
        usleep(averaging_period.total_microseconds() / 10);
    }

    scheduler.stop();
}

TEST(SchedulerTests, TestSubsecondSynchronicity)
{
    auto const accuracy = bpt::milliseconds(1);
    auto const averaging_intl = bpt::milliseconds(50);
    auto const this_time = bpt::from_iso_string("20160819T000000");

    std::vector<call_history> calls;

    run_scheduler(calls, averaging_intl, bpt::seconds(1));

    for(auto call : calls)
    {
        //std::cout << "call_time: " << call.call_time << " px_time: " << call.spread.time << " bid_px: " << call.spread.bid_px << '\n';
        ASSERT_LT(svr::fix::scheduler_base::to_utc_time(call.call_time), call.spread.time + accuracy);
        ASSERT_EQ(0, bpt::time_duration(call.spread.time - this_time).total_microseconds() % averaging_intl.total_microseconds());
    }

    ASSERT_GE(calls.size(), 20UL);
}

TEST(SchedulerTests, TestSubminuteSynchronicity)
{
    auto const accuracy = bpt::milliseconds(1);
    auto const averaging_intl = bpt::seconds(1);
    bpt::ptime const this_time = bpt::from_iso_string("20160819T000000");

    std::vector<call_history> calls;

    run_scheduler(calls, averaging_intl, bpt::seconds(5));

    for(auto call : calls)
    {
        //std::cout << "call_time: " << call.call_time << " px_time: " << call.spread.time << " bid_px: " << call.spread.bid_px << '\n';
        ASSERT_LT(svr::fix::scheduler_base::to_utc_time(call.call_time), call.spread.time + accuracy);
        ASSERT_EQ(0, bpt::time_duration(call.spread.time - this_time).total_microseconds() % averaging_intl.total_microseconds());
    }

    ASSERT_GE(calls.size(), 5UL);
}
