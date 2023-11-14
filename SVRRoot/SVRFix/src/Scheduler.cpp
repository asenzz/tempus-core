#include "Scheduler.hpp"
#include "MeanCalculator.hpp"

#include <common/Logging.hpp>

#include <boost/date_time/c_local_time_adjustor.hpp>

typedef boost::date_time::c_local_adjustor<bpt::ptime> local_adj;

namespace svr {
namespace fix {

void scheduler_base::start(bpt::time_duration const & period, mean_spread_calculator_base & source, interprocess_writer_base & destination)
{
    if(worker)
        throw std::runtime_error("Error: spread_data_writer_base already started.");
    work = true;
    worker = new std::thread(&scheduler_base::worker_proc, this, period, &source, &destination);
}


void scheduler_base::stop()
{
    if(!worker)
        throw std::runtime_error("Error: spread_data_writer_base is not started.");

    work = false;
    worker->join();
    delete worker;
    worker = nullptr;
}


void scheduler_base::worker_proc(bpt::time_duration const & period, mean_spread_calculator_base * source, interprocess_writer_base * destination)
{
    LOG4_INFO("scheduler worker thread started.");

    static const bpt::ptime epoch = bpt::from_iso_string("20100101T000000");

    while(work)
    {
        size_t const period_usec = period.total_microseconds();

        size_t const intervals_since_epoch =
            1 + (bpt::time_duration(bpt::microsec_clock::local_time() - epoch).total_microseconds() / period_usec);

        long signed wait_for_usec = period_usec - bpt::time_duration(bpt::microsec_clock::local_time() - epoch).total_microseconds() % period_usec;

        long signed const wait_interval_usec = 50000;

        while (wait_for_usec > wait_interval_usec)
        {
            std::this_thread::sleep_for( std::chrono::duration<double, std::micro>(wait_interval_usec) );
            wait_for_usec = period_usec - bpt::time_duration(bpt::microsec_clock::local_time() - epoch).total_microseconds() % period_usec;
            if(!work)
                return;
        }

        if(wait_for_usec > 10)
            std::this_thread::sleep_for( std::chrono::duration<double, std::micro>(wait_for_usec) );

        bid_ask_spread result;

        bpt::ptime const to_time = to_utc_time(epoch + bpt::microseconds(intervals_since_epoch * period_usec));

        if(source->calculate(to_time, result))
            destination->write(result);
    }

    LOG4_INFO("scheduler worker thread stopped.");
}

bpt::ptime scheduler_base::to_utc_time(bpt::ptime const & local_time)
{
    bpt::ptime const utc_time_diff = local_adj::utc_to_local(local_time);
    return local_time - (utc_time_diff - local_time);
}


}}