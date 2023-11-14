#ifndef SCHEDULER_HPP
#define SCHEDULER_HPP

#include "BidAskSpread.hpp"

#include <thread>
#include <atomic>

#include "MeanCalculator.hpp"
#include "InterprocessWriter.hpp"

namespace svr {
namespace fix {

class scheduler_base
{
public:
    void start(bpt::time_duration const & period, mean_spread_calculator_base & source, interprocess_writer_base & destination);
    void stop();

    static bpt::ptime to_utc_time(bpt::ptime const & local_time);
private:
    std::thread * worker {nullptr};

    void worker_proc(bpt::time_duration const & period, mean_spread_calculator_base * source, interprocess_writer_base * destination);

    std::atomic<bool> work;
};

}}

#endif /* SCHEDULER_HPP */
