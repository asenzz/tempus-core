#include "MeanCalculator.hpp"

#include <algorithm>
#include <mutex>

namespace svr {
namespace fix {


/******************************************************************************/
/*                            B A S E                                         */
/******************************************************************************/

mean_spread_calculator_base::mean_spread_calculator_base(bpt::time_duration averaging_period)
: averaging_period(averaging_period)
, retention_period(averaging_period * retend_intervals)
, values(retention_period.total_milliseconds() * quotes_per_second / 1000 )
, last_success_retention (averaging_period * 10)
{
}


mean_spread_calculator_base::~mean_spread_calculator_base()
{
}


void mean_spread_calculator_base::add_value(bid_ask_spread const & value)
{
    std::scoped_lock<svr::common::spin_lock> guard(lock);

    if(values.capacity() < 2)
        values.resize((values.size() + 1) * 1.5);

    values.push_back( value );

    while(values.front().time <= value.time - retention_period)
        values.pop_front();
}


bool mean_spread_calculator_base::calculate(bpt::ptime const & to_time, bid_ask_spread & result)
{
    std::scoped_lock<svr::common::spin_lock> guard(lock);

    if(values.empty())
        return false;

    bid_ask_spread tmp;

    tmp.time = to_time - averaging_period;
    auto interval_start = std::lower_bound(values.begin(), values.end(), tmp, [](bid_ask_spread const & other, bid_ask_spread const & comparand)->bool{return other.time <= comparand.time;});

    if(interval_start == values.end())
        return get_last_success(to_time, result);

    tmp.time = to_time;
    auto const interval_end   = std::upper_bound(interval_start, values.end(), tmp, [](bid_ask_spread const & other, bid_ask_spread const & comparand)->bool{return other.time < comparand.time;});

    if(interval_end == interval_start)
        return get_last_success(to_time, result);

    bool b_res = do_calculate(interval_start, interval_end, result);
    result.time = to_time;

    if(b_res)
        last_success = result;

    return b_res;
}

bool mean_spread_calculator_base::get_last_success(bpt::ptime const & to_time, bid_ask_spread & result) const
{
    if(!last_success)
        return false;

    if(to_time > last_success->time + last_success_retention)
        return false;

    result = *last_success;
    result.time = to_time;
    return true;
}


/******************************************************************************/
/*                            T W A P                                         */
/******************************************************************************/


twap_spread_calculator::twap_spread_calculator(bpt::time_duration const & averaging_period)
:mean_spread_calculator_base(averaging_period)
{
}

//  Formula to calculate the mean price
//
//              1          (px[n] + px[n+1])
// mean_px = -------  SUM ----------------- (t[n+1] - t[n])
//            t_all    N          2
//

bool twap_spread_calculator::do_calculate(my_cont_t::const_iterator start, my_cont_t::const_iterator const & end, bid_ask_spread & result)
{
    result = *start;

    if(start + 1 == end)
        return true;

    bpt::time_duration const t_all = (end - 1)->time - start->time;

    ++start;

    bid_ask_spread current_mean;
    for(; start != end; ++start)
    {
        double const new_px_weight = double(bpt::time_duration(start->time - (start-1)->time).total_microseconds()) / double(t_all.total_microseconds());

        current_mean.ask_px += new_px_weight * ( (start-1)->ask_px + start->ask_px ) / 2.0;
        current_mean.bid_px += new_px_weight * ( (start-1)->bid_px + start->bid_px ) / 2.0;
    }

    --start;
    current_mean.ask_qty = start->ask_qty;
    current_mean.bid_qty = start->bid_qty;
    current_mean.time = start->time;

    result = current_mean;

    return true;
}

}
}