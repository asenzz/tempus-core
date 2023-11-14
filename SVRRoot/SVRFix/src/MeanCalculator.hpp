#ifndef MEANCALCULATOR_HPP
#define MEANCALCULATOR_HPP

#include "BidAskSpread.hpp"
#include <boost/circular_buffer.hpp>
#include <boost/optional.hpp>
#include <common/spin_lock.hpp>
#include <memory>

namespace svr {
namespace fix {

class mean_spread_calculator_base
{
public:
    mean_spread_calculator_base(bpt::time_duration averaging_period);
    virtual ~mean_spread_calculator_base();

    void add_value(bid_ask_spread const & value);
    bool calculate(bpt::ptime const & to_time, bid_ask_spread & result);

    typedef boost::circular_buffer<bid_ask_spread> my_cont_t;
private:
    virtual bool do_calculate(my_cont_t::const_iterator start, my_cont_t::const_iterator const & end, bid_ask_spread & result) = 0;

    bpt::time_duration const averaging_period, retention_period;
    my_cont_t values;
    static size_t const quotes_per_second = 20;
    static size_t const retend_intervals = 50;
    svr::common::spin_lock lock;
    boost::optional<bid_ask_spread> last_success;
    bpt::time_duration const last_success_retention;

    bool get_last_success(bpt::ptime const & to_time, bid_ask_spread & result) const;

};

class twap_spread_calculator : public mean_spread_calculator_base
{
public:
    twap_spread_calculator(bpt::time_duration const & averaging_period);
private:
    virtual bool do_calculate(my_cont_t::const_iterator start, my_cont_t::const_iterator const & end, bid_ask_spread & result);
};

}
}

#endif /* MEANCALCULATOR_HPP */

