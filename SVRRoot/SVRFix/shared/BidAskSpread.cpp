#include <BidAskSpread.hpp>

namespace svr {
namespace fix {

bid_ask_spread::bid_ask_spread()
: bid_px(0), bid_qty(0), ask_px(0), ask_qty(0), time()
{
}


bid_ask_spread::bid_ask_spread(double bid_px, size_t bid_qty, double ask_px, size_t ask_qty, bpt::ptime const & time)
: bid_px(bid_px), bid_qty(bid_qty), ask_px(ask_px), ask_qty(ask_qty), time(time)
{}

bool bid_ask_spread::operator==(bid_ask_spread const & other) const
{
    return bid_qty == other.bid_qty
        && ask_qty == other.ask_qty
        && time == other.time
        && fabs(bid_px - other.bid_px) < std::numeric_limits<double>::epsilon()
        && fabs(ask_px - other.ask_px) < std::numeric_limits<double>::epsilon();
}


}}
