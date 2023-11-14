#ifndef BIDASKSPREAD_HPP
#define BIDASKSPREAD_HPP

#include <common/types.hpp>

namespace svr {
namespace fix {

struct bid_ask_spread
{
    double bid_px;
    size_t bid_qty;
    double ask_px;
    size_t ask_qty;
    bpt::ptime time;

    bid_ask_spread();
    bid_ask_spread(double bid_px, size_t bid_qty, double ask_px, size_t ask_qty, bpt::ptime const & time);
    bool operator==(bid_ask_spread const & other) const;
};

}}

#endif /* BIDASKSPREAD_HPP */

