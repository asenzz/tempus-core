#ifndef INTERPROCESSREADER_HPP
#define INTERPROCESSREADER_HPP

#include "BidAskSpread.hpp"

namespace svr {
namespace fix {

class mm_file_reader
{
public:
    mm_file_reader(std::string const & mm_file_name); // throws std::runtime_error if cannot open/read the mm file
    ~mm_file_reader();

    using bid_ask_spread_container = std::vector<bid_ask_spread>;

    bid_ask_spread_container read_all() const;
    bid_ask_spread_container read_new(bpt::ptime const & last_time) const;

private:
    struct Impl;
    Impl &pImpl;
};


}}

#endif /* INTERPROCESSREADER_HPP */

