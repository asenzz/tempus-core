#ifndef INTERPROCESSWRITER_HPP
#define INTERPROCESSWRITER_HPP

#include "BidAskSpread.hpp"
#include <memory>

namespace svr {
namespace fix {


class interprocess_writer_base
{
public:
    interprocess_writer_base()=default;
    interprocess_writer_base(interprocess_writer_base const &)=delete;
    void write(bid_ask_spread const & spread);
protected:
    virtual void do_write(bid_ask_spread const & spread) = 0;
};


//memory mapped file writer
class mm_file_writer : public interprocess_writer_base
{
public:
    mm_file_writer(std::string const & mm_file_name, size_t no_of_elements);
    ~mm_file_writer();

protected:
    void do_write(bid_ask_spread const & spread);

private:
    class Impl;
    Impl & pImpl;
};


}}

#endif /* INTERPROCESSWRITER_HPP */

