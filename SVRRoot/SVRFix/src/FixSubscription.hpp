#ifndef FIXSESSION_HPP
#define FIXSESSION_HPP

#include <string>
#include <atomic>
#include <memory>

#include <common/types.hpp>

#include <quickfix/SessionID.h>

#include "Scheduler.hpp"

namespace svr { namespace datamodel { class InputQueue; } }
using InputQueue_ptr = std::shared_ptr<svr::datamodel::InputQueue>;

namespace svr {
namespace fix {

class fix_subscription_container
{
public:
    fix_subscription_container(std::string const & senderCompId, std::string const & targetCompId);
    fix_subscription_container(fix_subscription_container const & other)=delete;
    ~fix_subscription_container();

    void create_unless_exists(InputQueue_ptr input_queue);
    void add_value(std::string const & symbol, bid_ask_spread const & spread);
private:
    class Impl;
    Impl & pImpl;
};


}}

#endif /* FIXSESSION_HPP */

