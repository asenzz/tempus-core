#ifndef FIXAPPLICATION_HPP
#define FIXAPPLICATION_HPP

#include <quickfix/Application.h>
#include <quickfix/MessageCracker.h>
#include <atomic>

#include "FixSubscription.hpp"

namespace svr {
namespace fix {

class FixConnector : public FIX::Application, public FIX::MessageCracker
{
public:
    FixConnector(FIX::SessionSettings & settings);
    FixConnector(const FixConnector& orig)=delete;
    virtual ~FixConnector();

    void run();
    void stop();
private:
    void onCreate( const FIX::SessionID& ) {};
    void onLogon( const FIX::SessionID& sessionID );
    void onLogout( const FIX::SessionID& sessionID );
    void toAdmin( FIX::Message&, const FIX::SessionID& );
    void toApp( FIX::Message&, const FIX::SessionID& ) noexcept(false);
    void fromAdmin( const FIX::Message&, const FIX::SessionID& ) noexcept(false){}
    void fromApp( const FIX::Message& message, const FIX::SessionID& sessionID ) noexcept(false);

    using FIX40::MessageCracker::onMessage;
    using FIX41::MessageCracker::onMessage;
    using FIX42::MessageCracker::onMessage;
    using FIX43::MessageCracker::onMessage;
    using FIX44::MessageCracker::onMessage;
    using FIX50::MessageCracker::onMessage;
    using FIX50SP1::MessageCracker::onMessage;
    using FIX50SP2::MessageCracker::onMessage;
    using FIXT11::MessageCracker::onMessage;

    void onMessage( const FIX44::MarketDataSnapshotFullRefresh&, const FIX::SessionID& );
    void onMessage( const FIX44::MarketDataIncrementalRefresh&, const FIX::SessionID& );

    std::atomic<bool> logged_in, done;
    FIX::SessionSettings & settings;
    FIX::SessionID md_session_id;
    std::shared_ptr<fix_subscription_container> subscriptions;
};

}
}

#endif /* FIXAPPLICATION_HPP */

