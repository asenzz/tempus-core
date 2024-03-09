#include "FixConnector.hpp"

#include <BidAskSpread.hpp>

#include <quickfix/fix44/MarketDataRequest.h>
#include <quickfix/fix44/MarketDataSnapshotFullRefresh.h>

#include <common/Logging.hpp>
#include <common/ScopeExit.hpp>
#include <thread>

#include <appcontext.hpp>

namespace svr {
namespace fix {

void FixConnector::run()
{
    auto message_at_exit = at_scope_exit([]() { LOG4_INFO("The FixConnector main loop exited."); });

    while (!logged_in) {
        std::this_thread::sleep_for(std::chrono::microseconds(1000));
        if (done) return;
    }

    subscriptions = std::shared_ptr<fix_subscription_container>(new fix_subscription_container(md_session_id.getSenderCompID(), md_session_id.getTargetCompID()));

    while (!done) {
        const auto iqs = APP.input_queue_service.get_all_queues_with_sign(true);
        for (const auto &iq: iqs) subscriptions->create_unless_exists(iq);
        for (auto i = 0U; i < 100; ++i) {
            if (done) return;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
}


void FixConnector::stop()
{
    done = true;
}


FixConnector::FixConnector(FIX::SessionSettings &settings)
        : logged_in(false), done(false), settings(settings)
{
}


FixConnector::~FixConnector()
{
}


void FixConnector::onLogon(const FIX::SessionID &sessionID)
{
    LOG4_INFO("Logged on successfully.");
    logged_in = true;
    md_session_id = sessionID;
}


void FixConnector::onLogout(const FIX::SessionID &sessionID)
{
    LOG4_INFO("Logged out successfully.");
    logged_in = false;
}


void FixConnector::fromApp(const FIX::Message &message, const FIX::SessionID &sessionID) noexcept(false)
{
    crack(message, sessionID);
}


void FixConnector::toApp(FIX::Message &message, const FIX::SessionID &sessionID) noexcept(false)
{
    try {
        FIX::PossDupFlag possDupFlag;
        message.getHeader().getField(possDupFlag);
        if (possDupFlag) throw FIX::DoNotSend();
    }
    catch (FIX::FieldNotFound &) {}
}


void FixConnector::toAdmin(FIX::Message &message, const FIX::SessionID &session_id)
{
    if (FIX::MsgType_Logon == message.getHeader().getField(FIX::FIELD::MsgType)) {
        message.setField(FIX::EncryptMethod(0));
        message.setField(FIX::Username(settings.get(session_id).getString("Login")));
        message.setField(FIX::Password(settings.get(session_id).getString("Password")));
        message.setField(FIX::ResetSeqNumFlag("Y"));
    }
}

namespace {
void read_bas(FIX44::MarketDataSnapshotFullRefresh::NoMDEntries const &mdEntries, bid_ask_spread &bas)
{
    FIX::MDEntryType mdEntryType;
    mdEntries.get(mdEntryType);

    FIX::MDEntryPx mdEntryPx;
    mdEntries.get(mdEntryPx);

    FIX::MDEntrySize mdEntrySize;
    mdEntries.get(mdEntrySize);

    switch (mdEntryType.getValue()) {
        case '0':
            bas.bid_px = mdEntryPx.getValue();
            bas.bid_qty = decltype(bas.bid_qty)(mdEntrySize.getValue());
            break;
        case '1':
            bas.ask_px = mdEntryPx.getValue();
            bas.ask_qty = decltype(bas.ask_qty)(mdEntrySize.getValue());
            break;
        default:
            throw std::invalid_argument("FixConnector::read_bas: invalid MDEntryType value passed.");
    }
}
}

void FixConnector::onMessage(const FIX44::MarketDataSnapshotFullRefresh &message, const FIX::SessionID &)
{
    bid_ask_spread bas;

    FIX44::MarketDataSnapshotFullRefresh::NoMDEntries mdEntries;

    message.getGroup(1, mdEntries);
    read_bas(mdEntries, bas);

    message.getGroup(2, mdEntries);
    read_bas(mdEntries, bas);

    FIX::SendingTime st;
    message.getHeader().getField(st);

    bas.time = bpt::ptime_from_tm(st.getValue().getTmUtc());
    bas.time += bpt::millisec(st.getValue().getMillisecond());

    FIX::Symbol symbol;
    message.get(symbol);

    subscriptions->add_value(symbol.getString(), bas);
}


void FixConnector::onMessage(const FIX44::MarketDataIncrementalRefresh &, const FIX::SessionID &)
{

}

}
}


