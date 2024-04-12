#include "FixSubscription.hpp"
#include "model/InputQueue.hpp"
#include "appcontext.hpp"

#include <quickfix/fix44/MarketDataRequest.h>
#include <quickfix/fix44/MarketDataSnapshotFullRefresh.h>
#include <quickfix/Session.h>
#include <ios>

#include <common/Logging.hpp>
#include <model/InputQueue.hpp>

namespace svr {
namespace fix {

/******************************************************************************/
/*                                                                            */
/*                      Helper class input_queue_writer                       */
/*                                                                            */
/******************************************************************************/

class input_queue_writer : public mm_file_writer
{
public:
    input_queue_writer(datamodel::InputQueue_ptr input_queue, std::string const & mm_file_name, size_t no_of_elements);
private:
    void do_write(bid_ask_spread const & spread);
    datamodel::InputQueue_ptr input_queue;
};


/******************************************************************************/
/*                                                                            */
/*                      Helper class fix_subscription                         */
/*                                                                            */
/******************************************************************************/

class fix_data_drain
{
public:
    fix_data_drain(datamodel::InputQueue_ptr input_queue, std::string const & mmf_name, bpt::time_duration const & period);
    fix_data_drain(const fix_data_drain& orig) = delete;

    void start();
    void stop();
    void add_value(bid_ask_spread const & spread);
private:
    bpt::time_duration const period;
    twap_spread_calculator mean_calc;
    input_queue_writer writer;
    scheduler_base scheduler;

    bool subscribeForMd(std::string const & symbol);
};

/******************************************************************************/
/*                                                                            */
/*                    fix_subscription_container::Impl                        */
/*                                                                            */
/******************************************************************************/

struct fix_subscription_container::Impl {
    typedef std::map<bpt::time_duration, fix_data_drain *> duration_to_drain_mapping;
    typedef std::map<std::string, duration_to_drain_mapping> my_cont_t;

    my_cont_t cont;
    std::string const senderCompId;
    std::string const targetCompId;

    Impl(std::string const & senderCompId, std::string const & targetCompId)
    : senderCompId(senderCompId)
    , targetCompId(targetCompId)
    {}

    ~Impl()
    {
        for(auto const & dsm : cont)
            for(auto const & duration_drain_pair : dsm.second)
            {
                duration_drain_pair.second->stop();
                delete duration_drain_pair.second;
            }
    }

    void create_unless_exists(datamodel::InputQueue_ptr input_queue)
    {
        std::string symbol = input_queue->get_logical_name();
        std::transform(std::execution::par_unseq, symbol.begin(), symbol.end(),symbol.begin(), ::toupper);

        auto iter_cont = cont.insert(my_cont_t::value_type(symbol, duration_to_drain_mapping()));

        auto iter_drain = iter_cont.first->second.find(input_queue->get_resolution());

        if(iter_drain != iter_cont.first->second.end())
            return;

        fix_data_drain * new_drain = new fix_data_drain(input_queue, input_queue->get_table_name(), input_queue->get_resolution());

        iter_cont.first->second.insert(duration_to_drain_mapping::value_type(input_queue->get_resolution(), new_drain));

        new_drain->start();

        subscribeForMd(symbol);
    }


    void add_value(std::string symbol, bid_ask_spread const & spread)
    {
        std::transform(std::execution::par_unseq, symbol.begin(), symbol.end(),symbol.begin(), ::toupper);

        auto iter_cont = cont.find(symbol);
        if(iter_cont == cont.end())
        {
            LOG4_ERROR("Missing entry in the container for symbol "  << symbol);
            return;
        }

        for( auto & drain : iter_cont->second )
            drain.second->add_value(spread);
    }


    bool subscribeForMd(std::string const & symbol)
    {
        std::string md_request_id  = "MD_";
        md_request_id += symbol;

        FIX::MDReqID mdReqID( md_request_id.c_str() );
        FIX::SubscriptionRequestType subType( FIX::SubscriptionRequestType_SNAPSHOT_AND_UPDATES );
        FIX::MarketDepth marketDepth( 1 );

        FIX44::MarketDataRequest message( mdReqID, subType, marketDepth );

        message.setField(FIX::MDUpdateType( 0 )); // 0=Full, 1=Incremental

        FIX44::MarketDataRequest::NoMDEntryTypes marketDataEntryGroup;
        marketDataEntryGroup.set(FIX::MDEntryType ( FIX::MDEntryType_BID ));
        message.addGroup( marketDataEntryGroup );
        marketDataEntryGroup.set(FIX::MDEntryType ( FIX::MDEntryType_OFFER ));
        message.addGroup( marketDataEntryGroup );

        FIX44::MarketDataRequest::NoRelatedSym symbolGroup;
        symbolGroup.set( FIX::Symbol( symbol) );
        message.addGroup( symbolGroup );

        message.getHeader().setField( FIX::SenderCompID(senderCompId) );
        message.getHeader().setField( FIX::TargetCompID(targetCompId) );

        return FIX::Session::sendToTarget( message );
    }
};

/******************************************************************************/
/*                                                                            */
/*                      fix_subscription_container                            */
/*                                                                            */
/******************************************************************************/

fix_subscription_container::fix_subscription_container(std::string const & senderCompId, std::string const & targetCompId)
:pImpl( * new Impl(senderCompId, targetCompId) )
{

}


fix_subscription_container::~fix_subscription_container()
{
    delete & pImpl;
}


void fix_subscription_container::create_unless_exists(datamodel::InputQueue_ptr input_queue)
{
    pImpl.create_unless_exists(input_queue);
}


void fix_subscription_container::add_value(std::string const & symbol, bid_ask_spread const & spread)
{
    pImpl.add_value(symbol, spread);
}

/******************************************************************************/
/*                                                                            */
/*                      Helper classes implementation                         */
/*                                                                            */
/******************************************************************************/


input_queue_writer::input_queue_writer(datamodel::InputQueue_ptr input_queue, std::string const & mm_file_name, size_t no_of_elements)
: mm_file_writer(mm_file_name, no_of_elements)
, input_queue(input_queue)
{
}


/******************************************************************************/


void input_queue_writer::do_write(bid_ask_spread const & spread)
{
    mm_file_writer::do_write(spread);
    input_queue->get_data().clear();

    datamodel::DataRow_ptr row = datamodel::DataRow_ptr(new svr::datamodel::DataRow(spread.time, spread.time, 0, {spread.ask_px, double(spread.ask_qty), spread.bid_px, double(spread.bid_qty)}));

    input_queue->get_data().push_back(row);
    APP.input_queue_service.save(input_queue);
}


fix_data_drain::fix_data_drain(datamodel::InputQueue_ptr input_queue, std::string const & mmf_name, bpt::time_duration const & period)
: period(period)
, mean_calc( period )
, writer( input_queue, mmf_name, 1000 )
, scheduler()
{
}


void fix_data_drain::start()
{
    scheduler.start(period, mean_calc, writer);
}


void fix_data_drain::stop()
{
    scheduler.stop();
}


void fix_data_drain::add_value(bid_ask_spread const & spread)
{
    mean_calc.add_value(spread);
}


}}
