//+------------------------------------------------------------------+
//|                                                tempus-trader.mq4 |
//|                                           Copyright 2019, Tempus |
//|                                              https://tempus.work |
//+------------------------------------------------------------------+
#property copyright     "Copyright 2019, Tempus"
#property link          "https://tempus.work"
#property version       "1.00"
#property strict
#property description   "A simple and naive trading expert."

input string Server_URL = "10.4.8.24:8080";
input double Trade_Lot = 0.01;
input int Slippage = 3;
input double Trading_Alpha = 55;
const double Stop_Loss_Pct = 2. * Trading_Alpha / 100.;
const double Take_Profit_Pct = 2. * (100. - Trading_Alpha) / 100.;
input string Username = "svrwave";
input string Password = "svrwave";
input int Dataset_ID = 100;
input ulong Forecast_Offset = 360; 
input double Position_Threshold = .1;
input ulong Stop_Hour = 12;
input ulong Stop_Day = 4;
input ulong StdDev_Period = 60;

// secs
input ulong Prediction_Wait = 12; // Prediction delay in seconds

const bool Average = true;
const string ServerUrl = Server_URL;

#define EXPERT_MAGIC 132435   // MagicNumber of the expert

#include <TempusGMT.mqh>
#include <TempusMvc.mqh>
#include <AveragePrice.mqh>
#include <tempus-constants.mqh>

#include <Trade\PositionInfo.mqh>
#include <Trade\Trade.mqh>

CPositionInfo  m_position;                   // trade position object
CTrade         m_trade;                      // trading object
MqlNet         net;

TempusController controller;
datetime pending_request_time = 0;

#define INIT_TEST

datetime G_local_time_of_last_request = 0;
datetime G_server_time_of_last_request = 0;

int iStdDev_handle = 0;

double indicator_value(const int handle, const int shift)
{
    double buffer[];
    while (CopyBuffer(handle, 0, shift, 1, buffer) != 1)
    {
        Print(StringFormat("Error in indicator_value(%d,%d): cannot copy value", handle, shift));
    }
    if (ArraySize(buffer) < 1) return(0.);
    return(buffer[0]);
}

  
int OnInit()
{
    LOG_INFO("", "Starting up ...");

    string input_queue = input_queue_name_from_symbol(_Symbol);
        
    if(!net.Open(Server_URL)) {
        Print("Incorrect server address");
        return(INIT_PARAMETERS_INCORRECT);
    }
    long status_code = 0;
    const string login_url = "mt4/login";
    const string post_payload = "username=" + Username + "&password=" + Password;
    net.Post(login_url, post_payload, status_code);
    if (status_code != 200) {
        LOG_ERROR("", "Authentication error occured " + string(status_code) + " " + ErrorDescription(status_code));
        return(INIT_PARAMETERS_INCORRECT);
    }

    iStdDev_handle = iStdDev(_Symbol, PERIOD_M1, StdDev_Period /* PeriodSeconds() / 60 */, 0, MODE_SMA, PRICE_WEIGHTED);
    //iATR_handle = iATR(_Symbol, PERIOD_M1, 60);

    controller.init(input_queue, PeriodSeconds(), 0, 0, 0, 0, 1);
    return(INIT_SUCCEEDED);
}


void OnDeinit(const int reason)
{
}


bool CheckOpenOrders()
{
   //const double price_ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   //const double price_bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   //const double price_spread = MathAbs(price_ask - price_bid);
   bool closed_order = false;
   for(int i = 0; i < PositionsTotal() ; ++i) {
      if (!m_position.SelectByIndex(i)) continue;
      if (m_position.Symbol() != _Symbol) continue;
      datetime position_time = 0;
      m_position.InfoInteger(POSITION_TIME, position_time);
      if (TimeCurrent() - position_time > PeriodSeconds()) {
        //ulong position_type = 0;
        //m_position.InfoInteger(POSITION_TYPE, position_type);
        m_trade.PositionClose(m_position.Ticket());
        closed_order = true;
      }
   }
   return closed_order;
}


void OnTick()
{
    const bool force_closed_order = CheckOpenOrders();

    const datetime current_bar_time = iTime(_Symbol, _Period, 0);
    const datetime current_server_time = TimeCurrent();
    const datetime current_local_time = GetWindowsLocalTime();

    MqlDateTime current_server_time_st{};
    TimeToStruct(current_server_time, current_server_time_st);
    if (current_server_time_st.day_of_week >= Stop_Day && current_server_time_st.hour >= Stop_Hour) return;
    // if (!(current_server_time_st.hour % 2)) return; // ??

    datetime anchor_time = 0;
    double anchor_price = 0;
    double predicted_price = 0;
    if (current_server_time >= current_bar_time + PeriodSeconds() - Forecast_Offset && current_bar_time >= pending_request_time) { // TODO double check, shouldn't it be, current_bar_time >= pending_request_time
        pending_request_time = current_bar_time + PeriodSeconds();
        controller.doRequest(net, pending_request_time, 1, Dataset_ID);
        LOG_DEBUG("", "Placed request for time " + TimeToString(pending_request_time, TIME_DATE_SECONDS) + " at server time " + TimeToString(current_server_time, TIME_DATE_SECONDS));
        
        TempusFigure results[];
        const datetime sleep_until = GetWindowsLocalTime() + Prediction_Wait;
        while ((!controller.getResults(net, pending_request_time, 1, Dataset_ID, results) || ArraySize(results) < 1) && GetWindowsLocalTime() < sleep_until) {};
        for (int bar_ix = ArraySize(results) - 1; bar_ix >= 0; --bar_ix) {
            if (results[bar_ix].tm != pending_request_time) {
                LOG_DEBUG("", "Skipping result " + results[bar_ix].to_string() + " not for request made for " + pending_request_time);
                continue;
            }
            anchor_time = results[bar_ix].tm - Forecast_Offset - PeriodSeconds(PERIOD_M1);
            anchor_price = iClose(_Symbol, PERIOD_M1, iBarShift(_Symbol, PERIOD_M1, anchor_time, false));
            predicted_price = results[bar_ix].cl;
            LOG_INFO("", "Received " + results[bar_ix].to_string() + " of initial request made at " + TimeToString(pending_request_time, TIME_DATE_SECONDS));
            break;
        }
    }
    
    if (predicted_price > 0) {
        const double this_position_price_difference = indicator_value(iStdDev_handle, 0);
        if (this_position_price_difference >= Position_Threshold) {
            MqlTick ticknow = {};
            ZeroMemory(ticknow);
            SymbolInfoTick(_Symbol, ticknow);
            const double price_ask = ticknow.ask; // SymbolInfoDouble(_Symbol, SYMBOL_ASK);
            const double price_bid = ticknow.bid; // SymbolInfoDouble(_Symbol, SYMBOL_BID);
            const double price_spread = MathAbs(price_ask - price_bid);
            Print("Predicted price is " + DoubleToString(predicted_price) + ", price bid is " + price_bid + ", price ask is " + price_ask + ", price spread " + price_spread);
            MqlTradeRequest request = {};
            ZeroMemory(request);
            MqlTradeResult result = {};
            ZeroMemory(result);
            request.action = TRADE_ACTION_DEAL;                     // Type of trade operation
            request.symbol = _Symbol;                               
            request.volume = Trade_Lot;                              
            request.deviation = Slippage;                           // Allowed deviation from the price
            request.magic = EXPERT_MAGIC + MathRand();              // MagicNumber of the order
            // Try a buy
            if (predicted_price > anchor_price) {
                request.type = ORDER_TYPE_BUY;                           // order type
                request.price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);   // price for opening
                request.sl = request.price - this_position_price_difference * Stop_Loss_Pct;
                request.tp = request.price + this_position_price_difference * Take_Profit_Pct;
            } else if (predicted_price < anchor_price) {
                request.type = ORDER_TYPE_SELL;                          // order type
                request.price = SymbolInfoDouble(_Symbol, SYMBOL_BID);   // price for opening
                request.sl = request.price + this_position_price_difference * Stop_Loss_Pct;
                request.tp = request.price - this_position_price_difference * Take_Profit_Pct;
            }
            //--- send the request
            if(!OrderSend(request, result)) 
                LOG_ERROR("", "OrderSend error " + string(_LastError));     // if unable to send the request, output the error code
            else 
                PrintFormat("Order sent, retcode=%u  deal=%I64u  order=%I64u", result.retcode, result.deal, result.order);
        } else {
            LOG_DEBUG("", "Price movement too small " + DoubleToString(this_position_price_difference));
        }
    }
}
//+------------------------------------------------------------------+
