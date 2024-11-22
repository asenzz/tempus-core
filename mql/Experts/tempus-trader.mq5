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

input string ServerUrl = "127.0.0.1:8080"; // IP address : Port
input float Trade_Lot = float(.1);
input ulong Slippage = 3;
input string Username = "svrwave"; // SVRWeb dataset user and password
input string Password = "svrwave";
input uint Dataset_ID = 100; // Dataset id
input float Horizon = float(.2); // partial of predicted time frame
input int Stop_Day = 4; // Stop trading after the N-th hour of the M-th day of the week
input int Stop_Hour = 18;
input uint StdDev_Period = 60; // Period in minutes to use for StdDev calculation
input float StdDev_Threshold = 0; // Start a position if StdDev is above this threshold
input float Stop_Loss_StdDev_Ratio = 1; // Ratio of StdDev from placement price to use as stop loss
input uint Prediction_Wait = 3; // Prediction delay in seconds
input double Spread_Threshold = 1e2; // Start a trade only if spread is below threshold
input bool Only_Sells = true; // Only start sell positions

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
const uint C_period_seconds = PeriodSeconds(PERIOD_CURRENT);
const uint C_forecast_offset = uint(C_period_seconds * Horizon);
const uint C_max_retries = 10;
const uint C_sleep_ms = 10;
const string C_dataset_id_str = string(Dataset_ID);
const bool Average = true; // Needed for tempus-constants.mqh
#define EXPERT_MAGIC 132435   // MagicNumber of the expert
#define NOSIGNAL 0xDeadBeef

#include <TempusGMT.mqh>
#include <TempusMvc.mqh>
#include <AveragePrice.mqh>
#include <tempus-constants.mqh>

MqlNet         net;

TempusController controller;
datetime pending_request_time = 0;

#define INIT_TEST

datetime G_local_time_of_last_request = 0;
datetime G_server_time_of_last_request = 0;

int iStdDev_handle = 0;

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
float indicator_value(const int handle, const int shift)
{
    double buffer[];
    while (CopyBuffer(handle, 0, shift, 1, buffer) != 1)
        Print(StringFormat("Error in indicator_value(%d,%d): cannot copy value", handle, shift));
    if (ArraySize(buffer) < 1) return(0.);
    return(buffer[0]);
}


//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int OnInit()
{
    LOG_INFO("", "Starting up ...");
    const string input_queue = input_queue_name_from_symbol(_Symbol);
    if(!net.Open(ServerUrl)) {
        Print("Incorrect server address");
        return(INIT_PARAMETERS_INCORRECT);
    }
    long status_code = 0;
    const string login_url = "mt4/login";
    const string post_payload = "username=" + Username + "&password=" + Password;
    net.Post(login_url, post_payload, status_code);
    if (status_code != 200) {
        LOG_ERROR("", "Authentication error occured " + string(status_code) + " " + ErrorDescription(int(status_code)));
        return(INIT_PARAMETERS_INCORRECT);
    }

    iStdDev_handle = iStdDev(_Symbol, PERIOD_M1, StdDev_Period /* PeriodSeconds() / 60 */, 0, MODE_SMA, PRICE_WEIGHTED);
//iATR_handle = iATR(_Symbol, PERIOD_M1, 60);

    controller.init(input_queue, PeriodSeconds(), 0, 0, 0, 0, 1);
    return(INIT_SUCCEEDED);
}


//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
}


//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool UpdatePosition(const string &symbol, const double price, const ulong position_ticket, const double volume, const ENUM_POSITION_TYPE type, const double tp, const double sl)
{
// Prepare the trade request
    MqlTradeRequest request;
    ZeroMemory(request);
    MqlTradeResult result;
    ZeroMemory(result);
    request.action = TRADE_ACTION_SLTP;
    request.symbol = symbol;
    request.volume = volume;
    request.position = position_ticket;
    request.type = type == POSITION_TYPE_BUY ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;
    request.price = price;
    request.tp = tp;
    request.sl = sl;
    request.deviation = Slippage; // Allowable price slippage in points
    request.type_filling = ORDER_FILLING_IOC; // Immediate or Cancel
      
// Send the trade request
    if (!OrderSend(request, result)) {
        LOG_ERROR("", "Failed to update position " + string(position_ticket) + " on symbol " + symbol +
                  ", OrderSend failed " + result.comment + " (retcode: " + string(result.retcode) + ")");
        return false;
    }

    if (result.retcode != TRADE_RETCODE_DONE) {
        LOG_ERROR("", "Failed to update position " + string(position_ticket) + " on symbol " + symbol +
                  ", trade failed " + result.comment + " (retcode: " + string(result.retcode) + ")");
        return false;
    }

    return true;
}
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void UpdatePositionSafe(const string &symbol, const double price, const ulong position_ticket, const double volume, const ENUM_POSITION_TYPE type, const double tp, const double sl)
{
    uint retries = 0;
    while (retries++ < C_max_retries) {
        if (UpdatePosition(symbol, price, position_ticket, volume, type, tp, sl)) break;
        Sleep(C_sleep_ms);
    }
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool ClosePosition(const string &symbol, const double price, const ulong position_ticket, const double volume, const ENUM_POSITION_TYPE type)
{
// Prepare the trade request
    MqlTradeRequest request;
    ZeroMemory(request);
    MqlTradeResult result;
    ZeroMemory(result);
    request.action = TRADE_ACTION_DEAL;
    request.symbol = symbol;
    request.volume = volume;
    request.position = position_ticket;
    request.type = type == POSITION_TYPE_BUY ? ORDER_TYPE_SELL : ORDER_TYPE_BUY;
    request.price = price;
    request.deviation = Slippage; // Allowable price slippage in points
    request.type_filling = ORDER_FILLING_IOC; // Immediate or Cancel

// Send the trade request
    if (!OrderSend(request, result)) {
        LOG_ERROR("", "Failed to close position " + string(position_ticket) + " on symbol " + symbol +
                  ", OrderSend failed " + result.comment + " (retcode: " + string(result.retcode) + ")");
        return false;
    }

    if (result.retcode != TRADE_RETCODE_DONE) {
        LOG_ERROR("", "Failed to close position " + string(position_ticket) + " on symbol " + symbol +
                  ", trade failed " + result.comment + " (retcode: " + string(result.retcode) + ")");
        return false;
    }

    return true;
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void ClosePositionSafe(const string &symbol, const double price, const ulong position_ticket, const double volume, const ENUM_POSITION_TYPE type)
{
    uint retries = 0;
    while (retries++ < C_max_retries) {
        if (ClosePosition(symbol, price, position_ticket, volume, type)) break;
        Sleep(C_sleep_ms);
    }
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CheckOpenOrders(const int signal_type, const double price, const double tp, const double sl, const datetime server_time)
{
    const int total_positions = PositionsTotal();
    bool closed_order = false;
    for (int i = total_positions - 1; i >= 0; --i) {
        // Get the position ticket
        const ulong position_ticket = PositionGetTicket(i);
        if (PositionSelectByTicket(position_ticket)) {
            // Check if the position symbol matches
            if (PositionGetString(POSITION_SYMBOL) == _Symbol) {
                // Get the opening time of the position
                const datetime position_time = PositionGetInteger(POSITION_TIME);
                const ENUM_POSITION_TYPE position_type = PositionGetInteger(POSITION_TYPE);
                // Check if the position was opened after the user-specified time
                if (server_time - position_time > C_period_seconds) {
                    const double position_volume = PositionGetDouble(POSITION_VOLUME);
                    if (position_type == signal_type) {
                        UpdatePosition(_Symbol, price, position_ticket, position_volume, position_type, tp, sl);
                    } else {
                        // Close the position
                        ClosePositionSafe(_Symbol, price, position_ticket, position_volume, position_type);
                        closed_order = true;
                    }
                }
            }
        } else {
            LOG_ERROR("", "Failed to select position with ticket " + string(position_ticket));
        }
    }
    return closed_order;
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void OnTick()
{
    const datetime current_bar_time = iTime(_Symbol, _Period, 0);
    const datetime current_server_time = TimeCurrent();
    const datetime current_local_time = GetWindowsLocalTime();

    MqlDateTime current_server_time_st{};
    TimeToStruct(current_server_time, current_server_time_st);
    if (current_server_time_st.day_of_week >= Stop_Day && current_server_time_st.hour >= Stop_Hour) return;

    double anchor_price = 0;
    double predicted_price = 0;
    if (current_server_time >= current_bar_time + PeriodSeconds() - C_forecast_offset && current_bar_time >= pending_request_time) { // TODO double check, shouldn't it be, current_bar_time >= pending_request_time
        pending_request_time = current_bar_time + PeriodSeconds();
        controller.doRequest(net, pending_request_time, 1, C_dataset_id_str);
        LOG_DEBUG("", "Placed request for time " + TimeToString(pending_request_time, TIME_DATE_SECONDS) + " at server time " + TimeToString(current_server_time, TIME_DATE_SECONDS));

        TempusFigure results[];
        const datetime sleep_until = GetWindowsLocalTime() + Prediction_Wait;
        while ((!controller.getResults(net, pending_request_time, 1, C_dataset_id_str, results) || ArraySize(results) < 1) && GetWindowsLocalTime() < sleep_until) { }; // Sleep(100);
        for (int bar_ix = ArraySize(results) - 1; bar_ix >= 0; --bar_ix) {
            if (results[bar_ix].tm != pending_request_time) {
                LOG_DEBUG("", "Skipping result " + results[bar_ix].to_string() + " not for request made for " + string(pending_request_time));
                continue;
            }
            anchor_price = iClose(_Symbol, PERIOD_M1, iBarShift(_Symbol, PERIOD_M1, results[bar_ix].tm - C_forecast_offset - PeriodSeconds(PERIOD_M1), false));
            predicted_price = results[bar_ix].cl;
            LOG_INFO("", "Received " + results[bar_ix].to_string() + " of initial request made at " + TimeToString(pending_request_time, TIME_DATE_SECONDS));
            break;
        }
    }

    if (predicted_price > 0) {
        int signal;
        if (predicted_price > anchor_price) signal = POSITION_TYPE_BUY;
        else if (predicted_price < anchor_price) signal = POSITION_TYPE_SELL;
        else signal = (int) NOSIGNAL;
        
        MqlTick ticknow;
        ZeroMemory(ticknow);
        SymbolInfoTick(_Symbol, ticknow);
        const double price_ask = ticknow.ask; // SymbolInfoDouble(_Symbol, SYMBOL_ASK);
        const double price_bid = ticknow.bid; // SymbolInfoDouble(_Symbol, SYMBOL_BID);
        const double current_dev = indicator_value(iStdDev_handle, 0);
        double stoploss;
        if (signal == POSITION_TYPE_BUY) {
            stoploss = price_ask - current_dev * Stop_Loss_StdDev_Ratio;
            CheckOpenOrders(signal, price_ask, predicted_price, stoploss, current_server_time);
        } else {
            stoploss = price_bid + current_dev * Stop_Loss_StdDev_Ratio;
            CheckOpenOrders(signal, price_bid, predicted_price, stoploss, current_server_time);
        }
        const double price_spread = MathAbs(price_ask - price_bid);
        LOG_DEBUG("", "Predicted price is " + DoubleToString(predicted_price) + ", price bid is " + DoubleToString(price_bid) + ", price ask is " + 
                    DoubleToString(price_ask) + ", price spread " + DoubleToString(price_spread));
        if (current_dev >= StdDev_Threshold && (Only_Sells && signal == POSITION_TYPE_SELL) && price_spread < Spread_Threshold) {
            MqlTradeRequest request;
            ZeroMemory(request);
            MqlTradeResult result;
            ZeroMemory(result);
            request.action = TRADE_ACTION_DEAL;                     // Type of trade operation
            request.symbol = _Symbol;
            request.volume = Trade_Lot;
            request.deviation = Slippage;                           // Allowed deviation from the price
            request.magic = EXPERT_MAGIC + MathRand();              // MagicNumber of the order
            if (signal == POSITION_TYPE_BUY) {
                request.type = ORDER_TYPE_BUY;                      // order type
                request.price = price_ask;                          // SymbolInfoDouble(_Symbol, SYMBOL_ASK);   // price for opening
            } else if (signal == POSITION_TYPE_SELL) {
                request.type = ORDER_TYPE_SELL;                          // order type
                request.price = price_bid; // SymbolInfoDouble(_Symbol, SYMBOL_BID);   // price for opening
            }
            request.sl = stoploss;
            request.tp = predicted_price;
            //--- send the request
            if(!OrderSend(request, result))
                LOG_ERROR("", "OrderSend error " + string(_LastError));     // if unable to send the request, output the error code
            else
                LOG_DEBUG("", "Order sent, retcode " + string(result.retcode) + " deal " + string(result.deal) + ", order " + string(result.order));
        } else {
            LOG_DEBUG("", "Price movement too small " + DoubleToString(current_dev));
        }
    } else {

    }
}
//+------------------------------------------------------------------+
