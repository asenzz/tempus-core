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
input double Trade_Lot_Multiplier = 1;
input ulong Slippage = 3;
input string Username = "svrwave"; // SVRWeb dataset user and password
input string Password = "svrwave";
input uint Dataset_ID = 100; // Dataset id
input double Horizon = .2; // partial of predicted time frame
input int Stop_Day = 6; // Stop trading after the N-th hour of the N-th day of the week, 0 is Sunday
input int Stop_Hour = 23;
input uint StdDev_Period = 60; // Period in minutes to use for StdDev calculation
input float StdDev_Threshold = 0; // Start a position if StdDev is above this threshold
input float Stop_Loss_StdDev_Ratio = 2; // Ratio of StdDev to use as stop loss
input float Initial_Wait = 1; // Initial wait for server to generate predictions, in seconds
input uint Prediction_Wait = 3; // Prediction delay, wait up to seconds including initial delay
input double Spread_Threshold = 1; // Start a trade only if spread is below threshold
input uint Leverage = 100;

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
const int C_retry_delay = 10; // ms
const int C_initial_wait_ms = (int) round(Initial_Wait * 1000);
const string C_dataset_id_str = string(Dataset_ID);
const bool Average = true; // Needed for tempus-constants.mqh
const string C_login_url = "mt4/login";
const string C_post_payload = "username=" + Username + "&password=" + Password;

#define EXPERT_MAGIC 132435   // MagicNumber of the expert
const int C_no_signal = (int) 0xDeadBeef;

#include <TempusGMT.mqh>
#include <TempusMVC.mqh>
#include <AveragePrice.mqh>
#include <tempus-constants.mqh>

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
const uint C_forecast_offset = uint(C_period_seconds * Horizon);
const string C_input_queue_name = input_queue_name_from_symbol(_Symbol);
const int C_max_req_retries = C_backtesting ? 2 : 5;
const uint C_max_retries = C_backtesting ? 2 : 10;
const uint C_sleep_ms = C_backtesting ? 0 : 10;
const double C_leverage_100 = Leverage * 100.;

MqlNet         net;

TempusController controller;
datetime G_pending_request_time = 0;

#define INIT_TEST

datetime G_local_time_of_last_request = 0;
datetime G_server_time_of_last_request = 0;
int iStdDev_handle = 0;

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
double indicator_value(const int handle, const int shift)
{
    double buffer[];
    while (CopyBuffer(handle, 0, shift, 1, buffer) != 1)
        LOG_ERROR("Cannot copy value" + IntegerToString(shift) + " from indicator handle " + IntegerToString(handle));
    if (ArraySize(buffer) < 1) {
        LOG_DEBUG("No values copied for indicator " + IntegerToString(handle) + ", shift " + IntegerToString(shift));
        return 0;
    }
    return buffer[0];
}



//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
double lot_size()
{
    return Trade_Lot_Multiplier * NormalizeDouble(AccountInfoDouble(ACCOUNT_EQUITY) / C_leverage_100, 2);
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int OnInit()
{
    LOG_INFO("Starting up ...");
    if(!net.Open(ServerUrl)) {
        Print("Incorrect server address");
        return INIT_PARAMETERS_INCORRECT;
    }
    long status_code = 0;
    net.Post(C_login_url, C_post_payload, status_code);
    if (status_code != 200) {
        LOG_ERROR("Authentication error occured " + IntegerToString(status_code) + " " + ErrorDescription(int(status_code)));
        return INIT_PARAMETERS_INCORRECT;
    }

    iStdDev_handle = iStdDev(_Symbol, PERIOD_M1, StdDev_Period /* PeriodSeconds() / 60 */, 0, MODE_SMA, PRICE_WEIGHTED);
// iATR_handle = iATR(_Symbol, PERIOD_M1, 60);

    if (!C_backtesting && !GlobalVariableCheck(C_one_minute_chart_identifier)) LOG_ERROR("One minute tempus data connector is not active!");

    controller.init(C_input_queue_name, C_period_seconds, 0, 0, 0, 0, 1);

    EventSetTimer(1);

    return INIT_SUCCEEDED;
}


//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    EventKillTimer();
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
    request.type_filling = ORDER_FILLING_IOC; // Immediate or Cancel
    request.price = price;
    request.tp = tp;
    request.sl = sl;
    request.deviation = Slippage; // Allowable price slippage in points

// Send the trade request
    if (!OrderSend(request, result)) {
        LOG_ERROR("Failed to update position " + IntegerToString(position_ticket) + " on symbol " + symbol +
                  ", OrderSend failed " + result.comment + " (retcode: " + IntegerToString(result.retcode) + ")");
        return false;
    }

    if (result.retcode != TRADE_RETCODE_DONE) {
        LOG_ERROR("Failed to update position " + IntegerToString(position_ticket) + " on symbol " + symbol +
                  ", trade failed " + result.comment + " (retcode: " + IntegerToString(result.retcode) + ")");
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
    while (++retries < C_max_retries) {
        if (UpdatePosition(symbol, price, position_ticket, volume, type, tp, sl)) break;
        if (!C_backtesting) Sleep(C_sleep_ms);
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
        LOG_ERROR("Failed to close position " + IntegerToString(position_ticket) + " on symbol " + symbol +
                  ", OrderSend failed " + result.comment + " (retcode: " + IntegerToString(result.retcode) + ")");
        return false;
    }

    if (result.retcode != TRADE_RETCODE_DONE) {
        LOG_ERROR("Failed to close position " + IntegerToString(position_ticket) + " on symbol " + symbol +
                  ", trade failed " + result.comment + " (retcode: " + IntegerToString(result.retcode) + ")");
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
    while (++retries < C_max_retries) {
        if (ClosePosition(symbol, price, position_ticket, volume, type)) break;
        Sleep(C_sleep_ms);
    }
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CheckOpenOrders(const int signal_type, const double price_bid, const double price_ask, const double tp, const double sl, const datetime server_time)
{
    const int total_positions = PositionsTotal();
    bool closed_order = false;
    for (int i = 0; i < total_positions; ++i) {
        const ulong position_ticket = PositionGetTicket(i);
        if (PositionSelectByTicket(position_ticket)) {
            if (PositionGetString(POSITION_SYMBOL) == _Symbol) {
                const datetime position_time = (datetime) PositionGetInteger(POSITION_TIME);
                if (server_time - position_time < C_period_seconds) continue;

                const ENUM_POSITION_TYPE position_type = (ENUM_POSITION_TYPE) PositionGetInteger(POSITION_TYPE);
                const double pos_price = position_type == POSITION_TYPE_BUY ? price_ask : price_bid;
                if (position_type == signal_type)
                    UpdatePositionSafe(_Symbol, pos_price, position_ticket, PositionGetDouble(POSITION_VOLUME), position_type, tp, sl);
                else
                    ClosePositionSafe(_Symbol, pos_price, position_ticket, PositionGetDouble(POSITION_VOLUME), position_type);
            }
        } else {
            LOG_ERROR("Failed to select position with ticket " + IntegerToString(position_ticket));
        }
    }
}


//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void OnTick()
{
    const datetime current_bar_time = iTime(_Symbol, _Period, 0);
    const datetime current_server_time = TimeCurrent();
    const datetime current_local_time = GetWindowsLocalTime();

    MqlDateTime current_server_time_st;
    TimeToStruct(current_server_time, current_server_time_st);
    if (current_server_time_st.day_of_week >= Stop_Day && current_server_time_st.hour >= Stop_Hour) return;
    const datetime next_bar_offset = current_bar_time + C_period_seconds - C_forecast_offset;
    aprice anchor_price, predicted_price;
    const bool make_request = C_backtesting ? current_server_time >= next_bar_offset : datetime(GlobalVariableGet(C_one_minute_chart_identifier)) >= next_bar_offset - C_period_m1_seconds;
    if (make_request && current_bar_time >= G_pending_request_time) {
        Sleep(100);// TODO Remove temporary fix
        G_pending_request_time = current_bar_time + C_period_seconds;
        int req_res, req_retries = 0;
        do req_res = controller.doRequest(net, G_pending_request_time, 1, C_dataset_id_str);
        while (req_res < 0 && ++req_retries < C_max_req_retries);

        if (req_res >= 0)
            LOG_DEBUG("Successfully placed request for time " + TimeToString(G_pending_request_time, C_time_mode) +
                      " at server time " + TimeToString(current_server_time, C_time_mode));

        TempusFigure results[];
        const datetime sleep_until = current_local_time + Prediction_Wait;
        if (!C_backtesting) Sleep(C_initial_wait_ms);
        while ((!controller.getResults(net, G_pending_request_time, 1, C_dataset_id_str, results) || ArraySize(results) < 1) && GetWindowsLocalTime() < sleep_until)
            if (!C_backtesting) Sleep(C_retry_delay);

        const int res_len = ArraySize(results);
        for (int bar_ix = 0; bar_ix < res_len; ++bar_ix) {
            if (results[bar_ix].tm != G_pending_request_time) {
                LOG_DEBUG("Skipping result " + results[bar_ix].to_string() + " not for request made for " + TimeToString(G_pending_request_time, C_time_mode));
                continue;
            }
            anchor_price = get_price(results[bar_ix].tm - C_forecast_offset - 1);
            predicted_price = results[bar_ix].cl;
            LOG_INFO("Received " + results[bar_ix].to_string() + " of initial request made at " + TimeToString(G_pending_request_time, C_time_mode));
            break;
        }
    }

    if (predicted_price.valid()) {
        int signal;
        if (predicted_price.ask > anchor_price.ask)
            signal = POSITION_TYPE_BUY;
        else if (predicted_price.bid < anchor_price.bid)
            signal = POSITION_TYPE_SELL;
        else
            signal = C_no_signal;

        MqlTick ticknow;
        ZeroMemory(ticknow);
        SymbolInfoTick(_Symbol, ticknow);
        const double price_ask = ticknow.ask;
        const double price_bid = ticknow.bid;
        const double current_dev = indicator_value(iStdDev_handle, 0);
        double stoploss, takeprofit;
        if (signal == POSITION_TYPE_BUY) {
            stoploss = price_ask - current_dev * Stop_Loss_StdDev_Ratio;
            takeprofit = predicted_price.ask;
        } else {
            stoploss = price_bid + current_dev * Stop_Loss_StdDev_Ratio;
            takeprofit = predicted_price.bid;
        }
        CheckOpenOrders(signal, price_bid, price_ask, takeprofit, stoploss, current_server_time);
        const double price_spread = MathAbs(price_ask - price_bid);
        LOG_DEBUG("Take profit price is " + DoubleToString(takeprofit, DOUBLE_PRINT_DECIMALS) + ", current price bid is " + DoubleToString(price_bid, DOUBLE_PRINT_DECIMALS) +
                  ", ask is " + DoubleToString(price_ask, DOUBLE_PRINT_DECIMALS) + ", current spread " + DoubleToString(price_spread, DOUBLE_PRINT_DECIMALS) + 
                  ", anchor price " + anchor_price.to_string());
        if (current_dev >= StdDev_Threshold && price_spread < Spread_Threshold && signal != C_no_signal) {
            MqlTradeRequest request;
            ZeroMemory(request);
            MqlTradeResult result;
            ZeroMemory(result);
            request.action = TRADE_ACTION_DEAL;                     // Type of trade operation
            request.symbol = _Symbol;
            request.volume = lot_size();
            request.deviation = Slippage;                           // Allowed deviation from the price
            request.magic = EXPERT_MAGIC + MathRand();              // MagicNumber of the order
            request.type_filling = ORDER_FILLING_IOC;
            if (signal == POSITION_TYPE_BUY) {
                request.type = ORDER_TYPE_BUY;                      // order type
                request.price = price_ask;                          // SymbolInfoDouble(_Symbol, SYMBOL_ASK);   // price for opening
            } else if (signal == POSITION_TYPE_SELL) {
                request.type = ORDER_TYPE_SELL;                     // order type
                request.price = price_bid;                          // price for opening
            }
            request.sl = stoploss;
            request.tp = takeprofit;
            //--- send the request
            if(!OrderSend(request, result))
                LOG_ERROR("OrderSend error " + IntegerToString(_LastError));     // if unable to send the request, output the error code
            else
                LOG_DEBUG("Order sent, retcode " + IntegerToString(result.retcode) + " deal " + IntegerToString(result.deal) + ", order " + IntegerToString(result.order));
        } else {
            LOG_DEBUG("Price movement too small " + DoubleToString(current_dev, DOUBLE_PRINT_DECIMALS) +
                      " or spread too high " + DoubleToString(price_spread, DOUBLE_PRINT_DECIMALS) + ", signal " + IntegerToString(signal));
        }
    }
}
//+------------------------------------------------------------------+

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void OnTimer()
{
    OnTick();
}
//+------------------------------------------------------------------+
