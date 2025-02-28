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
input int Stop_Day = 7; // Stop trading after the N-th hour of the N-th day of the week, 0 is Sunday [0,6]
input int Stop_Hour = 24; // Last hour of last day of the week [0,23]
input uint StdDev_Period = 60; // Period in minutes to use for StdDev calculation
input float StdDev_Threshold = 0; // Start a position if StdDev is above this threshold
input float Stop_Loss_StdDev_Ratio = 6; // Ratio of StdDev to use as stop loss
input float Initial_Wait = float(.3); // Initial wait for server to generate predictions, in seconds
input uint Prediction_Wait = 3; // Prediction delay, wait up to seconds including initial delay
input double Spread_Threshold = 1; // Start a trade only if spread is below threshold
input double Max_Risk = .1; // Maximum risk for stop loss
input double Start_Distance = 2; // Start a position when price is N times spread away from predicted price
input double Close_Periods = 1; // Force close a position N times current Period from opening time if positive. Risk increases the higher this coeficient is.
input bool Skip_Request = true;

#define EXPERT_MAGIC 132435   // MagicNumber of the expert
#define ANCHOR_DIRECTION

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

const int C_retry_delay = 10; // ms
const int C_initial_wait_ms = (int) round(Initial_Wait * 1000);
const string C_dataset_id_str = IntegerToString(Dataset_ID);
const bool Average = true; // Needed for tempus_constants.mqh
const string C_login_url = "tempus/login";
const string C_post_payload = "username=" + Username + "&password=" + Password;
const int C_no_signal = (int) 0xDeadBeef;

#include <tempus/time_util.mqh>
#include <tempus/components.mqh>
#include <tempus/price.mqh>
#include <tempus/constants.mqh>

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
const uint C_forecast_offset = uint(C_period_seconds * Horizon);
const string C_input_queue_name = input_queue_name_from_symbol(_Symbol);
const int C_max_req_retries = C_backtesting ? 2 : 5;
const uint C_sleep_ms = C_backtesting ? 0 : 10;
const double C_leverage_100 = C_leverage * 100.;
const bool C_early_stop = Stop_Day < 7 && Stop_Hour < 24;
const double C_risk_leverage = Max_Risk / C_leverage;
const ulong C_instance_magic = EXPERT_MAGIC + MathRand();

MqlNet         net;

tempus_controller controller;

#define INIT_TEST

datetime G_local_time_of_last_request = 0;
datetime G_server_time_of_last_request = 0;
int iStdDev_handle = 0;

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool get_current_tick(MqlTick &out)
{
    uint retries = 0;
    while (++retries < C_max_retries) if (SymbolInfoTick(_Symbol, out)) return true;
    LOG_ERROR("Failed getting current tick for symbol " + _Symbol);
    return false;
}

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

const double C_min_volume = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
const double C_max_volume = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
const double C_volume_step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
double lot_size()
{
    const double equity = AccountInfoDouble(ACCOUNT_EQUITY);
    double res = Trade_Lot_Multiplier * NormalizeDouble(equity / C_leverage_100, 2);
    res = MathRound(res / C_volume_step) * C_volume_step;
    if (res < C_min_volume) res = C_min_volume;
    else if (res > C_max_volume) res = C_max_volume;
    LOG_VERBOSE("Lot size " + DoubleToString(res, DOUBLE_PRINT_DECIMALS) + ", equity " + DoubleToString(equity, DOUBLE_PRINT_DECIMALS) + ", multiplier " + 
        DoubleToString(Trade_Lot_Multiplier, DOUBLE_PRINT_DECIMALS));
    return res;
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
bool update_position(const string &symbol, const MqlTick &price, const ulong position_ticket, const double volume, const ENUM_POSITION_TYPE type, const aprice &tp)
{
// Prepare the trade request
    MqlTradeRequest request;
    ZeroMemory(request);
    request.action = TRADE_ACTION_SLTP;
    request.symbol = symbol;
    request.volume = volume;
    request.position = position_ticket;
    request.type = type == POSITION_TYPE_BUY ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;
    request.type_filling = ORDER_FILLING_IOC; // Immediate or Cancel    
    if (type == POSITION_TYPE_BUY) {
        request.price = price.ask;
        request.tp = tp.ask;
        request.sl = PositionGetDouble(POSITION_SL);
    } else {
        request.price = price.bid;
        request.tp = tp.bid;
        request.sl = PositionGetDouble(POSITION_SL);
    }
    request.deviation = Slippage; // Allowable price slippage in points

// Send the trade request
    if (!send_order_safe(request)) {
        LOG_ERROR("Failed to update position " + IntegerToString(position_ticket) + " on symbol " + symbol);
        return false;
    }

    return true;
}


//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool close_position(const string &symbol, const MqlTick &price, const ulong position_ticket, const double volume, const ENUM_POSITION_TYPE type)
{
// Prepare the trade request
    MqlTradeRequest request;
    ZeroMemory(request);
    request.action = TRADE_ACTION_DEAL;
    request.symbol = symbol;
    request.volume = volume;
    request.position = position_ticket;
    if (type == POSITION_TYPE_BUY) {
        request.type = ORDER_TYPE_SELL;
        request.price = price.bid;
    } else {
        request.type = ORDER_TYPE_BUY;
        request.price = price.ask;
    }
    request.deviation = Slippage; // Allowable price slippage in points
    request.type_filling = ORDER_FILLING_IOC; // Immediate or Cancel
// Send the trade request
    if (!send_order_safe(request)) {
        LOG_ERROR("Failed to close position " + IntegerToString(position_ticket) + " on symbol " + symbol);
        return false;
    }

    return true;
}

//+------------------------------------------------------------------+
//| Update existing positions with a newly predicted price                                                                 |
//+------------------------------------------------------------------+
void update_positions(const MqlTick &current_price, const int signal, const aprice &tp, const double spread, const datetime server_time)
{
    static const datetime close_period = datetime(Close_Periods * C_period_seconds);
    const int total_positions = PositionsTotal();
    for (int i = 0; i < total_positions; ++i) {
        const ulong position_ticket = PositionGetTicket(i);
        if (PositionSelectByTicket(position_ticket)) {
            if (PositionGetString(POSITION_SYMBOL) == _Symbol) {
                if (server_time - datetime(PositionGetInteger(POSITION_TIME)) < close_period) continue;

                const double position_vol = PositionGetDouble(POSITION_VOLUME);
                const ENUM_POSITION_TYPE position_type = (ENUM_POSITION_TYPE) PositionGetInteger(POSITION_TYPE);
                
                close_position(_Symbol, current_price, position_ticket, position_vol, position_type);
                continue;
                
                const double open_price = PositionGetDouble(POSITION_PRICE_OPEN);
                
                if ((position_type == POSITION_TYPE_BUY && current_price.ask > open_price + spread && tp.ask <= current_price.ask + spread) ||
                    (position_type == POSITION_TYPE_SELL && current_price.bid < open_price - spread && tp.bid >= current_price.bid - spread)) 
                {
                    close_position(_Symbol, current_price, position_ticket, position_vol, position_type);
                } else if (
                    (position_type == POSITION_TYPE_BUY && (current_price.ask < tp.ask || open_price < tp.ask || current_price.ask > open_price + spread)) ||
                    (position_type == POSITION_TYPE_SELL && (current_price.bid > tp.bid || open_price > tp.bid || current_price.bid < open_price - spread))
                ) {
                    if (!price_equal(position_type == POSITION_TYPE_BUY ? tp.ask : tp.bid, PositionGetDouble(POSITION_TP)))
                        update_position(_Symbol, current_price, position_ticket, position_vol, position_type, tp);
                } else 
                    close_position(_Symbol, current_price, position_ticket, position_vol, position_type);
            }
        } else
            LOG_ERROR("Failed to select position with ticket " + IntegerToString(position_ticket));
    }
}


//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void OnTick()
{
    MqlTick ticknow;
    if (!get_current_tick(ticknow)) return;

    const double price_spread = ticknow.ask - ticknow.bid;
    const datetime current_server_time = TimeCurrent();
    static aprice last_prediction(0, 0);
    static int last_signal = C_no_signal;
    if (last_prediction.valid()) update_positions(ticknow, last_signal, last_prediction, price_spread, current_server_time);
    
    if (C_early_stop) {
        MqlDateTime current_server_time_st;
        TimeToStruct(current_server_time, current_server_time_st);
        if (current_server_time_st.day_of_week >= Stop_Day && current_server_time_st.hour >= Stop_Hour) return;
    }

    const datetime current_bar_time = iTime(_Symbol, _Period, 0);
    static const datetime offset = C_period_seconds - C_forecast_offset;
    const datetime next_bar_offset = current_bar_time + offset;
    const bool make_request = C_backtesting || Skip_Request ?
                              current_server_time >= next_bar_offset:
                              datetime(GlobalVariableGet(C_one_minute_chart_identifier)) >= next_bar_offset - C_period_m1_seconds;
    static aprice predicted_price(0, 0);
#ifdef ANCHOR_DIRECTION
    static aprice anchor_price(0, 0);
#endif
    static datetime request_time = 0;
    if (predicted_price.valid() && current_server_time > request_time) predicted_price.reset();

    if (make_request && current_bar_time >= request_time && predicted_price.valid() == false) {
        request_time = current_bar_time + C_period_seconds;
        if (Skip_Request == false) {
            int req_res, req_retries = 0;
            do req_res = controller.do_request(net, request_time, 1, C_dataset_id_str);
            while (req_res < 0 && ++req_retries < C_max_req_retries);
            if (req_res >= 0) LOG_VERBOSE("Successfully placed request for time " + TimeToString(request_time, C_time_mode) +
                                            " at server time " + TimeToString(current_server_time, C_time_mode));
        }
        price_row results[];
        const datetime sleep_until = get_windows_local_time() + Prediction_Wait;
        Sleep(C_initial_wait_ms);
        while ((!controller.get_results(net, request_time, 1, C_dataset_id_str, results) || ArraySize(results) < 1) && get_windows_local_time() < sleep_until)
            Sleep(C_retry_delay);

        const int res_len = ArraySize(results);
        for (int bar_ix = 0; bar_ix < res_len; ++bar_ix) {
            if (results[bar_ix].tm != request_time) {
                LOG_DEBUG("Skipping result " + results[bar_ix].to_string() + " not for request made for " + TimeToString(request_time, C_time_mode));
                continue;
            }
#ifdef ANCHOR_DIRECTION
            static const datetime forecast_offset_1 = C_forecast_offset - 1;
            anchor_price = get_price(results[bar_ix].tm - forecast_offset_1);
#endif
            predicted_price = results[bar_ix].cl;
            last_prediction = predicted_price;
            LOG_INFO("Received " + results[bar_ix].to_string() + " of initial request made at " + TimeToString(request_time, C_time_mode));
            break;
        }
    }

    if (predicted_price.valid()) {
        int signal;
#ifdef ANCHOR_DIRECTION
        const double buy_move = predicted_price.ask - anchor_price.ask;
        const double sell_move = anchor_price.bid - predicted_price.bid;
#else
        const double buy_move = predicted_price.ask - ticknow.ask;
        const double sell_move = ticknow.bid - predicted_price.bid;
#endif
        if (buy_move <= 0 && sell_move <= 0)
            signal = C_no_signal; // Spread tightened no signal
        else if (buy_move > sell_move)
            signal = POSITION_TYPE_BUY;
        else
            signal = POSITION_TYPE_SELL;
        last_signal = signal;
        const double current_dev = indicator_value(iStdDev_handle, 0);
        const aprice takeprofit(predicted_price.bid, predicted_price.ask);
        LOG_DEBUG("Take profit price is " + takeprofit.to_string() + ", current bid " + DoubleToString(ticknow.bid, DOUBLE_PRINT_DECIMALS) +
                  ", current ask " + DoubleToString(ticknow.ask, DOUBLE_PRINT_DECIMALS) + ", current spread " + DoubleToString(price_spread, DOUBLE_PRINT_DECIMALS));
        // Start new position
        if (current_dev >= StdDev_Threshold && 
            price_spread < Spread_Threshold &&
            ((signal == POSITION_TYPE_BUY && takeprofit.ask - ticknow.ask > price_spread * Start_Distance) ||
             (signal == POSITION_TYPE_SELL && ticknow.bid - takeprofit.bid > price_spread * Start_Distance))) {
            const double dev_stop_loss = MathMin(current_dev * Stop_Loss_StdDev_Ratio, C_risk_leverage * AccountInfoDouble(ACCOUNT_EQUITY));
            const aprice stoploss(ticknow.bid + dev_stop_loss, ticknow.ask - dev_stop_loss);
            MqlTradeRequest request;
            ZeroMemory(request);
            request.action = TRADE_ACTION_DEAL;                     // Type of trade operation
            request.symbol = _Symbol;
            request.volume = lot_size();
            request.deviation = Slippage;                           // Allowed deviation from the price
            request.magic = C_instance_magic;                       // MagicNumber of the order
            request.type_filling = ORDER_FILLING_IOC;
            if (signal == POSITION_TYPE_BUY) {
                request.type = ORDER_TYPE_BUY;                      // order type
                request.price = ticknow.ask;                        // SymbolInfoDouble(_Symbol, SYMBOL_ASK);   // price for opening
                request.sl = stoploss.ask;
                request.tp = takeprofit.ask;
            } else {                                                // if (signal == POSITION_TYPE_SELL)
                request.type = ORDER_TYPE_SELL;                     // order type
                request.price = ticknow.bid;                        // price for opening
                request.sl = stoploss.bid;
                request.tp = takeprofit.bid;
            }
            send_order_safe(request);
#ifdef ANCHOR_DIRECTION
            anchor_price.reset();
#endif
            predicted_price.reset();
        } else
            LOG_DEBUG("Price movement too small " + DoubleToString(current_dev, DOUBLE_PRINT_DECIMALS) +
                      " or spread too high " + DoubleToString(price_spread, DOUBLE_PRINT_DECIMALS) + ", signal " + IntegerToString(signal));
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
