//+------------------------------------------------------------------+
//|                                        tempus-market-connect.mq5 |
//|                                                       Jarko Asen |
//|                                                www.zarkoasen.com |
//+------------------------------------------------------------------+
#property copyright "Jarko Asen"
#property link      "www.zarkoasen.com"
#property version   "1.00"

//--- input parameters
input string    HTTP_URL = ""; // Use slower HTTP/S API for communication. Set to empty to disable HTTP/S communication.
input string    HTTP_User = "svrwave";
input string    HTTP_Password = "svrwave";
input bool      Use_Streams = true;  // Use Tempus Streaming messages protocol (faster)
input string    Adjacent_Symbol = "";
input double Horizon = .2; // partial of predicted time frame
input int Stop_Day = 7; // Stop trading after the N-th hour of the N-th day of the week, 0 is Sunday [0,6]
input int Stop_Hour = 24; // Last hour of last day of the week [0,23]
input bool Skip_Request = true;
input uint Dataset_ID = 100;
input double Max_Risk = .1; // Partial of running equity
input double Close_Periods = 1;
input uint Slippage = 3; // Allowable price slippage in points
input bool Make_Request = false; // Make a request to predict the coming period. The daemon can do this by itself, and it's faster that way.
input uint Prediction_Wait = 3; // Wait up to N seconds for a HTTP response
input double Initial_Wait = .3;
input uint StdDev_Period = 60; // Period in minutes to use for StdDev calculation
input float StdDev_Threshold = 0; // Start a position if StdDev is above this threshold
input float Stop_Loss_StdDev_Ratio = 6; // Ratio of StdDev to use as stop loss
input double Spread_Threshold = 1; // Start a trade only if spread is below threshold
input double Start_Distance = 1; // Start a position only if current price is X times Spread away from predicted price
input double Trade_Lot_Multiplier = 1; // Amount of current equity to invest to start a position



#include <tempus/constants.mqh>
#include <tempus/api.mqh>
#include <tempus/time_util.mqh>
#include <tempus/components.mqh>
#include <tempus/price.mqh>
#include <tempus/constants.mqh>


#define EXPERT_MAGIC 132435   // MagicNumber of the expert
#define ANCHOR_DIRECTION


const int C_retry_delay = 10; // ms
const string C_dataset_id_str = IntegerToString(Dataset_ID);
const string C_login_url = "tempus/login";
const string C_post_payload = "username=" + HTTP_User + "&password=" + HTTP_Password;
const int C_no_signal = (int) 0xDeadBeef;
const bool C_use_http = StringLen(HTTP_URL) > 0;
const uint C_forecast_offset = uint(C_period_seconds * Horizon);
const string C_input_queue_name = input_queue_name_from_symbol(_Symbol);
const int C_max_req_retries = C_backtesting ? 2 : 5;
const uint C_sleep_ms = C_backtesting ? 0 : 10;
const bool C_early_stop = Stop_Day < 7 && Stop_Hour < 24;
const double C_risk_leverage = Max_Risk / C_leverage;
const ulong C_instance_magic = EXPERT_MAGIC + MathRand();
const int C_initial_wait_ms = (int) round(Initial_Wait * 1000);
const int iStdDev_handle = iStdDev(_Symbol, PERIOD_M1, StdDev_Period, 0, MODE_SMA, PRICE_WEIGHTED);

datetime G_last_sent = 0, G_last_sent_1 = 0;
datetime G_local_time_of_last_request = 0, G_server_time_of_last_request = 0;

tempusapi *G_p_svr_client = NULL, *G_p_svr_client_1 = NULL;
streams_messaging *G_p_streams_msg = NULL;
MqlNet G_net_api;
tempus_controller G_controller;

const string C_server_queue_name = C_input_queue_name + input_queue_name_from_symbol(Adjacent_Symbol) + "_avg";

double indicator_value(const int handle, const int shift)
{
    double buffer[];
    while (CopyBuffer(handle, 0, shift, 1, buffer) != 1)
        LOG_ERROR("Cannot copy value " + IntegerToString(shift) + " from indicator handle " + IntegerToString(handle));
    if (ArraySize(buffer) < 1) {
        LOG_DEBUG("No values copied for indicator " + IntegerToString(handle) + ", shift " + IntegerToString(shift));
        return 0;
    }
    return buffer[0];
}


//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
    if (Use_Streams) {
        G_p_streams_msg = new streams_messaging(C_streams_root);
        if (!G_p_streams_msg) {
            LOG_ERROR("Streams initialization failed.");
            return INIT_FAILED;
        }
    }

    G_p_svr_client = new tempusapi(false, 0, true, Adjacent_Symbol, false, "", G_p_streams_msg, _Symbol, _Period, C_server_queue_name);
    G_p_svr_client_1 = new tempusapi(false, 0, true, Adjacent_Symbol, true, "", G_p_streams_msg, _Symbol, PERIOD_M1, C_server_queue_name); // Shortest period


#ifndef HISTORYTOSQL
    if (C_use_http && (!G_p_svr_client.connect(HTTP_URL) || !G_p_svr_client.login(HTTP_User, HTTP_Password) ||
         !G_p_svr_client_1.connect(HTTP_URL) || !G_p_svr_client_1.login(HTTP_User, HTTP_Password)))
        return INIT_PARAMETERS_INCORRECT;
#endif

    if (!G_p_svr_client.send_history()) return INIT_FAILED;
    if (!G_p_svr_client_1.send_history()) return INIT_FAILED;
    datetime stub_datetime;    
    G_p_svr_client.next_time_range(G_last_sent, stub_datetime);
    G_p_svr_client_1.next_time_range(G_last_sent_1, stub_datetime);

    EventSetTimer(1);

    return INIT_SUCCEEDED;
}


//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
//--- destroy timer
    EventKillTimer();

    if (G_p_svr_client) delete G_p_svr_client;
    if (G_p_svr_client_1) delete G_p_svr_client_1;
    if (G_p_streams_msg) delete G_p_streams_msg;
}

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
    request.deviation = Slippage;
    request.type_filling = ORDER_FILLING_IOC; // Immediate or Cancel
// Send the trade request
    if (!send_order_safe(request)) {
        LOG_ERROR("Failed to close position " + IntegerToString(position_ticket) + " on symbol " + symbol);
        return false;
    }

    return true;
}


//+------------------------------------------------------------------+
//|                                                                  |
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

#ifdef _DISABLE_ // Retest
                const double open_price = PositionGetDouble(POSITION_PRICE_OPEN);

                if ((position_type == POSITION_TYPE_BUY && current_price.ask > open_price + spread && tp.ask <= current_price.ask + spread) ||
                    (position_type == POSITION_TYPE_SELL && current_price.bid < open_price - spread && tp.bid >= current_price.bid - spread)) {
                    close_position(_Symbol, current_price, position_ticket, position_vol, position_type);
                } else if (
                    (position_type == POSITION_TYPE_BUY && (current_price.ask < tp.ask || open_price < tp.ask || current_price.ask > open_price + spread)) ||
                    (position_type == POSITION_TYPE_SELL && (current_price.bid > tp.bid || open_price > tp.bid || current_price.bid < open_price - spread))
                ) {
                    if (!price_equal(position_type == POSITION_TYPE_BUY ? tp.ask : tp.bid, PositionGetDouble(POSITION_TP)))
                        update_position(_Symbol, current_price, position_ticket, position_vol, position_type, tp);
                } else
                    close_position(_Symbol, current_price, position_ticket, position_vol, position_type);
#endif

            }
        } else
            LOG_ERROR("Failed to select position with ticket " + IntegerToString(position_ticket));
    }
}


MqlTick ticknow;
double price_spread = 0;

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void market_process()
{
    if (price_spread == 0) return;

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
    const bool do_predict = current_server_time >= next_bar_offset;

    static aprice predicted_price(0, 0);
#ifdef ANCHOR_DIRECTION
    static aprice anchor_price(0, 0);
#endif
    static datetime request_time = 0;
    if (predicted_price.valid() && current_server_time > request_time) predicted_price.reset();

    if (do_predict && current_bar_time >= request_time && predicted_price.valid() == false) {

        request_time = current_bar_time + C_period_seconds;
        
        const datetime last_complete_bar_time = iTime(_Symbol, _Period, 1);
        if (last_complete_bar_time > G_last_sent) {
            if (G_p_svr_client.send_history(G_last_sent, last_complete_bar_time)) {
               G_last_sent = last_complete_bar_time; 
            } else {
                LOG_ERROR("Failed sending bar data to queue.");
                return;
            }
        }
            
        if (current_server_time > G_last_sent_1) {
            if (G_p_svr_client_1.send_history(G_last_sent_1, current_server_time)) {
                G_last_sent_1 = current_server_time;
            } else {
                LOG_ERROR("Failed sending high resolution data to queue.");
                return;
            }
        }

        if (Make_Request && C_use_http) {
            int req_res, req_retries = 0;
            do req_res = G_controller.do_request(G_net_api, request_time, 1, C_dataset_id_str);
            while (req_res < 0 && ++req_retries < C_max_req_retries);
            if (req_res >= 0) LOG_VERBOSE("Successfully placed request for time " + TimeToString(request_time, C_time_mode) + 
                                            " at server time " + TimeToString(current_server_time, C_time_mode));
        }
        
        price_row results[];
        if (C_use_http) {
            const datetime sleep_until = get_windows_local_time() + Prediction_Wait;
            Sleep(C_initial_wait_ms);
            while ((!G_controller.get_results(G_net_api, request_time, 1, C_dataset_id_str, results) || ArraySize(results) < 1) && get_windows_local_time() < sleep_until)
                Sleep(C_retry_delay);
        }
        if (Use_Streams) G_p_streams_msg.get_results(Dataset_ID, request_time, 1, C_period_seconds, results);

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
        const double stddev = indicator_value(iStdDev_handle, 0);
        const aprice takeprofit(predicted_price.bid, predicted_price.ask);
        LOG_DEBUG("Take profit price is " + takeprofit.to_string() + ", current bid " + DoubleToString(ticknow.bid, DOUBLE_PRINT_DECIMALS) +
                  ", current ask " + DoubleToString(ticknow.ask, DOUBLE_PRINT_DECIMALS) + ", current spread " + DoubleToString(price_spread, DOUBLE_PRINT_DECIMALS));
        // Start new position
        if (stddev >= StdDev_Threshold &&
            price_spread < Spread_Threshold &&
            ((signal == POSITION_TYPE_BUY && takeprofit.ask - ticknow.ask > price_spread * Start_Distance) ||
             (signal == POSITION_TYPE_SELL && ticknow.bid - takeprofit.bid > price_spread * Start_Distance))) {
            const double dev_stop_loss = MathMin(stddev * Stop_Loss_StdDev_Ratio, C_risk_leverage * AccountInfoDouble(ACCOUNT_EQUITY));
            const aprice stoploss(ticknow.bid + dev_stop_loss, ticknow.ask - dev_stop_loss);
            MqlTradeRequest request;
            ZeroMemory(request);
            request.action = TRADE_ACTION_DEAL;                     // Type of trade operation
            request.symbol = _Symbol;
            request.volume = lot_size(Trade_Lot_Multiplier);
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
            LOG_DEBUG("Price movement too small " + DoubleToString(stddev, DOUBLE_PRINT_DECIMALS) +
                      " or spread too high " + DoubleToString(price_spread, DOUBLE_PRINT_DECIMALS) + ", signal " + IntegerToString(signal));
    }
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void OnTick()
{
    if (!get_current_tick(ticknow)) return;
    price_spread = ticknow.ask - ticknow.bid;
    market_process();
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void OnTimer()
{
    market_process();
}


//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void OnBookEvent(const string &symbol)
{
    market_process();
}

//+------------------------------------------------------------------+
