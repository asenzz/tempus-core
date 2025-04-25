//+------------------------------------------------------------------+
//|                                                  retracement.mq5 |
//|                                                       Jarko Asen |
//|                                                www.zarkoasen.com |
//+------------------------------------------------------------------+

#define ANCHOR_DIRECTION
#define EXPERT_MAGIC 132435   // MagicNumber of the expert


#include <tempus/time_util.mqh>
#include <tempus/components.mqh>
#include <tempus/price.mqh>
#include <tempus/constants.mqh>
#include <tempus/request_util.mqh>
#include <tempus/types.mqh>


//--- input parameters
input double Horizon = .2; // partial of predicted time frame
input double Trade_Lot_Multiplier = 1;
input double Start_Distance = 2; // Start a position when price is N times spread away from predicted price
input double Coef_Spread_Take_Profit = 1;
input ulong Slippage = 3;
input int Stop_Day = 7; // Stop trading after the N-th hour of the N-th day of the week, 0 is Sunday [0,6]
input int Stop_Hour = 24; // Last hour of last day of the week [0,23]
input uint StdDev_Period = 60; // Period in minutes to use for StdDev calculation
input double StdDev_Threshold = 0; // Start a position if StdDev is above this threshold
input double Max_Risk = .1; // Maximum risk for stop loss
input double Close_Periods = 1; // Force close a position N times current Period from opening time if positive. Risk increases the higher this coeficient is.
input double Spread_Threshold = 1; // Start a trade only if spread is below threshold
input double Stop_Loss_StdDev_Ratio = 6; // Ratio of StdDev to use as stop loss
input int MA_Period = 60; // MA period in minutes
input int ADX_Period = 180; // ADX MA period in minutes
input double Strength_Threshold = 30;
input double Trailing_Grid_Width = 1; // Grid width for trailing stop [1..10], 0 means no trailing stop

const bool C_early_stop = Stop_Day < 7 && Stop_Hour < 24;
const uint C_forecast_offset = uint(C_period_seconds * Horizon);
const double C_risk_leverage = Max_Risk / C_leverage;
const int C_no_signal = (int) 0xDeadBeef;
int G_iStdDev_handle = 0;
int G_iMA_handle = 0;
int G_iADX_handle = 0;
const ulong C_instance_magic = EXPERT_MAGIC + MathRand();


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
double get_indicator_value(const int handle, const int shift)
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
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
//--- create timer
    EventSetTimer(1);

//---

    G_iStdDev_handle = iStdDev(_Symbol, PERIOD_M1, StdDev_Period, 0, MODE_SMA, PRICE_WEIGHTED);
    G_iMA_handle = iMA(_Symbol, PERIOD_M1, MA_Period, 0, MODE_SMA, PRICE_WEIGHTED);
    G_iADX_handle = iADX(_Symbol, PERIOD_M1, ADX_Period);
    return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool update_position_tp(const string &symbol, const MqlTick &price, const ulong position_ticket, const double volume, const ENUM_POSITION_TYPE type, const aprice &tp)
{
// Prepare the trade request
    MqlTradeRequest request;
    ZeroMemory(request);
    request.action = TRADE_ACTION_SLTP;
    request.symbol = symbol;
    request.volume = volume;
    request.position = position_ticket;
    request.type_filling = ORDER_FILLING_IOC; // Immediate or Cancel
    request.deviation = Slippage; // Allowable price slippage in points
    if (type == POSITION_TYPE_BUY) {
        request.type = ORDER_TYPE_BUY;
        request.price = price.ask;
        request.tp = tp.ask;
        request.sl = PositionGetDouble(POSITION_SL);
    } else {
        request.type = ORDER_TYPE_SELL;
        request.price = price.bid;
        request.tp = tp.bid;
        request.sl = PositionGetDouble(POSITION_SL);
    }

// Send the trade request
    if (!send_order_safe(request)) {
        LOG_ERROR("Failed to update position " + IntegerToString(position_ticket) + " on symbol " + symbol);
        return false;
    }

    return true;
}

bool update_position_sl(const string &symbol, const MqlTick &price, const ulong position_ticket, const double volume, const ENUM_POSITION_TYPE type, const double sl)
{
// Prepare the trade request
    MqlTradeRequest request;
    ZeroMemory(request);
    request.action = TRADE_ACTION_SLTP;
    request.symbol = symbol;
    request.volume = volume;
    request.position = position_ticket;
    request.type_filling = ORDER_FILLING_IOC; // Immediate or Cancel
    request.tp = PositionGetDouble(POSITION_TP);
    request.sl = sl;
    request.deviation = Slippage; // Allowable price slippage in points
    if (type == POSITION_TYPE_BUY) {
        request.type = ORDER_TYPE_BUY;
        request.price = price.ask;
    } else {
        request.type = ORDER_TYPE_SELL;
        request.price = price.bid;
    }
    
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


double get_best_stop_loss(const ENUM_POSITION_TYPE position_type, const MqlTick &current_price, const double open_price)
{
    if (position_type == POSITION_TYPE_BUY && current_price.ask > open_price)
        return current_price.ask + MathFloor((current_price.ask - open_price) / Trailing_Grid_Width) * Trailing_Grid_Width;
    else if (position_type == POSITION_TYPE_SELL && current_price.bid < open_price)
        return current_price.bid - MathFloor((open_price - current_price.bid) / Trailing_Grid_Width) * Trailing_Grid_Width;
    else 
        return 0;
}        


//+------------------------------------------------------------------+
//| Update existing positions with a newly predicted price                                                                 |
//+------------------------------------------------------------------+
int update_positions(const MqlTick &current_price, const int signal, const aprice &tp, const double spread, const datetime server_time)
{
    static const datetime close_period = datetime(Close_Periods * C_period_seconds);
    const int total_positions = PositionsTotal();
    for (int i = 0; i < total_positions; ++i) {
        const ulong position_ticket = PositionGetTicket(i);
        if (PositionSelectByTicket(position_ticket)) {
            if (PositionGetString(POSITION_SYMBOL) == _Symbol) {
                const ENUM_POSITION_TYPE position_type = (ENUM_POSITION_TYPE) PositionGetInteger(POSITION_TYPE);
                const double position_vol = PositionGetDouble(POSITION_VOLUME);
                if (Trailing_Grid_Width > 0) {
                    const double open_price = PositionGetDouble(POSITION_PRICE_OPEN); // Move two lines above if managing TP
                    const double best_sl = get_best_stop_loss(position_type, current_price, open_price);
                    if (best_sl > 0) {
                        const double sl = PositionGetDouble(POSITION_SL);
                        if ((position_type == POSITION_TYPE_BUY && price_greater_than(best_sl, sl)) || 
                            (position_type == POSITION_TYPE_SELL && price_less_than(best_sl, sl)))
                            update_position_sl(_Symbol, current_price, position_ticket, position_vol, position_type, best_sl);
                    }
                }
                
                if (server_time - datetime(PositionGetInteger(POSITION_TIME)) < close_period) continue;
                close_position(_Symbol, current_price, position_ticket, position_vol, position_type);
                
#ifdef _DISABLED_
                if ((position_type == POSITION_TYPE_BUY && current_price.ask > open_price + spread && tp.ask <= current_price.ask + spread) ||
                    (position_type == POSITION_TYPE_SELL && current_price.bid < open_price - spread && tp.bid >= current_price.bid - spread)) {
                    close_position(_Symbol, current_price, position_ticket, position_vol, position_type);
                } else if (
                    (position_type == POSITION_TYPE_BUY && (current_price.ask < tp.ask || open_price < tp.ask || current_price.ask > open_price + spread)) ||
                    (position_type == POSITION_TYPE_SELL && (current_price.bid > tp.bid || open_price > tp.bid || current_price.bid < open_price - spread))
                ) {
                    if (!price_equal(position_type == POSITION_TYPE_BUY ? tp.ask : tp.bid, PositionGetDouble(POSITION_TP)))
                        update_position_tp(_Symbol, current_price, position_ticket, position_vol, position_type, tp);
                } else
                    close_position(_Symbol, current_price, position_ticket, position_vol, position_type);
#endif
            }
        } else
            LOG_ERROR("Failed to select position with ticket " + IntegerToString(position_ticket));
    }
    return total_positions;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
//--- destroy timer
    EventKillTimer();

}
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
//---
    MqlTick ticknow;
    if (!get_current_tick(ticknow)) return;

    const double price_spread = ticknow.ask - ticknow.bid;
    const datetime current_server_time = TimeCurrent();
    static aprice last_prediction(0, 0);
    static int last_signal = C_no_signal;
    int total_positions = C_no_signal;
    if (last_prediction.valid()) total_positions = update_positions(ticknow, last_signal, last_prediction, price_spread, current_server_time);

    if (C_early_stop) {
        MqlDateTime current_server_time_st;
        TimeToStruct(current_server_time, current_server_time_st);
        if (current_server_time_st.day_of_week >= Stop_Day && current_server_time_st.hour >= Stop_Hour) return;
    }

    const datetime current_bar_time = iTime(_Symbol, _Period, 0);
    static const datetime offset = C_period_seconds - C_forecast_offset;
    const datetime next_bar_offset = current_bar_time + offset;
    const bool make_request = current_server_time >= next_bar_offset;
    static aprice predicted_price(0, 0);
#ifdef ANCHOR_DIRECTION
    static aprice anchor_price(0, 0);
#endif
    static datetime request_time = 0;
    if (predicted_price.valid() && current_server_time > request_time) predicted_price.reset();

    if (make_request && current_bar_time >= request_time && predicted_price.valid() == false) {
        const double strength = get_indicator_value(G_iADX_handle, 0);
        if (strength > Strength_Threshold) return;
        
        request_time = current_bar_time + C_period_seconds;
        const double ma_bid = get_indicator_value(G_iMA_handle, 0);
        const double ma_ask = ma_bid + price_spread;
        predicted_price.set(ma_bid, ma_ask);
        last_prediction = predicted_price;
#ifdef ANCHOR_DIRECTION
        static const datetime forecast_offset_1 = C_forecast_offset - 1;
        anchor_price = get_price(current_server_time - forecast_offset_1);
#endif
        LOG_INFO("Received " + predicted_price.to_string() + " of initial request made at " + TimeToString(request_time, C_time_mode));
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
        const double current_dev = 1e3 * get_indicator_value(G_iStdDev_handle, 0) / ticknow.bid;
        const aprice takeprofit(predicted_price.bid, predicted_price.ask);
        LOG_DEBUG("Take profit price is " + takeprofit.to_string() + ", current bid " + DoubleToString(ticknow.bid, DOUBLE_PRINT_DECIMALS) +
                  ", current ask " + DoubleToString(ticknow.ask, DOUBLE_PRINT_DECIMALS) + ", current spread " + DoubleToString(price_spread, DOUBLE_PRINT_DECIMALS));
        if (total_positions == C_no_signal) total_positions = PositionsTotal();
        // Start new position
        if (total_positions < 100 &&
            current_dev >= StdDev_Threshold &&
            price_spread < Spread_Threshold &&
            ((signal == POSITION_TYPE_BUY && takeprofit.ask - ticknow.ask > price_spread * Start_Distance) || 
             (signal == POSITION_TYPE_SELL && ticknow.bid - takeprofit.bid > price_spread * Start_Distance))
             ) {
            const double dev_stop_loss = MathMin(current_dev * Stop_Loss_StdDev_Ratio, C_risk_leverage * AccountInfoDouble(ACCOUNT_EQUITY));
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
            LOG_DEBUG("Price movement too small " + DoubleToString(current_dev, DOUBLE_PRINT_DECIMALS) +
                      " or spread too high " + DoubleToString(price_spread, DOUBLE_PRINT_DECIMALS) + ", signal " + IntegerToString(signal));
    }

}
//+------------------------------------------------------------------+
//| Timer function                                                   |
//+------------------------------------------------------------------+
void OnTimer()
{
//---
    OnTick();
}
//+------------------------------------------------------------------+
//| BookEvent function                                               |
//+------------------------------------------------------------------+
void OnBookEvent(const string &symbol)
{
//---
    OnTick();
}
//+------------------------------------------------------------------+

double OnTester()
{
    const double eq_drawdown = TesterStatistics(STAT_EQUITY_DDREL_PERCENT);
    const double eq_drawdown_score = eq_drawdown > .01 ? 1. / eq_drawdown : 0;
    
    const double bn_drawdown = TesterStatistics(STAT_BALANCE_DDREL_PERCENT);
    const double bn_drawdown_score = bn_drawdown > .01 ? 1. / bn_drawdown : 0;

    const double profit = TesterStatistics(STAT_PROFIT);
    const double sign = profit < 0 ? -1e-3 : 1e-3;
    
    const double trades = TesterStatistics(STAT_TRADES);
    
    return sign * MathPow(MathAbs(profit), 2.9) * MathPow(eq_drawdown_score, .4) * MathPow(bn_drawdown_score, .4); // * MathPow(trades, .2);
}
