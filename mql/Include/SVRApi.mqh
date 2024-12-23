//+------------------------------------------------------------------+
//|                                                       SVRApi.mqh |
//|                        Copyright 2015, MetaQuotes Software Corp. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2015, MetaQuotes Software Corp."
#property link      "https://www.mql5.com"
#property version   "1.00"
#property strict

#include <hash.mqh>
#include <MqlNet.mqh>
#include <json.mqh>
#include <RequestUtils.mqh>
#include <TempusMVC.mqh>
#include <TempusGMT.mqh>
#include <AveragePrice.mqh>
#include <DSTPeriods.mqh>
#include <tempus-constants.mqh>

#define MAX_PACK_SIZE 0xFFFF
const uint C_retries_count = 10;

// Used for backtests, sleeps using actual local time
void sleep_local(const uint sleep)
{
    const datetime sleep_until = TimeLocal() + sleep;
    while (GetWindowsLocalTime() < sleep_until) {};
}


//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
datetime nearestBarTimeOrBefore(const datetime bar_time)
{
    LOG_INFO("", "Starting " + TimeToString(bar_time, C_time_mode));
    datetime barTimeIter = datetime(bar_time / C_period_seconds) * C_period_seconds;
    int barShift = iBarShift(_Symbol, _Period, barTimeIter, false);
    while (barShift == -1 || iTime(_Symbol, _Period, barShift) > bar_time) {
        barTimeIter -= C_period_seconds;
        barShift = iBarShift(_Symbol, _Period, barTimeIter, false);
    }
    const datetime result = iTime(_Symbol, _Period, barShift);
    LOG_INFO("", "Returning " + TimeToString(result, C_time_mode) + " for bar time " + TimeToString(bar_time, C_time_mode));
    return result;
}


//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class SVRApi
{
    MqlNet           net;
    RequestUtils     requestUtils;
    bool             histDataReconciled;
    uint const       packSize;
    string           AuxInputQueue;
    string           time_zone_info;
    bool             per_second;

    aprice           myHi[], myLo[], myOp[], myCl[], myAvg[], myAuxAvg[];
    uint             myVl[];
    datetime         myTm[];

    bool             DemoMode;
    int              DemoBars;
    bool             is_average;

    bool             send_history(const string &input_queue, const string &period, const datetime start_time, datetime end_time);

    int              prepare_history_avg(const datetime start_time, const datetime end_time);

    int              prepare_history_ohlc(const datetime start_time, const datetime end_time);

    uint             pack_history_ohlc(const uint bar_start, const uint bar_end, const string &delimiter, Hash &hash);

    uint             pack_history_avg(const uint bar_start, const uint bar_end, const string &delimiter, Hash &hash);

    int              findBar(const datetime start_time);

    int              bSearchBar(datetime start_time, uint left, uint right);

    bool             get_missing_period(datetime& time_from, datetime& time_to, const string &input_queue, const string &period, const string &function);

public:
                     SVRApi(const bool DemoMode_, const int DemoBars_, const bool isAverage_, const string AuxInputQueue_, const bool per_second, const string timeZoneInfo_);

                    ~SVRApi();

    bool             next_time_range(datetime& bar_start, datetime& bar_end, const string &input_queue, const string &period);
    //bool ReconcileHistoricalData(string input_queue, string period);

    bool             Connect(const string &URL);

    bool             Login(const string &Username, const string &Password);

    void             send_bar(const string &input_queue, const string &period, const bool finalize, const aprice &op, const aprice &hi, const aprice &lo, const aprice &cl, const uint vol, const datetime tm);
    void             send_bar(const string &input_queue, const string &period, const bool finalize, const AveragePrice &avg_price);
    void             send_bar(const string &input_queue, const string &period, const bool finalize, const AveragePrice &avg_price, const AveragePrice &avg_aux_price, const string &col_name, const string &col_aux_name);
    void             send_bars(const string &input_queue, const string &period, const bool finalize, const aprice &prices[], const uint &volumes[], const aprice &pricesaux[], const uint &volumesaux[], const datetime &times[], const string &col_name, const string &aux_col_name);
    void             send_bars(const string &input_queue, const string &period, const bool finalize, const aprice &prices[], const datetime &times[], const uint &volumes[]);
    bool             send_history(const string &input_queue, const string &period);

    static int       copy_rates_safe(const string &symbol, const ENUM_TIMEFRAMES time_frame, const datetime start, const datetime end, MqlRates &rates[]);
    static int       copy_ticks_safe(const string &symbol, MqlTick &ticks[], const uint flags, const datetime time_start, const datetime time_end);
    bool             history_reconciled() const;
};


//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool SVRApi::history_reconciled() const
{
    return histDataReconciled;
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
SVRApi::SVRApi(const bool DemoMode_, const int DemoBars_, const bool isAverage_, const string AuxInputQueue_ = "", const bool OneSecondData_ = false, const string timeZoneInfo_ = "") :
    histDataReconciled(false), packSize(MAX_PACK_SIZE), DemoMode(DemoMode_), DemoBars(DemoBars_), is_average(isAverage_), AuxInputQueue(AuxInputQueue_), per_second(OneSecondData_), time_zone_info(timeZoneInfo_)
{
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
SVRApi::~SVRApi()
{
}
//+------------------------------------------------------------------+


//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool SVRApi::Connect(const string &Url)
{
    return net.Open(Url);
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool SVRApi::Login(const string &username, const string &password)
{
    static const string LoginUrl = "mt4/login";
    const string PostBody = "username=" + username + "&password=" + password;
    long StatusCode;
    net.Post(LoginUrl, PostBody, StatusCode);
    return StatusCode == 200;
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void SVRApi::send_bar(
    const string &input_queue, const string &period, const bool finalize, const aprice &op, const aprice &hi, const aprice &lo, const aprice &cl, const uint vol, const datetime tm)
{
    string response;
    Hash param;

    const string time_str = TimeToString(tm, C_time_mode);
    param.hPutString("symbol", input_queue);
    param.hPutString("time", time_str);
    param.hPutString("open_bid", DoubleToString(op.bid, DOUBLE_PRINT_DECIMALS));
    param.hPutString("open_ask", DoubleToString(op.ask, DOUBLE_PRINT_DECIMALS));
    param.hPutString("high_bid", DoubleToString(hi.bid, DOUBLE_PRINT_DECIMALS));
    param.hPutString("high_ask", DoubleToString(hi.ask, DOUBLE_PRINT_DECIMALS));
    param.hPutString("low_bid", DoubleToString(lo.bid, DOUBLE_PRINT_DECIMALS));
    param.hPutString("low_ask", DoubleToString(lo.ask, DOUBLE_PRINT_DECIMALS));
    param.hPutString("close_bid", DoubleToString(cl.bid, DOUBLE_PRINT_DECIMALS));
    param.hPutString("close_ask", DoubleToString(cl.ask, DOUBLE_PRINT_DECIMALS));
    param.hPutString("Volume", IntegerToString(vol));
    param.hPutString("isFinal", finalize ? "true" : "false");
    param.hPutString("period", period);
    param.hPutString("clientTime", TimeToString(TimeLocal(), C_time_mode));
    static const string queue_call = "queue";
    static const string send_tick_call = "sendTick";
    if (!net.RpcCall(queue_call, send_tick_call, param, response)) return;

    if (!requestUtils.isSuccessfulResponse(response)) LOG_ERROR("", "Server returned: " + requestUtils.getErrorMessage(response));
#ifdef DEBUG_CONNECTOR
    LOG_DEBUG("", "Finalized bar at " + time_str);
#endif
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void SVRApi::send_bar(const string &input_queue, const string &period, const bool finalize, const AveragePrice &avg_price)
{
    string response;
    Hash param;

    uint bar = 0;
    const string time_str = TimeToString(avg_price.tm, C_time_mode);
    param.hPutString("symbol", input_queue);
    param.hPutString("time", time_str);
    param.hPutString(input_queue + "_bid", DoubleToString(avg_price.value.bid, DOUBLE_PRINT_DECIMALS));
    param.hPutString(input_queue + "_ask", DoubleToString(avg_price.value.ask, DOUBLE_PRINT_DECIMALS));
    param.hPutString("Volume", IntegerToString(avg_price.volume));
    param.hPutString("isFinal", finalize ? "true" : "false");
    param.hPutString("period", period);
    param.hPutString("clientTime", TimeToString(TimeCurrent(), C_time_mode));

    static const string queue_call = "queue";
    static const string send_tick_call = "sendTick";
    if (!net.RpcCall(queue_call, send_tick_call, param, response)) return;
    if (!requestUtils.isSuccessfulResponse(response)) LOG_ERROR("", "Server returned " + requestUtils.getErrorMessage(response));
#ifdef DEBUG_CONNECTOR
    LOG_INFO("", "Finalized bar at " + time_str + ", price " + avg_price.value.to_string());
#endif
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void
SVRApi::send_bar(
    const string &input_queue,
    const string &period,
    const bool finalize,
    const AveragePrice &avg_price,
    const AveragePrice &avg_price_aux,
    const string &col_name,
    const string &aux_col_name)
{
    string response;
    Hash param;

    const string time_str = TimeToString(avg_price.tm, C_time_mode);
    param.hPutString("symbol", input_queue);
    param.hPutString("time", time_str);
    param.hPutString(col_name + "_bid", DoubleToString(avg_price.value.bid, DOUBLE_PRINT_DECIMALS));
    param.hPutString(col_name + "_ask", DoubleToString(avg_price.value.ask, DOUBLE_PRINT_DECIMALS));
    param.hPutString(aux_col_name + "_bid", DoubleToString(avg_price_aux.value.bid, DOUBLE_PRINT_DECIMALS));
    param.hPutString(aux_col_name + "_ask", DoubleToString(avg_price_aux.value.ask, DOUBLE_PRINT_DECIMALS));
    param.hPutString("Volume", IntegerToString(avg_price.volume));
    param.hPutString("isFinal", finalize ? "true" : "false");
    param.hPutString("period", period);
    param.hPutString("clientTime", TimeToString(TimeCurrent(), C_time_mode));

    static const string queue_call = "queue";
    static const string send_tick_call = "sendTick";
    if (!net.RpcCall(queue_call, send_tick_call, param, response)) return;

    if (!requestUtils.isSuccessfulResponse(response)) LOG_ERROR("", "Server returned " + requestUtils.getErrorMessage(response));
#ifdef DEBUG_CONNECTOR
    LOG_INFO("", "Finalized bar at: " + time_str);
#endif
}


//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void SVRApi::send_bars(const string &input_queue, const string &period, const bool finalize, const aprice &prices[], const datetime &times[], const uint &volumes[])
{
    const int price_len = ArraySize(prices);
    if (price_len != ArraySize(times)) {
        LOG_ERROR("", "Prices length " + string(price_len) + " does not equal times length " + string(ArraySize(times)));
        return;
    }
    const string column_bid = input_queue + "_bid";
    const string column_ask = input_queue + "_ask";
    Hash param;
    param.hPutString("period", period);
    param.hPutString("clientTime", TimeToString(TimeLocal(), C_time_mode));
    param.hPutString("symbol", input_queue);
    param.hPutString("isFinal", finalize ? "true" : "false");
    for (int i = 0; i < price_len; ++i) {
        param.hPutString("time", TimeToString(times[i], C_time_mode));
        param.hPutString(column_bid, DoubleToString(prices[i].bid, DOUBLE_PRINT_DECIMALS));
        param.hPutString(column_ask, DoubleToString(prices[i].ask, DOUBLE_PRINT_DECIMALS));
        param.hPutString("Volume", IntegerToString(volumes[i]));
        static const string queue_call = "queue";
        static const string send_tick = "sendTick";
        string response;
        if (!net.RpcCall(queue_call, send_tick, param, response)) return;
        if (!requestUtils.isSuccessfulResponse(response)) LOG_ERROR("", "Server returned " + requestUtils.getErrorMessage(response));
#ifdef DEBUG_CONNECTOR
        LOG_DEBUG("", "Finalized bar at " + time_str + ", price " + prices[i].to_string());
#endif
    }
}


//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void SVRApi::send_bars(
    const string &input_queue,
    const string &period,
    const bool finalize,
    const aprice &prices[],
    const uint &volumes[],
    const aprice &pricesaux[],
    const uint &volumesaux[],
    const datetime &times[],
    const string &col_name,
    const string &aux_col_name)
{
    const int price_len = ArraySize(prices);
    if (price_len < 1 || price_len != ArraySize(times) || price_len != ArraySize(pricesaux)) {
        LOG_ERROR("", "Prices length " + string(price_len) + " does not equal times length " + string(ArraySize(times)) + ", aux prices length " + string(ArraySize(pricesaux)));
        return;
    }
    static const string queue = "queue";
    static const string send_tick = "sendTick";
    const string current_time_str = TimeToString(TimeLocal(), C_time_mode);
    const string col_name_bid = col_name + "_bid";
    const string col_name_ask = col_name + "_ask";
    const string aux_col_name_bid = aux_col_name + "_bid";
    const string aux_col_name_ask = aux_col_name + "_ask";
    Hash param;
    param.hPutString("isFinal", finalize ? "true" : "false");
    param.hPutString("symbol", input_queue);
    param.hPutString("period", period);
    param.hPutString("clientTime", current_time_str);
    for (int i = 0; i < price_len; ++i) {
        string response;
        int bar = 0;
        param.hPutString("time", TimeToString(times[i], C_time_mode));
        param.hPutString(col_name_bid, DoubleToString(prices[i].bid, DOUBLE_PRINT_DECIMALS));
        param.hPutString(col_name_ask, DoubleToString(prices[i].ask, DOUBLE_PRINT_DECIMALS));
        param.hPutString(aux_col_name_bid, DoubleToString(pricesaux[i].bid, DOUBLE_PRINT_DECIMALS));
        param.hPutString(aux_col_name_ask, DoubleToString(pricesaux[i].ask, DOUBLE_PRINT_DECIMALS));
        param.hPutString("Volume", IntegerToString(volumes[i] + volumesaux[i]));
        if (!net.RpcCall(queue, send_tick, param, response)) return; // TODO Send multiple hashes in single RPC
        if (!requestUtils.isSuccessfulResponse(response))
            LOG_ERROR("", "Server returned: " + requestUtils.getErrorMessage(response));
#ifdef DEBUG_CONNECTOR
        else
            LOG_DEBUG("", "Finalized bar at " + time_str + ", price " + prices[i].to_string());
#endif
    }
}


//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool SVRApi::next_time_range(datetime& time_from, datetime& time_to, const string &input_queue, const string &period)
{
    static const string funName = "getNextTimeRangeToBeSent";
    const bool result = get_missing_period(time_from, time_to, input_queue, period, funName);
    if (result) LOG_INFO("", "Requested interval from " + TimeToString(time_from, C_time_mode) + " to " + TimeToString(time_to, C_time_mode));
    else LOG_DEBUG("", "Done.");
    return result;
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class IntervalRetryCounter
{
    uint             barNo_;
    uint             retryCount_;
    const uint       maxRetryCount_;

public:
                     IntervalRetryCounter(const uint bar): maxRetryCount_(3)
    {
        retryCount_ = 0;
        barNo_ = bar;
    }

    bool             retry()
    {
        return retryCount_++ < maxRetryCount_;
    }

    ulong            bar() const
    {
        return barNo_;
    }

    void             bar(const uint bar)
    {
        retryCount_ = 0;
        barNo_ = bar;
    }
};

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int SVRApi::bSearchBar(datetime start_time, uint left, uint right)
{
    int count = int(left - right), it, step;
    while (count > 0) {
        it = (int) left;
        step = count / 2;
        it -= step;
        if (GetTargetTime(it) < start_time) {
            left = --it;
            count -= step + 1;
        } else
            count = step;
    }
    return (int) left;
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int SVRApi::findBar(const datetime start_time)
{
    const int e = Bars(_Symbol, _Period) - 1;
    if (e < 1) {
        LOG_ERROR("", "No bars for symbol " + _Symbol + " were found!");
        return 0;
    }
    const datetime lastBar = GetTargetTime(e);

    if (lastBar > start_time) return e;
    uint timeRangeLimit = uint((GetTargetTime(0) - start_time) / C_period_seconds);

    timeRangeLimit = MathMin(timeRangeLimit, e);

    return bSearchBar(start_time, timeRangeLimit, 0);

}


//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool SVRApi::send_history(const string &input_queue, const string &period)
{
    datetime start_time, end_time;
    while(next_time_range(start_time, end_time, input_queue, period))
        if (!send_history(input_queue, period, start_time, end_time)) return false;

    return true;
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int SVRApi::copy_ticks_safe(const string &symbol, MqlTick &ticks[], const uint flags, const datetime time_start, const datetime time_end)
{
    int last_error, copied_ct;
    uint retries = 0;
    do {
        copied_ct = CopyTicksRange(symbol, ticks, flags, time_start * 1000, time_end * 1000);
        if (copied_ct > 0) return copied_ct;
        last_error = GetLastError();
        ResetLastError();
        if (!C_backtesting) Sleep(1);
    } while (copied_ct < 1 && ++retries < C_retries_count &&
             (last_error == ERR_HISTORY_TIMEOUT || last_error == ERR_HISTORY_LOAD_ERRORS || last_error == ERR_HISTORY_NOT_FOUND || ERR_NOT_ENOUGH_MEMORY));
    LOG_ERROR("",
              "Failed to copy ticks for " + symbol + ", from " + TimeToString(time_start, C_time_mode) + ", to " +
              TimeToString(time_end, C_time_mode) + ", error " + ErrorDescription(last_error) + ", symbol " + symbol);
    return ArraySize(ticks);
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int SVRApi::copy_rates_safe(const string &symbol, const ENUM_TIMEFRAMES time_frame, const datetime start, const datetime end, MqlRates &rates[])
{
    int copied_ct = CopyRates(symbol, time_frame, start, end, rates);
    int last_error;
    if (copied_ct < 0) {
        last_error = GetLastError();
        ResetLastError();
        uint retries = 0;
        while (last_error == ERR_HISTORY_LOAD_ERRORS || last_error == ERR_HISTORY_TIMEOUT || last_error == ERR_HISTORY_NOT_FOUND || ERR_NOT_ENOUGH_MEMORY) {
            LOG_ERROR("", "Retrying last_error " + ErrorDescription(last_error) + " for " + symbol);
            Sleep(1);
            copied_ct = CopyRates(symbol, time_frame, start, end, rates);
            if (copied_ct > 0) break;
            last_error = GetLastError();
            ResetLastError();
            ++retries;
            if (retries > C_retries_count) break;
        }
    }
    return copied_ct;
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
// TODO Implement aux input column as commented out below
int SVRApi::prepare_history_avg(const datetime start_time, const datetime end_time)
{
    if (ArraySize(myHi) > 0) ArrayFree(myHi);
    if (ArraySize(myLo) > 0) ArrayFree(myLo);
    if (ArraySize(myOp) > 0) ArrayFree(myOp);
    if (ArraySize(myCl) > 0) ArrayFree(myCl);

    if (start_time >= end_time) {
        LOG_ERROR("", "Illegal time range " + TimeToString(start_time, C_time_mode) + " until " + TimeToString(end_time, C_time_mode));
        return 0;
    }
    const int start_bar = iBarShift(_Symbol, _Period, start_time, false);
    if (start_bar < 0) {
        LOG_ERROR("", "Bar for start time " + TimeToString(start_time, C_time_mode) + " not found: " + ErrorDescription(GetLastError()));
        ResetLastError();
        return 0;
    } else if (iTime(_Symbol, _Period, start_bar) != start_time) {
        LOG_ERROR("", "Start time " + TimeToString(start_time, C_time_mode) + " does not equal bar time " + TimeToString(iTime(_Symbol, _Period, start_bar), C_time_mode));
    }
    const int end_bar = iBarShift(_Symbol, _Period, end_time, false);
    if (end_bar < 0) {
        LOG_ERROR("", "Bar for end time " + TimeToString(end_time, C_time_mode) + " not found: " + ErrorDescription(GetLastError()));
        ResetLastError();
        return 0;
    } else if (iTime(_Symbol, _Period, end_bar) != end_time)
        LOG_INFO("", "End time " + TimeToString(end_time, C_time_mode) + " does not equal bar time " + TimeToString(iTime(_Symbol, _Period, end_bar), C_time_mode));

// end_bar is smaller than start_bar
    const uint bar_ct = (per_second ? C_period_seconds : 1) * (1 + start_bar - end_bar);
    LOG_INFO("", "Preparing up to " + string(bar_ct) + " bars for period " + TimeToString(start_time, C_time_mode)
             + " to " + TimeToString(end_time, C_time_mode) + ", symbol " + _Symbol);
    ArrayResize(myAvg, bar_ct);
    ArrayResize(myTm, bar_ct);
    ArrayResize(myVl, bar_ct);
    int ok_bars = 0;
    static const int available_bars = iBars(_Symbol, _Period);
    aprice prev_tick_price = get_rate(start_bar, e_rate_type::price_open);
    for (int bar_ix = start_bar; bar_ix >= end_bar; --bar_ix) {
        const datetime bar_time = iTime(_Symbol, _Period, bar_ix);
        const datetime next_bar_time = bar_ix < 1 ? bar_time + C_period_seconds : iTime(_Symbol, _Period, bar_ix - 1);
        const datetime prev_bar_time = bar_ix >= available_bars ? bar_time - C_period_seconds : iTime(_Symbol, _Period, bar_ix + 1);
        const string bar_time_str = TimeToString(bar_time, C_time_mode);
        if (bar_time == 0) {
            LOG_ERROR("", "Bar time or price not found for " + _Symbol + ", from " + bar_time_str + ", to " + TimeToString(next_bar_time, C_time_mode) +
                      ", error " + ErrorDescription(GetLastError()));
            ResetLastError();
            continue;
        }
        MqlTick ticks[];
        const int ticks_count = copy_ticks_safe(_Symbol, ticks, COPY_TICKS_ALL, prev_bar_time, next_bar_time);
        if (ticks_count < 1 && (iHigh(_Symbol, _Period, bar_ix) != iLow(_Symbol, _Period, bar_ix)
                                || iOpen(_Symbol, _Period, bar_ix) != iClose(_Symbol, _Period, bar_ix)
                                || iOpen(_Symbol, _Period, bar_ix) != prev_tick_price.bid)) {
            LOG_ERROR("", "Inconsistency between ticks and bars at " + bar_time_str + ", high doesn't equal low but no ticks for bar copied. ");
#ifndef TOLERATE_DATA_INCONSISTENCY
            break;
#endif
            LOG_ERROR("", "Using one minute rates for " + bar_time_str + " which is imprecise!");
            MqlRates rates[];
            const int rates_ct = copy_rates_safe(_Symbol, PERIOD_M1, bar_time, next_bar_time - 1, rates);
            const AveragePrice current_price(rates, rates_ct, bar_time);
            if (per_second) {
                const int last_bar = MathMin(ArraySize(myAvg), C_period_seconds + ok_bars);
                for (int i = ok_bars; i < last_bar; ++i) {
                    myAvg[i] = current_price.value;
                    myTm[i] = current_price.tm;
                    myVl[i] = current_price.volume;
                }
            } else {
                myAvg[ok_bars] = current_price.value;
                myTm[ok_bars] = current_price.tm;
                myVl[ok_bars] = current_price.volume;
                ++ok_bars;
            }
            prev_tick_price = current_price.close_price;
            continue;
        }
        if (per_second) {
            prev_tick_price = persec_prices(prev_tick_price, ticks, bar_time, C_period_seconds, myAvg, myTm, myVl, ok_bars);
            if (ticks_count < 1) {
                LOG_ERROR("", "No ticks found for " + bar_time_str + " using close price for bar " + IntegerToString(bar_ix));
                prev_tick_price = get_rate(bar_ix, e_rate_type::price_close);
            }
            ok_bars += C_period_seconds;
        } else {
            const AveragePrice current_price(ticks, bar_time, C_period_seconds, prev_tick_price);
            if (!current_price.value.valid()) {
                LOG_ERROR("", "Illegal price for " + TimeToString(bar_time, C_time_mode));
#ifdef TOLERATE_DATA_INCONSISTENCY
                continue;
#else
                break;
#endif
            }
            myAvg[ok_bars] = current_price.value;
            myTm[ok_bars] = current_price.tm;
            myVl[ok_bars] = current_price.volume;
            prev_tick_price = current_price.close_price;
            ++ok_bars;
        }
        if (ok_bars % 10000 == 0) LOG_DEBUG("", "Collected " + IntegerToString(ok_bars) + " bars.");
    }
    ArrayRemove(myAvg, ok_bars);
    ArrayRemove(myTm, ok_bars);
    ArrayRemove(myVl, ok_bars);
    LOG_INFO("", "Successfully prepared " + IntegerToString(ok_bars) + " bars out of " + IntegerToString(bar_ct) + " possible bars from " +
             TimeToString(start_time, C_time_mode) + " until " + TimeToString(end_time, C_time_mode));
    return ok_bars;
}


/*
    const ulong iterPeriod = per_second ? 1 : C_period_seconds;
    const ulong max_values_ct = per_second ? end_time - start_time : (end_time - start_time) / C_period_seconds;
    double tmp_avg[], tempAuxBid[], tempCommonBid[];
    datetime tmp_time[], tempAuxTime[], tempCommonTime[];
    ArrayResize(tmp_avg, max_values_ct);
    ArrayResize(tmp_time, max_values_ct);
    ArrayResize(tempAuxBid, max_values_ct);
    ArrayResize(tempAuxTime, max_values_ct);
    ArrayResize(tempCommonBid, max_values_ct);
    ArrayResize(tempCommonTime, max_values_ct);
    ulong main_successful = 0, aux_successful = 0;

    //PrepareHistoryPackage sends from number_of_bars to 1
    const string main_symbol = _Symbol;
    datetime iterDate = end_time;
    LOG_INFO("prepare_history_avg", "From " + string(start_time) + " until " + string(end_time));
    for (iterDate = start_time; iterDate < end_time; iterDate += iterPeriod) {
        MqlTick ticks[];
        ArraySetAsSeries(ticks, true);
        const int copied_ct = CopyTicks(main_symbol, ticks, COPY_TICKS_INFO, iterDate, iterDate + iterPeriod);
        if (copied_ct < 1) {
            LOG_ERROR("", "No bars copied for " + main_symbol + " from " + string(DSTCorrectedIterDate) + " to " + string(DSTCorrectedIterDate + C_period_seconds));
            continue;
        }

        AveragePrice current_price(iOpen(main_symbol, PERIOD_M1, 0), ticks, copied_ct, iterDate);

        if (current_price.value != 0) {
            //LOG_INFO("prepare_history_avg", "values " + string(values));
            ArrayFill(tmp_avg, successful, 1, current_price.value);
            ArrayFill(tmp_time, successful, 1, current_price.tm);
            successful++;
        }
        iterDate -= next;
    }

    int common_ct = 0;
    if (AuxInputQueue != "") {
        iterDate = end_time;
        LOG_INFO("prepare_history_avg", "Aux from " + string(start_time) + " until " + string(end_time));
        while(iterDate > start_time) {
            MqlRates rates[];
            ArraySetAsSeries(rates, true);
            datetime offsetIterDate = GetUTCTime(iterDate);
            datetime DSTCorrectedIterDate = offsetIterDate + getDSTOffsetFromNow(offsetIterDate);
            //LOG_INFO("prepare_history_avg", "iterDate DSTCorrectedIterData " + string(iterDate) + " " + string(DSTCorrectedIterDate));

            const int copied_ct = copy_rates_safe(AuxInputQueue, PERIOD_M1, DSTCorrectedIterDate, DSTCorrectedIterDate + C_period_seconds - 1, rates);
            if(copied_ct < 1) {
                iterDate -= next;
                LOG_ERROR("", "No bars copied for " + AuxInputQueue + " from " + string(DSTCorrectedIterDate) + " until " + string(DSTCorrectedIterDate + C_period_seconds));
                continue;
            }

            AveragePrice current_price(rates, copied_ct, GetTargetDateTimeFromDate(offsetIterDate));
            //LOG_INFO("prepare_history_avg", "GetTargetDateTimeFromDate(offset) " + string(GetTargetDateTimeFromDate(offsetIterDate)));

            if (current_price.value != 0 && current_price.tm == tmp_time[aux_successful]) {
                //LOG_INFO("prepare_history_avg", "values " + string(values));
                ArrayFill(tempAuxBid, aux_successful, 1, current_price.value);
                ArrayFill(tempAuxTime, aux_successful, 1, current_price.tm);
                aux_successful++;
            }
            iterDate -= next;
        }
        for (int i = 0; i < successful; ++i) {
            for (int j = 0; j < aux_successful; ++j) {
                if (tmp_time[i] == tempAuxTime[j]) {
                    ArrayFill(tempCommonBid, common_ct, 1, tmp_avg[i]);
                    ArrayFill(tempCommonTime, common_ct, 1, tmp_time[i]);
                    common_ct++;
                }
            }
        }
        if (aux_successful != successful)
            LOG_ERROR("", "aux_successful " + string(aux_successful) + " != successful " + string(successful));
        if (aux_successful != common_ct)
            LOG_ERROR("", "aux_successful " + string(aux_successful) + " != common_ct " + string(common_ct));
        ArrayResize(myAuxAvg, aux_successful + offset);
        ArrayCopy(myAuxAvg, tempAuxBid, offset, 0, aux_successful);
    } else {
        for (int i = 0; i < successful; ++i) {
            ArrayFill(tempCommonBid, common_ct, 1, tmp_avg[i]);
            ArrayFill(tempCommonTime, common_ct, 1, tmp_time[i]);
            common_ct++;
        }
    }

    ArrayResize(myAvg, common_ct + offset);
    ArrayCopy(myAvg, tempCommonBid, offset, 0, common_ct);

    ArrayResize(myTm, common_ct + offset);
    ArrayCopy(myTm, tempCommonTime, offset, 0, common_ct);
    ArrayResize(myVl, common_ct + offset);
    VolumeMQL4(_Symbol, _Period, myVl, offset, 0, common_ct);
    LOG_INFO("", "Successful " + string(common_ct) + " bars out of " + string(values) + " possible bars from " + string(start_time) + " until " + string(end_time));
    return common_ct;
}
*/

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int SVRApi::prepare_history_ohlc(const datetime start_time, const datetime end_time)
{
    if (per_second) {
        LOG_ERROR("", "No OHLC when sending high frequency data!");
        return 0;
    }
    if (ArraySize(myAvg) > 0) ArrayFree(myAvg);
    if (ArraySize(myAuxAvg) > 0) ArrayFree(myAuxAvg);
    LOG_INFO("", "Preparing from " + TimeToString(start_time, C_time_mode) + " until " + TimeToString(end_time, C_time_mode));
    const int start_bar = iBarShift(_Symbol, _Period, start_time, false);
    if (start_bar < 0) {
        LOG_ERROR("", "Bar for start time " + TimeToString(start_time) + " not found: " + ErrorDescription(GetLastError()));
        ResetLastError();
        return 0;
    }
    const int end_bar = iBarShift(_Symbol, _Period, end_time, false);
    if (end_bar < 0) {
        LOG_ERROR("", "Bar for end time " + TimeToString(end_time) + " not found: " + ErrorDescription(GetLastError()));
        ResetLastError();
        return 0;
    }
// end_bar is smaller than start_bar
    const int bar_ct = 1 + start_bar - end_bar;
    ArrayResize(myTm, bar_ct);
    ArrayResize(myHi, bar_ct);
    ArrayResize(myLo, bar_ct);
    ArrayResize(myOp, bar_ct);
    ArrayResize(myCl, bar_ct);
    ArrayResize(myTm, bar_ct);
    ArrayResize(myVl, bar_ct);
    MqlRates rates[];
    ArraySetAsSeries(rates, true);
    CopyRates(_Symbol, _Period, end_bar, bar_ct, rates);
    for (int i = 0; i < bar_ct; ++i) {
        myOp[i].set(rates[i], e_rate_type::price_open);
        myHi[i].set(rates[i], e_rate_type::price_high);
        myLo[i].set(rates[i], e_rate_type::price_low);
        myCl[i].set(rates[i], e_rate_type::price_close);
        myTm[i] = rates[i].time;
        myVl[i] = (uint) rates[i].tick_volume;
    }
    LOG_INFO("", "Copied " + string(bar_ct) + " bars from " + TimeToString(start_time, C_time_mode) + " until " + TimeToString(end_time, C_time_mode));
    return bar_ct;
}

#define MAX_RETRIES 10

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool SVRApi::send_history(const string &input_queue, const string &period, const datetime start_time, datetime end_time)
{
    if (DemoMode) end_time = per_second ? end_time - DemoBars : end_time - DemoBars * C_period_seconds;
    const int copied_ct = is_average ? prepare_history_avg(start_time, end_time) : prepare_history_ohlc(start_time, end_time);
    if (copied_ct < 1) {
        LOG_ERROR("", "Failed preparing history bars.");
        return false;
    }
    LOG_INFO("", "Prepared " + string(copied_ct) + " price bars.");
    const int bar_count = ArraySize(myTm);

#ifdef HISTORYTOSQL
    LOG_INFO("", "Writing history to MQL5/Files");
    const int file_handle = FileOpen("history_bars_" + input_queue + "_" + string(per_second ? 1 : C_period_seconds) + ".sql", FILE_WRITE | FILE_ANSI | FILE_CSV);
    const string time_current_str = TimeToString(TimeLocal(), C_time_mode);
    if (file_handle != INVALID_HANDLE) {
        FileSeek(file_handle, 0, SEEK_END);
        if (is_average)
            for (int bar_ix = 0; bar_ix < bar_count; ++bar_ix)
                FileWrite(file_handle, TimeToString(myTm[bar_ix], C_time_mode), time_current_str, IntegerToString(myVl[bar_ix]),
                          DoubleToString(myAvg[bar_ix].bid, DOUBLE_PRINT_DECIMALS), DoubleToString(myAvg[bar_ix].ask, DOUBLE_PRINT_DECIMALS));
        else
            for (int bar_ix = 0; bar_ix < bar_count; ++bar_ix)
                FileWrite(file_handle, TimeToString(myTm[bar_ix], C_time_mode), time_current_str, IntegerToString(myVl[bar_ix]),
                          DoubleToString(myOp[bar_ix].bid, DOUBLE_PRINT_DECIMALS), DoubleToString(myOp[bar_ix].ask, DOUBLE_PRINT_DECIMALS),
                          DoubleToString(myHi[bar_ix].bid, DOUBLE_PRINT_DECIMALS), DoubleToString(myHi[bar_ix].ask, DOUBLE_PRINT_DECIMALS),
                          DoubleToString(myLo[bar_ix].bid, DOUBLE_PRINT_DECIMALS), DoubleToString(myLo[bar_ix].ask, DOUBLE_PRINT_DECIMALS),
                          DoubleToString(myCl[bar_ix].bid, DOUBLE_PRINT_DECIMALS), DoubleToString(myCl[bar_ix].ask, DOUBLE_PRINT_DECIMALS));
        FileFlush(file_handle);
        FileClose(file_handle);
    }
    return false;
#endif

    string header = "Time:date;";
    if (is_average) {
        header += input_queue + "_bid:double;" + input_queue + "_ask:double;";
        if (AuxInputQueue != "") header += AuxInputQueue + "_bid:double;" + AuxInputQueue + "_ask:double;";
        header += "Volume:int";
    } else
        header += "High:double;Low:double;Open:double;Close:double;Volume:int";

    const string delimiter = ";";
    int bar = 0;
    const int bar_count_1 = bar_count - 1;
    while (bar < bar_count_1) {
        Hash params;
        params.hPutString("barFrom", string(bar));
        params.hPutString("symbol", input_queue);
        params.hPutString("period", period);
        params.hPutString("header", header);
        params.hPutString("delimiter", delimiter);
        const uint finished_at = is_average ? pack_history_avg(bar, bar_count_1, delimiter, params) : pack_history_ohlc(bar, bar_count_1, delimiter, params);
        params.hPutString("barTo", string(finished_at));
        string response;
        int retries = 0;
        const string queue_call = "queue";
        const string history_data = "historyData";
        while ((!net.RpcCall(queue_call, history_data, params, response) || !requestUtils.isSuccessfulResponse(response)) && retries < MAX_RETRIES) {
            if (StringLen(response) > 0)
                LOG_ERROR("", "Server returned: " + requestUtils.getErrorMessage(response));
            if (retries < MAX_RETRIES)
                LOG_ERROR("", "Error while sending historical data, retrying.");
            else
                LOG_ERROR("", "Error while sending historical data. Skipping bars from " + string(bar) + " to " + string(finished_at));
            ++retries;
        }
        if (retries < MAX_RETRIES) LOG_INFO("", "Successfully sent " + string(finished_at) + " bars.");
        if (finished_at == bar_count_1) break;
        else bar = (int) finished_at;
    }
    if (ArraySize(myAuxAvg) > 0) ArrayFree(myAuxAvg);
    if (ArraySize(myAvg) > 0) ArrayFree(myAvg);
    if (ArraySize(myCl) > 0) ArrayFree(myCl);
    if (ArraySize(myHi) > 0) ArrayFree(myHi);
    if (ArraySize(myLo) > 0) ArrayFree(myLo);
    if (ArraySize(myOp) > 0) ArrayFree(myOp);
    if (ArraySize(myTm) > 0) ArrayFree(myTm);
    if (ArraySize(myVl) > 0) ArrayFree(myVl);
    return true;
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
// TODO Add time zone information
uint SVRApi::pack_history_ohlc(const uint bar_start, const uint bar_end, const string &delimiter, Hash &hash)
{
    uint bar_ix = bar_start;
    const string time_zone_delim = time_zone_info + delimiter;
    for (uint pack_bar = 0; bar_ix <= bar_end && pack_bar < packSize; ++bar_ix, ++pack_bar)
        hash.hPutString(IntegerToString(bar_ix),
                        TimeToString(myTm[bar_ix], C_time_mode) + time_zone_delim +
                        DoubleToString(myHi[bar_ix].bid, DOUBLE_PRINT_DECIMALS) + delimiter + DoubleToString(myHi[bar_ix].ask, DOUBLE_PRINT_DECIMALS) + delimiter +
                        DoubleToString(myLo[bar_ix].bid, DOUBLE_PRINT_DECIMALS) + delimiter + DoubleToString(myLo[bar_ix].ask, DOUBLE_PRINT_DECIMALS) + delimiter +
                        DoubleToString(myOp[bar_ix].bid, DOUBLE_PRINT_DECIMALS) + delimiter + DoubleToString(myOp[bar_ix].ask, DOUBLE_PRINT_DECIMALS) + delimiter +
                        DoubleToString(myCl[bar_ix].bid, DOUBLE_PRINT_DECIMALS) + delimiter + DoubleToString(myCl[bar_ix].ask, DOUBLE_PRINT_DECIMALS) + delimiter +
                        IntegerToString(myVl[bar_ix], DOUBLE_PRINT_DECIMALS));
    return bar_ix;
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
uint SVRApi::pack_history_avg(const uint bar_start, const uint bar_end, const string &delimiter, Hash &hash)
{
    uint bar_ix = bar_start;
    if (AuxInputQueue != "") {
        for (uint pack_bar = 0; bar_ix <= bar_end && pack_bar < packSize; ++bar_ix, ++pack_bar)
            hash.hPutString(
                IntegerToString(bar_ix), TimeToString(myTm[bar_ix], C_time_mode) + delimiter +
                DoubleToString(myAvg[bar_ix].bid, DOUBLE_PRINT_DECIMALS) + delimiter + DoubleToString(myAvg[bar_ix].ask, DOUBLE_PRINT_DECIMALS) + delimiter +
                DoubleToString(myAuxAvg[bar_ix].bid, DOUBLE_PRINT_DECIMALS) + delimiter + DoubleToString(myAuxAvg[bar_ix].ask, DOUBLE_PRINT_DECIMALS) + delimiter +
                IntegerToString(myVl[bar_ix]));
    } else {
        for (uint pack_bar = 0; bar_ix <= bar_end && pack_bar < packSize; ++bar_ix, ++pack_bar)
            hash.hPutString(
                IntegerToString(bar_ix), TimeToString(myTm[bar_ix], C_time_mode) + delimiter +
                DoubleToString(myAvg[bar_ix].bid, DOUBLE_PRINT_DECIMALS) + delimiter + DoubleToString(myAvg[bar_ix].ask, DOUBLE_PRINT_DECIMALS) + delimiter +
                IntegerToString(myVl[bar_ix]));
    }
    return bar_ix;
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
/*
bool SVRApi::ReconcileHistoricalData(string input_queue, string period)
{
    datetime time_from, time_to;
    bool recon_finished = false;
    string funName = "reconcileHistoricalData";
    bool result = get_missing_period(time_from, time_to, input_queue, period, funName, recon_finished);

    if(!result) {
        LOG_ERROR ("SVRApi::ReconcileHistoricalData", "Historical data reconciliation aborted");
        return false;
    }

    if (recon_finished) {
        LOG_INFO ("SVRApi::ReconcileHistoricalData", "Historical data reconciled");
        histDataReconciled = true;
        return false;
    }

    //Do next reconcilation step
    if(time_from == time_to) return true;

    result = send_history(input_queue, period, time_from, time_to, false, packSize);
    if (!result) LOG_ERROR("SVRApi::ReconcileHistoricalData", "Historical data reconciliation aborted");

    return result;
}
*/

//+------------------------------------------------------------------+
//| Offer time from, time to period, and receive a request that is
//| a subset of the offered time period
//+------------------------------------------------------------------+
bool SVRApi::get_missing_period(datetime& time_from, datetime& time_to, const string &input_queue, const string &period, const string &function)
{
    string response;
    Hash params;
    params.hPutString("symbol", input_queue);
    params.hPutString("period", per_second ? "1" : period);
    const int bars = C_bars_offered < iBars(_Symbol, _Period) ? C_bars_offered : iBars(_Symbol, _Period);
    time_to = iTime(_Symbol, _Period, 1);
    if (per_second) time_to += C_period_seconds - 1;
    MqlTick first_tick[];
    const datetime last_bar_time = iTime(_Symbol, _Period, bars - 1);
    CopyTicks(_Symbol, first_tick, COPY_TICKS_ALL, last_bar_time * 1000, 1);
    if (ArraySize(first_tick) < 1) {
        LOG_ERROR("", "No ticks found for " + _Symbol + ", period " + IntegerToString(C_period_seconds) + " sec, bar" + IntegerToString(bars - 1) +
                  ", time " + TimeToString(last_bar_time, C_time_mode));
        return false;
    }
    time_from = first_tick[0].time;
#ifdef HISTORYTOSQL
    return true;
#endif
    const string str_offer_to_time = TimeToString(time_to, C_time_mode);
    const string str_offer_from_time = TimeToString(time_from, C_time_mode);
    params.hPutString("offeredTimeFrom", str_offer_from_time);
    params.hPutString("offeredTimeTo", str_offer_to_time);
    LOG_INFO("", "Offering " + IntegerToString(per_second ? bars * C_period_seconds : bars) + " bars for " + input_queue + ", symbol " + _Symbol +
             ", from " + str_offer_from_time + " until " + str_offer_to_time);
    static const string queue_call = "queue";
    if (!net.RpcCall(queue_call, function, params, response)) {
        LOG_ERROR("", "Queue call failed, bailing.");
        return false;
    }
    LOG_DEBUG("", "About to call isSuccessfulResponse");
    const bool success = requestUtils.isSuccessfulResponse(response);
    LOG_DEBUG("", "isSuccessfulResponse returned " + string(success));
    if (!success) {
        LOG_ERROR("", "Server returned " + requestUtils.getErrorMessage(response));
    } else {
        string strReqTimeFrom, strReqTimeTo;
        JSONObject *jo = requestUtils.getResultObject(response);
        if (!jo || !jo.getString("timeFrom", strReqTimeFrom)) {
            LOG_DEBUG("", "Reconciliation complete, server response " + response);
            return false;
        }
        jo.getString("timeTo", strReqTimeTo);
        const datetime reqTimeFrom = StringToTime(strReqTimeFrom);
        const datetime reqTimeTo = StringToTime(strReqTimeTo);
        if (reqTimeFrom > reqTimeTo) {
            LOG_ERROR("", "Server returned illegal times, from " + strReqTimeFrom + " to " + strReqTimeTo);
            return false;
        } else {
            LOG_INFO("", "Server requested period from " + strReqTimeFrom + " to " + strReqTimeTo);
        }
        time_from = reqTimeFrom;
        time_to = reqTimeTo;
    }

    return success;
}
//+------------------------------------------------------------------+
