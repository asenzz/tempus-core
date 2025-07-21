//+------------------------------------------------------------------+
//|                                               tempus_api.mql.mqh |
//|                                                       Jarko Asen |
//|                                                www.zarkoasen.com |
//+------------------------------------------------------------------+
#property copyright "Jarko Asen"
#property link      "www.zarkoasen.com"

#include <tempus/api.mqh>

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
datetime nearest_bar_time_or_before(const datetime bar_time)
{
    const ENUM_TIMEFRAMES period = _Period;
    const int period_seconds = PeriodSeconds(period);
    const string symbol = _Symbol;
    LOG_INFO("Starting " + TimeToString(bar_time, C_time_mode));
    datetime bar_time_iter = datetime(bar_time / period_seconds) * period_seconds;
    int      bar_shift    = iBarShift(symbol, period, bar_time_iter, false);
    while(bar_shift == -1 || iTime(symbol,period, bar_shift) > bar_time) {
        bar_time_iter -= period_seconds;
        bar_shift     = iBarShift(symbol, period, bar_time_iter, false);
    }
    const datetime result = iTime(symbol, period, bar_shift);
    LOG_INFO("Returning " + TimeToString(result, C_time_mode) + " for bar time " + TimeToString(bar_time, C_time_mode));
    return result;
}


static const uint tempusapi::package_len = MAX_PACK_SIZE;
static const string tempusapi::delimiter   = ";"; // For HTTTP REST fields
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
tempusapi::tempusapi(
    const bool demo_mode_, const int demo_bars_, const bool is_average_, const string aux_queue_name_, const bool per_second_,
    const string &timezone_, streams_messaging *const p_streams_messaging_, const string &symbol_, const ENUM_TIMEFRAMES period_, const string &input_queue_) :
    demo_mode(demo_mode_),
    demo_bars(demo_bars_),
    is_average(is_average_),
    aux_queue_name(aux_queue_name_),
    per_second(per_second_),
    timezone(timezone_),
    p_streams_messaging(p_streams_messaging_),
    symbol(symbol_),
    period(period_),
    period_sec(PeriodSeconds(period_)),
    period_str(IntegerToString(period_sec)),
    input_queue(input_queue_)
{
    LOG_VERBOSE("Period " + IntegerToString(period) + ", period seconds " + IntegerToString(period_sec) + ", period string " + period_str + ", input queue " + input_queue);
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
tempusapi::~tempusapi()
{
}
//+------------------------------------------------------------------+

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool tempusapi::connect(const string &url)
{
    return net.Open(url);
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool tempusapi::login(const string &username, const string &password)
{
    static const string login_url = "tempus/login";
    const string        post_body = "username=" + username + "&password=" + password;
    long                status_code;
    net.Post(login_url, post_body, status_code);
    return use_http = status_code == 200;
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void tempusapi::send_bar(const bool finalize, const aprice &op, const aprice &hi, const aprice &lo, const aprice &cl, const uint vol, const datetime tm)
{
    if (use_http) {
        string response;
        Hash   param;

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
        param.hPutString("period", period_str);
        param.hPutString("clientTime", TimeToString(TimeLocal(), C_time_mode));
        static const string queue_call     = "queue";
        static const string send_tick_call = "sendTick";
        if(!net.RpcCall(queue_call, send_tick_call, param, response)) return;

        if(!request_utils.is_response_successful(response))
            LOG_ERROR("Server returned: " + request_utils.get_error_msg(response));
#ifdef DEBUG_CONNECTOR
        LOG_DEBUG("Sent bar at " + time_str + " using HTTP API.");
#endif
    } else {
        LOG_ERROR("Not implemented");
    }
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void tempusapi::send_bar(const bool finalize, const AveragePrice &avg_price)
{
    if (use_http) {
        string response;
        Hash   param;

        uint         bar      = 0;
        const string time_str = TimeToString(avg_price.tm, C_time_mode);
        param.hPutString("symbol", input_queue);
        param.hPutString("time", time_str);
        param.hPutString(input_queue + "_bid", DoubleToString(avg_price.value.bid, DOUBLE_PRINT_DECIMALS));
        param.hPutString(input_queue + "_ask", DoubleToString(avg_price.value.ask, DOUBLE_PRINT_DECIMALS));
        param.hPutString("Volume", IntegerToString(avg_price.volume));
        param.hPutString("isFinal", finalize ? "true" : "false");
        param.hPutString("period", period_str);
        param.hPutString("clientTime", TimeToString(TimeCurrent(), C_time_mode));

        static const string queue_call     = "queue";
        static const string send_tick_call = "sendTick";
        if(!net.RpcCall(queue_call, send_tick_call, param, response)) return;
        if(!request_utils.is_response_successful(response))
            LOG_ERROR("Server returned " + request_utils.get_error_msg(response));
#ifdef DEBUG_CONNECTOR
        LOG_INFO("Sent using HTTP bar at " + time_str + ", price " + avg_price.value.to_string());
#endif
    } else {
        LOG_ERROR("Not implemented!");
    }
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void tempusapi::send_bar(const bool finalize, const AveragePrice &avg_price, const AveragePrice &avg_price_aux, const string &col_name, const string &aux_col_name)
{
    if (use_http) {
        string response;
        Hash   param;

        const string time_str = TimeToString(avg_price.tm, C_time_mode);
        param.hPutString("symbol", input_queue);
        param.hPutString("time", time_str);
        param.hPutString(col_name + "_bid", DoubleToString(avg_price.value.bid, DOUBLE_PRINT_DECIMALS));
        param.hPutString(col_name + "_ask", DoubleToString(avg_price.value.ask, DOUBLE_PRINT_DECIMALS));
        param.hPutString(aux_col_name + "_bid", DoubleToString(avg_price_aux.value.bid, DOUBLE_PRINT_DECIMALS));
        param.hPutString(aux_col_name + "_ask", DoubleToString(avg_price_aux.value.ask, DOUBLE_PRINT_DECIMALS));
        param.hPutString("Volume", IntegerToString(avg_price.volume));
        param.hPutString("isFinal", finalize ? "true" : "false");
        param.hPutString("period", period_str);
        param.hPutString("clientTime", TimeToString(TimeCurrent(), C_time_mode));

        static const string queue_call     = "queue";
        static const string send_tick_call = "sendTick";
        if(!net.RpcCall(queue_call, send_tick_call, param, response)) return;

        if(!request_utils.is_response_successful(response))
            LOG_ERROR("Server returned " + request_utils.get_error_msg(response));
#ifdef DEBUG_CONNECTOR
        LOG_INFO("Sent bar using HTtp at " + time_str);
#endif
    } else {
        LOG_ERROR("Not implemented");
    }
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void tempusapi::send_bars(const bool finalize, const aprice &prices[], const datetime &times_[], const uint &volumes_[])
{
    if (use_http) {
        const int price_len   = ArraySize(prices);
        const int times_len   = ArraySize(times_);
        const int volumes_len = ArraySize(volumes_);
        if(price_len != times_len || price_len != volumes_len) {
            LOG_ERROR("Prices length " + string(price_len) + " does not equal times length " + IntegerToString(times_len) + ", volumes length " + IntegerToString(volumes_len));
            return;
        }
        const string column_bid = input_queue + "_bid";
        const string column_ask = input_queue + "_ask";
        Hash         param;
        param.hPutString("period", period_str);
        param.hPutString("clientTime", TimeToString(TimeLocal(), C_time_mode));
        param.hPutString("symbol", input_queue);
        param.hPutString("isFinal", finalize ? "true" : "false");
        for(int i = 0; i < price_len; ++i) {
            param.hPutString("time", TimeToString(times_[i], C_time_mode));
            param.hPutString(column_bid, DoubleToString(prices[i].bid, DOUBLE_PRINT_DECIMALS));
            param.hPutString(column_ask, DoubleToString(prices[i].ask, DOUBLE_PRINT_DECIMALS));
            param.hPutString("Volume", IntegerToString(volumes_[i]));
            static const string queue_call = "queue";
            static const string send_tick  = "sendTick";
            string              response;
            if(!net.RpcCall(queue_call, send_tick, param, response))
                return;
            if(!request_utils.is_response_successful(response))
                LOG_ERROR("Server returned " + request_utils.get_error_msg(response));
#ifdef DEBUG_CONNECTOR
            LOG_DEBUG("Finalized bar at " + time_str + ", price " + prices[i].to_string());
#endif
        }
    }
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void tempusapi::send_bars(
    const bool      finalize,
    const aprice   &prices[],
    const uint     &volumes_[],
    const aprice   &pricesaux[],
    const uint     &volumesaux[],
    const datetime &times_[],
    const string   &col_name,
    const string   &aux_col_name)
{
    if (use_http) {
        const int price_len = ArraySize(prices);
        if(price_len < 1 || price_len != ArraySize(times) || price_len != ArraySize(pricesaux)) {
            LOG_ERROR("Prices length " + string(price_len) + " does not equal times length " + string(ArraySize(times_)) + ", aux prices length " + string(ArraySize(pricesaux)));
            return;
        }
        static const string queue            = "queue";
        static const string send_tick        = "sendTick";
        const string        current_time_str = TimeToString(TimeLocal(), C_time_mode);
        const string        col_name_bid     = col_name + "_bid";
        const string        col_name_ask     = col_name + "_ask";
        const string        aux_col_name_bid = aux_col_name + "_bid";
        const string        aux_col_name_ask = aux_col_name + "_ask";
        Hash                param;
        param.hPutString("isFinal", finalize ? "true" : "false");
        param.hPutString("symbol", input_queue);
        param.hPutString("period", period_str);
        param.hPutString("clientTime", current_time_str);
        for(int i = 0; i < price_len; ++i) {
            string response;
            int    bar = 0;
            param.hPutString("time", TimeToString(times_[i], C_time_mode));
            param.hPutString(col_name_bid, DoubleToString(prices[i].bid, DOUBLE_PRINT_DECIMALS));
            param.hPutString(col_name_ask, DoubleToString(prices[i].ask, DOUBLE_PRINT_DECIMALS));
            param.hPutString(aux_col_name_bid, DoubleToString(pricesaux[i].bid, DOUBLE_PRINT_DECIMALS));
            param.hPutString(aux_col_name_ask, DoubleToString(pricesaux[i].ask, DOUBLE_PRINT_DECIMALS));
            param.hPutString("Volume", IntegerToString(volumes_[i] + volumesaux[i]));
            if(!net.RpcCall(queue, send_tick, param, response))
                return;   // TODO Send multiple hashes in single RPC
            if(!request_utils.is_response_successful(response))
                LOG_ERROR("Server returned: " + request_utils.get_error_msg(response));
#ifdef DEBUG_CONNECTOR
            else
                LOG_DEBUG("Finalized bar at " + time_str + ", price " + prices[i].to_string());
#endif
        }
    }
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool tempusapi::next_time_range(datetime &time_from, datetime &time_to)
{
    bool result = true;
    static const string function = "getNextTimeRangeToBeSent";

    const int bars = MathMin(C_bars_offered, iBars(symbol, period));
    time_to        = iTime(symbol, period, 1);
    if (per_second) time_to += period_sec - 1;
    MqlTick        first_tick[];
    const datetime last_bar_time = iTime(symbol, period, bars - 1);
    CopyTicks(symbol, first_tick, COPY_TICKS_ALL, last_bar_time * 1000, 1);
    if(ArraySize(first_tick) < 1) {
        LOG_ERROR("No ticks found for " + symbol + ", period " + IntegerToString(period_sec) + " sec, bar" + IntegerToString(bars - 1) +
                  ", time " + TimeToString(last_bar_time, C_time_mode));
        return false;
    }
    time_from = first_tick[0].time;

#ifdef HISTORYTOSQL
    return true;
#endif

    if (use_http) {
        string response;
        Hash   params;
        params.hPutString("symbol", input_queue);
        params.hPutString("period", per_second ? "1" : period_str);
        const string str_offer_to_time   = TimeToString(time_to, C_time_mode);
        const string str_offer_from_time = TimeToString(time_from, C_time_mode);
        params.hPutString("offeredTimeFrom", str_offer_from_time);
        params.hPutString("offeredTimeTo", str_offer_to_time);
        LOG_INFO("Offering " + IntegerToString(per_second ? bars * period_sec : bars) + " bars for " + input_queue + ", symbol " + symbol +
                 ", from " + str_offer_from_time + " until " + str_offer_to_time);
        static const string queue_call = "queue";
        if(!net.RpcCall(queue_call, function, params, response)) {
            LOG_ERROR("Queue call failed, bailing.");
            return false;
        }

        if(!request_utils.is_response_successful(response)) {
            LOG_ERROR("Server returned " + request_utils.get_error_msg(response));
        } else {
            string      str_req_time_from, str_req_time_to;

            JSONObject *jo = request_utils.get_result_obj(response);

            if(!jo || !jo.getString("timeFrom", str_req_time_from)) {
                LOG_DEBUG("Reconciliation complete, server response " + response);
                return false;
            }
            if (!jo.getString("timeTo", str_req_time_to)) {
                LOG_ERROR("Reconciliation time to is missing, time from is " + str_req_time_from);
                return false;
            }

            const datetime req_time_from = StringToTime(str_req_time_from);
            const datetime req_time_to   = StringToTime(str_req_time_to);
            if (req_time_from > req_time_to || !req_time_from || !req_time_to) {
                LOG_ERROR("Server returned illegal times, from " + str_req_time_from + " to " + str_req_time_to);
                return false;
            }

            LOG_INFO("Server requested period from " + str_req_time_from + " to " + str_req_time_to);
            time_from = req_time_from;
            time_to   = req_time_to;
        }
    }

    if (p_streams_messaging && !p_streams_messaging.offer_range(input_queue, time_from, time_to)) {
        LOG_ERROR("Offer time range failed.");
        return false;
    }

    LOG_INFO("Requested interval from " + TimeToString(time_from, C_time_mode) + " to " + TimeToString(time_to, C_time_mode));

    return result;
}


//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int tempusapi::bsearch_bar(const datetime start_time, int left, const int right)
{
    int count = left - right, it, step;
    while(count > 0) {
        it    = left;
        step  = count / 2;
        it   -= step;
        if(GetTargetTime(it) < start_time) {
            left   = --it;
            count -= step + 1;
        } else
            count = step;
    }
    return (int)left;
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int tempusapi::find_bar(const datetime start_time)
{
    const int e = Bars(symbol, period) - 1;
    if(e < 1) {
        LOG_ERROR("No bars for symbol " + symbol + " were found!");
        return 0;
    }
    const datetime last_bar = GetTargetTime(e);

    if (last_bar > start_time) return e;
    uint time_range_limit = uint((GetTargetTime(0) - start_time) / period_sec);

    time_range_limit = MathMin(time_range_limit, e);

    return bsearch_bar(start_time, time_range_limit, 0);
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool tempusapi::send_history(const datetime earliest_time)
{
    datetime start_time, end_time;
    while(next_time_range(start_time, end_time)) {
        if (end_time < earliest_time) return false;
        if (start_time < earliest_time) start_time = earliest_time;
        if (!send_history(start_time, end_time)) return false;
    }
    return true;
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int tempusapi::copy_ticks_safe(const string &symbol, MqlTick &ticks[], const uint flags, const datetime time_start, const datetime time_end)
{
    int  last_error, copied_ct;
    uint retries = 0;
    do {
        copied_ct = CopyTicksRange(symbol, ticks, flags, time_start * 1000, time_end * 1000);
        if(copied_ct > 0)
            return copied_ct;
        last_error = GetLastError();
        ResetLastError();
        Sleep(1);
    } while(copied_ct < 1 && ++retries < C_retries_count &&
            (last_error == ERR_HISTORY_TIMEOUT || last_error == ERR_HISTORY_LOAD_ERRORS || last_error == ERR_HISTORY_NOT_FOUND || ERR_NOT_ENOUGH_MEMORY));
    LOG_ERROR(
        "Failed to copy ticks for " + symbol + ", from " + TimeToString(time_start, C_time_mode) + ", to " +
        TimeToString(time_end, C_time_mode) + ", error " + ErrorDescription(last_error) + ", symbol " + symbol);
    return ArraySize(ticks);
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int tempusapi::copy_rates_safe(const string &symbol, const ENUM_TIMEFRAMES time_frame, const datetime start, const datetime end, MqlRates &rates[])
{
    int copied_ct = CopyRates(symbol, time_frame, start, end, rates);
    int last_error;
    if(copied_ct < 0) {
        last_error = GetLastError();
        ResetLastError();
        uint retries = 0;
        while(last_error == ERR_HISTORY_LOAD_ERRORS || last_error == ERR_HISTORY_TIMEOUT || last_error == ERR_HISTORY_NOT_FOUND || ERR_NOT_ENOUGH_MEMORY) {
            LOG_ERROR("Retrying last_error " + ErrorDescription(last_error) + " for " + symbol);
            Sleep(1);
            copied_ct = CopyRates(symbol, time_frame, start, end, rates);
            if(copied_ct > 0)
                break;
            last_error = GetLastError();
            ResetLastError();
            ++retries;
            if(retries > C_retries_count) break;
        }
    }
    return copied_ct;
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
// TODO Implement aux input column as commented out below
uint tempusapi::prepare_history_avg(const datetime start_time, const datetime end_time)
{
    if(ArraySize(hi_bars) > 0)
        ArrayFree(hi_bars);
    if(ArraySize(lo_bars) > 0)
        ArrayFree(lo_bars);
    if(ArraySize(open_bars) > 0)
        ArrayFree(open_bars);
    if(ArraySize(close_bars) > 0)
        ArrayFree(close_bars);

    if(start_time >= end_time) {
        LOG_ERROR("Illegal time range " + TimeToString(start_time, C_time_mode) + " until " + TimeToString(end_time, C_time_mode));
        return 0;
    }
    const int start_bar = iBarShift(symbol, period, start_time, false);
    if(start_bar < 0) {
        LOG_ERROR("Bar for start time " + TimeToString(start_time, C_time_mode) + " not found: " + ErrorDescription(GetLastError()));
        ResetLastError();
        return 0;
    } else if(iTime(symbol, period, start_bar) != start_time)
        LOG_INFO("Start time " + TimeToString(start_time, C_time_mode) + " does not equal bar time " + TimeToString(iTime(symbol, period, start_bar), C_time_mode));

    const int end_bar = iBarShift(symbol, period, end_time, false);
    if(end_bar < 0) {
        LOG_ERROR("Bar for end time " + TimeToString(end_time, C_time_mode) + " not found " + ErrorDescription(GetLastError()));
        ResetLastError();
        return 0;
    } else if(iTime(symbol, period, end_bar) != end_time)
        LOG_INFO("End time " + TimeToString(end_time, C_time_mode) + " does not equal bar time " + TimeToString(iTime(symbol, period, end_bar), C_time_mode));

// end_bar is smaller than start_bar
    const uint bar_ct = (per_second ? period_sec : 1) * (1 + start_bar - end_bar);
    LOG_INFO("Preparing up to " + IntegerToString(bar_ct) + " bars for period " + TimeToString(start_time, C_time_mode) + " to " + 
        TimeToString(end_time, C_time_mode) + ", symbol " + symbol);
    ArrayResize(avg_bars, bar_ct);
    ArrayResize(times, bar_ct);
    ArrayResize(volumes, bar_ct);
    uint              ok_bars         = 0;
    static const int available_bars  = iBars(symbol, period);
    aprice           prev_tick_price = get_rate(start_bar, e_rate_type::price_open);
    for(int bar_ix = start_bar; bar_ix >= end_bar; --bar_ix) {
        datetime bar_time      = iTime(symbol, period, bar_ix);
        if (bar_time > end_time) {
            LOG_ERROR("Bar " + IntegerToString(bar_ix) + " time is after end time " + TimeToString(end_time, C_time_mode) + ", skipping.");
            continue;
        }
        if (bar_time < start_time) bar_time = start_time;
        const string   bar_time_str  = TimeToString(bar_time, C_time_mode);

        datetime next_bar_time;
        if (per_second) 
            next_bar_time = bar_time + period_sec;
        else 
            next_bar_time = bar_ix < 1 ? iTime(symbol, period, 0) - (bar_ix - 1) * period_sec : iTime(symbol, period, bar_ix - 1);
        if (next_bar_time > end_time) next_bar_time = end_time;
        if (next_bar_time <= bar_time) {
            LOG_ERROR("Next bar time " + TimeToString(next_bar_time, C_time_mode) + ", less or equal bar time " + TimeToString(bar_time, C_time_mode) + ", skipping.");
            continue;
        }
        
        const datetime prev_bar_time = bar_ix >= available_bars ? bar_time - period_sec : iTime(symbol, period, bar_ix + 1);

        if(bar_time == 0) {
            LOG_ERROR("Bar time or price not found for " + symbol + ", from " + bar_time_str + ", to " + TimeToString(next_bar_time, C_time_mode) +
                      ", error " + ErrorDescription(GetLastError()));
            ResetLastError();
            continue;
        }
        MqlTick   ticks[];
        const int ticks_count = copy_ticks_safe(symbol, ticks, COPY_TICKS_ALL, prev_bar_time, next_bar_time);
        if (ticks_count < 1 && // handle no tick data available
            (iHigh(symbol, period, bar_ix) != iLow(symbol, period, bar_ix) || iOpen(symbol, period, bar_ix) != iClose(symbol, period, bar_ix) || iOpen(symbol, period, bar_ix) != prev_tick_price.bid)) {
            LOG_ERROR("Inconsistency between ticks and bars at " + bar_time_str + ", high doesn't equal low but no ticks for bar copied. ");
#ifndef TOLERATE_DATA_INCONSISTENCY
            break;
#endif
            LOG_ERROR("Using one minute rates for " + bar_time_str + " which is imprecise!");
            MqlRates           rates[];
            const int          rates_ct = copy_rates_safe(symbol, PERIOD_M1, bar_time, next_bar_time - 1, rates);
            const AveragePrice current_price(rates, rates_ct, bar_time);
            if(per_second) {
                const uint last_bar = MathMin(ArraySize(avg_bars), period_sec + ok_bars);
                for (uint i = ok_bars; i < last_bar; ++i) {
                    avg_bars[i] = current_price.value;
                    times[i]    = current_price.tm;
                    volumes[i]  = current_price.volume;
                }
            } else {
                avg_bars[ok_bars] = current_price.value;
                times[ok_bars]    = current_price.tm;
                volumes[ok_bars]  = current_price.volume;
                ++ok_bars;
            }
            prev_tick_price = current_price.close_price;
            continue;
        }
        if (per_second) {
            const uint n_bars = uint(bar_time + period_sec > next_bar_time ? next_bar_time - bar_time : period_sec);
            prev_tick_price = persec_prices(prev_tick_price, ticks, bar_time, n_bars, avg_bars, times, volumes, ok_bars);
            if(ticks_count < 1) {
                LOG_ERROR("No ticks found for " + bar_time_str + " using close price for bar " + IntegerToString(bar_ix));
                prev_tick_price = get_rate(bar_ix, e_rate_type::price_close);
            }
            ok_bars += n_bars;
        } else {
            const AveragePrice current_price(ticks, bar_time, period_sec, prev_tick_price);
            if(!current_price.value.valid()) {
                LOG_ERROR("Illegal price for " + TimeToString(bar_time, C_time_mode));
#ifdef TOLERATE_DATA_INCONSISTENCY
                continue;
#else
                break;
#endif
            }
            avg_bars[ok_bars] = current_price.value;
            times[ok_bars]    = current_price.tm;
            volumes[ok_bars]  = current_price.volume;
            prev_tick_price   = current_price.close_price;
            ++ok_bars;
        }
        if (ok_bars % C_print_progress_every == 0)
            LOG_DEBUG("Collected " + IntegerToString(ok_bars) + " bars.");
    }
    ArrayRemove(avg_bars, ok_bars);
    ArrayRemove(times, ok_bars);
    ArrayRemove(volumes, ok_bars);
    LOG_INFO("Successfully prepared " + IntegerToString(ok_bars) + " bars out of " + IntegerToString(bar_ct) + " possible bars from " +
             TimeToString(start_time, C_time_mode) + " until " + TimeToString(end_time, C_time_mode));
    return ok_bars;
}


//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
uint tempusapi::prepare_history_ohlc(const datetime start_time, const datetime end_time)
{
    if(per_second) {
        LOG_ERROR("No OHLC when sending high frequency data!");
        return 0;
    }
    if(ArraySize(avg_bars) > 0) ArrayFree(avg_bars);
    if(ArraySize(aux_avg_bars) > 0) ArrayFree(aux_avg_bars);
    LOG_INFO("Preparing from " + TimeToString(start_time, C_time_mode) + " until " + TimeToString(end_time, C_time_mode));
    const int start_bar = iBarShift(symbol, period, start_time, false);
    if(start_bar < 0) {
        LOG_ERROR("Bar for start time " + TimeToString(start_time) + " not found: " + ErrorDescription(GetLastError()));
        ResetLastError();
        return 0;
    }
    const int end_bar = iBarShift(symbol, period, end_time, false);
    if(end_bar < 0) {
        LOG_ERROR("Bar for end time " + TimeToString(end_time) + " not found: " + ErrorDescription(GetLastError()));
        ResetLastError();
        return 0;
    }
    if (start_bar < end_bar) {
        LOG_ERROR("Start bar " + IntegerToString(start_bar) + " smaller than " + IntegerToString(end_bar));
        return 0;
    }
// end_bar is smaller than start_bar
    const uint bar_ct = 1 + start_bar - end_bar;
    ArrayResize(times, bar_ct);
    ArrayResize(hi_bars, bar_ct);
    ArrayResize(lo_bars, bar_ct);
    ArrayResize(open_bars, bar_ct);
    ArrayResize(close_bars, bar_ct);
    ArrayResize(times, bar_ct);
    ArrayResize(volumes, bar_ct);
    MqlRates rates[];
    ArraySetAsSeries(rates, true);
    CopyRates(symbol, period, end_bar, bar_ct, rates);
    for(uint i = 0; i < bar_ct; ++i) {
        open_bars[i].set(rates[i], e_rate_type::price_open);
        hi_bars[i].set(rates[i], e_rate_type::price_high);
        lo_bars[i].set(rates[i], e_rate_type::price_low);
        close_bars[i].set(rates[i], e_rate_type::price_close);
        times[i]   = rates[i].time;
        volumes[i] = uint32_t(rates[i].tick_volume);
    }
    LOG_INFO("Copied " + IntegerToString(bar_ct) + " bars from " + TimeToString(start_time, C_time_mode) + " until " + TimeToString(end_time, C_time_mode));
    return bar_ct;
}


//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool tempusapi::send_history(const datetime start_time, datetime end_time)
{
    if (demo_mode) end_time = per_second ? end_time - demo_bars : end_time - demo_bars * period_sec;
    const uint copied_ct = is_average ? prepare_history_avg(start_time, end_time) : prepare_history_ohlc(start_time, end_time);
    if (copied_ct < 1) {
        LOG_ERROR("Failed preparing history bars.");
        return false;
    }
    LOG_INFO("Prepared " + IntegerToString(copied_ct) + " price bars.");
    const int bar_count = ArraySize(times);

#ifdef HISTORYTOSQL
    LOG_INFO("Writing history to MQL5/Files");
    const int    file_handle      = FileOpen("history_bars_" + input_queue + "_" + IntegerToString(per_second ? 1 : period_sec) + ".sql", FILE_WRITE | FILE_ANSI | FILE_CSV);
    const string time_current_str = TimeToString(TimeLocal(), C_time_mode);
    if(file_handle != INVALID_HANDLE) {
        FileSeek(file_handle, 0, SEEK_END);
        if(is_average)
            for(int bar_ix = 0; bar_ix < bar_count; ++bar_ix)
                FileWrite(file_handle, TimeToString(times[bar_ix], C_time_mode), time_current_str, IntegerToString(volumes[bar_ix]),
                          DoubleToString(avg_bars[bar_ix].bid, DOUBLE_PRINT_DECIMALS), DoubleToString(avg_bars[bar_ix].ask, DOUBLE_PRINT_DECIMALS));
        else
            for(int bar_ix = 0; bar_ix < bar_count; ++bar_ix)
                FileWrite(file_handle, TimeToString(times[bar_ix], C_time_mode), time_current_str, IntegerToString(volumes[bar_ix]),
                          DoubleToString(open_bars[bar_ix].bid, DOUBLE_PRINT_DECIMALS), DoubleToString(open_bars[bar_ix].ask, DOUBLE_PRINT_DECIMALS),
                          DoubleToString(hi_bars[bar_ix].bid, DOUBLE_PRINT_DECIMALS), DoubleToString(hi_bars[bar_ix].ask, DOUBLE_PRINT_DECIMALS),
                          DoubleToString(lo_bars[bar_ix].bid, DOUBLE_PRINT_DECIMALS), DoubleToString(lo_bars[bar_ix].ask, DOUBLE_PRINT_DECIMALS),
                          DoubleToString(close_bars[bar_ix].bid, DOUBLE_PRINT_DECIMALS), DoubleToString(close_bars[bar_ix].ask, DOUBLE_PRINT_DECIMALS));
        FileFlush(file_handle);
        FileClose(file_handle);
    }
    return false;
#endif

    if (use_http) {
        string header = "Time:date;";
        if(is_average) {
            header += input_queue + "_bid:double;" + input_queue + "_ask:double;";
            if(aux_queue_name != "")
                header += aux_queue_name + "_bid:double;" + aux_queue_name + "_ask:double;";
            header += "Volume:int";
        } else
            header += "High:double;Low:double;Open:double;Close:double;Volume:int";

        const int    bar_count_1 = bar_count - 1;
        int          bar         = 0;
        while(bar < bar_count) {
            Hash params;
            params.hPutString("barFrom", IntegerToString(bar));
            params.hPutString("symbol", input_queue);
            params.hPutString("period", period_str);
            params.hPutString("header", header);
            params.hPutString("delimiter", delimiter);
            const uint finished_at = is_average ? pack_history_avg(bar, bar_count_1, params) : pack_history_ohlc(bar, bar_count_1, params);
            params.hPutString("barTo", IntegerToString(finished_at));
            string       response;
            uint         retries     = 0;
            const string queue_call   = "queue";
            const string history_data = "historyData";
            while ((!net.RpcCall(queue_call, history_data, params, response) || !request_utils.is_response_successful(response)) && retries < C_max_retries) {
                if(StringLen(response) > 0)
                    LOG_ERROR("Server returned: " + request_utils.get_error_msg(response));
                if(retries < C_max_retries)
                    LOG_ERROR("Error while sending historical data, retrying.");
                else
                    LOG_ERROR("Error while sending historical data. Skipping bars from " + IntegerToString(bar) + " to " + IntegerToString(finished_at));
                ++retries;
            }
            if (retries < C_max_retries) LOG_INFO("Successfully sent " + IntegerToString(finished_at) + " bars.");
            bar = int(finished_at);
            if (bar == bar_count_1) break;
        }
    }

    if (p_streams_messaging) {
        string columns[2];
        columns[0] = input_queue + "_bid";
        columns[1] = input_queue + "_ask";
        p_streams_messaging.send_queue(input_queue, columns, times, volumes, avg_bars, 0, bar_count);
    }

    if(ArraySize(aux_avg_bars) > 0)
        ArrayFree(aux_avg_bars);
    if(ArraySize(avg_bars) > 0)
        ArrayFree(avg_bars);
    if(ArraySize(close_bars) > 0)
        ArrayFree(close_bars);
    if(ArraySize(hi_bars) > 0)
        ArrayFree(hi_bars);
    if(ArraySize(lo_bars) > 0)
        ArrayFree(lo_bars);
    if(ArraySize(open_bars) > 0)
        ArrayFree(open_bars);
    if(ArraySize(times) > 0)
        ArrayFree(times);
    if(ArraySize(volumes) > 0)
        ArrayFree(volumes);
    return true;
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
// TODO Add time zone information
uint tempusapi::pack_history_ohlc(const uint bar_start, const uint bar_end, Hash &hash)
{
    uint         bar_ix          = bar_start;
    const string time_zone_delim = timezone + delimiter;
    for(uint pack_bar = 0; bar_ix <= bar_end && pack_bar < package_len; ++bar_ix, ++pack_bar)
        hash.hPutString(IntegerToString(bar_ix),
                        TimeToString(times[bar_ix], C_time_mode) + time_zone_delim +
                        DoubleToString(hi_bars[bar_ix].bid, DOUBLE_PRINT_DECIMALS) + delimiter + DoubleToString(hi_bars[bar_ix].ask, DOUBLE_PRINT_DECIMALS) + delimiter +
                        DoubleToString(lo_bars[bar_ix].bid, DOUBLE_PRINT_DECIMALS) + delimiter + DoubleToString(lo_bars[bar_ix].ask, DOUBLE_PRINT_DECIMALS) + delimiter +
                        DoubleToString(open_bars[bar_ix].bid, DOUBLE_PRINT_DECIMALS) + delimiter + DoubleToString(open_bars[bar_ix].ask, DOUBLE_PRINT_DECIMALS) + delimiter +
                        DoubleToString(close_bars[bar_ix].bid, DOUBLE_PRINT_DECIMALS) + delimiter + DoubleToString(close_bars[bar_ix].ask, DOUBLE_PRINT_DECIMALS) + delimiter +
                        IntegerToString(volumes[bar_ix], DOUBLE_PRINT_DECIMALS));
    return bar_ix;
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
uint tempusapi::pack_history_avg(const uint bar_start, const uint bar_end, Hash &hash)
{
    uint bar_ix = bar_start;
    if(aux_queue_name != "") {
        for(uint pack_bar = 0; bar_ix <= bar_end && pack_bar < package_len; ++bar_ix, ++pack_bar)
            hash.hPutString(
                IntegerToString(bar_ix), TimeToString(times[bar_ix], C_time_mode) + delimiter +
                DoubleToString(avg_bars[bar_ix].bid, DOUBLE_PRINT_DECIMALS) + delimiter + DoubleToString(avg_bars[bar_ix].ask, DOUBLE_PRINT_DECIMALS) + delimiter +
                DoubleToString(aux_avg_bars[bar_ix].bid, DOUBLE_PRINT_DECIMALS) + delimiter + DoubleToString(aux_avg_bars[bar_ix].ask, DOUBLE_PRINT_DECIMALS) + delimiter +
                IntegerToString(volumes[bar_ix]));
    } else {
        for(uint pack_bar = 0; bar_ix <= bar_end && pack_bar < package_len; ++bar_ix, ++pack_bar)
            hash.hPutString(
                IntegerToString(bar_ix), TimeToString(times[bar_ix], C_time_mode) + delimiter +
                DoubleToString(avg_bars[bar_ix].bid, DOUBLE_PRINT_DECIMALS) + delimiter + DoubleToString(avg_bars[bar_ix].ask, DOUBLE_PRINT_DECIMALS) + delimiter +
                IntegerToString(volumes[bar_ix]));
    }
    return bar_ix;
}


//+------------------------------------------------------------------+
