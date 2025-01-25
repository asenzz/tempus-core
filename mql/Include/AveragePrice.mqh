//+------------------------------------------------------------------+
//|                                                 AveragePrice.mqh |
//|                                        Copyright 2019, Tempus CM |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2021, Tempus CM"
#property link      "https://www.mql5.com"
#property version   "1.00"
#property strict
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
#include "MqlLog.mqh"
#include "CompatibilityMQL4.mqh"
#include <WinUser32.mqh>
#include <stdlib.mqh>


enum e_rate_type {
    price_open = 0,
    price_high,
    price_low,
    price_close,
    price_hlc
};


//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void add_rate_spread(double &rate, const double spread_digits)
{
    rate += spread_digits / pow(10, _Digits);
}


class aprice {
    public:
    double bid, ask;

    string to_string() const
    {
        return "bid " + DoubleToString(bid, DOUBLE_PRINT_DECIMALS) + ", ask " + DoubleToString(ask, DOUBLE_PRINT_DECIMALS);
    }
    
    double spread() const
    {
        return ask - bid;
    }

    bool valid() const
    {
        return ask > 0 && bid > 0 && MathIsValidNumber(bid) && MathIsValidNumber(ask);
    }

    void set(const MqlRates &rate, const e_rate_type type)
    {
        switch (type) {
        case e_rate_type::price_open:
            ask = bid = rate.open;
            break;

        case e_rate_type::price_high:
            ask = bid = rate.high;
            break;

        case e_rate_type::price_low:
            ask = bid = rate.low;
            break;

        case e_rate_type::price_close:
            ask = bid = rate.close;
            break;

        case e_rate_type::price_hlc:
            ask = bid = (rate.high + rate.low + 2. * rate.close) / 4.;
            break;

        default:
            LOG_ERROR("Unknown value type " + string(type));
            DebugBreak();
        }

        add_rate_spread(ask, rate.spread);
    }

    void set(const MqlTick &tick)
    {
        bid = tick.bid;
        ask = tick.ask;
    }
    
    void set(const double bid_, const double ask_)
    {
        bid = bid_;
        ask = ask_;
    }
    
    void reset()
    {
        set(0, 0);
    }

    aprice *operator += (const aprice &o)
    {
        bid += o.bid;
        ask += o.ask;
        return &this;
    }

    aprice *operator += (const MqlTick &o)
    {
        ask += o.ask;
        bid += o.bid;
        return &this;
    }
        
    aprice *operator /= (const int o)
    {
        bid /= o;
        ask /= o;
        return &this;
    }

    aprice operator * (const int o)
    {
        return aprice(bid * o, ask * o);
    }

    aprice(const MqlRates &rate, const e_rate_type type) 
    { 
        set(rate, type);
    }

    aprice(const aprice &o) 
    {
        bid = o.bid;
        ask = o.ask;
    }

    aprice() { reset(); }

    aprice (const double bid_, const double ask_) : bid(bid_), ask(ask_) {}
};


aprice get_price(const datetime at)
{
    static const uint max_retries = 10;
    uint retries = 0;
    datetime start_time = iTime(_Symbol, _Period, iBarShift(_Symbol, _Period, at) + 1);
    if (start_time == at) start_time -= C_period_seconds;
    const ulong atms = at * 1000;
    MqlTick ticks[];
    do {
        CopyTicksRange(_Symbol, ticks, COPY_TICKS_ALL, start_time * 1000, atms);
        start_time -= C_period_seconds;
        ++retries;
    } while(ArraySize(ticks) < 1 && retries < max_retries);
    aprice res;
    if (ArraySize(ticks) < 1) return res;
    res.set(ticks[ArraySize(ticks) - 1]);
    return res;
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
aprice get_rate(const string &symbol, const int shift, const e_rate_type type)
{
    MqlRates rate[];
    ArraySetAsSeries(rate, true);
    const int copied = CopyRates(symbol, _Period, shift, 1, rate);
    if (copied < 1) {
        LOG_ERROR("Failed copying rates for " + IntegerToString(shift));
        return aprice();
    }
    aprice result(rate[0], type);
    return result;
}


//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
aprice get_rate(const int shift, const e_rate_type type)
{
    return get_rate(_Symbol, shift, type);
}


//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
struct AveragePrice {
    aprice close_price;
    datetime tm;
    aprice value;
    uint volume;
    
    AveragePrice(const MqlRates &rates[], const int size);
    AveragePrice(const MqlRates &rates[], const int size, const datetime time_set);
    AveragePrice(const MqlTick &ticks[], const datetime bar_time, const int duration_sec, const aprice &start_price);
    AveragePrice();
    ~AveragePrice();
};

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
aprice calc_msec_twap(const MqlTick &ticks[], int &tick_ix, const int ticks_len, const datetime time_iter, aprice &twap_price, uint &volume, const aprice &cur_price_)
{
    aprice cur_price = cur_price_;
    if (ticks[tick_ix].time != time_iter) LOG_ERROR("Starting tick does not equal time iter!");
    int last_ms = 0;
    const int start_tick_ix = tick_ix;
    for (; tick_ix < ticks_len && ticks[tick_ix].time == time_iter; ++tick_ix) {
        const int cur_ms = int(ticks[tick_ix].time_msc % MILLISECONDS_IN_SECOND);
        twap_price += cur_price * (cur_ms - last_ms);
        last_ms = cur_ms;
        cur_price.set(ticks[tick_ix]);
    }
    twap_price += cur_price * (MILLISECONDS_IN_SECOND - last_ms);
    twap_price /= MILLISECONDS_IN_SECOND;
    volume = tick_ix - start_tick_ix;
    return cur_price;
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
aprice persec_prices( // returns last processed tick price
    const aprice &start_price,
    const MqlTick &ticks[],
    const datetime bar_time,
    const int duration_sec,
    aprice &prices[],
    datetime &times[],
    uint &volumes[],
    const int start_out)
{
    if (duration_sec < 1) {
        LOG_ERROR("Duration seconds is illegal " + string(duration_sec));
        return aprice();
    }
    datetime time_iter = bar_time;
    const int ticks_len = ArraySize(ticks);
    aprice last_tick_price;
    if (ticks_len > 0 && (ticks[0].time <= bar_time || !start_price.valid()))
        last_tick_price.set(ticks[0]);
    else
        last_tick_price = start_price;

    const int end_out = start_out + duration_sec;
    if (ticks_len < 1) {
        LOG_ERROR("Ticks array is empty, copying open price " + last_tick_price.to_string());
        for (int i = start_out; i < end_out; ++i, ++time_iter) {
            prices[i] = last_tick_price;
            times[i] = time_iter;
            volumes[i] = 0;
        }
        return last_tick_price;
    }

    for (int tick_ix = 0, price_ix = start_out; price_ix < end_out; ++price_ix, ++time_iter) {
        times[price_ix] = time_iter;
        
        for (; tick_ix < ticks_len && ticks[tick_ix].time < time_iter; ++tick_ix) last_tick_price.set(ticks[tick_ix]);
        
        if (tick_ix >= ticks_len || ticks[tick_ix].time > time_iter) {
            volumes[price_ix] = 0;
            prices[price_ix] = last_tick_price;
        } else if (ticks[tick_ix].time == time_iter)
            last_tick_price = calc_msec_twap(ticks, tick_ix, ticks_len, time_iter, prices[price_ix], volumes[price_ix], last_tick_price);
    }

    return last_tick_price;
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
AveragePrice::AveragePrice(const MqlRates &rates[], const int size)
{
    if (ArraySize(rates) < 1) {
        LOG_ERROR("Rates are empty, user provided length " + IntegerToString(size));
        return;
    }
    AveragePrice(rates, size, rates[0].time);
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
AveragePrice::AveragePrice(const MqlRates &rates[], const int size, const datetime time_set)
{
    if (ArraySize(rates) < 1) {
        LOG_ERROR("Rates for " + TimeToString(time_set, C_time_mode) + " are empty");
        return;
    }
    value.set(rates[0], e_rate_type::price_open);
    volume = 0;
    // Time is set to user defined
    tm = time_set;
    for (int i = 0; i < size; ++i) {
        if (rates[i].time < time_set) continue;
        value += aprice(rates[i], e_rate_type::price_hlc);
        volume += (uint) rates[i].tick_volume;
    }
    close_price.set(rates[0], e_rate_type::price_close);
    value /= 1 + size;
}


//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
AveragePrice::AveragePrice(const MqlTick &ticks[], const datetime bar_time, const int duration_sec, const aprice &start_price)
{
    volume = 0;
    tm = bar_time;
    const int ticks_len = ArraySize(ticks);
    if (ticks_len < 1) {
        if (start_price.valid()) {
            value = start_price;
            close_price = start_price;
        } else {
            value.reset();
            close_price.reset();
        }
        LOG_ERROR("Ticks array is empty!");
        return;
    }
   
    int tick_ix = 0;
    aprice last_tick_price;
    if (ticks[tick_ix].time <= bar_time || !start_price.valid())
        last_tick_price.set(ticks[tick_ix]);
    else
        last_tick_price = start_price;        
        
    for (datetime time_iter = bar_time; time_iter < bar_time + duration_sec; ++time_iter) {
        for (; tick_ix < ticks_len && ticks[tick_ix].time < time_iter; ++tick_ix) last_tick_price.set(ticks[tick_ix]);
              
        if (tick_ix >= ticks_len || ticks[tick_ix].time > time_iter) value += last_tick_price;
        else if (ticks[tick_ix].time == time_iter) {
            aprice cur_price;
            uint cur_vol;
            last_tick_price = calc_msec_twap(ticks, tick_ix, ticks_len, time_iter, cur_price, cur_vol, last_tick_price);
            value += cur_price;
            volume += cur_vol;
        }
    }

    value /= duration_sec;
    close_price = last_tick_price;
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
AveragePrice::~AveragePrice()
{
}
//+------------------------------------------------------------------+
//|
//+------------------------------------------------------------------+
AveragePrice::AveragePrice()
{
}
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
