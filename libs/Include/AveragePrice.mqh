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
#include <WinUser32.mqh> 
#include <stdlib.mqh>

#define TIME_DATE_SECONDS (TIME_DATE | TIME_SECONDS)

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
struct AveragePrice
{
    double closePrice;
    AveragePrice(MqlRates &rates[], long size);
    AveragePrice(MqlRates &rates[], long size, datetime timeToSet);
    AveragePrice (const MqlTick &ticks[], const datetime barTime, const ulong durationSecs, const double startPrice);
    AveragePrice();
    ~AveragePrice();
    datetime tm;
    double value;
    long volume;
};

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void calc_msec_twap(const MqlTick &ticks[], ulong &tickIdx, const ulong ticksLen, const datetime timeIter, double &lastPrice)
{
    // Millisecond is the highest resolution MQL5 provides
    double msecPrices[1000];
    // Iterate through ticks for the current second
    ulong lastTickMsec = 0;
    if (ticks[tickIdx].time != timeIter) LOG_ERROR("", "Starting tick does not equal time iter!");
    const ulong startTickIdx = tickIdx;
    while (tickIdx < ticksLen && ticks[tickIdx].time == timeIter) {
        const ulong curTickMsec = ticks[tickIdx].time_msc % 1000;
//        if (ticks[tickIdx].time_msc / 1000 != ticks[tickIdx].time) 
//            LOG_ERROR("generateSecondPrices", "Tick " + string(tickIdx) + " time " + TimeToString(ticks[tickIdx].time, TIME_DATE_SECONDS) + " to msec time " + TimeToString(ticks[tickIdx].time_msc / 1000, TIME_DATE_SECONDS));
        ArrayFill(msecPrices, lastTickMsec, curTickMsec - lastTickMsec, lastPrice);
        lastPrice = ticks[tickIdx].bid;
        lastTickMsec = curTickMsec;
        ++tickIdx;
    }
    ArrayFill(msecPrices, lastTickMsec, ArraySize(msecPrices) - lastTickMsec, lastPrice);
    
    // Calculate the time-weighted average price
    lastPrice = 0;
    for (ulong t = 0; t < ArraySize(msecPrices); ++t) lastPrice += msecPrices[t];
    lastPrice /= double(ArraySize(msecPrices));
    //LOG_INFO("", "Used " + string(tickIdx - startTickIdx) + " ticks for TWAP at " + TimeToString(timeIter, TIME_DATE_SECONDS) + " price " + DoubleToString(lastPrice, 16));
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void
generateSecondPrices(
    const double startPrice,
    const MqlTick &ticks[],
    const datetime barTime,
    const ulong durationSecs,
    double &prices[],
    datetime &times[],
    ulong &volumes[])
{
    ArrayResize(prices, durationSecs);
    ArrayResize(times, durationSecs);
    ArrayResize(volumes, durationSecs);
    datetime timeIter = barTime;    
    const long ticksLen = ArraySize(ticks);
    if (ticksLen < 1) {
        ArrayFill(prices, 0, ArraySize(prices), startPrice);
        ArrayFill(volumes, 0, ArraySize(volumes), 0);
        for (ulong t = 0; t < durationSecs && timeIter < barTime + durationSecs; ++timeIter, ++t) times[t] = timeIter;
        LOG_ERROR("generateSecondPrices", "Ticks array is empty, copying start price!");
        return;
    }
    ulong tickIdx = 0, priceIx = 0;
    double lastPrice = startPrice == -1 ? (ArraySize(ticks) > 0 ? ticks[0].bid : 0) : startPrice; // Bar open price
    while (timeIter < barTime + durationSecs) {
        // Fast-forward tick to current time
        while (tickIdx < ticksLen && ticks[tickIdx].time < timeIter) lastPrice = ticks[tickIdx++].bid;
        if (tickIdx >= ticksLen || ticks[tickIdx].time > timeIter) {
            prices[priceIx] = lastPrice;
            volumes[priceIx] = 0;
            LOG_DEBUG("", "Ignoring tick for price " + string(priceIx) + " " + string(prices[priceIx]) + " because its later than iterator at " + TimeToString(timeIter, TIME_DATE_SECONDS) + " or tick index " + string(tickIdx) + " is past " + string(ticksLen));
        } else if (ticks[tickIdx].time == timeIter) {
            const ulong lastTick = tickIdx;
            double curPrice = lastPrice;
            calc_msec_twap(ticks, tickIdx, ticksLen, timeIter, curPrice);
            const ulong ticksTwap = tickIdx - lastTick;
            LOG_DEBUG("", "Calculated TWAP from " + string(ticksTwap) + " ticks, " + DoubleToString(lastPrice, TIME_DATE_SECONDS) + " for " + TimeToString(timeIter, TIME_DATE_SECONDS) + " tick index " + string(tickIdx) + " price index " + string(priceIx));
            prices[priceIx] = curPrice;
            if (tickIdx > 0) lastPrice = ticks[tickIdx - 1].bid;
            else LOG_ERROR("", "Tick index is zero, price is not initialized properly!");
            volumes[priceIx] = ticksTwap;
        }
        times[priceIx] = timeIter;
        ++priceIx;
        ++timeIter;
    }
    while (priceIx < ArraySize(prices)) prices[priceIx++] = lastPrice;

    const long barIx = iBarShift(_Symbol, _Period, barTime, false);
    const double barClose = iClose(_Symbol, _Period, barIx);
    if (lastPrice != barClose)
        LOG_ERROR("generateSecondPrices", "Last tick price " + DoubleToString(lastPrice, 16) + " doesn't equal bar " + IntegerToString(barIx)  + " at time " + TimeToString(barTime, TIME_DATE_SECONDS) + " close price " + DoubleToString(barClose, 16) + " last tick at " + TimeToString(ticks[ArraySize(ticks) - 1].time, TIME_DATE_SECONDS) + " price " + DoubleToString(ticks[ArraySize(ticks) - 1].bid));
    if (priceIx < ArraySize(times)) ArrayRemove(times, priceIx);
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
AveragePrice::AveragePrice(MqlRates &rates[], long size)
{
    this.value = size > 0 ? rates[0].open : 0.;
    this.volume = 0;

    // Time is set to the last one
    this.tm = rates[0].time;
    for (long i = 0; i < size; ++i) {
        this.value += rates[i].close + rates[i].low + rates[i].high;
        this.volume += rates[i].tick_volume;
    }
    this.closePrice = rates[size - 1].close;
    this.value = this.value / ((size > 0 ? 1. : 0) + size * 3.);
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
AveragePrice::AveragePrice(MqlRates &rates[], long size, datetime timeToSet)
{
    if (ArraySize(rates) < 1) {
        LOG_ERROR("", "Rates for " + TimeToString(timeToSet, TIME_DATE_SECONDS) + " are empty");
        return;
    }
    this.value = 0.;
    this.volume = 0;

    // Time is set to user defined
    this.tm = timeToSet;
    long rate_ct = 0;
    for(long i = 0; i < size; i++) {
        if (rates[i].time < timeToSet) continue;
        this.value += rates[i].close + rates[i].low + rates[i].high;
        this.volume += rates[i].tick_volume;
        ++rate_ct;
    }
    this.closePrice = rates[ArraySize(rates) - 1].close;
    this.value = this.value / (double(size) * 3.);
}


//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
AveragePrice::AveragePrice(const MqlTick &ticks[], const datetime barTime, const ulong durationSecs, const double startPrice)
{
    const ulong ticksLen = ArraySize(ticks);
    if (ticksLen < 1) {
        this.value = startPrice;
        this.closePrice = startPrice;
        this.tm = barTime;
        LOG_ERROR("", "Ticks array is empty!");
        return;
    }
    double prices[];
    datetime times[];
    ulong volumes[];
    generateSecondPrices(startPrice, ticks, barTime, durationSecs, prices, times, volumes);
    this.tm = barTime;
    this.value = 0;
    this.volume = 0;
    for (long t = 0; t < ArraySize(prices); ++t) {
        this.value += prices[t];
        this.volume += volumes[t];
    }
    this.value /= double(ArraySize(prices));
    this.closePrice = ticks[ArraySize(ticks) - 1].bid;
/*
    static int file_handle;
    if (barTime == StringToTime("2021-09-27 00:00:00")) {
        file_handle = FileOpen("prices1s.log", FILE_WRITE | FILE_CSV);
        if (file_handle != INVALID_HANDLE) {
            FileSeek(file_handle, 0, SEEK_END);
        }
    }
    for (long t = 0; t < ArraySize(prices); ++t) {
        if (barTime == StringToTime("2021-09-27 00:00:00")) {
            string msg = "Price 1s " + string(t) + ", start time " + TimeToString(barTime, TIME_DATE_SECONDS) + "\t" + TimeToString(times[t], TIME_DATE_SECONDS) + "\t" + DoubleToString(prices[t], 16);
            FileWrite(file_handle, msg);
        }
    }
    if (barTime == StringToTime("2021-09-27 00:00:00")) {
        string msg = "barTime\t" + TimeToString(this.tm, TIME_DATE_SECONDS) + "\t" + DoubleToString(this.value, 16);
        FileWrite(file_handle, msg);
        FileFlush(file_handle);
        FileClose(file_handle);
    }
*/
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
