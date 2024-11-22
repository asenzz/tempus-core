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

#define MAX_DOUBLE_PRECISION 16
#define MAX_PACK_SIZE 32768
#define MAX_TICKS_PER_SECOND 20


// Used for backtests, sleeps using actual local time
void sleep_local(const ulong sleep)
{
    const datetime sleep_until = GetWindowsLocalTime() + sleep;
    while (GetWindowsLocalTime() < sleep_until) {};
}


//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
datetime 
nearestBarTimeOrBefore(const datetime barTime)
{
    LOG_INFO("nearestBarTimeOrBefore", "Starting " + TimeToString(barTime, TIME_DATE_SECONDS));
    datetime barTimeIter = datetime(barTime / PeriodSeconds()) * PeriodSeconds();
    long barShift = iBarShift(Symbol(), Period(), barTimeIter, false);
    while (barShift == -1 || iTime(Symbol(), Period(), barShift) > barTime) {
        barTimeIter -= PeriodSeconds();
        barShift = iBarShift(Symbol(), Period(), barTimeIter, false);
    }
    const datetime result = iTime(Symbol(), Period(), barShift);
    LOG_INFO("", "Returning " + TimeToString(result, TIME_DATE_SECONDS) + " for bar time " + TimeToString(barTime, TIME_DATE_SECONDS));
    return result;
} 


//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class SVRApi
{
private:
    MqlNet net;
    RequestUtils requestUtils;
    bool histDataReconciled;
    ulong const packSize;
    string AuxInputQueue;
    string timeZoneInfo;
    bool OneSecondData;

    double myHi[], myLo[], myOp[], myCl[];
    double myBid[], myAuxBid[];
    long myVl[];
    datetime myTm[];

    bool DemoMode;
    long DemoBars;
    bool isAverage;

    ulong prepareHistoryPackage(const ulong barStart, const ulong barFinish, const string &delimiter, Hash &hash);

    bool SendHistory(const string &inputQueue, const string &period, const datetime startDate, const datetime endDate);

    long PrepareHistoryBidPrices(const datetime startTime, const datetime endTime);

    long PrepareHistoryBidPricesAvg(const datetime startTime, const datetime endTime);

    long PrepareHistoryBidPricesOHLC(const datetime startTime, const datetime endTime);

    ulong prepareAverageHistoryPackage(const ulong barStart, const ulong barFinish, const string &delimiter, Hash &hash);

    ulong findBar(const datetime startTime);

    ulong bSearchBar(const datetime startTime, const ulong left, const ulong right);

    bool GetCustomTimeRange(datetime& timeFrom, datetime& timeTo, const string &inputQueue, const string &period, const string &function, bool &recon_finished);

public:
    SVRApi(const bool DemoMode_, const long DemoBars_, const bool isAverage_, const string AuxInputQueue_, const bool OneSecondData, const string timeZoneInfo_);

    ~SVRApi();

    bool GetNextTimeRange(datetime& barStart, datetime& barFinish, const string &inputQueue, const string &period);
    //bool ReconcileHistoricalData(string inputQueue, string period);

    bool Connect(const string &URL);

    bool Login(const string &Username, const string &Password);

    void SendBar(const string &inputQueue, const string &period, const bool finalize, const double op, const double hi, const double lo, const double cl, const long vol, const datetime tm);
    void SendBar(const string &inputQueue, const string &period, const bool finalize, const AveragePrice &avg_price);
    void SendBar(const string &inputQueue, const string &period, const bool finalize, const AveragePrice &avg_price, const AveragePrice &avg_aux_price, const string &col_name, const string &col_aux_name);
    void SendBars(const string &inputQueue, const string &period, const bool finalize, const double &prices[], const ulong &volumes[], const double &pricesaux[], const ulong &volumesaux[], const datetime &times[], const string &col_name, const string &aux_col_name);
    void SendBars(const string &inputQueue, const string &period, const bool finalize, const double &prices[], const datetime &times[], const ulong &volumes[]);
    bool SendHistory(const string &inputQueue, const string &period);

    static long CopyRatesSafe(const string &symbol, const ENUM_TIMEFRAMES time_frame, const datetime copied_period_start, const datetime copied_period_end, MqlRates &rates[]);
    static long CopyTicksSafe(const string &symbol, MqlTick &ticks[], const uint flags, const datetime timeStart, const datetime timeEnd);
    bool getHistDataReconciled() const
    {
        return histDataReconciled;
    }
};

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
SVRApi::SVRApi(const bool DemoMode_, const long DemoBars_, const bool isAverage_, const string AuxInputQueue_ = "", const bool OneSecondData_ = false, const string timeZoneInfo_ = "") : 
    histDataReconciled(false), packSize(MAX_PACK_SIZE), DemoMode(DemoMode_), DemoBars(DemoBars_), isAverage(isAverage_), AuxInputQueue(AuxInputQueue_), OneSecondData(OneSecondData_), timeZoneInfo(timeZoneInfo_)
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
    return (net.Open(Url));
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool SVRApi::Login(const string &username, const string &password)
{
    string LoginUrl = "mt4/login";
    string PostBody = "username=" + username + "&password=" + password;
    long StatusCode;
    net.Post(LoginUrl, PostBody, StatusCode);
    return (StatusCode == 200);
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void SVRApi::SendBar(const string &inputQueue, const string &period, const bool finalize, const double op, const double hi, const double lo, const double cl, const long vol, const datetime tm)
{
    string response;
    Hash param;

    const string strTm = TimeToString(tm, TIME_DATE_SECONDS);
    param.hPutString("symbol", inputQueue);
    param.hPutString("time", strTm);
    param.hPutString("open", string(op));
    param.hPutString("high", string(hi));
    param.hPutString("low", string(lo));
    param.hPutString("close", string(cl));
    param.hPutString("Volume", string(vol));
    param.hPutString("isFinal", finalize ? "true" : "false");
    param.hPutString("period", period);
    param.hPutString("clientTime", TimeToString(TimeCurrent(), TIME_DATE_SECONDS));
    const string queue_call = "queue";
    const string send_tick_call = "sendTick";
    if (!net.RpcCall(queue_call, send_tick_call, param, response))
        return;

    if (StringLen(response) > 0 && !requestUtils.isSuccessfulResponse(response))
        LOG_ERROR("SVRApi::SendBar", "Server returned: " + requestUtils.getErrorMessage(response));

    LOG_INFO("SVRApi::SendBar", "Finalized bar at: " + strTm);
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void SVRApi::SendBar(const string &inputQueue, const string &period, const bool finalize, const AveragePrice &avg_price)
{
    string response;
    Hash param;

    ulong bar = 0;
    const string strTm = TimeToString(avg_price.tm, TIME_DATE_SECONDS);
    param.hPutString("symbol", inputQueue);
    param.hPutString("time", strTm);
    param.hPutString(inputQueue + "_bid", string(avg_price.value));
    param.hPutString("Volume", string(avg_price.volume));
    param.hPutString("isFinal", finalize ? "true" : "false");
    param.hPutString("period", period);
    param.hPutString("clientTime", TimeToString(TimeCurrent(), TIME_DATE_SECONDS));

    const string queue_call = "queue";
    const string send_tick_call = "sendTick";
    if (!net.RpcCall(queue_call, send_tick_call, param, response))
        return;

    if (StringLen(response) > 0 && !requestUtils.isSuccessfulResponse(response))
        LOG_ERROR("SVRApi::SendBar", "Server returned: " + requestUtils.getErrorMessage(response));

    LOG_INFO("SVRApi::SendBar", "Finalized bar at: " + strTm + " price " + string(avg_price.value));
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void
SVRApi::SendBar(
    const string &inputQueue,
    const string &period,
    const bool finalize,
    const AveragePrice &avg_price,
    const AveragePrice &avg_price_aux,
    const string &col_name,
    const string &aux_col_name)
{
    string response;
    Hash param;

    const string strTm = TimeToString(avg_price.tm, TIME_DATE_SECONDS);
    param.hPutString("symbol", inputQueue);
    param.hPutString("time", strTm);
    param.hPutString(col_name, string(avg_price.value));
    param.hPutString(aux_col_name, string(avg_price_aux.value));
    param.hPutString("Volume", string(avg_price.volume));
    param.hPutString("isFinal", finalize ? "true" : "false");
    param.hPutString("period", period);
    param.hPutString("clientTime", TimeToString(TimeCurrent(), TIME_DATE_SECONDS));

    const string queue_call = "queue";
    const string send_tick_call = "sendTick";
    if (!net.RpcCall(queue_call, send_tick_call, param, response)) return;

    if (StringLen(response) > 0 && !requestUtils.isSuccessfulResponse(response))
        LOG_ERROR("SVRApi::SendBar", "Server returned: " + requestUtils.getErrorMessage(response));

    LOG_INFO("SVRApi::SendBar", "Finalized bar at: " + strTm);
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void SVRApi::SendBars(const string &inputQueue, const string &period, const bool finalize, const double &prices[], const datetime &times[], const ulong &volumes[])
{
    const ulong pricesLen = ArraySize(prices);
    if (pricesLen != ArraySize(times)) {
        LOG_ERROR("SVRApi::SendBars", "Prices length " + string(pricesLen) + " does not equal times length " + string(ArraySize(times)));
        return;
    }

    for (int i = 0; i < pricesLen; ++i) {
        string response;
        Hash param;
    
        ulong bar = 0;
        const string strTm = TimeToString(times[i], TIME_DATE_SECONDS);
        param.hPutString("symbol", inputQueue);
        param.hPutString("time", strTm);
        param.hPutString(inputQueue + "_bid", DoubleToString(prices[i], 16)); // Column name
        param.hPutString("Volume", IntegerToString(volumes[i]));
        param.hPutString("isFinal", finalize ? "true" : "false");
        param.hPutString("period", period);
        param.hPutString("clientTime", TimeToString(TimeCurrent(), TIME_DATE_SECONDS));
    
        const string queue_call = "queue";
        const string send_tick = "sendTick";
        if (!net.RpcCall(queue_call, send_tick, param, response)) return;
    
        if (StringLen(response) > 0 && !requestUtils.isSuccessfulResponse(response))
            LOG_ERROR("SVRApi::SendBar", "Server returned: " + requestUtils.getErrorMessage(response));
        LOG_INFO("SVRApi::SendBar", "Finalized bar at: " + strTm + " price " + string(prices[i]));
    }
}


//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void SVRApi::SendBars(
    const string &inputQueue,
    const string &period, 
    const bool finalize, 
    const double &prices[],
    const ulong &volumes[],
    const double &pricesaux[],
    const ulong &volumesaux[],
    const datetime &times[],
    const string &col_name,
    const string &aux_col_name)
{
    const ulong pricesLen = ArraySize(prices);
    if (pricesLen != ArraySize(times) || pricesLen != ArraySize(pricesaux)) {
        LOG_ERROR("SVRApi::SendBars", "Prices length " + string(pricesLen) + " does not equal times length " + string(ArraySize(times)) + ", aux prices length " + string(ArraySize(pricesaux)));
        return;
    }

    for (int i = 0; i < pricesLen; ++i) {
        string response;
        Hash param;
    
        ulong bar = 0;
        const string strTm = TimeToString(times[i], TIME_DATE_SECONDS);
        param.hPutString("symbol", inputQueue);
        param.hPutString("time", strTm);
        param.hPutString(col_name, DoubleToString(prices[i], 16));
        param.hPutString(aux_col_name, DoubleToString(pricesaux[i], 16));
        param.hPutString("Volume", IntegerToString(volumes[i] + volumesaux[i]));
        param.hPutString("isFinal", finalize ? "true" : "false");
        param.hPutString("period", period);
        param.hPutString("clientTime", TimeToString(TimeCurrent(), TIME_DATE_SECONDS));
    
        const string queue = "queue";
        const string send_tick = "sendTick";
        if (!net.RpcCall(queue, send_tick, param, response)) return;
    
        if (StringLen(response) > 0 && !requestUtils.isSuccessfulResponse(response))
            LOG_ERROR("", "Server returned: " + requestUtils.getErrorMessage(response));
        else
            LOG_INFO("", "Finalized bar at: " + strTm + " price " + string(prices[i]));
    }
}


//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool SVRApi::GetNextTimeRange(datetime& timeFrom, datetime& timeTo, const string &inputQueue, const string &period)
{
    const string funName = "getNextTimeRangeToBeSent";
    bool reconFinished = false;
    const bool result = GetCustomTimeRange(timeFrom, timeTo, inputQueue, period, funName, reconFinished);
    if (result)
        LOG_INFO("SVRApi::GetNextTimeRange", "The requested interval is: " + TimeToString(timeFrom, TIME_DATE_SECONDS) + " to " + TimeToString(timeTo, TIME_DATE_SECONDS));
    return result;
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class IntervalRetryCounter
{
private:
    ulong barNo_;
    ulong retryCount_;
    const ulong maxRetryCount_;
public:
    IntervalRetryCounter(const ulong barNo): maxRetryCount_(3)
    {
        retryCount_ = 0;
        barNo_ = barNo;
    }

    bool retry()
    {
        return retryCount_++ < maxRetryCount_;
    }

    ulong barNo() const
    {
        return barNo_;
    }

    void barNo(const ulong barNo)
    {
        retryCount_ = 0;
        barNo_ = barNo;
    }
};

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
ulong SVRApi::bSearchBar(datetime startTime, ulong left, ulong right)
{
    ulong count = left - right, it, step;
    while( count > 0) {
        it = left;
        step = count / 2;
        it -= step;
        if (GetTargetTime(it) < startTime) {
            left = --it;
            count -= step + 1;
        } else
            count = step;
    }
    return left;
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
ulong SVRApi::findBar(const datetime startTime)
{
    const ulong e = Bars(_Symbol, _Period) - 1;
    const datetime lastBar = GetTargetTime(e);

    if(lastBar > startTime) return e;
    ulong timeRangeLimit = ulong((GetTargetTime(0) - startTime) / PeriodSeconds());

    timeRangeLimit = MathMin(timeRangeLimit, e);

    return bSearchBar(startTime, timeRangeLimit, 0);

}


//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool SVRApi::SendHistory(const string &inputQueue, const string &period)
{
    datetime startTime, endTime;
    while(GetNextTimeRange(startTime, endTime, inputQueue, period)) {
        if (!SendHistory(inputQueue, period, startTime, endTime)) return false;
        LOG_INFO("SVRApi::SendHistory", "Successfuly sent bars." );
    }
    return true;
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
long SVRApi::CopyTicksSafe(const string &symbol, MqlTick &ticks[], const uint flags, const datetime timeStart, const datetime timeEnd)
{
    const ulong timeStartMsec = timeStart * MILLISECONDS_IN_SECOND;
    const ulong maxTicksCount = (timeEnd - timeStart) * MAX_TICKS_PER_SECOND;
    long copiedCt = CopyTicks(symbol, ticks, flags, timeStartMsec, maxTicksCount);
    if (copiedCt < 1) {
        uint retries = 0;
        while (copiedCt == -1 && retries++ < 10 && (GetLastError() == ERR_HISTORY_TIMEOUT || GetLastError() == ERR_HISTORY_LOAD_ERRORS || GetLastError() == ERR_HISTORY_NOT_FOUND))
            copiedCt = CopyTicks(symbol, ticks, flags, timeStartMsec, maxTicksCount);
        if (copiedCt < 1) {
            LOG_ERROR("SVRApi::CopyTicksSafe", "No ticks copied for " + symbol + " from " + TimeToString(timeStart) + " to " + TimeToString(timeEnd));
            return 0;
        }
    }

    ulong tickIx = 0;
    while (tickIx < ArraySize(ticks) && ticks[tickIx].time < timeEnd) ++tickIx;
    if (tickIx < ArraySize(ticks)) {
        LOG_DEBUG("SVRApi::CopyTicksSafe", "Trimming " + string(ArraySize(ticks) - tickIx) + " ticks.");
        ArrayRemove(ticks, tickIx, WHOLE_ARRAY);
    }
    return ArraySize(ticks);
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
long SVRApi::CopyRatesSafe(const string &symbol, const ENUM_TIMEFRAMES time_frame, const datetime copied_period_start, const datetime copied_period_end, MqlRates &rates[])
{
    string copied_symbol = symbol;
    StringToUpper(copied_symbol);

    long copied_ct = CopyRates(copied_symbol, time_frame, copied_period_start, copied_period_end, rates);
    long lastError = GetLastError();
    long retries = 0;
    while (lastError == ERR_HISTORY_LOAD_ERRORS || lastError == ERR_HISTORY_TIMEOUT || lastError == ERR_HISTORY_NOT_FOUND) {
        LOG_ERROR("", "Retrying lastError " + ErrorDescription(lastError) + " for " + copied_symbol);
        ResetLastError();
        Sleep(50);
        copied_ct = CopyRates(copied_symbol, time_frame, copied_period_start, copied_period_end, rates);
        lastError = GetLastError();
        ++retries;
        if (retries > 5) break;
    }
    if (lastError != 0) {
        LOG_ERROR("", "lastError " + string(lastError) + " for " + copied_symbol);
        ResetLastError();
        copied_ct = CopyRates(copied_symbol, time_frame, copied_period_start, copied_period_end, rates);
    }

    return copied_ct;
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
// TODO Implement aux input column
long SVRApi::PrepareHistoryBidPricesAvg(const datetime startTime, const datetime endTime)
{
    if (startTime >= endTime) {
        LOG_ERROR("SVRApi::PrepareHistoryBidPricesAvg", "Illegal time range " + TimeToString(startTime, TIME_DATE_SECONDS) + " until " + TimeToString(endTime, TIME_DATE_SECONDS));
        return 0;
    }
    const long startBar = iBarShift(_Symbol, _Period, startTime, false);
    if (startBar < 0) {
        LOG_ERROR("SVRApi::PrepareHistoryBidPricesAvg", "Bar for start time " + TimeToString(startTime, TIME_DATE_SECONDS) + " not found: " + ErrorDescription(GetLastError()));
        return 0;
    } else if (iTime(_Symbol, _Period, startBar) != startTime) {
        LOG_ERROR("", "Start time " + TimeToString(startTime, TIME_DATE_SECONDS) + " does not equal bar time " + TimeToString(iTime(_Symbol, _Period, startBar), TIME_DATE_SECONDS));
    }
    const long endBar = iBarShift(_Symbol, _Period, endTime, false);
    if (endBar < 0) {
        LOG_ERROR("SVRApi::PrepareHistoryBidPricesAvg", "Bar for end time " + TimeToString(endTime, TIME_DATE_SECONDS) + " not found: " + ErrorDescription(GetLastError()));
        return 0;
    } else if (iTime(_Symbol, _Period, endBar) != endTime) {
        LOG_INFO("", "End time " + TimeToString(endTime, TIME_DATE_SECONDS) + " does not equal bar time " + TimeToString(iTime(_Symbol, _Period, endBar), TIME_DATE_SECONDS));
    }
    // endBar is smaller than startBar
    ulong totalBarCount = 1 + startBar - endBar;
    if (OneSecondData) totalBarCount *= PeriodSeconds();
    LOG_INFO("SVRApi::PrepareHistoryBidPricesAvg", "Preparing up to " + totalBarCount + " bars for period " + TimeToString(startTime, TIME_DATE_SECONDS) + " to " + TimeToString(endTime, TIME_DATE_SECONDS));
    double tempBid[];
    datetime tempTime[];
    ulong tempVolume[];
    ArrayResize(tempBid, totalBarCount);
    ArrayResize(tempTime, totalBarCount);
    ArrayResize(tempVolume, totalBarCount);
    ulong okBars = 0;
    double prevTickPrice = iOpen(_Symbol, _Period, startBar);
    for (long barIx = startBar; barIx >= endBar; --barIx) {
        MqlTick ticks[];
        const datetime barTime = iTime(_Symbol, _Period, barIx);
        const string barTimeStr = TimeToString(barTime, TIME_DATE_SECONDS);
        if (barTime == 0) {
            LOG_ERROR("SVRApi::PrepareHistoryBidPricesAvg", "Bar time or price not found for " + _Symbol + " from " + barTimeStr + " to " + TimeToString(barTime + PeriodSeconds(), TIME_DATE_SECONDS) + " error: " + ErrorDescription(GetLastError()));
            continue;
        }
        const long ticksCount = CopyTicksSafe(_Symbol, ticks, COPY_TICKS_ALL, iTime(_Symbol, _Period, barIx + 1), barTime + PeriodSeconds());
        if (ticksCount < 1) {
            if (iHigh(_Symbol, _Period, barIx) != iLow(_Symbol, _Period, barIx) || iOpen(_Symbol, _Period, barIx) != iClose(_Symbol, _Period, barIx)) {
                LOG_ERROR("SVRApi::PrepareHistoryBidPricesAvg", "Inconsistency between ticks and bars at " + barTimeStr + ", high doesn't equal low but no ticks for bar copied.");
                if (!OneSecondData) {
                    LOG_INFO("SVRApi::PrepareHistoryBidPricesAvg", "Using one minute rates for " + barTimeStr + " which is imprecise!");
                    MqlRates rates[];
                    CopyRatesSafe(_Symbol, ENUM_TIMEFRAMES::PERIOD_M1,  barTime, barTime + PeriodSeconds() - 1, rates);
                    AveragePrice currentPrice(rates, ArraySize(rates), barTime);
                    ArrayFill(tempBid, okBars, 1, currentPrice.value);
                    ArrayFill(tempTime, okBars, 1, currentPrice.tm);
                    prevTickPrice = currentPrice.closePrice;
                    ++okBars;
                    continue;
                }
            }
        }
        if (!OneSecondData) {
            AveragePrice currentPrice(ticks, barTime, PeriodSeconds(), prevTickPrice);
            if (currentPrice.value <= 0) {
                LOG_ERROR("SVRApi::PrepareHistoryBidPricesAvg", "Illegal price for " + TimeToString(barTime, TIME_DATE_SECONDS));
                continue;
            }
            ArrayFill(tempBid, okBars, 1, currentPrice.value);
            ArrayFill(tempTime, okBars, 1, currentPrice.tm);
            prevTickPrice = currentPrice.closePrice;
            ++okBars;
        } else {
            double prices[];
            datetime times[];
            ulong volumes[];
            if (ticksCount < 1) prevTickPrice = (iHigh(_Symbol, _Period, barIx) + iLow(_Symbol, _Period, barIx) + iClose(_Symbol, _Period, barIx)) / 3.;
            generateSecondPrices(prevTickPrice, ticks, barTime, PeriodSeconds(), prices, times, volumes);
            ArrayCopy(tempBid, prices, okBars);
            ArrayCopy(tempTime, times, okBars);
            if (ticksCount > 0) prevTickPrice = ticks[ArraySize(ticks) - 1].bid;
            else {
                LOG_ERROR("SVRApi::PrepareHistoryBidPricesAvg", "No ticks found for " + barTimeStr + " using close price for bar " + string(barIx));
                prevTickPrice = iClose(_Symbol, PERIOD_CURRENT, barIx);
            }
                
            okBars += ArraySize(prices);
        }    
    }

    ArrayResize(myBid, okBars);
    ArrayCopy(myBid, tempBid, 0, 0, okBars);
    
    ArrayResize(myTm, okBars);
    ArrayCopy(myTm, tempTime, 0, 0, okBars);
    
    ArrayResize(myVl, okBars);
    ArrayCopy(myVl, tempVolume, 0, 0, okBars);
    
    LOG_INFO("SVRApi::PrepareHistoryBidPricesAvg", "Successfully prepared " + string(okBars) + " bars out of " + string(totalBarCount) + " possible bars from " + TimeToString(startTime) + " until " + TimeToString(endTime));
    return okBars;
}


/*
    const ulong iterPeriod = OneSecondData ? 1 : PeriodSeconds();
    const ulong max_values_ct = OneSecondData ? endTime - startTime : (endTime - startTime) / PeriodSeconds();
    double tempBid[], tempAuxBid[], tempCommonBid[];
    datetime tempTime[], tempAuxTime[], tempCommonTime[];
    ArrayResize(tempBid, max_values_ct);
    ArrayResize(tempTime, max_values_ct);
    ArrayResize(tempAuxBid, max_values_ct);
    ArrayResize(tempAuxTime, max_values_ct);
    ArrayResize(tempCommonBid, max_values_ct);
    ArrayResize(tempCommonTime, max_values_ct);
    ulong main_successful = 0, aux_successful = 0;

    //PrepareHistoryPackage sends from number_of_bars to 1
    const string main_symbol = Symbol();
    datetime iterDate = endTime;
    LOG_INFO("PrepareHistoryBidPricesAvg", "From " + string(startTime) + " until " + string(endTime));
    for (iterDate = startTime; iterDate < endTime; iterDate += iterPeriod) {
        MqlTick ticks[];
        ArraySetAsSeries(ticks, true);
        const long copied_ct = CopyTicks(main_symbol, ticks, COPY_TICKS_INFO, iterDate, iterDate + iterPeriod);
        if (copied_ct < 1) {
            LOG_ERROR("", "No bars copied for " + main_symbol + " from " + string(DSTCorrectedIterDate) + " to " + string(DSTCorrectedIterDate + PeriodSeconds()));
            continue;
        }

        AveragePrice currentPrice(iOpen(main_symbol, PERIOD_M1, 0), ticks, copied_ct, iterDate);

        if (currentPrice.value != 0) {
            //LOG_INFO("PrepareHistoryBidPricesAvg", "values " + string(values));
            ArrayFill(tempBid, successful, 1, currentPrice.value);
            ArrayFill(tempTime, successful, 1, currentPrice.tm);
            successful++;
        }
        iterDate -= next;
    }

    long common_ct = 0;
    if (AuxInputQueue != "") {
        iterDate = endTime;
        LOG_INFO("PrepareHistoryBidPricesAvg", "Aux from " + string(startTime) + " until " + string(endTime));
        while(iterDate > startTime) {
            MqlRates rates[];
            ArraySetAsSeries(rates, true);
            datetime offsetIterDate = GetUTCTime(iterDate);
            datetime DSTCorrectedIterDate = offsetIterDate + getDSTOffsetFromNow(offsetIterDate);
            //LOG_INFO("PrepareHistoryBidPricesAvg", "iterDate DSTCorrectedIterData " + string(iterDate) + " " + string(DSTCorrectedIterDate));

            const long copied_ct = CopyRatesSafe(AuxInputQueue, PERIOD_M1, DSTCorrectedIterDate, DSTCorrectedIterDate + PeriodSeconds() - 1, rates);
            if(copied_ct < 1) {
                iterDate -= next;
                LOG_ERROR("", "No bars copied for " + AuxInputQueue + " from " + string(DSTCorrectedIterDate) + " until " + string(DSTCorrectedIterDate + PeriodSeconds()));
                continue;
            }

            AveragePrice currentPrice(rates, copied_ct, GetTargetDateTimeFromDate(offsetIterDate));
            //LOG_INFO("PrepareHistoryBidPricesAvg", "GetTargetDateTimeFromDate(offset) " + string(GetTargetDateTimeFromDate(offsetIterDate)));

            if (currentPrice.value != 0 && currentPrice.tm == tempTime[aux_successful]) {
                //LOG_INFO("PrepareHistoryBidPricesAvg", "values " + string(values));
                ArrayFill(tempAuxBid, aux_successful, 1, currentPrice.value);
                ArrayFill(tempAuxTime, aux_successful, 1, currentPrice.tm);
                aux_successful++;
            }
            iterDate -= next;
        }
        for (long i = 0; i < successful; ++i) {
            for (long j = 0; j < aux_successful; ++j) {
                if (tempTime[i] == tempAuxTime[j]) {
                    ArrayFill(tempCommonBid, common_ct, 1, tempBid[i]);
                    ArrayFill(tempCommonTime, common_ct, 1, tempTime[i]);
                    common_ct++;
                }
            }
        }
        if (aux_successful != successful)
            LOG_ERROR("", "aux_successful " + string(aux_successful) + " != successful " + string(successful));
        if (aux_successful != common_ct)
            LOG_ERROR("", "aux_successful " + string(aux_successful) + " != common_ct " + string(common_ct));
        ArrayResize(myAuxBid, aux_successful + offset);
        ArrayCopy(myAuxBid, tempAuxBid, offset, 0, aux_successful);
    } else {
        for (long i = 0; i < successful; ++i) {
            ArrayFill(tempCommonBid, common_ct, 1, tempBid[i]);
            ArrayFill(tempCommonTime, common_ct, 1, tempTime[i]);
            common_ct++;
        }
    }

    ArrayResize(myBid, common_ct + offset);
    ArrayCopy(myBid, tempCommonBid, offset, 0, common_ct);

    ArrayResize(myTm, common_ct + offset);
    ArrayCopy(myTm, tempCommonTime, offset, 0, common_ct);
    ArrayResize(myVl, common_ct + offset);
    VolumeMQL4(_Symbol, _Period, myVl, offset, 0, common_ct);
    LOG_INFO("", "Successful " + string(common_ct) + " bars out of " + string(values) + " possible bars from " + string(startTime) + " until " + string(endTime));
    return common_ct;
}
*/

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
long SVRApi::PrepareHistoryBidPricesOHLC(const datetime startTime, const datetime endTime)
{
    if (OneSecondData) {
        LOG_ERROR("PrepareHistoryBidPricesOHLC", "No OHLC when sending high frequency data!");
        return 0;
    }
    LOG_INFO("PrepareHistoryBidPricesOHLC", "Preparing from " + TimeToString(startTime) + " until " + TimeToString(endTime, TIME_DATE_SECONDS));    
    const long startBar = iBarShift(Symbol(), Period(), startTime, false);
    if (startBar < 0) {
        LOG_ERROR("SVRApi::PrepareHistoryBidPricesOHLC", "Bar for start time " + TimeToString(startTime) + " not found: " + ErrorDescription(GetLastError()));
        return 0;
    }
    const long endBar = iBarShift(Symbol(), Period(), endTime, false);
    if (endBar < 0) {
        LOG_ERROR("SVRApi::PrepareHistoryBidPricesOHLC", "Bar for end time " + TimeToString(endTime) + " not found: " + ErrorDescription(GetLastError()));
        return 0;
    }
    // endBar is smaller than startBar
    const ulong totalBarCount = 1 + startBar - endBar;
    double tempOpen[], tempHigh[], tempLow[], tempClose[];
    datetime tempTime[];
    ulong tempVl[];
    ArrayResize(myTm, totalBarCount);
    ArrayResize(myHi, totalBarCount);
    ArrayResize(myLo, totalBarCount);
    ArrayResize(myOp, totalBarCount);
    ArrayResize(myCl, totalBarCount);
    ArrayResize(myTm, totalBarCount);
    ArrayResize(myVl, totalBarCount);
    MqlRates rates[];
    ArraySetAsSeries(rates, true);
    ulong i = 0;
    for (ulong barIx = endBar; barIx <= startBar && i < totalBarCount; ++barIx) {
        ArrayFill(myCl, i, 1, iClose(Symbol(), Period(), barIx));
        ArrayFill(myOp, i, 1, iOpen(Symbol(), Period(), barIx));
        ArrayFill(myHi, i, 1, iHigh(Symbol(), Period(), barIx));
        ArrayFill(myLo, i, 1, iLow(Symbol(), Period(), barIx));
        ArrayFill(myTm, i, 1, iTime(Symbol(), Period(), barIx));
        ArrayFill(myVl, i, 1, iVolume(Symbol(), Period(), barIx));
        ++i;
    }
    LOG_INFO("", "Copied " + string(totalBarCount) + " bars from " + TimeToString(startTime) + " until " + TimeToString(endTime));
    return totalBarCount;
}

#define MAX_RETRIES 10

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool SVRApi::SendHistory(const string &inputQueue, const string &period, const datetime startTime, datetime endTime)
{
    if (DemoMode) endTime = OneSecondData ? endTime - DemoBars : endTime - DemoBars * PeriodSeconds();

    long copiedCt = 0;
    if (isAverage)
        copiedCt = PrepareHistoryBidPricesAvg(startTime, endTime);
    else
        copiedCt = PrepareHistoryBidPricesOHLC(startTime, endTime);

    if (copiedCt < 1) {
        LOG_ERROR("", "Failed preparing history bars.");
        return false;
    }
    
    LOG_INFO("SVRApi::SendHistory", "Prepared " + string(copiedCt) + " bid price bars.");

    const ulong totalBarsCount = ArraySize(myTm);

#ifdef HISTORYTOSQL
    LOG_INFO("SVRApi::SendHistory", "Writing history to MQL5/Files/history_bars.sql");
    int file_handle = FileOpen("history_bars_" + inputQueue + "_" + string(int(PERIOD_CURRENT)) +".sql", FILE_WRITE|FILE_ANSI|FILE_CSV);
    if (file_handle != INVALID_HANDLE) {
        FileSeek(file_handle, 0, SEEK_END);
        if (isAverage)
            for(ulong barIx = 0; barIx < totalBarsCount; ++barIx)
                FileWrite(file_handle,
                          TimeToString(myTm[barIx], TIME_DATE_SECONDS),
                          TimeToString(TimeCurrent(), TIME_DATE_SECONDS),
                          IntegerToString(myVl[barIx]),
                          DoubleToString(myBid[barIx], MAX_DOUBLE_PRECISION));
        else
            for(ulong barIx = 0; barIx < totalBarsCount; ++barIx)
                FileWrite(file_handle,
                          TimeToString(myTm[barIx], TIME_DATE_SECONDS),
                          TimeToString(TimeCurrent(), TIME_DATE_SECONDS),
                          IntegerToString(myVl[barIx]),
                          DoubleToString(myOp[barIx], MAX_DOUBLE_PRECISION),
                          DoubleToString(myHi[barIx], MAX_DOUBLE_PRECISION),
                          DoubleToString(myLo[barIx], MAX_DOUBLE_PRECISION),
                          DoubleToString(myCl[barIx], MAX_DOUBLE_PRECISION));
        FileFlush(file_handle);
        FileClose(file_handle);
    }
    return false;
#endif
    string header = "Time:date;";
    if (isAverage) { // Meta Trader 5 supports only bid quotes
        header += inputQueue + "_bid:double;";
        if (AuxInputQueue != "") header += AuxInputQueue + "_bid:double;";
        header += "Volume:long";
    } else {
        header += "High:double;Low:double;Open:double;Close:double;Volume:long";
    }

    const string delimiter = ";";
    ulong barNo = 0;
    
    while (barNo + 1 < totalBarsCount) {
        Hash params;
        params.hPutString("barFrom", string(barNo));
        params.hPutString("symbol", inputQueue);
        params.hPutString("period", period);
        params.hPutString("header", header);
        params.hPutString("delimiter", delimiter);
        
        ulong finishedAt;
        if (isAverage)
            finishedAt = prepareAverageHistoryPackage(barNo, totalBarsCount - 1, delimiter, params);
        else
            finishedAt = prepareHistoryPackage(barNo, totalBarsCount - 1, delimiter, params);
        params.hPutString("barTo", string(finishedAt));

        string response;
        ulong retriesCt = 0;
        const string queue_call = "queue";
        const string history_data = "historyData";
        while (!net.RpcCall(queue_call, history_data, params, response) || !requestUtils.isSuccessfulResponse(response) && retriesCt < MAX_RETRIES) {
            if (StringLen(response) > 0) {
                LOG_ERROR("SVRApi::SendHistory", "Server returned: " + requestUtils.getErrorMessage(response) + " params: " + net.ToJSON(params));
            }
            if (retriesCt + 1 < MAX_RETRIES)
                LOG_ERROR("SVRApi::SendHistory", "Error while sending historical data, retrying.");
            else
                LOG_ERROR("SVRApi::SendHistory", "Error while sending historical data. Skipping bars from " + string(barNo) + " to " + string(finishedAt));
            ++retriesCt;
        } 
        if (retriesCt < MAX_RETRIES) LOG_INFO("SVRApi::SendHistory", "Successfully sent " + string(finishedAt) + " bars.");
        if (finishedAt == totalBarsCount - 1) return true;
        else barNo = finishedAt;
    }
    ArrayFree(myAuxBid);
    ArrayFree(myBid);
    ArrayFree(myCl);
    ArrayFree(myHi);
    ArrayFree(myLo);
    ArrayFree(myOp);
    ArrayFree(myTm);
    ArrayFree(myVl);

    return true;
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
// TODO Add time zone information
ulong SVRApi::prepareHistoryPackage(const ulong barStart, const ulong barFinish, const string &delimiter, Hash &hash)
{
    ulong barIx = barStart;
    for(ulong packedBars = 0; barIx <= barFinish && packedBars < packSize; ++barIx, ++packedBars)
        hash.hPutString(IntegerToString(barIx), TimeToString(myTm[barIx], TIME_DATE_SECONDS) + timeZoneInfo + delimiter + DoubleToString(myHi[barIx], MAX_DOUBLE_PRECISION) + delimiter + DoubleToString(myLo[barIx], MAX_DOUBLE_PRECISION) + delimiter +
            DoubleToString(myOp[barIx], MAX_DOUBLE_PRECISION) + delimiter + DoubleToString(myCl[barIx], MAX_DOUBLE_PRECISION) + delimiter + IntegerToString(myVl[barIx], MAX_DOUBLE_PRECISION));

    return barIx;
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
ulong SVRApi::prepareAverageHistoryPackage(const ulong barStart, const ulong barFinish, const string &delimiter, Hash &hash)
{
    ulong barIx = barStart;
    for(ulong packedBars = 0; barIx <= barFinish && packedBars < packSize; ++barIx, ++packedBars) {
        hash.hPutString(IntegerToString(barIx), AuxInputQueue != "" ? 
                              TimeToString(myTm[barIx], TIME_DATE_SECONDS) + delimiter +
                              DoubleToString(myBid[barIx], MAX_DOUBLE_PRECISION) + delimiter +
                              DoubleToString(myAuxBid[barIx], MAX_DOUBLE_PRECISION) + delimiter +
                              IntegerToString(myVl[barIx])
                              :                              
                              TimeToString(myTm[barIx], TIME_DATE_SECONDS) + delimiter +
                              DoubleToString(myBid[barIx], MAX_DOUBLE_PRECISION) + delimiter +
                              IntegerToString(myVl[barIx]));
    }
    return barIx;
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
/*
bool SVRApi::ReconcileHistoricalData(string inputQueue, string period)
{
    datetime timeFrom, timeTo;
    bool recon_finished = false;
    string funName = "reconcileHistoricalData";
    bool result = GetCustomTimeRange(timeFrom, timeTo, inputQueue, period, funName, recon_finished);

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
    if(timeFrom == timeTo) return true;

    result = SendHistory(inputQueue, period, timeFrom, timeTo, false, packSize);
    if (!result) LOG_ERROR("SVRApi::ReconcileHistoricalData", "Historical data reconciliation aborted");

    return result;
}
*/

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool SVRApi::GetCustomTimeRange(datetime& timeFrom, datetime& timeTo, const string &inputQueue, const string &period, const string &function, bool &recon_finished)
{
    string response;
    Hash params;

    params.hPutString("symbol", inputQueue);
    params.hPutString("period", OneSecondData ? "1" : period);
    const long bars = BARS_OFFERED < iBars(Symbol(), Period()) ? BARS_OFFERED : iBars(Symbol(), Period());
    timeTo = iTime(Symbol(), Period(), 1);
    if (OneSecondData) timeTo += PeriodSeconds() - 1;
    timeFrom = iTime(Symbol(), Period(), bars - 1);
#ifdef HISTORYTOSQL
    return true;
#endif
    string strOfferToTime = TimeToString(timeTo, TIME_DATE_SECONDS);
    const string strOfferFromTime = TimeToString(timeFrom, TIME_DATE_SECONDS);
    params.hPutString("offeredTimeFrom", strOfferFromTime);
    params.hPutString("offeredTimeTo", strOfferToTime);
    LOG_INFO("SVRApi::GetCustomTimeRange", "Offering " + string(OneSecondData ? bars * PeriodSeconds() : bars) + " bars for " + inputQueue + " symbol " + Symbol() + " from " + strOfferFromTime + " until " + strOfferToTime);

    const string queue_call = "queue";
    if (!net.RpcCall(queue_call, function, params, response)) return(false);

    const bool success = requestUtils.isSuccessfulResponse(response);
    if (!success) {
        LOG_ERROR("SVRApi::GetCustomTimeRange", "Server returned: " + requestUtils.getErrorMessage(response));
    } else {
        string strReqTimeFrom, strReqTimeTo;
        JSONObject* jo = requestUtils.getResultObject(response);

        if(!jo.getString("timeFrom", strReqTimeFrom)) recon_finished = true;
        jo.getString("timeTo", strReqTimeTo);

        const datetime reqTimeFrom = StringToTime(strReqTimeFrom);
        const datetime reqTimeTo = StringToTime(strReqTimeTo);

        if (reqTimeFrom > reqTimeTo) {
            LOG_ERROR("SVRApi::GetCustomTimeRange", "Server returned illegal times, from " + strReqTimeFrom + " to " + strReqTimeTo);
            return false;
        } else {
            LOG_INFO("SVRApi::GetCustomTimeRange", "Server requested period from " + strReqTimeFrom + " to " + strReqTimeTo);
        }
        timeFrom = reqTimeFrom;
        timeTo = reqTimeTo;
    }
    return(success);
}
//+------------------------------------------------------------------+
