//+------------------------------------------------------------------+
//|                                                    tempus_time_util.mqh |
//|                                           Copyright 2019, Tempus |
//|                                              https://tempus.work |
//+------------------------------------------------------------------+

#property copyright "Copyright 2019, Tempus"
#property link      "https://tempus.work"
#property strict

ENUM_TIMEFRAMES _TargetPeriod = 0;
int _TimeOffset = 0;
int _OffsetHours = 0;
int _PeriodHours = 0;
bool _DisableDST = false;


#include <tempus/dst_periods.mqh>
#include <tempus/log.mqh>

#import "Kernel32.dll"
void GetSystemTime(int& TimeArray[]); // Working fine!
void GetLocalTime(int& TimeArray[]); // Working fine!
#import


// Used for backtests, sleeps using actual local time
void sleep_local(const uint sleep)
{
    const datetime sleep_until = TimeLocal() + sleep;
    while(get_windows_local_time() < sleep_until) {};
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
datetime GetWindowsSystemTime()
{
    int TimeArray[4];
    GetSystemTime(TimeArray);
    int year = TimeArray[0] & 65535;
    int month = TimeArray[0] >> 16;
    int day = TimeArray[1] >> 16;
    int hour = TimeArray[2] & 65535;
    int minute = TimeArray[2] >> 16;
    int second = TimeArray[3] & 65535;

    return StringToTime( FormatDateTime(year, month, day, hour, minute, second) );
}


//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
datetime get_windows_local_time()
{
    if (C_backtesting) {
        int TimeArray[4];
        GetLocalTime(TimeArray);
        return StringToTime( FormatDateTime(
            TimeArray[0] & 65535, 
            TimeArray[0] >> 16, 
            TimeArray[1] >> 16, 
            TimeArray[2] & 65535,
            TimeArray[2] >> 16,
            TimeArray[3] & 65535)
        );
    } else
        return TimeLocal();
}


//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
string FormatDateTime(const int year, const int iMonth, const int iDay, const int iHour, const int iMinute, const int iSecond)
{
    string month = IntegerToString(iMonth + 100);
    month = StringSubstr(month, 1);
    string day = IntegerToString(iDay + 100);
    day = StringSubstr(day, 1);
    string hour = IntegerToString(iHour + 100);
    hour = StringSubstr(hour, 1);
    string minute = IntegerToString(iMinute + 100);
    minute = StringSubstr(minute, 1);
    string second = IntegerToString(iSecond + 100);
    second = StringSubstr(second, 1);
    return StringFormat("%d.%s.%s %s:%s:%s", year, month, day, hour, minute, second);
}


//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
void TempusGMTInit(const ENUM_TIMEFRAMES targetPeriod, const int timeOffset, const bool DisableDST_)
{
    LOG_INFO("Server time to local time difference " + TimeToString(TimeTradeServer() - TimeLocal(), C_time_mode));
    _TargetPeriod = targetPeriod;
    _TimeOffset = timeOffset;
    _OffsetHours = _TimeOffset / MINUTES_IN_HOUR;
    _PeriodHours = PeriodSeconds(targetPeriod) / SECONDS_IN_HOUR;
    _DisableDST = DisableDST_;
}
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
bool CheckInit()
{
    if (_TargetPeriod == 0 || _TimeOffset == 0) return false;
    return true;
}


//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int getOffsetWithDST(const datetime targetTime)
{
    return _DisableDST ? _OffsetHours : _OffsetHours - wasDSTActiveThen(targetTime);
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int getOffsetWithDST(const int index)
{
    return getOffsetWithDST(iTime(_Symbol, _TargetPeriod, index));
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool isLastHourOfPeriod(const long currentHour)
{
    if (_TargetPeriod != PERIOD_H4) return false;
    return currentHour < (_PeriodHours - 1) ? false : true;
}

//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
// Time[N]
// Returns UTC candles time in accordance to server time [03:00-06:59], [07:00-10:59], ... ,[23:00, 02:59]
datetime GetUTCServerTime(const int index)
{
    if (_TargetPeriod != PERIOD_H4) {
        const datetime cur_time = iTime(Symbol(), _TargetPeriod, index);
        return cur_time - SECONDS_IN_HOUR * getOffsetWithDST(cur_time);
    }

    const int diff = (int) (iTime(_Symbol, PERIOD_H1, 0) - iTime(Symbol(), PERIOD_H4, 0)) / SECONDS_IN_HOUR;
    if(isLastHourOfPeriod(diff)) {
        const datetime result = iTime(_Symbol, PERIOD_H1, index * _PeriodHours + diff - getOffsetWithDST(index));
        return result != 0 ? result : iTime(Symbol(), PERIOD_H4, index - 1) - (_PeriodHours - getOffsetWithDST(index)) * SECONDS_IN_HOUR;
    } else {
        const datetime result = iTime(_Symbol, PERIOD_H1, index * _PeriodHours + diff + _PeriodHours - getOffsetWithDST(index));
        return result != 0 ? result : iTime(Symbol(), PERIOD_H4, index) - (_PeriodHours - getOffsetWithDST(index)) * SECONDS_IN_HOUR;
    }
}

// Returns UTC candles normal times [00:00-03:59], [04:00-07:59], ..., [20:00-23:59]
datetime GetTargetTime(const int index)
{
    if (_TargetPeriod == PERIOD_H4) {
        return GetUTCServerTime(index) - getOffsetWithDST(index) * SECONDS_IN_HOUR; // TODO Buggy check again!
    } else {
        const datetime cur_time = iTime(Symbol(), _TargetPeriod, index);
        return cur_time - getOffsetWithDST(cur_time) * SECONDS_IN_HOUR;
    }
}

// Used when iterating over dates - in tempusapi.h to send history data
// Start time of candles differ with 1hour with data in server time: ([04:00-07:59] -> [03:00-06:59])
datetime GetUTCTime(const datetime date)
{
    if (_TargetPeriod == PERIOD_H4) {
        return date - getOffsetWithDST(date) * SECONDS_IN_HOUR;
    } else {
        return date;
    }
}

//save UTC data with proper hours for UTC chart times : ([03:00-06:59] -> [00:00-03:59])
datetime GetTargetDateTimeFromDate(const datetime date)
{
    if (_TargetPeriod == PERIOD_H4) {
        return GetUTCTime(date);
    } else {
        return date;
    }
}
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
// Open[N]
double GetTargetOpen(const int index)
{
    if (_TargetPeriod != PERIOD_H4) return iOpen(_Symbol, _TargetPeriod, index);

    const int shift = (int) (iTime(_Symbol, PERIOD_H1, 0) - GetUTCServerTime(0)) / SECONDS_IN_HOUR;
    return iOpen(_Symbol, PERIOD_H1, index * _PeriodHours + shift);
}
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
// Close[N]
double GetTargetClose(const int index)
{
    if (_TargetPeriod != PERIOD_H4) return iClose(Symbol(), _TargetPeriod, index);

    if (index == 0)
        return iClose(Symbol(), PERIOD_H1, 0);
    else {
        const int shift = (int) (iTime(Symbol(), PERIOD_H1, 0) - GetUTCServerTime(0)) / SECONDS_IN_HOUR;
        const int total = index * _PeriodHours + shift - (_PeriodHours - 1);
        return iClose(Symbol(), PERIOD_H1, index * _PeriodHours + shift - getOffsetWithDST(index));
    }
}
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
// High[N]
double GetTargetHigh(const int index)
{
    if (_TargetPeriod != PERIOD_H4) return iHigh(Symbol(), _TargetPeriod, index);

    const int shift = (int) (iTime(Symbol(), PERIOD_H1, 0) - GetUTCServerTime(0)) / SECONDS_IN_HOUR;
    double max_value = iHigh(Symbol(), PERIOD_H1, index * _PeriodHours + shift);
    for(int i = 1; i < _PeriodHours; ++i) {
        const int total = index * _PeriodHours + shift - i;
        const double new_val = iHigh(Symbol(), PERIOD_H1, index * _PeriodHours + shift - i);
        if(new_val > max_value)
            max_value = new_val;
    }

    return max_value;
}
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
// Low[N]
double GetTargetLow(const int index)
{
    if (_TargetPeriod != PERIOD_H4) return iLow(Symbol(), _TargetPeriod, index);

    const int shift = (int) (iTime(Symbol(), PERIOD_H1, 0) - GetUTCServerTime(0)) / SECONDS_IN_HOUR;

    double min_value = iHigh(Symbol(), PERIOD_H1, index * _PeriodHours + shift);
    for(int i = 1; i < _PeriodHours; ++i) {
        double new_val = iLow(Symbol(), PERIOD_H1, index * _PeriodHours + shift - i);
        if(new_val < min_value)
            min_value = new_val;
    }
    return min_value;
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
long GetTargetVolume(const int index)
{
    return 1;
//return Volume[index]; // TODO Implement properly!
}
//+------------------------------------------------------------------+
