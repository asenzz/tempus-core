//+------------------------------------------------------------------+
//|                                            CompatibilityMQL4.mqh |
//|                                  Copyright 2021, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2021, MetaQuotes Ltd."
#property link      "https://www.mql5.com"

#property copyright "keiji"
#property copyright "DC2008"
#property link      "https://www.mql5.com"

#define SECONDS_IN_HOUR         3600
#define SECONDS_IN_MINUTE       60
#define MINUTES_IN_HOUR         60
#define MILLISECONDS_IN_SECOND  1000

//--- Declaration of constants
#define OP_BUY 0           //Buy 
#define OP_SELL 1          //Sell 
#define OP_BUYLIMIT 2      //Pending order of BUY LIMIT type 
#define OP_SELLLIMIT 3     //Pending order of SELL LIMIT type 
#define OP_BUYSTOP 4       //Pending order of BUY STOP type 
#define OP_SELLSTOP 5      //Pending order of SELL STOP type 
//---
#define MODE_OPEN 0
#define MODE_CLOSE 3
#define MODE_VOLUME 4
#define MODE_REAL_VOLUME 5
#define MODE_TRADES 0
#define MODE_HISTORY 1
#define SELECT_BY_POS 0
#define SELECT_BY_TICKET 1
//---
#define DOUBLE_VALUE 0
#define FLOAT_VALUE 1
#define LONG_VALUE INT_VALUE
//---
#define CHART_BAR 0
#define CHART_CANDLE 1
//---
#define MODE_ASCEND 0
#define MODE_DESCEND 1
//---
#define MODE_LOW 1
#define MODE_HIGH 2
#define MODE_TIME 5
#define MODE_BID 9
#define MODE_ASK 10
#define MODE_POINT 11
#define MODE_DIGITS 12
#define MODE_SPREAD 13
#define MODE_STOPLEVEL 14
#define MODE_LOTSIZE 15
#define MODE_TICKVALUE 16
#define MODE_TICKSIZE 17
#define MODE_SWAPLONG 18
#define MODE_SWAPSHORT 19
#define MODE_STARTING 20
#define MODE_EXPIRATION 21
#define MODE_TRADEALLOWED 22
#define MODE_MINLOT 23
#define MODE_LOTSTEP 24
#define MODE_MAXLOT 25
#define MODE_SWAPTYPE 26
#define MODE_PROFITCALCMODE 27
#define MODE_MARGINCALCMODE 28
#define MODE_MARGININIT 29
#define MODE_MARGINMAINTENANCE 30
#define MODE_MARGINHEDGED 31
#define MODE_MARGINREQUIRED 32
#define MODE_FREEZELEVEL 33
//---
#define EMPTY -1


//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void SetIndexStyle(int index,
                       int type,
                       int style = EMPTY,
                       int width = EMPTY,
                       color clr = CLR_NONE)
{
    if (type != EMPTY)
        PlotIndexSetInteger(index, PLOT_DRAW_TYPE, type);
    if(width > -1)
        PlotIndexSetInteger(index, PLOT_LINE_WIDTH, width);
    if(clr != CLR_NONE)
        PlotIndexSetInteger(index, PLOT_LINE_COLOR, clr);
    if(style != EMPTY)    
        PlotIndexSetInteger(index, PLOT_LINE_STYLE, style);
}


//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
string input_queue_name_from_symbol(string symbol_name)
{
    StringReplace(symbol_name, ".", "");
    StringReplace(symbol_name, " ", "");
    StringReplace(symbol_name, ",", "");
    StringToLower(symbol_name);
    return symbol_name;
}


//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool ObjectCreateMQL4(
    const string &name,
    const ENUM_OBJECT type,
    const int window,
    datetime time1,
    double price1,
    datetime time2 = 0,
    double price2 = 0,
    datetime time3 = 0,
    double price3 = 0)
{
    return(ObjectCreate(0, name, type, window, time1, price1, time2, price2, time3, price3));
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool ObjectDeleteMQL4(const string &name)
{
    return(ObjectDelete(0, name));
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int TimeYearMQL4(datetime date)
{
    MqlDateTime tm;
    TimeToStruct(date, tm);
    return(tm.year);
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void VolumeMQL4(
    const string &symbolName,
    const ENUM_TIMEFRAMES sourcePeriod,
    long &dstArray[],
    const int dstOffset,
    const int srcOffset,
    const int count)
{
//ArraySetAsSeries(dstArray, true);
    CopyTickVolume(symbolName, sourcePeriod, srcOffset, count, dstArray);
    long zeroArray[];
    ArrayResize(zeroArray, dstOffset);
    ArrayFill(zeroArray, 0, WHOLE_ARRAY, 0);
    ArrayInsert(dstArray, zeroArray, 0, 0);
}
//+------------------------------------------------------------------+
