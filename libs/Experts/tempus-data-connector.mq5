//+------------------------------------------------------------------+
//|                                            Tempus data connector |
//|                                                     Papakaya LTD |
//|                                         https://www.papakaya.com |
//+------------------------------------------------------------------+
#property copyright "Papakaya LTD"
#property link      "https://www.papakaya.com"
#property version   "001.000"
#property strict

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+

// For one second data enable one second data and set connector on a M1 chart



input string    Username       = "svrwave";             //Username
input string    Password       = "svrwave";             //Password
input string    ServerUrl      = "10.4.8.24:8080";      //Server URL
input string    AuxInput       = "";                    //Auxilliary input
input bool      DemoMode       = false;
input long      DemoBars       = 0;
input bool      Average        = true;
input long      TimeOffset     = 0;
input bool      DisableDST     = true;
input bool      OneSecondData  = true;
input string    TimeZoneInfo   = "EEST";
input long      ForecastDelay  = 360;


#include <tempus-constants.mqh>
#include <SVRApi.mqh>
#include <AveragePrice.mqh>
#include <TempusGMT.mqh>


const long TimeOffsetSecs = TimeOffset * 60;
const string StrPeriod = OneSecondData ? "1" : string(PeriodSeconds());


string InputQueueFinal;
string InputQueue;

SVRApi svrClient(DemoMode, DemoBars, Average, AuxInput, OneSecondData, TimeZoneInfo);

//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
int OnInit()
{
    if (OneSecondData && Period() != PERIOD_M1) {
        LOG_ERROR("init checks: Period", "Chart time frame must be 1 minute in order to send 1 second data!");
        return(INIT_PARAMETERS_INCORRECT);
    }    
    if (AuxInput != "" && !Average) {
        LOG_ERROR("init checks: Aux Input", "Aux input data can only be sent when average is enabled!");
        return(INIT_PARAMETERS_INCORRECT);
    }
    disableDST = DisableDST;
    InputQueue = Symbol();
    StringReplace(InputQueue, ".", "");
    StringReplace(InputQueue, " ", "");
    StringReplace(InputQueue, ",", "");
    StringToLower(InputQueue);
    InputQueueFinal = AuxInput == "" ? InputQueue : InputQueue + AuxInput;
    if (Average) InputQueueFinal = InputQueueFinal + "_avg";
    StringToLower(InputQueueFinal);

    SetPerfLogging(true);

    TempusGMTInit(Period(), TimeOffset, DisableDST);
    
#ifdef RUN_TEST
    MqlTick ticks[];
    const string main_symbol = Symbol();
    const datetime barTime = iTime(main_symbol, Period(), 1);
    const long ticksCount = svrClient.CopyTicksSafe(main_symbol, ticks, COPY_TICKS_ALL, barTime, barTime + PeriodSeconds());
    for (int i = 0; i < ticksCount; ++i) 
        LOG_INFO("", "Tick " + string(i) + " has time " + TimeToString(ticks[i].time, TIME_DATE_SECONDS) + "." + string(ticks[i].time_msc % 1000));
    double prices[];
    datetime times[];
    ulong volumes[];
    generateSecondPrices(iOpen(_Symbol, _Period, 1), ticks, barTime, PeriodSeconds(), prices, times, volumes);
    for (int i = 0; i < ArraySize(prices); ++i) LOG_INFO("", "Second " + string(i) + ", price " + string(prices[i]) + ", time " + TimeToString(times[i], TIME_SECONDS|TIME_DATE) + ", tick volume " + volumes[i]);
    
    LOG_INFO("init checks: Time", "DSTActive: " + string(wasDSTActiveThen(iTime(Symbol(), PERIOD_M1, 0))) +
             " H1(0): " + string(iTime(Symbol(), PERIOD_H1, 0)) + " H1(1): " + string(iTime(Symbol(), PERIOD_H1, 1)) +
             " H4(0): " + string(iTime(Symbol(), PERIOD_H4, 0)) + " H4(1): " + string(iTime(Symbol(), PERIOD_H4, 1))
            );
    LOG_INFO("init checks: TempusGMT", "DSTActive: " + string(wasDSTActiveThen(iTime(Symbol(), Period(), 0))) +
             " TargetTime(0): " + string(GetTargetTime(0)) + " TargetTime(1): " + string(GetTargetTime(1)));
    LOG_INFO("", "Test error description: " + ErrorDescription(GetLastError()));
    
    static int file_handle = FileOpen("history_bars_" + _Symbol + "_" + StrPeriod + ".sql", FILE_WRITE|FILE_ANSI|FILE_CSV);
    if (file_handle != INVALID_HANDLE) {
        FileSeek(file_handle, 0, SEEK_END);
        FileWrite(file_handle,
                      TimeToString(TimeCurrent(), TIME_DATE_SECONDS),
                      TimeToString(TimeCurrent(), TIME_DATE_SECONDS),
                      IntegerToString(100),
                      DoubleToString(1.2345, MAX_DOUBLE_PRECISION));
        FileFlush(file_handle);
        FileClose(file_handle);
    }
    return(INIT_SUCCEEDED);
#endif

#ifndef HISTORYTOSQL
    if (!svrClient.Connect(ServerUrl))
        return(INIT_PARAMETERS_INCORRECT);
    if (!svrClient.Login(Username, Password))
        return(INIT_PARAMETERS_INCORRECT);
#endif
    if (!svrClient.SendHistory(InputQueueFinal, StrPeriod))
        return(INIT_FAILED);

    EventSetTimer(1);

    return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void OnDeinit(const long reason)
{
    EventKillTimer();
}

//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
void OnTick()
{
    const long bar_index = DemoMode == false ? 1 : DemoBars + 1;
    const datetime target_time = iTime(Symbol(), Period(), bar_index);
    static datetime last_finalized = 0;

    if (target_time <= last_finalized) return;

    // So decompose starts in time on Tempus
    if (Period() != PERIOD_M1 && datetime(GlobalVariableGet(one_minute_chart_identifier)) < target_time + 2 * PeriodSeconds() - ForecastDelay) return;
    
    LOG_INFO("", "Target time " + TimeToString(target_time, TIME_DATE_SECONDS) + ", last finalized " + TimeToString(last_finalized, TIME_DATE_SECONDS));
    
    if (Average) {
        long endBar = last_finalized ? iBarShift(Symbol(), Period(), last_finalized, false) - 1 : bar_index;
        if (endBar < bar_index) endBar = bar_index;
        for (long sendBarIdx = bar_index; sendBarIdx <= endBar; ++sendBarIdx) {
            if (OneSecondData)
                sendAverageSeconds(bar_index);
            else
                sendAverage(bar_index);
        }
    } else
        svrClient.SendBar(InputQueueFinal, StrPeriod, true, 
                            iOpen(Symbol(), Period(), bar_index), 
                            iHigh(Symbol(), Period(), bar_index), 
                            iLow(Symbol(), Period(), bar_index), 
                            iClose(Symbol(), Period(), bar_index), 
                            iVolume(Symbol(), Period(), bar_index), 
                            target_time);
        
    last_finalized = target_time;
    GlobalVariableSet(chart_identifier, last_finalized);
}


//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void sendAverage(const uint bar_index)
{
    const datetime barTime = iTime(Symbol(), Period(), bar_index);
    const datetime barBeforeTime =  iTime(Symbol(), Period(), bar_index + 1);
    const datetime time_to = barTime + PeriodSeconds() - 1;
    const bool do_aux = AuxInput != "";
    MqlTick ticks[];
    const long copied_ct = SVRApi::CopyTicksSafe(_Symbol, ticks, COPY_TICKS_ALL, barBeforeTime, barTime + PeriodSeconds());
    long aux_copied_ct = 0;
    if (do_aux) {
        LOG_ERROR("", "Aux not supported!"); // TODO Implement if needed!
        //aux_copied_ct = SVRApi::CopyTicksSafe(AuxInput, PERIOD_M1, barTime, time_to, rates); // Safer for non current chart
    } else {
        aux_copied_ct = 1;
    }
    if (copied_ct > 0 && aux_copied_ct > 0) {
        AveragePrice current_price(ticks, GetTargetTime(bar_index), PeriodSeconds(), iOpen(Symbol(), PERIOD_CURRENT, bar_index));
        if (do_aux) {
            LOG_ERROR("", "Not implemented!");
            /*
            if (aux_copied_ct != copied_ct) LOG_ERROR("", "aux_copied_ct " + string(aux_copied_ct) + " != " + string(copied_ct) + " copied_ct");
            AveragePrice current_aux_price(ratesAux, aux_copied_ct, barTime);
            svrClient.SendBar(InputQueueFinal, StrPeriod, true, current_price, current_aux_price, InputQueue, AuxInput);
            LOG_INFO("", "Sent Tick for Time: " + string(current_price.tm) + " Value: " + string(current_price.value) + " Aux value: " + string(current_aux_price.value));
            */
        } else {
            svrClient.SendBar(InputQueueFinal, StrPeriod, true, current_price);
            LOG_INFO("", "Sent bar for time: " + string(current_price.tm) + " Value: " + string(current_price.value));
        }
    } else LOG_ERROR("", "Failed to get history data for the symbol " + Symbol() + " " + AuxInput);
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void sendAverageSeconds(const long bar_index)
{
    MqlTick ticks[];
    double prices[];
    datetime times[];
    ulong volumes[];
    
    const string main_symbol = Symbol();
    const datetime barTime = iTime(main_symbol, Period(), bar_index);
    const datetime barBeforeTime =  iTime(main_symbol, Period(), bar_index + 1);
    const bool do_aux = AuxInput != "";
    SVRApi::CopyTicksSafe(main_symbol, ticks, COPY_TICKS_ALL, barBeforeTime, barTime + PeriodSeconds());
    generateSecondPrices(iOpen(main_symbol, PERIOD_CURRENT, bar_index + 1), ticks, barTime, PeriodSeconds(), prices, times, volumes);
    if (do_aux) {
        MqlTick ticksaux[];
        double pricesaux[];
        datetime timesaux[];
        ulong volumesaux[];
        SVRApi::CopyTicksSafe(AuxInput, ticksaux, COPY_TICKS_ALL, barBeforeTime, barTime + PeriodSeconds());
        generateSecondPrices(iOpen(AuxInput, PERIOD_CURRENT, bar_index + 1), ticksaux, barTime, PeriodSeconds(), pricesaux, timesaux, volumesaux);
        svrClient.SendBars(InputQueueFinal, StrPeriod, true, prices, volumes, pricesaux, volumesaux, timesaux, InputQueue, AuxInput);
        LOG_INFO("sendAverageSeconds", "Sent " + string(ArraySize(pricesaux)) + " aux bars for time " + TimeToString(barTime, TIME_DATE_SECONDS));
    } else {
        svrClient.SendBars(InputQueueFinal, StrPeriod, true, prices, times, volumes);
        LOG_INFO("sendAverageSeconds", "Sent " + string(ArraySize(prices)) +  " bars for time " + TimeToString(barTime, TIME_DATE_SECONDS));
    }
}
    

//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
void OnTimer()
{
    /*
    static bool timerSetup = false;
    if(!timerSetup)
    {
       long currentSeconds = TimeCurrent() - Time[0];
       if( (currentSeconds >= PeriodSeconds() / 2) && (currentSeconds < (PeriodSeconds() / 2 + 3))) // Do the reconciliation in the first 3 seconds of the second half of a period
       {
          timerSetup = true;
          EventSetTimer(PeriodSeconds());
       }
       else
          return;
    }

    datetime start = TimeCurrent();

    bool result = true;
    while(result && TimeSeconds(TimeLocal()) < 55)
    {
       result = svrClient.ReconcileHistoricalData(InputQueueFinal, StrPeriod);
    }
    if( svrClient.getHistDataReconciled() )
       EventKillTimer();
    */
}
//+------------------------------------------------------------------+

//+------------------------------------------------------------------+
