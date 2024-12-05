//+------------------------------------------------------------------+
//|                                            Tempus data connector |
//|                                                     Papakaya LTD |
//|                                         https://www.papakaya.com |
//+------------------------------------------------------------------+
#property copyright "Papakaya LTD"
#property link      "https://www.papakaya.com"
// #property version   "001.000"
#property strict

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+

// For one second data enable one second data and set connector on a M1 chart
input string    Username       = "svrwave";             //Username
input string    Password       = "svrwave";             //Password
input string    ServerUrl      = "127.0.0.1:8080";      //Server URL
input string    AuxInputName   = "";                    //Auxilliary input
input bool      DemoMode       = false;
input int       DemoBars       = 0;
input bool      Average        = true;
input int       TimeOffset     = 0;
input bool      DisableDST     = true;
input bool      OneSecondData  = false; // Output one second data
input string    TimeZoneInfo   = "UTC"; // Time zone
input double    PredictHorizon = .2; // Forecast delay in seconds, Horizon .2 = 720 seconds on a 1 hour time frame


#include <tempus-constants.mqh>
#include <SVRApi.mqh>
#include <AveragePrice.mqh>
#include <TempusGMT.mqh>


const int C_forecast_delay = int(PredictHorizon * C_period_seconds);
const int C_time_offset_secs = TimeOffset * 60;
const string C_period_secs_str = OneSecondData ? "1" : string(C_period_seconds);
const int C_tick_bar_index = DemoMode == false ? 1 : DemoBars + 1;
const datetime C_forecast_delay_period = 2 * C_period_seconds - C_forecast_delay;
const bool C_period_isnt_m1 = _Period != PERIOD_M1;


string server_queue_name;
string input_queue_name;

SVRApi svr_client(DemoMode, DemoBars, Average, AuxInputName, OneSecondData, TimeZoneInfo);

//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
int OnInit()
{
    LOG_INFO("", "Period is " + TimeToString(C_period_seconds, TIME_DATE_SECONDS));

    if (OneSecondData && _Period != PERIOD_M1) {
        LOG_ERROR("", "Chart time frame must be 1 minute in order to send 1 second data!");
        return(INIT_PARAMETERS_INCORRECT);
    }    
    if (AuxInputName != "" && !Average) {
        LOG_ERROR("", "Aux input data can only be sent when average is enabled!");
        return(INIT_PARAMETERS_INCORRECT);
    }
    disableDST = DisableDST;
    input_queue_name = _Symbol;
    StringReplace(input_queue_name, ".", "");
    StringReplace(input_queue_name, " ", "");
    StringReplace(input_queue_name, ",", "");
    
    StringToLower(input_queue_name);
    server_queue_name = AuxInputName == "" ? input_queue_name : input_queue_name + AuxInputName;
    if (Average) server_queue_name = server_queue_name + "_avg";
    StringToLower(server_queue_name);

#ifdef RUN_TEST
    MqlTick ticks[];
    const string main_symbol = _Symbol;
    const datetime bar_time = iTime(main_symbol, _Period, 1);
    const int ticksCount = svr_client.copy_ticks_safe(main_symbol, ticks, COPY_TICKS_ALL, bar_time, bar_time + C_period_seconds);
    for (int i = 0; i < ticksCount; ++i) 
        LOG_INFO("", "Tick " + string(i) + " has time " + TimeToString(ticks[i].time, TIME_DATE_SECONDS) + "." + string(ticks[i].time_msc % 1000));
    aprice prices[C_period_seconds];
    datetime times[C_period_seconds];
    uint volumes[C_period_seconds];
    persec_prices(get_rate(1, e_rate_type::price_open), ticks, bar_time, C_period_seconds, prices, times, volumes, 0);
    for (int i = 0; i < ArraySize(prices); ++i)
        LOG_INFO(, "Second " + string(i) + ", price " + prices[i].to_string() + ", time " + TimeToString(times[i], TIME_SECONDS|TIME_DATE) + ", tick volume " + volumes[i]);
    
    LOG_INFO("init checks: Time", "DSTActive: " + string(wasDSTActiveThen(iTime(_Symbol, PERIOD_M1, 0))) +
             " H1(0): " + string(iTime(_Symbol, PERIOD_H1, 0)) + " H1(1): " + string(iTime(_Symbol, PERIOD_H1, 1)) +
             " H4(0): " + string(iTime(_Symbol, PERIOD_H4, 0)) + " H4(1): " + string(iTime(_Symbol, PERIOD_H4, 1)));
    LOG_INFO("init checks: TempusGMT", "DSTActive: " + string(wasDSTActiveThen(iTime(_Symbol, _Period, 0))) +
             " TargetTime(0): " + string(GetTargetTime(0)) + " TargetTime(1): " + string(GetTargetTime(1)));
    LOG_INFO("", "Test error description: " + ErrorDescription(GetLastError()));
    
    static int file_handle = FileOpen("history_bars_" + _Symbol + "_" + C_period_secs_str + ".sql", FILE_WRITE|FILE_ANSI|FILE_CSV);
    if (file_handle != INVALID_HANDLE) {
        FileSeek(file_handle, 0, SEEK_END);
        FileWrite(file_handle, 
            TimeToString(TimeCurrent(), TIME_DATE_SECONDS), TimeToString(TimeCurrent(), TIME_DATE_SECONDS), IntegerToString(100), DoubleToString(1.2345, DOUBLE_PRINT_DECIMALS));
        FileFlush(file_handle);
        FileClose(file_handle);
    }
    return(INIT_SUCCEEDED);
#endif

#ifndef HISTORYTOSQL
    if (!svr_client.Connect(ServerUrl))
        return(INIT_PARAMETERS_INCORRECT);
    if (!svr_client.Login(Username, Password))
        return(INIT_PARAMETERS_INCORRECT);
#endif
    if (!svr_client.send_history(server_queue_name, C_period_secs_str))
        return(INIT_FAILED);

    EventSetTimer(1);

    return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    EventKillTimer();
}

//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
void OnTick()
{
    const datetime target_time = iTime(_Symbol, _Period, C_tick_bar_index);
    static datetime last_finalized = 0;

    if (target_time <= last_finalized) return;
    // So decompose starts in time on Tempus
    if (C_period_isnt_m1 && datetime(GlobalVariableGet(C_one_minute_chart_identifier)) < target_time + C_forecast_delay_period) return;

#ifdef DEBUG_CONNECTOR
    LOG_INFO("", "Target time " + TimeToString(target_time, TIME_DATE_SECONDS) + ", last finalized " + TimeToString(last_finalized, TIME_DATE_SECONDS));
#endif
    
    if (Average) {
        int end_bar = last_finalized ? iBarShift(_Symbol, _Period, last_finalized, false) - 1 : C_tick_bar_index;
        if (end_bar < C_tick_bar_index) end_bar = C_tick_bar_index;
        for (int send_bar_ix = C_tick_bar_index; send_bar_ix <= end_bar; ++send_bar_ix) {
            if (OneSecondData)
                send_average_secs(C_tick_bar_index);
            else
                send_avg(C_tick_bar_index);
        }
    } else {
        MqlRates rates[];
        CopyRates(_Symbol, _Period, C_tick_bar_index, 1, rates);
        if (ArraySize(rates) < 1) 
            LOG_ERROR("", "No rates found for bar " + IntegerToString(C_tick_bar_index));
        else 
            svr_client.send_bar(server_queue_name, C_period_secs_str, true,  
                            aprice(rates[0], e_rate_type::price_open), 
                            aprice(rates[0], e_rate_type::price_high), 
                            aprice(rates[0], e_rate_type::price_low),
                            aprice(rates[0], e_rate_type::price_close),
                            (uint) rates[0].tick_volume, rates[0].time);
    }
        
    last_finalized = target_time;
    GlobalVariableSet(C_chart_identifier, last_finalized);
}


//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void send_avg(const int bar_index)
{
    const datetime bar_time = iTime(_Symbol, _Period, bar_index);
    const datetime bar_before_time =  iTime(_Symbol, _Period, bar_index + 1);
    const datetime time_to = bar_time + C_period_seconds - 1;
    const bool do_aux = AuxInputName != "";
    MqlTick ticks[];
    const int copied_ct = SVRApi::copy_ticks_safe(_Symbol, ticks, COPY_TICKS_ALL, bar_before_time, bar_time + C_period_seconds);
    int aux_copied_ct = 0;
    if (do_aux) {
        LOG_ERROR("", "Aux not supported!"); // TODO Implement if needed!
        //aux_copied_ct = SVRApi::copy_ticks_safe(AuxInputName, PERIOD_M1, bar_time, time_to, rates); // Safer for non current chart
    } else
        aux_copied_ct = 1;
    
    if (copied_ct > 0 && aux_copied_ct > 0) {
        const AveragePrice current_price(ticks, GetTargetTime(bar_index), C_period_seconds, get_rate(bar_index, e_rate_type::price_open));
        if (do_aux) {
            LOG_ERROR("", "Not implemented!");
            /*
            if (aux_copied_ct != copied_ct) LOG_ERROR("", "aux_copied_ct " + string(aux_copied_ct) + " != " + string(copied_ct) + " copied_ct");
            AveragePrice current_aux_price(ratesAux, aux_copied_ct, bar_time);
            svr_client.send_bar(server_queue_name, C_period_secs_str, true, current_price, current_aux_price, input_queue_name, AuxInputName);
            LOG_INFO("", "Sent Tick for Time: " + string(current_price.tm) + " Value: " + string(current_price.value) + " Aux value: " + string(current_aux_price.value));
            */
        } else {
            svr_client.send_bar(server_queue_name, C_period_secs_str, true, current_price);
            LOG_INFO("", "Sent bar for time " + TimeToString(current_price.tm, TIME_DATE_SECONDS) + ", value " + current_price.value.to_string());
        }
    } else 
        LOG_ERROR("", "Failed to get history data for the symbol " + _Symbol + " " + AuxInputName);
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void send_average_secs(const int bar_index)
{
    MqlTick ticks[];
    aprice prices[];
    datetime times[];
    uint volumes[];
    ArrayResize(prices, C_period_seconds);
    ArrayResize(times, C_period_seconds);
    ArrayResize(volumes, C_period_seconds);
    
    const string main_symbol = _Symbol;
    const datetime bar_time = iTime(main_symbol, _Period, bar_index), bar_before_time = iTime(main_symbol, _Period, bar_index + 1);
    const datetime bar_time_end = bar_time + C_period_seconds;
    const bool do_aux = AuxInputName != "";    
    SVRApi::copy_ticks_safe(main_symbol, ticks, COPY_TICKS_ALL, bar_before_time, bar_time_end);
    persec_prices(get_rate(bar_index, e_rate_type::price_open), ticks, bar_time, C_period_seconds, prices, times, volumes, 0);
    if (do_aux) {
        MqlTick ticksaux[];
        aprice pricesaux[];
        datetime timesaux[];
        uint volumesaux[];
        ArrayResize(pricesaux, C_period_seconds);
        ArrayResize(timesaux, C_period_seconds);
        ArrayResize(volumesaux, C_period_seconds);
        SVRApi::copy_ticks_safe(AuxInputName, ticksaux, COPY_TICKS_ALL, bar_before_time, bar_time_end);
        persec_prices(get_rate(AuxInputName, bar_index, e_rate_type::price_open), ticksaux, bar_time, C_period_seconds, pricesaux, timesaux, volumesaux, 0);
        svr_client.send_bars(server_queue_name, C_period_secs_str, true, prices, volumes, pricesaux, volumesaux, timesaux, input_queue_name, AuxInputName);
        LOG_DEBUG("", "Sent " + C_period_seconds_str + " aux bars for time " + TimeToString(bar_time, TIME_DATE_SECONDS));
    } else {
        svr_client.send_bars(server_queue_name, C_period_secs_str, true, prices, times, volumes);
        LOG_DEBUG("", "Sent " + C_period_seconds_str + " bars for time " + TimeToString(bar_time, TIME_DATE_SECONDS));
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
       result = svr_client.ReconcileHistoricalData(server_queue_name, C_period_secs_str);
    }
    if( svr_client.history_reconciled() )
       EventKillTimer();
    */
}
//+------------------------------------------------------------------+

//+------------------------------------------------------------------+
