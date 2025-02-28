//+------------------------------------------------------------------+
//|                                            Tempus data connector |
//|                                                     Papakaya LTD |
//|                                         https://www.papakaya.com |
//+------------------------------------------------------------------+
#property copyright "Papakaya LTD"
#property link      "https://www.papakaya.com"
// #property version   "001.000"
#property strict

#define TEMPUS_CONNECTOR

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+

// For one second data enable one second data and set connector on a M1 chart
input string    Username       = "svrwave";             // Username
input string    Password       = "svrwave";             // Password
input string    ServerUrl      = "127.0.0.1:8080";      // Server URL
input string    AuxInputName   = "";                    // Auxilliary input
input bool      DemoMode       = false;
input int       DemoBars       = 0;
input bool      Average        = true;
input int       TimeOffset     = 0;
input bool      DisableDST     = true;
input bool      OneSecondData  = false; // Output one second data
input string    TimeZoneInfo   = "UTC"; // Time zone
input double    PredictHorizon = .2; // Forecast delay in seconds, Horizon .2 = 720 seconds on a 1 hour time frame
input bool      Use_Streams    = false;

#include <tempus/constants.mqh>
#include <tempus/api.mqh>
#include <tempus/price.mqh>
#include <tempus/time_util.mqh>


//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
const int C_forecast_delay = int(PredictHorizon * C_period_seconds);
const int C_time_offset_secs = TimeOffset * 60;
const int C_tick_bar_index = DemoMode == false ? 1 : DemoBars + 1;
const datetime C_forecast_delay_period = 2 * C_period_seconds - C_forecast_delay;
const bool C_period_is_not_M1 = _Period != PERIOD_M1;
const bool C_do_aux = AuxInputName != "";
const string C_input_queue_name = input_queue_name_from_symbol(_Symbol);
const string C_server_queue_name = C_input_queue_name + input_queue_name_from_symbol(AuxInputName) + (Average ? "_avg" : "");

streams_messaging *G_p_streams_msg = Use_Streams ? new streams_messaging(C_streams_root) : NULL;
tempusapi svr_client(DemoMode, DemoBars, Average, AuxInputName, OneSecondData, TimeZoneInfo, G_p_streams_msg, _Symbol, _Period, C_server_queue_name);


//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
int OnInit()
{
    if (OneSecondData && _Period != PERIOD_M1) {
        LOG_ERROR("Chart time frame must be 1 minute in order to send 1 second data!");
        return(INIT_PARAMETERS_INCORRECT);
    }
    if (AuxInputName != "" && !Average) {
        LOG_ERROR("Aux input data can only be sent when average is enabled!");
        return(INIT_PARAMETERS_INCORRECT);
    }
    disableDST = DisableDST;

#ifdef RUN_TEST
    MqlTick ticks[];
    const datetime bar_time = iTime(_Symbol, _Period, 1);
    const int ticksCount = svr_client.copy_ticks_safe(_Symbol, ticks, COPY_TICKS_ALL, bar_time, bar_time + C_period_seconds);
    for (int i = 0; i < ticksCount; ++i)
        LOG_INFO("Tick " + string(i) + " has time " + TimeToString(ticks[i].time, C_time_mode) + "." + string(ticks[i].time_msc % 1000));
    aprice prices[C_period_seconds];
    datetime times[C_period_seconds];
    uint volumes[C_period_seconds];
    persec_prices(get_rate(1, e_rate_type::price_open), ticks, bar_time, C_period_seconds, prices, times, volumes, 0);
    for (int i = 0; i < ArraySize(prices); ++i)
        LOG_INFO(, "Second " + string(i) + ", price " + prices[i].to_string() + ", time " + TimeToString(times[i], C_time_mode) + ", tick volume " + volumes[i]);

    LOG_INFO("DSTActive: " + string(wasDSTActiveThen(iTime(_Symbol, PERIOD_M1, 0))) +
             " H1(0): " + string(iTime(_Symbol, PERIOD_H1, 0)) + " H1(1): " + string(iTime(_Symbol, PERIOD_H1, 1)) +
             " H4(0): " + string(iTime(_Symbol, PERIOD_H4, 0)) + " H4(1): " + string(iTime(_Symbol, PERIOD_H4, 1)));
    LOG_INFO("DSTActive: " + string(wasDSTActiveThen(iTime(_Symbol, _Period, 0))) +
             " TargetTime(0): " + string(GetTargetTime(0)) + " TargetTime(1): " + string(GetTargetTime(1)));
    LOG_INFO("Test error description: " + ErrorDescription(GetLastError()));

    static const int file_handle = FileOpen("history_bars_" + _Symbol + "_" + C_period_secs_str + ".sql", FILE_WRITE | FILE_ANSI | FILE_CSV);
    if (file_handle != INVALID_HANDLE) {
        FileSeek(file_handle, 0, SEEK_END);
        FileWrite(file_handle,
                  TimeToString(TimeCurrent(), C_time_mode), TimeToString(TimeCurrent(), C_time_mode), IntegerToString(100), DoubleToString(1.2345, DOUBLE_PRINT_DECIMALS));
        FileFlush(file_handle);
        FileClose(file_handle);
    }
    return(INIT_SUCCEEDED);
#endif

#ifndef HISTORYTOSQL
    if (!svr_client.connect(ServerUrl))
        return INIT_PARAMETERS_INCORRECT;
    if (!svr_client.login(Username, Password))
        return INIT_PARAMETERS_INCORRECT;
#endif
    const string period_str = OneSecondData ? C_one_str : C_period_seconds_str;
    if (!svr_client.send_history()) return INIT_FAILED;

    EventSetTimer(C_period_seconds);

    return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    EventKillTimer();
    delete G_p_streams_msg;
}

//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
void OnTick()
{
    const datetime target_time = iTime(_Symbol, _Period, C_tick_bar_index);
    static datetime last_finalized = 0;

    if (target_time <= last_finalized) return;
    // So decompose starts in time on Tempus
    if (C_period_is_not_M1 && datetime(GlobalVariableGet(C_one_minute_chart_identifier)) < target_time + C_forecast_delay_period) return;

#ifdef DEBUG_CONNECTOR
    LOG_INFO("Target time " + TimeToString(target_time, C_time_mode) + ", last finalized " + TimeToString(last_finalized, C_time_mode));
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
            LOG_ERROR("No rates found for bar " + IntegerToString(C_tick_bar_index));
        else
            svr_client.send_bar(true,
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
    MqlTick ticks[];
    const int copied_ct = tempusapi::copy_ticks_safe(_Symbol, ticks, COPY_TICKS_ALL, bar_before_time, bar_time + C_period_seconds);
    int aux_copied_ct = 0;
    if (C_do_aux) {
        LOG_ERROR("Aux not supported!"); // TODO Implement if needed!
        //aux_copied_ct = tempusapi::copy_ticks_safe(AuxInputName, PERIOD_M1, bar_time, time_to, rates); // Safer for non current chart
    } else
        aux_copied_ct = 1;

    if (copied_ct > 0 && aux_copied_ct > 0) {
        const AveragePrice current_price(ticks, GetTargetTime(bar_index), C_period_seconds, get_rate(bar_index, e_rate_type::price_open));
        if (C_do_aux) {
            LOG_ERROR("Not implemented!");
            /*
            if (aux_copied_ct != copied_ct) LOG_ERROR("aux_copied_ct " + string(aux_copied_ct) + " != " + string(copied_ct) + " copied_ct");
            AveragePrice current_aux_price(ratesAux, aux_copied_ct, bar_time);
            svr_client.send_bar(C_server_queue_name, C_period_secs_str, true, current_price, current_aux_price, C_input_queue_name, AuxInputName);
            LOG_INFO("Sent Tick for Time: " + string(current_price.tm) + " Value: " + string(current_price.value) + " Aux value: " + string(current_aux_price.value));
            */
        } else {
            svr_client.send_bar(true, current_price);
            LOG_INFO("Sent bar for time " + TimeToString(current_price.tm, C_time_mode) + ", value " + current_price.value.to_string());
        }
    } else
        LOG_ERROR("Failed to get history data for the symbol " + _Symbol + " " + AuxInputName);
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

    const datetime bar_time = iTime(_Symbol, _Period, bar_index), bar_before_time = iTime(_Symbol, _Period, bar_index + 1);
    const datetime bar_time_end = bar_time + C_period_seconds;
    tempusapi::copy_ticks_safe(_Symbol, ticks, COPY_TICKS_ALL, bar_before_time, bar_time_end);
    persec_prices(get_rate(bar_index, e_rate_type::price_open), ticks, bar_time, C_period_seconds, prices, times, volumes, 0);
    if (C_do_aux) {
        MqlTick ticksaux[];
        aprice pricesaux[];
        datetime timesaux[];
        uint volumesaux[];
        ArrayResize(pricesaux, C_period_seconds);
        ArrayResize(timesaux, C_period_seconds);
        ArrayResize(volumesaux, C_period_seconds);
        tempusapi::copy_ticks_safe(AuxInputName, ticksaux, COPY_TICKS_ALL, bar_before_time, bar_time_end);
        const aprice init_price = get_rate(AuxInputName, bar_index, e_rate_type::price_open);
        persec_prices(init_price, ticksaux, bar_time, C_period_seconds, pricesaux, timesaux, volumesaux, 0);
        svr_client.send_bars(true, prices, volumes, pricesaux, volumesaux, timesaux, C_input_queue_name, AuxInputName);
        LOG_DEBUG("Sent " + C_period_seconds_str + " aux bars for time " + TimeToString(bar_time, C_time_mode));
    } else {
        svr_client.send_bars(true, prices, times, volumes);
        LOG_DEBUG("Sent " + C_period_seconds_str + " bars for time " + TimeToString(bar_time, C_time_mode));
    }
}


//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
void OnTimer()
{
    const datetime current_server_time = TimeCurrent();
    MqlDateTime current_server_time_st;
    TimeToStruct(current_server_time, current_server_time_st);
    if (current_server_time_st.day_of_week != 5 && current_server_time_st.hour >= 22) return;
    
    OnTick();
}
//+------------------------------------------------------------------+

//+------------------------------------------------------------------+
