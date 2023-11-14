//+------------------------------------------------------------------+
//|                                                 Tempus indicator |
//|                                                         Papakaya |
//+------------------------------------------------------------------+
#property copyright "Papakaya LTD"
#property link      "https://www.papakaya.com"
//#property version   "000.900"
#property strict
#property indicator_chart_window
#property indicator_buffers 2
#property indicator_plots   2
//--- plot Open
#property indicator_label1  "BuyArrows"
#property indicator_type1   DRAW_ARROW
#property indicator_color1  clrBlue
#property indicator_style1  STYLE_SOLID
#property indicator_width1  1
//--- plot High
#property indicator_label2  "SellArrows"
#property indicator_type2   DRAW_ARROW
#property indicator_color2  clrBlue
#property indicator_style2  STYLE_SOLID
#property indicator_width2  1


//================================================

input ulong    BarNumber                = 1;
input string   Username                 = "svrwave";
input string   Password                 = "svrwave";
input string   ServerUrl                = "10.4.8.24:8080";
input string   DataSet                  = "100";
input bool     RequestHigh              = false;
input bool     RequestLow               = false;
input bool     RequestOpen              = false;
input bool     RequestClose             = false;
input ulong    PauseBeforePolling       = 20;
input ulong    MaxPendingRequests       = 3;
input bool     KeepHistoryBars          = true;
input bool     DemoMode                 = false;
input bool     Average                  = true;
input int      TimeOffset               = 0;
input bool     DisableDST               = true;
input ulong    ForecastOffset           = 360;


#include <TempusMvc.mqh>
#include <AveragePrice.mqh>
#include <TempusGMT.mqh>
#include <tempus-constants.mqh>


int            FBWidth  = 1;
color          FBColorUp, FBColorDown;
long           ChartType;
datetime       Poll_time;
int            CurrentRequestedBarsNumber = 0;
int            CurrentRecievedBarsNumber = 0;


MqlNet         net;
TempusController controller;
TempusGraph    view;
datetime pending_requests[];
#define INIT_TEST


static datetime G_last_request_time = 0;
static datetime G_soonest_response_time = 0;

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int OnInit()
{
    GlobalVariableSet(chart_predictions_identifier, 0.);
    ChartSetInteger(0, CHART_SHOW_GRID, 0);
    ChartSetInteger(0, CHART_SHOW_PERIOD_SEP, 1);
    ChartSetInteger(0, CHART_AUTOSCROLL, 1);
    ChartSetInteger(0, CHART_SHIFT, 1);
    ChartSetDouble(0, CHART_SHIFT_SIZE, 20);
    ChartSetInteger(0, CHART_FOREGROUND, false);


    string input_queue = input_queue_name_from_symbol(Symbol());
    LOG_INFO("", "Starting up ...");
    TempusGMTInit(_Period, TimeOffset, DisableDST);

    if(!net.Open(ServerUrl)) {
        Print("Incorrect server address");
        return(INIT_PARAMETERS_INCORRECT);
    }
    long StatusCode;
    net.Post("mt4/login", "username=" + Username + "&password=" + Password, StatusCode);
    if (StatusCode != 200) {
        LOG_ERROR("", "Authentication error occured " + string(StatusCode) + " " + ErrorDescription(StatusCode));
        return(INIT_PARAMETERS_INCORRECT);
    }

    controller.init(input_queue, PeriodSeconds(), RequestHigh, RequestLow, RequestOpen, RequestClose, Average);
    view.init(BarNumber, PeriodSeconds(), KeepHistoryBars, DemoMode, Average);

#ifdef INIT_TEST
    ArrayResize(pending_requests, MaxPendingRequests, MaxPendingRequests);
    if (ArraySize(pending_requests) < 1) {
        LOG_ERROR("", "Failed setting request size array to " + string(MaxPendingRequests));
        return(INIT_PARAMETERS_INCORRECT);
    }
    if (ArraySize(pending_requests) > 1) ArrayFill(pending_requests, 1, ArraySize(pending_requests) - 1, 0);
    const datetime current_bar_time = DemoMode == false ? GetTargetTime(0) : GetTargetTime(BarNumber);
    controller.doRequest(net, current_bar_time, BarNumber, DataSet);
    LOG_DEBUG("", "Initial request for time " + string(current_bar_time) + " placed.");
    pending_requests[0] = current_bar_time;

    TempusFigure results[];
    if(controller.getResults(net, current_bar_time, BarNumber, DataSet, results)) {
        const int last_bar_ix = ArraySize(results) - 1;
        if (last_bar_ix < 0) {
            LOG_DEBUG("", "No results received yet!");
        } else {
            view.redraw(results, current_bar_time, Average);
            LOG_INFO("", "Received high: " + StringFormat("%.5f", results[last_bar_ix].hi) +
                     " low: " + StringFormat("%.5f", results[last_bar_ix].lo) + " of initial request at " + TimeToString(pending_requests[0], TIME_DATE_SECONDS));
            pending_requests[0] = 0;
        }
    } else {
        LOG_DEBUG("", "Response for initial request for time " + TimeToString(pending_requests[0]) + " not ready yet.");
        G_last_request_time = current_bar_time;
        G_soonest_response_time = TimeLocal() + PauseBeforePolling;
    }

#endif 

    EventSetTimer(30);

    IndicatorSetString(INDICATOR_SHORTNAME,"Tempus direction indicator");
    IndicatorSetInteger(INDICATOR_DIGITS, 0);

    return(INIT_SUCCEEDED);
}

//================================================

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    EventKillTimer();
    view.close();
}



//================================================

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int doCalculate(const int rates_total)
{
    const datetime current_bar_time = PeriodSeconds() + iTime(Symbol(), Period(), DemoMode ? BarNumber : 0);

    if (current_bar_time > G_last_request_time
        && datetime(GlobalVariableGet(one_minute_chart_identifier)) > current_bar_time - ForecastOffset
        && datetime(GlobalVariableGet(chart_identifier)) >= current_bar_time - 2 * PeriodSeconds()) {
        int request_ix = 0;
        while (request_ix < MaxPendingRequests && pending_requests[request_ix]) ++request_ix;

        if (request_ix == MaxPendingRequests) { // Find oldest request
            request_ix = 0;
            for(int ir = 0; ir < MaxPendingRequests; ++ir)
                if(pending_requests[ir] < pending_requests[request_ix])
                    request_ix = ir;
        }

        controller.doRequest(net, current_bar_time, BarNumber, DataSet);
        pending_requests[request_ix] = current_bar_time;
        G_last_request_time = current_bar_time;

        if (!G_soonest_response_time) 
            G_soonest_response_time = TimeLocal() + PauseBeforePolling;
        else if (TimeLocal() + PauseBeforePolling < G_soonest_response_time) 
            G_soonest_response_time = TimeLocal() + PauseBeforePolling;

        LOG_INFO("", "Requested " + string(BarNumber) + " bar(s) starting from " +
                 string(current_bar_time) + " request #" + string(request_ix) + ", soonest response " + string(G_soonest_response_time));
    }


    bool redrawn = false;

    static datetime next_results_time = 0;
    if (TimeLocal() > next_results_time) {
        for (ulong request_ix = 0; request_ix < MaxPendingRequests; ++request_ix) {
            if (pending_requests[request_ix] && G_soonest_response_time && TimeLocal() >= G_soonest_response_time) {
                LOG_DEBUG("", "Retrieving response " + string(pending_requests[request_ix]) + " request #" + string(request_ix));
                TempusFigure results[];
                if(controller.getResults(net, pending_requests[request_ix], BarNumber, DataSet, results)) {
                    const int last_bar_ix = ArraySize(results) - 1;
                    if (last_bar_ix < 0) {
                        LOG_DEBUG("", "Results are empty!");
                        continue;
                    }
    
                    LOG_DEBUG("", "Received bars from " + TimeToString(results[0].tm, TIME_DATE_SECONDS) + " to " + TimeToString(results[last_bar_ix].tm, TIME_DATE_SECONDS));
                    if (results[0].tm >= current_bar_time) {
                        redrawn |= view.redraw(results, pending_requests[request_ix], Average);
                    } else
                        LOG_ERROR("", "Received old bar with time value " +  TimeToString(results[0].tm, TIME_DATE_SECONDS) + ". Current bar is " + TimeToString(current_bar_time, TIME_DATE_SECONDS) + ". Skipping it");
    
                    pending_requests[request_ix] = 0;
                } else {
                    next_results_time = TimeLocal() + PauseBeforePolling;
                    LOG_DEBUG("", "Response for request #" + string(request_ix) + " " + string(pending_requests[request_ix]) + " not ready yet.");
                }
            }
        }
    }

    if (redrawn)
        G_soonest_response_time = 0;
    else
        view.fadeFigure();

    return rates_total;
}

//+------------------------------------------------------------------+
//| Custom indicator iteration function                              |
//+------------------------------------------------------------------+
int OnCalculate(const int rates_total,
                const int prev_calculated,
                const datetime &time[],
                const double &open[],
                const double &high[],
                const double &low[],
                const double &close[],
                const long &tick_volume[],
                const long &volume[],
                const int &spread[])
{
    return doCalculate(rates_total);
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void OnTimer(void)
{
    doCalculate(0);
}

//=========================================================================================================
