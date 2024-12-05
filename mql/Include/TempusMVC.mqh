//+-------------------------------------------------------------------+
//| This is a MVC helper file file for the Tempus Connectors project  |
//|                                                                   |
//|                                                                   |
//+-------------------------------------------------------------------+
#property copyright "Tempus Connectors"
#property strict

#include <MqlNet.mqh>
#include <hash.mqh>
#include <json.mqh>
#include <CompatibilityMQL4.mqh>
#include <MqlLog.mqh>
#include <AveragePrice.mqh>
#include <SVRApi.mqh>

//+-------------------------------------------------------------------+
//|                                                                   |
//|                        V I E W                                    |
//|                                                                   |
//+-------------------------------------------------------------------+

struct TempusFigure
{
    double           op, hi, lo, cl;
    datetime         tm;

                     TempusFigure(): tm(0), op(0), hi(0), lo(0), cl(0) {}

    bool             empty()
    {
        return tm == 0 || op == 0 || hi == 0 || lo == 0 || cl == 0;
    }
    
    string  to_string()
    {
        return string(tm) + " " + string(hi) + " " + string(lo) + " " + string(op) + " " + string(cl);
    }
};

//+-------------------------------------------------------------------+

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class TempusGraph
{
    double hi[], lo[];
    color clr_up, clr_down, clr_line;
    TempusFigure figures[];
    int fig_ct, fig_res;
    datetime cur_time;
    bool keep_history;
    float predict_offset;

    void place_figure(const TempusFigure &fig);
    
public:
    void init(const int fig_num_, const int fig_res_, const bool keep_history_ = false, const bool demo_mode = false, const bool averageMode = false);
    void close();
    bool redraw(const TempusFigure &new_figs[], const datetime req_time, const bool average);
    bool fadeFigure();

    bool getFigure(const int index, TempusFigure &result) const;    
    
    TempusGraph(const float predict_offset_);
};

//+-------------------------------------------------------------------+
//|                                                                   |
//|                     C O N T R O L L E R                           |
//|                                                                   |
//+-------------------------------------------------------------------+

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class TempusController
{
    int               fig_res;
    string            valueColumns;
    bool              average;

    // Not used anymore. Will be deleted on code cleanup
    void doRequest    (MqlNet &mqlNet, const datetime timeCoord, const string &dataset);
    bool              getResults(MqlNet &mqlNet, const datetime timeCoord, const string dataset, TempusFigure &fig);
    double            queryForecast(MqlNet &mqlNet, const datetime timeCoord, const ushort OHLC, const string dataset, const bool request);
public:
    void              init(const string &symbol, const int _figRes = PERIOD_CURRENT, const bool requestHigh = false, const bool requestLow = false, const bool requestOpen = false, const bool requestClose = false, const bool _average = true);

    ulong doRequest   (MqlNet &mqlNet, const datetime timeStart, const ulong bars, const string &dataset);
    bool              getResults(MqlNet &mqlNet, const datetime timeStart, const ulong bars, const string dataset, TempusFigure &fig[]);
};

//+-------------------------------------------------------------------+
//|                                                                   |
//|                     I M P L E M E N T A T I O N                   |
//|                                                                   |
//+-------------------------------------------------------------------+

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void TempusController::init(const string &symbol, const int _figRes, const bool requestHigh, const bool requestLow, const bool requestOpen, const bool requestClose, const bool _average)
{
    if(_average) {
        valueColumns = symbol + "_avg_bid";
    } else {
        if(requestHigh) valueColumns += "high,";
        if(requestLow) valueColumns += "low,";
        if(requestOpen) valueColumns += "open,";
        if(requestClose) valueColumns += "close,";
        StringSetCharacter(valueColumns, StringLen(valueColumns) - 1, 0);
        StringSetLength(valueColumns, StringLen(valueColumns) - 1);
        if(requestHigh && requestLow && requestLow && requestOpen) valueColumns = "";
    }
    average = _average;
    fig_res = _figRes;
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
ulong TempusController::doRequest(MqlNet &mqlNet, const datetime timeStart, const ulong bars, const string &dataset)
{
    Hash params;
    params.hPutString("dataset", dataset);
    params.hPutString("value_time_start", TimeToString(timeStart, TIME_DATE_SECONDS));
    params.hPutString("value_time_end", TimeToString(timeStart + fig_res * bars, TIME_DATE_SECONDS));
    params.hPutString("value_columns", valueColumns);
    string response;
    ulong finalResult = -1;
    LOG_DEBUG("", "Sending request " + TimeToString(timeStart, TIME_DATE_SECONDS) + " RPC call.");
    const string request_call = "request";
    const string function_name = "makeMultivalRequest";
    mqlNet.RpcCall(request_call, function_name, params, response);

    if(StringLen(response) <= 0) return finalResult;
    JSONParser parser;
    JSONValue *jv = parser.parse(response);
    if (!jv) {
        LOG_ERROR("", "JSON Value for " + string(timeStart) + " is null.");
        return finalResult;
    }

    if (!jv.isObject()) {
        LOG_ERROR("", "Received value for "  + string(timeStart) + " is not object");
        delete jv;
        return finalResult;
    }
    
    JSONObject *jo = jv;
    if (!jo.getValue("error").isNull()) {
        finalResult = -1;
        string err_msg = jo.getString("error");
        LOG_ERROR("", err_msg);
    } else if (!jo.getObject("result").isNull()) {
        JSONObject *result = jo.getObject("result");
        finalResult = result.getInt("request_id");
        LOG_VERBOSE ("", "Request id " + string(finalResult));
    }
    delete jv;
    return finalResult;
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool TempusController::getResults(MqlNet &mqlNet, const datetime timeStart, const ulong bars, const string dataset, TempusFigure &figs[])
{
    Hash params;
    params.hPutString("resolution", string(fig_res));
    LOG_VERBOSE("", "Getting results for " + string(timeStart));

    params.hPutString("dataset", dataset);
    params.hPutString("value_time_start", TimeToString(timeStart, TIME_DATE_SECONDS));
    params.hPutString("value_time_end", TimeToString(timeStart + fig_res * bars, TIME_DATE_SECONDS));

    string response;
    bool result = false;

    const string request = "request";
    const string get_multival_results = "getMultivalResults";
    mqlNet.RpcCall(request, get_multival_results, params, response);

    if(StringLen(response) <= 0) {
        LOG_ERROR("", "Response empty for " + string(timeStart));
        return result;
    }
    
    JSONParser parser;
    LOG_VERBOSE("", "Parsing " + string(response));

    JSONValue *jv = parser.parse(response);
    if (!jv || !jv.isObject()) {
        LOG_ERROR("", "Failed parsing response " + response + " for time " + TimeToString(timeStart, TIME_DATE_SECONDS));
        return result;
    }

    JSONObject *jo = jv;
    if (!jo.getValue("error").isNull()) {
        string err = jo.getString("error");
        LOG_ERROR("", "Received error " + err);
        delete jv;
        return result;
    } 
    if (jo.getArray("result").isNull()) {
        delete jv;
        LOG_ERROR("", "Result array in response for " + TimeToString(timeStart, TIME_DATE_SECONDS) + " is null.");
        return result;
    }
    
    JSONArray *res = jo.getArray("result");
    if (res.size() < 1) {
        delete jv;
        return false;
    }
    
    ArrayResize(figs, res.size());

    int i = 0, j = 0;
    for(; i < res.size() && j < long(bars); ++i, ++j) {
        figs[j].tm = StringToTime(res.getObject(i).getString("tm"));
        if(figs[j].tm < timeStart) {
            --j;
            LOG_ERROR("", "Skipping invalid result " + string(i) + " with time " + string(figs[j].tm));
            continue;
        }
        if (average) figs[j].hi = figs[j].lo = figs[j].op = figs[j].cl = res.getObject(i).getDouble("x");
        else {
            figs[j].hi = res.getObject(i).getDouble("h");
            figs[j].lo = res.getObject(i).getDouble("l");
            figs[j].op = res.getObject(i).getDouble("o");
            figs[j].cl = res.getObject(i).getDouble("c");
        }
        result = true;

        LOG_VERBOSE("", "Received figure " + string(i) + ", " + figs[i].to_string());
    }
    if (ArraySize(figs) != j) ArrayResize(figs, j);

    
    return result;
}

//+-------------------------------------------------------------------+
//+-------------------------------------------------------------------+
//+-------------------------------------------------------------------+

static const string TempusGraphAvgLineName = "avgLastRequest-price";
static const string TempusGraphTimeLineName = "avgLastRequest-time";
static const string TempusPredictSign = "lastResponseSign";


bool TempusGraph::getFigure(const int index, TempusFigure &result) const
{
    if (index >= ArraySize(figures)) return false;
    result = figures[index];
    return true;
}


TempusGraph::TempusGraph(const float predict_offset_) : clr_line(clrDodgerBlue), predict_offset(predict_offset_)
{
    clr_up   = (color) ChartGetInteger(0, CHART_COLOR_CHART_UP);
    clr_down = (color) ChartGetInteger(0, CHART_COLOR_CHART_DOWN);
}


void TempusGraph::init(const int fig_num_, const int fig_res_, const bool keep_history_, const bool demo_mode, const bool averageMode)
{
    fig_ct = fig_num_;
    fig_res = fig_res_;
    keep_history = keep_history_;

    // ArrayFill(hi, 0, 0, 0);
    // ArrayFill(lo, 0, 0, 0);        
    ArrayInitialize(hi, 0);
    ArrayInitialize(lo, 0);
    ArraySetAsSeries(hi, true);
    ArraySetAsSeries(lo, true);
    SetIndexBuffer(0, hi, INDICATOR_DATA);
    SetIndexBuffer(1, lo, INDICATOR_DATA);

    //PlotIndexSetInteger(0, PLOT_SHIFT, 0);
    //PlotIndexSetInteger(1, PLOT_SHIFT, 0);
    PlotIndexSetDouble(0, PLOT_EMPTY_VALUE, 0);
    PlotIndexSetDouble(1, PLOT_EMPTY_VALUE, 0);

    PlotIndexSetInteger(0, PLOT_ARROW, 233);
    PlotIndexSetInteger(1, PLOT_ARROW, 234);

    SetIndexStyle(0, DRAW_ARROW, STYLE_SOLID, 2, clrBlue);
    PlotIndexSetString(0, PLOT_LABEL, _Symbol + " Buy");
    SetIndexStyle(1, DRAW_ARROW, STYLE_SOLID, 2, clrRed);
    PlotIndexSetString(1, PLOT_LABEL, _Symbol + " Sell");

    LOG_DEBUG("", "View inited.");
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void TempusGraph::close()
{
    ArrayInitialize(hi, 0);
    ArrayInitialize(lo, 0);

    ObjectDelete(0, TempusGraphAvgLineName);
    ObjectDelete(0, TempusGraphTimeLineName);
    ObjectDelete(0, TempusPredictSign);
    Comment("");
}

//+------------------------------------------------------------------+
//| Append new figures to RAM                                        |
//+------------------------------------------------------------------+
bool TempusGraph::redraw(const TempusFigure &new_figs[], const datetime req_time, const bool average)
{
    const int new_figs_size = ArraySize(new_figs);
    if (new_figs_size < 1) {
        LOG_DEBUG("", "New figures are empty!");
        return false;
    }
    // Append new figures
    const int prev_figs_size = ArraySize(figures);
    ArrayResize(figures, prev_figs_size + new_figs_size);
    ArrayCopy(figures, new_figs, prev_figs_size);

    return fadeFigure();
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool TempusGraph::fadeFigure()
{
    datetime lasttm = 0;
    static double prev_drawn_price = 0;
    datetime last_anchor_time = 0;
    double last_anchor_price = 0;
    double last_price = 0;
    const int period_seconds = PeriodSeconds();
    const int period_seconds_m1 = PeriodSeconds(PERIOD_M1);
    const datetime offset_start = datetime(predict_offset * period_seconds - period_seconds_m1);
    for(int i = 0; i < ArraySize(figures); ++i) {
        if (figures[i].empty()) {
            LOG_ERROR("", "Skipping unitialized figure " + string(i) + " " + figures[i].to_string());
            continue;
        }
        const datetime anchor_time = figures[i].tm - offset_start;
        const double anchor_price = iClose(_Symbol, PERIOD_M1, iBarShift(_Symbol, PERIOD_M1, anchor_time, false));
        //LOG_DEBUG("", "Anchor time " + TimeToString(anchor_time, TIME_DATE_SECONDS) + ", anchor close price " + DoubleToString(anchor_price));
        if (figures[i].tm > lasttm) {
            last_anchor_time = anchor_time;
            last_anchor_price = anchor_price;
            last_price = figures[i].op;
            lasttm = figures[i].tm;
        }
        const int ind_ix = iBarShift(_Symbol, PERIOD_CURRENT, figures[i].tm); // (iTime(_Symbol, Period(), 0) - figures[i].tm) / fig_res;
        if(ind_ix < 0 || ind_ix >= ArraySize(hi) || ind_ix >= ArraySize(lo)) {
            LOG_DEBUG("", "Figure " + figures[i].to_string() + " index " + string(ind_ix) + " out of bounds " + string(ArraySize(hi) - 1));
            continue;
        }
        if (ind_ix == 0) {
            hi[ind_ix] = 0;
            lo[ind_ix] = 0;
            continue;
        }
        if (figures[i].op > anchor_price) {
            hi[ind_ix] = anchor_price;
//            hi[ind_ix] = 0;
            lo[ind_ix] = 0;
        } else {
            hi[ind_ix] = 0;
//            lo[ind_ix] = 0;
            lo[ind_ix] = anchor_price;
        }
    }
    bool redrawn = false;
    if (last_price <= 0 || last_price == prev_drawn_price) {
        LOG_VERBOSE("", "No new price " + string(last_price) + ", previous " + string(prev_drawn_price));
        return redrawn;
    }
    GlobalVariableSet(C_chart_predictions_identifier, last_price);
    GlobalVariableSet(C_chart_prediction_anchor_identifier, last_anchor_price);

/*
    if (ObjectFind(0, TempusGraphAvgLineName) <= 0 && !ObjectCreate(0, TempusGraphAvgLineName, OBJ_HLINE, 0, 0, last_price)) {
        LOG_ERROR("", "Creating graph line failed.");
    } else {
        ObjectSetInteger(0, TempusGraphAvgLineName, OBJPROP_COLOR, clrBlueViolet);
        ObjectSetInteger(0, TempusGraphAvgLineName, OBJPROP_STYLE, STYLE_SOLID);
    }
    
    if (!ObjectSetDouble(0, TempusGraphAvgLineName, OBJPROP_PRICE, last_price)) {
        LOG_ERROR("", "Failed setting new price " + string(last_price) + " to hline.");
        return redrawn;
    }

    if (ObjectFind(0, TempusGraphTimeLineName) <= 0 && !ObjectCreate(0, TempusGraphTimeLineName, OBJ_VLINE, 0, lasttm, 0)) {
        LOG_ERROR("", "Failed drawing vline");
        return redrawn;
    } else {
        ObjectSetInteger(0, TempusGraphTimeLineName, OBJPROP_COLOR, clrBlueViolet);
        ObjectSetInteger(0, TempusGraphTimeLineName, OBJPROP_STYLE, STYLE_SOLID);
    }
    
    if (!ObjectSetInteger(0, TempusGraphTimeLineName, OBJPROP_TIME, lasttm)) {
        LOG_ERROR("", "Failed setting time to hline.");
        return redrawn;
    }
*/
    prev_drawn_price = last_price;
    LOG_VERBOSE("", "Drawn price " + DoubleToString(last_price, 15));
/*
    bool prev_incorrect = false;
    for (long i = 0; i < ArraySize(figures); ++i) {
        const datetime prev_time = lasttm - PeriodSeconds();
        if (figures[i].tm == prev_time) {
            const double prev_anchor_price = iClose(_Symbol, PERIOD_M1, iBarShift(_Symbol, PERIOD_M1, prev_time - C_predict_offset * PeriodSeconds() - PeriodSeconds(PERIOD_M1)));
            MqlTick prev_ticks[];
            const long copied_ct = SVRApi::copy_ticks_safe(_Symbol, prev_ticks, COPY_TICKS_ALL, prev_time, prev_time + PeriodSeconds());
            if (copied_ct < 1) {
               LOG_ERROR("", "No ticks copied for prev time " + TimeToString(prev_time, TIME_DATE_SECONDS));
               continue;
            }
            const AveragePrice prev_average_price(prev_ticks, prev_time, PeriodSeconds(), iOpen(_Symbol, PERIOD_CURRENT, iBarShift(_Symbol, PERIOD_H1, prev_time)));
            const double prev_actual_price = prev_average_price.value;
            if (figures[i].op > prev_anchor_price != prev_actual_price > prev_anchor_price) {
                prev_incorrect = true;
                break;
            }
        }
    }
*/
    const bool sign_buy = last_price > last_anchor_price;
    if (true) {
        const string comment_text = (sign_buy ? "BUY " : "SELL ") + " at " + DoubleToString(last_anchor_price, 5) + ", starting time " + 
            TimeToString(last_anchor_time + PeriodSeconds(PERIOD_M1), TIME_DATE_SECONDS) + " expected hit price from " + 
            TimeToString(lasttm, TIME_DATE_SECONDS) + ", until " + TimeToString(lasttm + PeriodSeconds(), TIME_DATE_SECONDS) + "\nPredicted price " + DoubleToString(last_price, 5) + ", predicted movement " + DoubleToString(last_price - last_anchor_price) + "\nLast update server time: " + 
            TimeToString(TimeCurrent(), TIME_DATE_SECONDS) + ", last update local time " + TimeToString(TimeLocal(), TIME_DATE_SECONDS);
        Comment(comment_text);
    } else
        Comment("Please wait . . .");

    ObjectDelete(0, TempusPredictSign);
    const ENUM_OBJECT arrow_type = sign_buy ? OBJ_ARROW_BUY : OBJ_ARROW_SELL;
    if (!ObjectCreate(0, TempusPredictSign, arrow_type, 0, lasttm, last_anchor_price))
        LOG_ERROR("", "Failed creating signal display!");
        
    ObjectSetDouble(0, TempusPredictSign, OBJPROP_PRICE, last_anchor_price);    
    ObjectSetInteger(0, TempusPredictSign, OBJPROP_TIME, lasttm);
    ObjectSetInteger(0, TempusPredictSign, OBJPROP_FONTSIZE, 20);
    ObjectSetInteger(0, TempusPredictSign, OBJPROP_COLOR, sign_buy ? clrBlue : clrRed);

    return redrawn;
}


//


//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void TempusController::doRequest(MqlNet &mqlNet, const datetime timeCoord, const string &dataset)
{
    if(average)
        queryForecast(mqlNet, timeCoord, 4, dataset, true);
    else {
        queryForecast(mqlNet, timeCoord, 0, dataset, true);
        queryForecast(mqlNet, timeCoord, 1, dataset, true);
        queryForecast(mqlNet, timeCoord, 2, dataset, true);
        queryForecast(mqlNet, timeCoord, 3, dataset, true);
    }
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool TempusController::getResults(MqlNet &mqlNet, const datetime timeCoord, const string dataset, TempusFigure &fig)
{
    bool result = false;
    LOG_VERBOSE("", "Getting results " + string(average));

    TempusFigure temp;
    if(average)
        temp.op = temp.hi = temp.lo = temp.cl = queryForecast(mqlNet, timeCoord, 4, dataset, false);
    else {
        temp.op = queryForecast(mqlNet, timeCoord, 0, dataset, false );
        temp.hi = queryForecast(mqlNet, timeCoord, 1, dataset, false );
        temp.lo = queryForecast(mqlNet, timeCoord, 2, dataset, false );
        temp.cl = queryForecast(mqlNet, timeCoord, 3, dataset, false );
    }
    temp.tm = timeCoord;

    result = temp.op > 0 && temp.hi > 0 && temp.lo > 0 && temp.cl > 0;

    if( result ) {
        fig = temp;
        return true;
    }
    return false;
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
double TempusController::queryForecast(MqlNet &mqlNet, const datetime timeCoord, const ushort OHLC, const string dataset, const bool request)
{
    if (OHLC > 4) return(0);

//filling Hash
    Hash param;
    param.hPutString("resolution", string(fig_res));
    param.hPutString("dataset", dataset);
    string value_column;
    switch (OHLC) {
        case 0: value_column = "open"; break;
        case 1: value_column = "high"; break;
        case 2: value_column = "low"; break;
        case 4: value_column = "bid"; break;
        default: value_column = "close"; break;
    }    
    param.hPutString("value_column", value_column);
    param.hPutString("value_time", TimeToString(timeCoord, TIME_DATE_SECONDS));

    string response;
    double finalResult = -1;

    const string request_call = "request";
    const string send_tick_call = "sendTick";
    const string request_post_payload = request ? "makeForecastRequest" : "getForecastResult";
    mqlNet.RpcCall(request_call, request_post_payload, param, response);
    LOG_VERBOSE("", "Got response");

    if(StringLen(response) > 0) {
        JSONParser parser;

        LOG_VERBOSE("", "Parsing response: " + response);
        JSONValue *jv = parser.parse(response);

        if (jv != NULL && jv.isObject()) {
            JSONObject *jo = jv;

            if (!jo.getValue("error").isNull()) {
                LOG_VERBOSE("", "Received error.");
                string err = jo.getString("error");
                if (request && StringFind(err, "Cannot make forecast request because it already exists") != -1 )
                    finalResult = -1;
                else if (StringFind(err, "Response is not ready yet") == -1)
                    LOG_ERROR("TempusController::queryForecast", err);
            } else {
                if (!jo.getObject("result").isNull()) {
                    LOG_VERBOSE("", "Received result.");
                    JSONObject *result = jo.getObject("result");
                    finalResult = result.getDouble(request ? "request_id" : "x");
                    LOG_VERBOSE ("TempusController::queryForecast", (request ? "request_id:" : "forecasted_value:") + string(finalResult));
                }
            }
        }
        LOG_VERBOSE("", "Done.");

        delete jv;
    }

    return(finalResult);
}
//+------------------------------------------------------------------+
