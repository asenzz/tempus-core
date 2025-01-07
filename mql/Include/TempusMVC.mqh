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

struct TempusFigure {
    aprice           op, hi, lo, cl;
    datetime         tm;

                     TempusFigure(): tm(0) {}

    bool             empty()
    {
        return tm == 0 || op.valid() || hi.valid() || lo.valid() || cl.valid();
    }

    string           to_string()
    {
        return TimeToString(tm, C_time_mode) + ", " + hi.to_string() + ", " + lo.to_string() + ", " + op.to_string() + " " + cl.to_string();
    }
};

//+-------------------------------------------------------------------+

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class TempusGraph
{
    double           hi[], lo[];
    color            clr_up, clr_down, clr_line;
    TempusFigure     figures[];
    uint             fig_ct, resolution;
    datetime         cur_time;
    bool             keep_history;
    float            predict_offset;

    void             place_figure(const TempusFigure &fig);

public:
    void             init(const uint fig_num_, const uint fig_res_, const bool keep_history_ = false, const bool demo_mode = false, const bool averageMode = false);
    void             close();
    bool             redraw(const TempusFigure &new_figs[], const datetime req_time, const bool average);
    bool             fadeFigure();

    bool             getFigure(const int index, TempusFigure &result) const;

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
    string res_bid_column, res_ask_column, res_open_bid_column, res_open_ask_column, res_high_bid_column, res_high_ask_column, res_low_bid_column,
           res_low_ask_column, res_close_bid_column, res_close_ask_column;

    uint             resolution;
    string resolution_str;
    string           value_columns;
    bool             average;

    // Not used anymore. Will be deleted on code cleanup
    void             doRequest(MqlNet &mql_net, const datetime r_time, const string &dataset);
    bool             getResults(MqlNet &mql_net, const datetime r_time, const string &dataset, TempusFigure &fig);
    aprice           queryForecast(MqlNet &mql_net, const datetime r_time, const ushort OHLC, const string &dataset, const bool request);
public:
    void             init(const string &symbol, const uint fig_res_ = PERIOD_CURRENT, const bool request_high = false, const bool request_low = false,
                          const bool request_open = false, const bool request_close = false, const bool request_average = true);

    int              doRequest(MqlNet &mql_net, const datetime time_start, const uint bars, const string &dataset);
    bool             getResults(MqlNet &mql_net, const datetime time_start, const uint bars, const string &dataset, TempusFigure &fig[]);
};

//+-------------------------------------------------------------------+
//|                                                                   |
//|                     I M P L E M E N T A T I O N                   |
//|                                                                   |
//+-------------------------------------------------------------------+

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void TempusController::init(
    const string &symbol,
    const uint fig_res_,
    const bool request_high,
    const bool request_low,
    const bool request_open,
    const bool request_close,
    const bool request_average)
{
    res_bid_column = symbol + "_avg_bid";
    res_ask_column = symbol + "_avg_ask";
    res_open_bid_column = symbol + "_open_bid";
    res_open_ask_column = symbol + "_open_ask";
    res_high_bid_column = symbol + "_high_bid";
    res_high_ask_column = symbol + "_high_ask";
    res_low_bid_column = symbol + "_low_bid";
    res_low_ask_column = symbol + "_low_ask";
    res_close_bid_column = symbol + "_close_bid";
    res_close_ask_column = symbol + "_close_ask";

    if(request_average) {
        value_columns = res_bid_column + "," + res_ask_column;
    } else {
        value_columns = "";
        if(request_open) value_columns += res_open_bid_column + "," + res_open_ask_column;
        if(request_high) value_columns += res_high_bid_column + "," + res_high_ask_column;
        if(request_low) value_columns += res_low_bid_column + "," + res_low_ask_column;
        if(request_close) value_columns += res_close_bid_column + "," + res_close_ask_column;
    }

    average = request_average;
    resolution = fig_res_;
    resolution_str = IntegerToString(resolution);
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int TempusController::doRequest(MqlNet &mql_net, const datetime time_start, const uint bars, const string &dataset)
{
    Hash params;
    params.hPutString("dataset", dataset);
    params.hPutString("value_time_start", TimeToString(time_start, C_time_mode));
    params.hPutString("value_time_end", TimeToString(time_start + resolution * bars, C_time_mode));
    params.hPutString("value_columns", value_columns);
    params.hPutString("resolution", C_period_time_str);
    string response;
    int final_result = -1;
    LOG_DEBUG("Sending request " + TimeToString(time_start, C_time_mode) + " RPC call.");
    static const string request_call = "request";
    static const string function_name = "makeMultivalRequest";
    mql_net.RpcCall(request_call, function_name, params, response);
    if(StringLen(response) <= 0) return final_result;

    JSONParser parser;
    JSONValue *jv = parser.parse(response);
    if (!jv) {
        LOG_ERROR("JSON Value for " + TimeToString(time_start, C_time_mode) + " is null.");
        return final_result;
    }

    if (!jv.isObject()) {
        LOG_ERROR("Received value for "  + TimeToString(time_start, C_time_mode) + " is not object");
        delete jv;
        return final_result;
    }

    JSONObject *jo = jv;
    if (jo.getValue("error").isNull()) {
        LOG_DEBUG("Error is null");
        JSONObject *result = jo.getObject("result");
        final_result = result.getInt("request_id");
        LOG_VERBOSE("Request id " + IntegerToString(final_result));
    } else {
        LOG_ERROR("Error is not null.");
        const string err_msg = jo.getString("error");
        if (StringFind(err_msg, "already exists", 0) != -1) final_result = 0;
        LOG_ERROR(err_msg);
    }
    delete jv;

    return final_result;
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool TempusController::getResults(MqlNet &mql_net, const datetime time_start, const uint bars, const string &dataset, TempusFigure &figs[])
{
    Hash params;
    params.hPutString("resolution", C_period_time_str);
    LOG_VERBOSE("Getting results for " + TimeToString(time_start, C_time_mode));
    params.hPutString("dataset", dataset);
    params.hPutString("value_time_start", TimeToString(time_start, C_time_mode));
    params.hPutString("value_time_end", TimeToString(time_start + resolution * bars, C_time_mode));

    string response;
    bool result = false;
    static const string request = "request";
    static const string get_multival_results = "getMultivalResults";
    mql_net.RpcCall(request, get_multival_results, params, response);

    if (StringLen(response) <= 0) {
        LOG_ERROR("Response empty for " + TimeToString(time_start, C_time_mode));
        return result;
    }
    JSONParser parser;
    LOG_VERBOSE("Parsing " + response);

    JSONValue *jv = parser.parse(response);
    if (!jv || !jv.isObject()) {
        LOG_ERROR("Failed parsing response " + response + " for time " + TimeToString(time_start, C_time_mode));
        return result;
    }

    JSONObject *jo = jv;
    JSONValue *jv_error = jo.getValue("error");
    if (jv_error != NULL && !jv_error.isNull()) {
        const string err = jv_error.getString();
        LOG_ERROR("Received error message, " + err);
        delete jv;
        return true;
    }

    if (jo.getArray("result").isNull()) {
        delete jv;
        LOG_ERROR("Result array in response for " + TimeToString(time_start, C_time_mode) + " is null.");
        return result;
    }

    JSONArray *res = jo.getArray("result");
    if (res.size() < 1) {
        LOG_ERROR("Parsed no bars from response.");
        delete jv;
        return false;
    }

    ArrayResize(figs, res.size());
    uint j = 0;
    for (int i = 0; i < res.size() && j < bars; ++i) {
        JSONObject *jo_row = res.getObject(i);
        figs[j].tm = StringToTime(jo_row.getString("tm"));
        if(figs[j].tm < time_start) {
            LOG_ERROR("Skipping invalid result " + IntegerToString(i) + " with time " + TimeToString(figs[j].tm, C_time_mode));
            continue;
        }
        if (average) {
            string bid, ask;
            if (jo_row.getString(res_bid_column, bid) == false) {
                LOG_ERROR("Column " + res_bid_column + " is missing, discarding row " + jo_row.getString("tm") + ", " + IntegerToString(i) + ", " + jo_row.toString());
                continue;
            }
            if (jo_row.getString(res_ask_column, ask) == false) {
                LOG_ERROR("Column " + res_ask_column + " is missing, discarding row " + jo_row.getString("tm") + ", " + IntegerToString(i) + ", " + jo_row.toString());
                continue;
            }
            figs[j].cl.set(StringToDouble(bid), StringToDouble(ask));
        } else {
            figs[j].op.set(jo_row.getDouble(res_open_bid_column), jo_row.getDouble(res_open_ask_column));
            figs[j].hi.set(jo_row.getDouble(res_high_bid_column), jo_row.getDouble(res_high_ask_column));
            figs[j].lo.set(jo_row.getDouble(res_low_bid_column), jo_row.getDouble(res_low_ask_column));
            figs[j].cl.set(jo_row.getDouble(res_close_bid_column), jo_row.getDouble(res_close_ask_column));
        }
        result = true;
        LOG_VERBOSE("Received figure " + IntegerToString(i) + ", " + figs[j].to_string());
        ++j;
    }
    if (ArraySize(figs) != j) ArrayResize(figs, j);
    LOG_VERBOSE("Parsed " + IntegerToString(j) + " figures from response.");
    return result;
}

//+-------------------------------------------------------------------+
//+-------------------------------------------------------------------+
//+-------------------------------------------------------------------+

static const string TempusGraphAvgLineName = "avgLastRequest-price";
static const string TempusGraphTimeLineName = "avgLastRequest-time";
static const string TempusPredictSign = "lastResponseSign";


//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool TempusGraph::getFigure(const int index, TempusFigure &result) const
{
    if (index >= ArraySize(figures)) return false;
    result = figures[index];
    return true;
}


//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
TempusGraph::TempusGraph(const float predict_offset_) : clr_line(clrDodgerBlue), predict_offset(predict_offset_)
{
    clr_up   = (color) ChartGetInteger(0, CHART_COLOR_CHART_UP);
    clr_down = (color) ChartGetInteger(0, CHART_COLOR_CHART_DOWN);
}


//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void TempusGraph::init(const uint fig_num_, const uint fig_res_, const bool keep_history_, const bool demo_mode, const bool averageMode)
{
    fig_ct = fig_num_;
    resolution = fig_res_;
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

    LOG_DEBUG("View inited.");
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
        LOG_DEBUG("New figures are empty!");
        return false;
    }
// Append new figures
    const int prev_figs_size = ArraySize(figures);
    ArrayResize(figures, prev_figs_size + new_figs_size);
    for (int i = prev_figs_size; i < new_figs_size; ++i) figures[i] = new_figs[i - prev_figs_size];

    return fadeFigure();
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool TempusGraph::fadeFigure()
{
#ifdef DRAWING_PORTED
    datetime lasttm = 0;
    static double prev_drawn_price = 0;
    datetime last_anchor_time = 0;
    aprice last_anchor_price;
    aprice last_price;
    const int period_seconds_m1 = PeriodSeconds(PERIOD_M1);
    const datetime offset_start = datetime(predict_offset * C_period_seconds - 1);
    for(int i = 0; i < ArraySize(figures); ++i) {
        if (figures[i].empty()) {
            LOG_ERROR("Skipping unitialized figure " + string(i) + " " + figures[i].to_string());
            continue;
        }
        const datetime anchor_time = figures[i].tm - offset_start;
        const aprice anchor_price = get_price(anchor_time);
        //LOG_DEBUG("Anchor time " + TimeToString(anchor_time, C_time_mode) + ", anchor close price " + DoubleToString(anchor_price));
        if (figures[i].tm > lasttm) {
            last_anchor_time = anchor_time;
            last_anchor_price = anchor_price;
            last_price = figures[i].op;
            lasttm = figures[i].tm;
        }
        const int ind_ix = iBarShift(_Symbol, PERIOD_CURRENT, figures[i].tm); // (iTime(_Symbol, Period(), 0) - figures[i].tm) / resolution;
        if(ind_ix < 0 || ind_ix >= ArraySize(hi) || ind_ix >= ArraySize(lo)) {
            LOG_DEBUG("Figure " + figures[i].to_string() + " index " + string(ind_ix) + " out of bounds " + string(ArraySize(hi) - 1));
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
        LOG_VERBOSE("No new price " + string(last_price) + ", previous " + string(prev_drawn_price));
        return redrawn;
    }
    GlobalVariableSet(C_chart_predictions_identifier, last_price);
    GlobalVariableSet(C_chart_prediction_anchor_identifier, last_anchor_price);

    /*
        if (ObjectFind(0, TempusGraphAvgLineName) <= 0 && !ObjectCreate(0, TempusGraphAvgLineName, OBJ_HLINE, 0, 0, last_price)) {
            LOG_ERROR("Creating graph line failed.");
        } else {
            ObjectSetInteger(0, TempusGraphAvgLineName, OBJPROP_COLOR, clrBlueViolet);
            ObjectSetInteger(0, TempusGraphAvgLineName, OBJPROP_STYLE, STYLE_SOLID);
        }

        if (!ObjectSetDouble(0, TempusGraphAvgLineName, OBJPROP_PRICE, last_price)) {
            LOG_ERROR("Failed setting new price " + string(last_price) + " to hline.");
            return redrawn;
        }

        if (ObjectFind(0, TempusGraphTimeLineName) <= 0 && !ObjectCreate(0, TempusGraphTimeLineName, OBJ_VLINE, 0, lasttm, 0)) {
            LOG_ERROR("Failed drawing vline");
            return redrawn;
        } else {
            ObjectSetInteger(0, TempusGraphTimeLineName, OBJPROP_COLOR, clrBlueViolet);
            ObjectSetInteger(0, TempusGraphTimeLineName, OBJPROP_STYLE, STYLE_SOLID);
        }

        if (!ObjectSetInteger(0, TempusGraphTimeLineName, OBJPROP_TIME, lasttm)) {
            LOG_ERROR("Failed setting time to hline.");
            return redrawn;
        }
    */
    prev_drawn_price = last_price;
    LOG_VERBOSE("Drawn price " + DoubleToString(last_price, 15));
    /*
        bool prev_incorrect = false;
        for (long i = 0; i < ArraySize(figures); ++i) {
            const datetime prev_time = lasttm - PeriodSeconds();
            if (figures[i].tm == prev_time) {
                const double prev_anchor_price = iClose(_Symbol, PERIOD_M1, iBarShift(_Symbol, PERIOD_M1, prev_time - C_predict_offset * PeriodSeconds() - PeriodSeconds(PERIOD_M1)));
                MqlTick prev_ticks[];
                const long copied_ct = SVRApi::copy_ticks_safe(_Symbol, prev_ticks, COPY_TICKS_ALL, prev_time, prev_time + PeriodSeconds());
                if (copied_ct < 1) {
                   LOG_ERROR("No ticks copied for prev time " + TimeToString(prev_time, C_time_mode));
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
                                    TimeToString(last_anchor_time + PeriodSeconds(PERIOD_M1), C_time_mode) + " expected hit price from " +
                                    TimeToString(lasttm, C_time_mode) + ", until " + TimeToString(lasttm + PeriodSeconds(), C_time_mode) + "\nPredicted price " + DoubleToString(last_price, 5) + ", predicted movement " + DoubleToString(last_price - last_anchor_price) + "\nLast update server time: " +
                                    TimeToString(TimeCurrent(), C_time_mode) + ", last update local time " + TimeToString(TimeLocal(), C_time_mode);
        Comment(comment_text);
    } else
        Comment("Please wait . . .");

    ObjectDelete(0, TempusPredictSign);
    const ENUM_OBJECT arrow_type = sign_buy ? OBJ_ARROW_BUY : OBJ_ARROW_SELL;
    if (!ObjectCreate(0, TempusPredictSign, arrow_type, 0, lasttm, last_anchor_price))
        LOG_ERROR("Failed creating signal display!");

    ObjectSetDouble(0, TempusPredictSign, OBJPROP_PRICE, last_anchor_price);
    ObjectSetInteger(0, TempusPredictSign, OBJPROP_TIME, lasttm);
    ObjectSetInteger(0, TempusPredictSign, OBJPROP_FONTSIZE, 20);
    ObjectSetInteger(0, TempusPredictSign, OBJPROP_COLOR, sign_buy ? clrBlue : clrRed);

    return redrawn;
#endif
    return false;
}


//


//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void TempusController::doRequest(MqlNet &mql_net, const datetime r_time, const string &dataset)
{
    if(average) {
        queryForecast(mql_net, r_time, 4, dataset, true);
        queryForecast(mql_net, r_time, 5, dataset, true);
    } else {
        queryForecast(mql_net, r_time, 0, dataset, true);
        queryForecast(mql_net, r_time, 1, dataset, true);
        queryForecast(mql_net, r_time, 2, dataset, true);
        queryForecast(mql_net, r_time, 3, dataset, true);
    }
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool TempusController::getResults(MqlNet &mql_net, const datetime r_time, const string &dataset, TempusFigure &fig)
{
    bool result = false;
    LOG_VERBOSE("Getting results " + string(average));

    TempusFigure temp;
    if(average) {
        temp.cl = queryForecast(mql_net, r_time, 4, dataset, false);
    } else {
        temp.op = queryForecast(mql_net, r_time, 0, dataset, false );
        temp.hi = queryForecast(mql_net, r_time, 1, dataset, false );
        temp.lo = queryForecast(mql_net, r_time, 2, dataset, false );
        temp.cl = queryForecast(mql_net, r_time, 3, dataset, false );
    }
    temp.tm = r_time;

    result = temp.op.valid() && temp.hi.valid() && temp.lo.valid() && temp.cl.valid();

    if( result ) {
        fig = temp;
        return true;
    }
    return false;
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
aprice TempusController::queryForecast(MqlNet &mql_net, const datetime r_time, const ushort OHLC, const string &dataset, const bool request)
{
#ifdef NONEEDNOW
    if (OHLC > 5) return(0);

//filling Hash
    Hash param;
    param.hPutString("resolution", resolution_str);
    param.hPutString("dataset", dataset);
    string value_column;
    switch (OHLC) {
    case 1:
        value_column = "high";
        break;
    case 2:
        value_column = "low";
        break;
    case 4:
        value_column = "bid";
        break;
    case 5:
        value_column = "ask";
        break;
    default:
        value_column = "close";
        break;
    }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
    param.hPutString("value_column", value_column);
    param.hPutString("value_time", TimeToString(r_time, C_time_mode));

    string response;
    double final_result = -1;

    const string request_call = "request";
    const string send_tick_call = "sendTick";
    const string request_post_payload = request ? "makeForecastRequest" : "getForecastResult";
    mql_net.RpcCall(request_call, request_post_payload, param, response);
    LOG_VERBOSE("Got response");

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
    if(StringLen(response) > 0) {
        JSONParser parser;
        LOG_VERBOSE("Parsing response: " + response);
        JSONValue *jv = parser.parse(response);
        if (jv != NULL && jv.isObject()) {
            JSONObject *jo = jv;
            if (!jo.getValue("error").isNull()) {
                LOG_VERBOSE("Received error.");
                string err = jo.getString("error");
                if (request && StringFind(err, "Cannot make forecast request because it already exists") != -1 )
                    final_result = -1;
                else if (StringFind(err, "Response is not ready yet") == -1)
                    LOG_ERROR(err);
            } else {
                if (!jo.getObject("result").isNull()) {
                    LOG_VERBOSE("Received result.");
                    JSONObject *result = jo.getObject("result");
                    final_result = result.getDouble(request ? "request_id" : "x");
                    LOG_VERBOSE ("TempusController::queryForecast", (request ? "request_id:" : "forecasted_value:") + string(final_result));
                }
            }
        }
        LOG_VERBOSE("Done.");
        delete jv;
    }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
    return(final_result);
#endif

    return aprice();
}
//+------------------------------------------------------------------+
