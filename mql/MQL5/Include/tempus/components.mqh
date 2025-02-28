//+-------------------------------------------------------------------+
//| This is a MVC helper file file for the Tempus Connectors project  |
//|                                                                   |
//|                                                                   |
//+-------------------------------------------------------------------+
#property copyright "Tempus Connectors"
#property strict

#include <tempus/hash.mqh>
#include <tempus/json.mqh>
#include <tempus/net.mqh>
#include <tempus/compatibility.mqh>
#include <tempus/log.mqh>
#include <tempus/price.mqh>

struct price_row {
    aprice           op, hi, lo, cl;
    datetime         tm;
    
                     price_row(): tm(0) {}

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
class tempus_graph
{
    double           hi[], lo[];
    color            clr_up, clr_down, clr_line;
    price_row        figures[];
    uint             fig_ct, resolution;
    datetime         cur_time;
    bool             keep_history;
    float            predict_offset;

    void             place_figure(const price_row &fig);

public:
    void             init(const uint fig_num_, const uint fig_res_, const bool keep_history_ = false, const bool demo_mode = false, const bool averageMode = false);
    void             close();
    bool             redraw(const price_row &new_figs[], const datetime req_time, const bool average);
    bool             fade_figure();

    bool             get_figure(const int index, price_row &result) const;

                     tempus_graph(const float predict_offset_);
};

//+-------------------------------------------------------------------+
//|                                                                   |
//|                     C O N T R O L L E R                           |
//|                                                                   |
//+-------------------------------------------------------------------+

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class tempus_controller
{
    string res_bid_column, res_ask_column, res_open_bid_column, res_open_ask_column, res_high_bid_column, res_high_ask_column, res_low_bid_column,
           res_low_ask_column, res_close_bid_column, res_close_ask_column;

    uint             resolution;
    string           resolution_str;
    string           value_columns;
    bool             average;

    // Not used anymore. Will be deleted on code cleanup
    void             do_request(MqlNet &mql_net, const datetime r_time, const string &dataset);
    bool             get_results(MqlNet &mql_net, const datetime r_time, const string &dataset, price_row &fig);
    aprice           query_forecast(MqlNet &mql_net, const datetime r_time, const ushort OHLC, const string &dataset, const bool request);
public:
    void             init(const string &symbol, const uint fig_res_ = PERIOD_CURRENT, const bool request_high = false, const bool request_low = false,
                          const bool request_open = false, const bool request_close = false, const bool request_average = true);

    int              do_request(MqlNet &mql_net, const datetime time_start, const uint bars, const string &dataset);
    bool             get_results(MqlNet &mql_net, const datetime time_start, const uint bars, const string &dataset, price_row &fig[]);
};

#include <tempus/components_impl.mqh>