//+------------------------------------------------------------------+
//|                                                       tempus_api.mqh |
//|                        Copyright 2015, MetaQuotes Software Corp. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2015, MetaQuotes Software Corp."
#property link "https://www.mql5.com"
#property version "1.00"
#property strict

#include <tempus/hash.mqh>
#include <tempus/json.mqh>

#include <tempus/price.mqh>
#include <tempus/dst_periods.mqh>
#include <tempus/net.mqh>
#include <tempus/request_util.mqh>
#include <tempus/time_util.mqh>
#include <tempus/constants.mqh>
#include <tempus/types.mqh>
#include <tempus/smp.mqh>
#include <tempus/components.mqh>


#define MAX_PACK_SIZE 0xFFFF

const uint C_retries_count = 10;


//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class tempusapi
{
    MqlNet            net;
    string            symbol;
    ENUM_TIMEFRAMES   period;
    uint              period_sec;
    string            period_str, input_queue;
    
    RequestUtils      request_utils;
    static const uint package_len;
    static const string delimiter;
    const string      aux_queue_name, timezone;
    const bool        per_second;
    aprice            hi_bars[], lo_bars[], open_bars[], close_bars[], avg_bars[], aux_avg_bars[]; // Temporary arrays used to buffer data before sending
    
    uint32_t          volumes[];
    datetime          times[];
    const bool        demo_mode;
    const int         demo_bars;
    const bool        is_average, use_stream;
    bool              use_http;
    streams_messaging *p_streams_messaging;

    uint             prepare_history_avg(const datetime start_time, const datetime end_time);

    uint             prepare_history_ohlc(const datetime start_time, const datetime end_time);

    uint             pack_history_ohlc(const uint bar_start, const uint bar_end, Hash &hash);

    uint             pack_history_avg(const uint bar_start, const uint bar_end, Hash &hash);

    int              find_bar(const datetime start_time);

    int              bsearch_bar(const datetime start_time, int left, const int right);

public:
    tempusapi(
        const bool demo_mode_, const int demo_bars_, const bool is_average_, const string aux_queue_name_, const bool per_second_,
        const string &timezone_, streams_messaging *const p_streams_messaging_, const string &symbol, const ENUM_TIMEFRAMES period_, const string &input_queue_);

    ~tempusapi();

    bool             next_time_range(datetime &bar_start, datetime &bar_end);

    bool             connect(const string &url);

    bool             login(const string &user, const string &password);

    void             send_bar(const bool finalize, const aprice &op, const aprice &hi, const aprice &lo, const aprice &cl, const uint vol, const datetime tm);
    void             send_bar(const bool finalize, const AveragePrice &avg_price);
    void             send_bar(const bool finalize, const AveragePrice &avg_price, const AveragePrice &avg_aux_price, const string &col_name, const string &col_aux_name);
    void             send_bars(const bool finalize, const aprice &prices[], const uint &volumes[], const aprice &pricesaux[], const uint &volumesaux[], const datetime &times[], const string &col_name, const string &aux_col_name);
    void             send_bars(const bool finalize, const aprice &prices[], const datetime &times[], const uint &volumes[]);
    bool             send_history();
    bool             send_history(const datetime start_time, datetime end_time);

    static int       copy_rates_safe(const string &symbol, const ENUM_TIMEFRAMES time_frame, const datetime start, const datetime end, MqlRates &rates[]);
    static int       copy_ticks_safe(const string &symbol, MqlTick &ticks[], const uint flags, const datetime time_start, const datetime time_end);
};


#include <tempus/api_impl.mqh>
//+------------------------------------------------------------------+
