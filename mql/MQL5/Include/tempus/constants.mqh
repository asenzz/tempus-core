//+------------------------------------------------------------------+
//|                                             tempus_constants.mqh |
//|                                  Copyright 2021, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2021, MetaQuotes Ltd."
#property link      "https://www.mql5.com"

// #define DEBUG_CONNECTOR
#define HISTORYTOSQL
// #define RUN_TEST
#define TOLERATE_DATA_INCONSISTENCY

// For all bars
const int C_time_mode = TIME_DATE | TIME_SECONDS;
const int C_bars_offered = INT_MAX; // Maximum offered bars limit
const int C_period_seconds = PeriodSeconds();
const uint C_period_m1_seconds = PeriodSeconds(PERIOD_M1);
const string C_one_str = IntegerToString(1);
const string C_period_seconds_str = IntegerToString(C_period_seconds);
const string C_period_time_str = TimeToString(datetime(C_period_seconds), TIME_SECONDS);// Limited to 24 hours period

#ifdef TEMPUS_CONNECTOR
const string C_chart_identifier = _Symbol + "_" + IntegerToString(C_period_seconds) + (Average ? "_avg_" : "_") + ServerUrl;
const string C_chart_predictions_identifier = "Predictions" + "_" + C_chart_identifier;
const string C_chart_prediction_anchor_identifier = "Anchor" + "_" + C_chart_identifier;
const string C_one_minute_chart_identifier = _Symbol + "_60_avg_" + ServerUrl;
#endif

const datetime C_zero_time = datetime(0);
const bool C_backtesting = MQLInfoInteger(MQL_TESTER);
const long C_leverage = AccountInfoInteger(ACCOUNT_LEVERAGE);
#define DOUBLE_PRINT_DECIMALS 15
const uint C_max_retries = C_backtesting ? 2 : 10; // for HTTP calls
const string C_streams_root = "Z:\\dev\\shm\\tempus";
const uint C_print_progress_every = 10000;
const string C_queue_prefix = "q_svrwave_";
const int C_send_wait = 5; // Wait seconds to confirm sent order