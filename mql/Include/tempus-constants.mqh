//+------------------------------------------------------------------+
//|                                             tempus-constants.mqh |
//|                                  Copyright 2021, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2021, MetaQuotes Ltd."
#property link      "https://www.mql5.com"

// #define DEBUG_CONNECTOR
#define HISTORYTOSQL
// #define RUN_TEST

// For all bars
#define BARS_OFFERED INT_MAX

const int C_period_seconds = PeriodSeconds();
const string C_period_seconds_str = IntegerToString(C_period_seconds);
const string C_chart_identifier = _Symbol + "_" + string(C_period_seconds) + (Average ? "_avg_" : "_") + ServerUrl;
const string C_chart_predictions_identifier = "Predictions" + "_" + C_chart_identifier;
const string C_chart_prediction_anchor_identifier = "Anchor" + "_" + C_chart_identifier;
const string C_one_minute_chart_identifier = _Symbol + "_60_avg_" + ServerUrl;
const datetime C_zero_time = datetime(0);

#define TIME_DATE_SECONDS (TIME_DATE | TIME_SECONDS)
#define DOUBLE_PRINT_DECIMALS 15
