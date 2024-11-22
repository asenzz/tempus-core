//+------------------------------------------------------------------+
//|                                             tempus-constants.mqh |
//|                                  Copyright 2021, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2021, MetaQuotes Ltd."
#property link      "https://www.mql5.com"

#define DEBUG_CONNECTOR
// #define HISTORYTOSQL
//#define RUN_TEST

// For one minute
// #define BARS_OFFERED (60 * (20000 + 1000) + 6000000 /* decon tail */) // one minute bars are used for one second data

// For one hour
// #define BARS_OFFERED ((20000 + 1000) + 100000 /* decon tail */)

// For all bars
#define BARS_OFFERED uint(-1)

const string chart_identifier = Symbol() + "_" + string(PeriodSeconds()) + (Average ? "_avg_" : "_") + ServerUrl;
const string chart_predictions_identifier = "Predictions" + "_" + chart_identifier;
const string chart_prediction_anchor_identifier = "Anchor" + "_" + chart_identifier;
const string one_minute_chart_identifier = Symbol() + "_60_avg_" + ServerUrl;
const datetime c_zero_time = datetime(0);
