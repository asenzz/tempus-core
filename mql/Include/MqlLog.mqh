//+------------------------------------------------------------------+
//|                                                       MqlLog.mqh |
//|                                                     Papakaya LTD |
//|                                         https://www.papakaya.com |
//+------------------------------------------------------------------+
#property copyright "Papakaya LTD"
#property link      "https://www.papakaya.com"
#property strict

#include "tempus-constants.mqh"

// #define LOG_TO_FILE

//+------------------------------------------------------------------+

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void set_perf_logging(const bool value)
{
    LogPerfEnabled = value;
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void set_debug_logging(const bool value)
{
    LogDebugEnabled = value;
}

// Logging options ordered in priority
bool LogErrorEnabled = true;
bool LogInfoEnabled = true;
bool LogDebugEnabled = true;
bool LogVerboseEnabled = true;
bool LogPerfEnabled = true;
bool LogLineEnabled = true;


//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void log_msg(const string type, const string &method, const string &file, const string line, const string message)
{
    const string msg_str = TimeToString(TimeLocal(), C_time_mode) + ": " + type + method + (LogLineEnabled ? "(" + file + "." + string(line) + ")" : "") + (StringLen(method) > 0 ? ":" : "") + " " + message;
#ifdef LOG_TO_FILE
    static int file_handle = FileOpen("connector.log", FILE_SHARE_WRITE | FILE_CSV);
    if(file_handle != INVALID_HANDLE) {
        FileSeek(file_handle, 0, SEEK_END);
        FileWrite(file_handle, msg_str);
        //FileFlush(file_handle);
        //FileClose(file_handle);
    }
#endif
    Print(msg_str);
}

#define LOG_ERROR(message) if (LogErrorEnabled || LogInfoEnabled || LogDebugEnabled) log_msg("ERROR: ", __FUNCTION__, __FILE__, string(__LINE__), message )
#define LOG_SYS_ERR(message) if (LogErrorEnabled || LogInfoEnabled || LogDebugEnabled) log_msg("ERROR: ", __FUNCTION__, __FILE__, string(__LINE__), message + " Sys: " + IntegerToString(GetLastError()) + ", " + ErrorDescription(GetLastError()))
#define LOG_INFO(message) if (LogInfoEnabled || LogDebugEnabled || LogVerboseEnabled) log_msg("INFO: ", __FUNCTION__, __FILE__, string(__LINE__), message )
#define LOG_DEBUG(message) if (LogDebugEnabled || LogVerboseEnabled) log_msg("DEBUG: ", __FUNCTION__, __FILE__, string(__LINE__), message)
#define LOG_VERBOSE(message) if (LogVerboseEnabled) log_msg("VERBOSE: ", __FUNCTION__, __FILE__, string(__LINE__), message)
#define LOG_PERF(message) if (LogPerfEnabled) log_msg("PERF: ", __FUNCTION__, __FILE__, string(__LINE__), message)

//+------------------------------------------------------------------+
