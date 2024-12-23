//+------------------------------------------------------------------+
//|                                                       MqlLog.mqh |
//|                                                     Papakaya LTD |
//|                                         https://www.papakaya.com |
//+------------------------------------------------------------------+
#property copyright "Papakaya LTD"
#property link      "https://www.papakaya.com"
#property strict

#include "tempus-constants.mqh"

//#define LOG_TO_FILE

//+------------------------------------------------------------------+

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void SetPerfLogging(const bool value)
{
    LogPerfEnabled = value;
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void SetDebugLogging(const bool value)
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
void LOG_MESSAGE(const string type, const string &method, const string &file, const string line, const string message)
{
    const string strMsg = TimeToString(TimeLocal(), TIME_SECONDS|TIME_DATE) + ": " + type + method + (LogLineEnabled ? "(" + file + "." + string(line) + ")" : "") + (StringLen(method) > 0 ? ":" : "") + " " + message;
#ifdef LOG_TO_FILE
    static int file_handle = FileOpen("connector.log", FILE_SHARE_WRITE | FILE_CSV);
    if(file_handle != INVALID_HANDLE) {
        FileSeek(file_handle, 0, SEEK_END);
        FileWrite(file_handle, strMsg);
        //FileFlush(file_handle);
        //FileClose(file_handle);
    }
#endif
    Print(strMsg);
}

#define LOG_ERROR(stub, message) if (LogErrorEnabled || LogInfoEnabled || LogDebugEnabled) LOG_MESSAGE("ERROR: ", __FUNCTION__, __FILE__, string(__LINE__), message )
#define LOG_SYS_ERR(stub, message) if (LogErrorEnabled || LogInfoEnabled || LogDebugEnabled) LOG_MESSAGE("ERROR: ", __FUNCTION__, __FILE__, string(__LINE__), message + " Sys: " + IntegerToString(GetLastError()) + ", " + ErrorDescription(GetLastError()))
#define LOG_INFO(stub, message) if (LogInfoEnabled || LogDebugEnabled || LogVerboseEnabled) LOG_MESSAGE("INFO: ", __FUNCTION__, __FILE__, string(__LINE__), message )
#define LOG_DEBUG(stub, message) if (LogDebugEnabled || LogVerboseEnabled) LOG_MESSAGE("DEBUG: ", __FUNCTION__, __FILE__, string(__LINE__), message)
#define LOG_VERBOSE(stub, message) if (LogVerboseEnabled) LOG_MESSAGE("VERBS: ", __FUNCTION__, __FILE__, string(__LINE__), message)
#define LOG_PERF(stub, message) if (LogPerfEnabled) LOG_MESSAGE("PERF: ", __FUNCTION__, __FILE__, string(__LINE__), message)

//+------------------------------------------------------------------+
