//+------------------------------------------------------------------+
//|                                                 RequestUtils.mqh |
//|                        Copyright 2015, MetaQuotes Software Corp. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2015, MetaQuotes Software Corp."
#property link      "https://www.mql5.com"
#property strict


#include <hash.mqh>
#include <MqlNet.mqh>
#include <json.mqh>

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class RequestUtils
{

    JSONValue* jv;

    void Dispose()
    {
        if (CheckPointer(jv) == POINTER_DYNAMIC) delete jv;
    }
public:

    RequestUtils() : jv(NULL) {}
    ~RequestUtils()
    {
        Dispose();
    }

    bool is_response_successful(const string &response);
    string get_error_msg(const string &response);
    JSONObject *get_result_obj(const string &response);
    JSONValue *get_result_val(const string &response);
    JSONArray *get_result_array(const string &response);
};

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool RequestUtils::is_response_successful(const string &response)
{
    if(StringLen(response) < 1) return true;

    JSONParser parser;
    bool is_successful = false;
    Dispose();
    jv = parser.parse(response);
    if (jv != NULL && jv.isObject()) {
        JSONObject *jo = jv;
        is_successful = jo.getValue("error").isNull();
    }
    return is_successful;
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
string RequestUtils::get_error_msg(const string &response)
{
    if (StringLen(response) < 1) return "";

    string error_msg;
    JSONParser parser;

    Dispose();
    jv = parser.parse(response);

    if (jv != NULL && jv.isObject()) {
        JSONObject *jo = jv;
        if (jo.getValue("error").isNull())
            LOG_DEBUG("No error message");
        else
            error_msg = jo.getString("error");
    } else
        LOG_ERROR("Cannot parse error message from response " + response);
        
    return error_msg;
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
JSONObject* RequestUtils::get_result_obj(const string &response)
{

    JSONObject *result = NULL;
    if (StringLen(response) < 1) {
        LOG_ERROR("Empty response string.");
        return result;
    }
    JSONParser parser;
    Dispose();
    jv = parser.parse(response);
    if (jv != NULL && jv.isObject()) {
        JSONObject *jo = jv;
        if (jo.getObject("result").isNull()) {
            LOG_ERROR("No result object");
        } else {
            result = jo.getObject("result");
            LOG_VERBOSE("Result object is " + result.getString("timeFrom"));
        }
    } else
        LOG_ERROR("Cannot read result object from response " + response);
    return result;
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
JSONValue* RequestUtils::get_result_val(const string &response)
{

    JSONValue *result = NULL;
    if (StringLen(response) < 1) return result;
    JSONParser parser;
    Dispose();
    jv = parser.parse(response);
    if (jv != NULL && jv.isObject()) {
        JSONObject *jo = jv;
        if (jo.getValue("result").isNull()) 
            LOG_ERROR("No result value");
        else 
            result = jo.getValue("result");
    } else 
        LOG_ERROR("Cannot read result value from response: " + response);
    return result;
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
JSONArray* RequestUtils::get_result_array(const string &response)
{

    JSONArray *result = NULL;
    if (StringLen(response) < 1) return result;
    JSONParser parser;
    Dispose();
    jv = parser.parse(response);
    if (jv != NULL && jv.isObject()) {
        JSONObject *jo = jv;
        if (jo.isArray() && !jo.getArray("result").isNull())
            result = jo.getArray("result");
        else
            LOG_ERROR("No result array");
    } else
        LOG_ERROR("Cannot read result array from response: " + response);
    return result;
}
//+------------------------------------------------------------------+