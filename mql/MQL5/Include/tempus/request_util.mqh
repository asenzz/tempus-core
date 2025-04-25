//+------------------------------------------------------------------+
//|                                                 tempus_request_util.mqh |
//|                        Copyright 2015, MetaQuotes Software Corp. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2015, MetaQuotes Software Corp."
#property link      "https://www.mql5.com"
#property strict


#include <tempus/hash.mqh>
#include <tempus/net.mqh>
#include <tempus/json.mqh>

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
string to_string(const MqlTradeRequest &request)
{
    return StringFormat(
               "Request action %d" +
               ", magic %d" +
               ", order %d" +
               ", symbol %s" +
               ", volume %.2f" +
               ", price %.5f" +
               ", stop limit %.5f" +
               ", stop loss %.5f" +
               ", take profit %.5f" +
               ", deviation %d" +
               ", type %d" +
               ", type filling %d" +
               ", type time %d" +
               ", expiration %d" +
               ", comment %s" +
               ", position %d" +
               ", position by %d",
               request.action,
               request.magic,
               request.order,
               request.symbol,
               request.volume,
               request.price,
               request.stoplimit,
               request.sl,
               request.tp,
               request.deviation,
               request.type,
               request.type_filling,
               request.type_time,
               request.expiration,
               request.comment,
               request.position,
               request.position_by
           );
}


//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
// Helper function to provide error description
string result_description(const int retcode)
{
    switch(retcode) {
    case TRADE_RETCODE_REQUOTE:
        return "Requote error";
    case TRADE_RETCODE_TIMEOUT:
        return "Timeout error";
    case TRADE_RETCODE_CONNECTION:
        return "Connection error";
    case TRADE_RETCODE_INVALID:
        return "Invalid request";
    case TRADE_RETCODE_PLACED:
        return "Order placed successfully";
    case TRADE_RETCODE_DONE:
        return "Order done successfully";
    case TRADE_RETCODE_REJECT:
        return "Order rejected";
    case TRADE_RETCODE_CANCEL:
        return "Order canceled";
    case TRADE_RETCODE_PRICE_CHANGED:
        return "Price changed";
    case TRADE_RETCODE_PRICE_OFF:
        return "No quotes";
    case TRADE_RETCODE_INVALID_STOPS:
        return "Invalid stops";
    default:
        return "Unknown error";
    }
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool send_order_safe(const MqlTradeRequest &request)
{
    uint attempt = 0;
    MqlTradeResult result;
    while (attempt < C_max_retries) {
        // Reset the result structure before each attempt
        ZeroMemory(result);
        if (OrderSend(request, result)) { // && (result.retcode == TRADE_RETCODE_DONE || result.retcode == TRADE_RETCODE_PLACED)) {
            LOG_DEBUG("Send successful, ticket " + IntegerToString(result.order) + ", deal " + IntegerToString(result.deal) + ", attempt " + IntegerToString(attempt + 1));
            return true;
        }
        LOG_ERROR("Send attempt " + IntegerToString(attempt + 1) + " failed with " + IntegerToString(result.retcode) + ", " + result_description(result.retcode));
        if(result.retcode == TRADE_RETCODE_REQUOTE || result.retcode == TRADE_RETCODE_TIMEOUT || result.retcode == TRADE_RETCODE_CONNECTION) {
            ++attempt;
            Sleep(C_send_wait);
        } else break;
    }
    LOG_ERROR("Send failed after " + IntegerToString(attempt) + " attempts.");
    return false;
}


//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class RequestUtils
{

    JSONValue*       jv;

    void             Dispose()
    {
        if (CheckPointer(jv) == POINTER_DYNAMIC) delete jv;
    }
public:

                     RequestUtils() : jv(NULL) {}
                    ~RequestUtils()
    {
        Dispose();
    }

    bool             is_response_successful(const string &response);
    string           get_error_msg(const string &response);
    JSONObject       *get_result_obj(const string &response);
    JSONValue        *get_result_val(const string &response);
    JSONArray        *get_result_array(const string &response);
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
//+------------------------------------------------------------------+
