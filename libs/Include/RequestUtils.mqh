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

class RequestUtils{

   JSONValue* jv;

   void Dispose() { if (CheckPointer(jv) == POINTER_DYNAMIC) delete jv; }
public:

   RequestUtils() : jv(NULL){}
   ~RequestUtils() { Dispose(); }
   
   bool isSuccessfulResponse(string response);
   string getErrorMessage(string response);
   JSONObject *getResultObject(string response);
   JSONValue *getResultValue(string response);
   JSONArray *getResultArray(string response);  
};

bool RequestUtils::isSuccessfulResponse(string response){

   if(StringLen(response) == 0 ){
      return(false);
   }
   
   JSONParser parser;
   bool isSuccessful = false;

   Dispose();
   jv = parser.parse(response);

   if (jv != NULL && jv.isObject()) {
      JSONObject *jo = jv;
      if (!jo.getValue("error").isNull()) {
         isSuccessful = false;
      } else {
         isSuccessful = true;
      }
   }
      
   return(isSuccessful);
}

string RequestUtils::getErrorMessage(string response){
   if(isSuccessfulResponse(response)){
      return("");
   }
   string errorMessage;
   if (StringLen(response) > 0) {        
        JSONParser parser;

        Dispose();
        jv = parser.parse(response);

        if (jv != NULL && jv.isObject()) {
            JSONObject *jo = jv;
            if (!jo.getValue("error").isNull()) {
                errorMessage = jo.getString("error");
            } else{
               Print("No error message");
            }
        } else{
            LOG_ERROR("RequestUtils::getErrorMessage", "Cannot read error message from response: " + response);
        }               
    }
    return(errorMessage);
}

JSONObject* RequestUtils::getResultObject(string response){
   
   JSONObject *result = NULL;
   if(StringLen(response) == 0){
      return(result);
   }
   
   JSONParser parser;   

   Dispose();
   jv = parser.parse(response);

   if (jv != NULL && jv.isObject()) {
      JSONObject *jo = jv;
      if (!jo.getObject("result").isNull()) {
          result = jo.getObject("result");
      } else{
         Print("No result object");
      }
   } else{
      Print("Cannot read result object from response: " + response);
   }    
      
   return (result);
}

JSONValue* RequestUtils::getResultValue(string response){
   
   JSONValue *result = NULL;
   if(StringLen(response) == 0){
      return(result);
   }
   
   JSONParser parser;
   
   Dispose();
   jv = parser.parse(response);

   if (jv != NULL && jv.isObject()) {
      JSONObject *jo = jv;
      if (!jo.getValue("result").isNull()) {
          result = jo.getValue("result");
      } else{
         Print("No result value");
      }
   } else{
      Print("Cannot read result value from response: " + response);
   }  
      
   return (result);
}

JSONArray* RequestUtils::getResultArray(string response){
   
   JSONArray *result = NULL;
   if(StringLen(response) == 0){
      return(result);
   }
   
   JSONParser parser;
   Dispose();
   jv = parser.parse(response);

   if (jv != NULL && jv.isObject()) {
      JSONObject *jo = jv;
      if (jo.isArray() && !jo.getArray("result").isNull()) {
          result = jo.getArray("result");
      } else{
         Print("No result array");
      }
   } else{
      Print("Cannot read result array from response: " + response);
   }
        
   return (result);
}