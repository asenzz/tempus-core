//+------------------------------------------------------------------+
//|                                                  JSONExample.mq4 |
//|                        Copyright 2015, MetaQuotes Software Corp. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2015, MetaQuotes Software Corp."
#property link      "https://www.mql5.com"
#property version   "1.00"
#property strict

#include <hash.mqh>
#include <json.mqh>

//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart() {
    string s = "{ \"firstName\": \"John\", \"lastName\": \"Smith\", \"age\": 25, " +
            "\"address\": { \"streetAddress\": \"21 2nd Street\", \"city\": \"New York\", \"state\": \"NY\", \"postalCode\": \"10021\" }," +
            " \"phoneNumber\": [ { \"type\": \"home\", \"number\": \"212 555-1234\" }, { \"type\": \"fax\", \"number\": \"646 555-4567\" } ]," +
            " \"gender\":{ \"type\":\"male\" }  }";

    JSONParser *parser = new JSONParser();

    JSONValue *jv = parser.parse(s);

    if (jv == NULL) {
        Print("error:" + (string) parser.getErrorCode() + parser.get_error_msg());
    } else {


        Print("PARSED:" + jv.toString());

        if (jv.isObject()) { // check root value is an object. (it can be an array)

            JSONObject *jo = jv;

            // Direct access - will throw null pointer if wrong getter used.
            Print("firstName:" + jo.getString("firstName"));
            Print("city:" + jo.getObject("address").getString("city"));
            Print("phone:" + jo.getArray("phoneNumber").getObject(0).getString("number"));

            // Safe access in case JSON data is missing or different.
            if (jo.getString("firstName", s)) Print("firstName = " + s);

            // Loop over object keys
            JSONIterator *it = new JSONIterator(jo);
            for (; it.hasNext(); it.next()) {
                Print("loop:" + it.key() + " = " + it.val().toString());
            }
            delete it;
        }
        delete jv;
    }
    delete parser;

}
//+------------------------------------------------------------------+
