//+------------------------------------------------------------------+
//|                                                   MqlNetTest.mq4 |
//|                        Copyright 2015, MetaQuotes Software Corp. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2015, MetaQuotes Software Corp."
#property link      "https://www.mql5.com"
#property version   "1.00"
#property strict

#include <hash.mqh>
#include <MqlNet.mqh>
#include <json.mqh>
#include <SVRApi.mqh>

input string Username = "mql4_test";
input string Password = "svrwave";
input string Host = "172.16.78.1:8080";
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+

SVRApi svrApi;

int OnStart() {

    TestParseUrl();

    if (!svrApi.Connect(Host)) {
        Print("Cannot connect to Server!");
        return (0);
    }

    if (!svrApi.Login(Username, Password)) {
        Print("Error while Logging in");
        return (0);
    }

    
    //svrApi.SendBar();    
    //datetime from, to;
    //svrApi.GetNextTimeRangeToBeSent(from, to);    
    svrApi.SendHistory();

    //Print(__FUNCTION__ + " From: " + TimeToString(from));
    //Print(__FUNCTION__ + " To: " + TimeToString(to));

    return (0);
}


//+------------------------------------------------------------------+
//|                                     |
//+------------------------------------------------------------------+

bool TestParseUrl()
{
   string url = "aaabbb";

   ParsedUrl purl;

   if (!ParseUrl(url, purl))
      return false;

   if(purl.Host != "aaabbb" || purl.Port != 80 || purl.isSecure != false)
   {
      PrintFormat("Assertion failed: File: %s Line: %i", __FILE__, __LINE__);
      return false;
   }

   url = "www.aaa.bbb-cc:934782/www/page.jsp&param=&dns";
   if (!ParseUrl(url, purl))
      return false;

   if(purl.Host != "www.aaa.bbb-cc" || purl.Port != 934782 || purl.isSecure != false)
   {
      PrintFormat("Assertion failed: File: %s Line: %i", __FILE__, __LINE__);
      return false;
   }

   url = "https://www.aaa.bbb-cc:934782/www/page.jsp&param=&dns";
   if (!ParseUrl(url, purl))
      return false;

   if(purl.Host != "www.aaa.bbb-cc" || purl.Port != 934782 || purl.isSecure != true)
   {
      PrintFormat("Assertion failed: File: %s Line: %i", __FILE__, __LINE__);
      return false;
   }

   url = "https://0.0.0.0";
   if (!ParseUrl(url, purl))
      return false;

   if(purl.Host != "0.0.0.0" || purl.Port != 443 || purl.isSecure != true)
   {
      PrintFormat("Assertion failed: File: %s Line: %i", __FILE__, __LINE__);
      return false;
   }

   return true;
}
