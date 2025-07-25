//+------------------------------------------------------------------+
//|                                                     TestIndi.mq5 |
//|                                        Copyright © 2018, Amr Ali |
//|                             https://www.mql5.com/en/users/amrali |
//+------------------------------------------------------------------+
#property copyright "Copyright © 2018, Amr Ali"
#property link      "https://www.mql5.com/en/users/amrali"
#property version "2.15"
#property description "project description."
#property indicator_chart_window
#property indicator_type1 DRAW_NONE

// Uncomment the following line to see debug messages
// #define PRINT_TZ_DETAILS

#include "SessionHours.mqh"
//+------------------------------------------------------------------+
//| input variables                                                  |
//+------------------------------------------------------------------+
input int    LocalOpenHourSydney     = 9;
input int    LocalOpenHourTokyo      = 8;
input int    LocalOpenHourFrankfurt  = 8;
input int    LocalOpenHourLondon     = 8;
input int    LocalOpenHourNewYork    = 8;
input int    LocalCloseHourSydney    = 17;
input int    LocalCloseHourTokyo     = 17;
input int    LocalCloseHourFrankfurt = 17;
input int    LocalCloseHourLondon    = 17;
input int    LocalCloseHourNewYork   = 17;
//---
input int    FontSize                = 8;
input int    Y_Dist                  = 30;    // Vertical shift
input int    X_Dist                  = 12;    // Horizontal shift
input bool   UseGoldSymbol           = true;  // Load XAUUSD symbol for estimation of the server's TZ/DST

//+------------------------------------------------------------------+
//| global variables                                                 |
//+------------------------------------------------------------------+
CSessionHours syd(ZONE_ID_SYDNEY);
CSessionHours tok(ZONE_ID_TOKYO);
CSessionHours frf(ZONE_ID_FRANKFURT);
CSessionHours lon(ZONE_ID_LONDON);
CSessionHours nyc(ZONE_ID_NEWYORK);
//+------------------------------------------------------------------+
//| Custom indicator initialization function                         |
//+------------------------------------------------------------------+
int OnInit()
  {
//--- change default session times
   syd.BeginLocalTime(LocalOpenHourSydney, 0);
   tok.BeginLocalTime(LocalOpenHourTokyo, 0);
   frf.BeginLocalTime(LocalOpenHourFrankfurt, 0);
   lon.BeginLocalTime(LocalOpenHourLondon, 0);
   nyc.BeginLocalTime(LocalOpenHourNewYork, 0);

   syd.EndLocalTime(LocalCloseHourSydney, 0);
   tok.EndLocalTime(LocalCloseHourTokyo, 0);
   frf.EndLocalTime(LocalCloseHourFrankfurt, 0);
   lon.EndLocalTime(LocalCloseHourLondon, 0);
   nyc.EndLocalTime(LocalCloseHourNewYork, 0);

//--- set the option of using Gold symbol for estimation of the server's TZ/DST
   CTimeZoneInfo::SetUsingGoldSymbol(UseGoldSymbol);
//---
   EventSetTimer(1);
   return (INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   ObjectsDeleteAll(0,"time_zone_label");
   EventKillTimer();
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void OnTimer()
  {
   syd.RefreshTime();
   tok.RefreshTime();
   frf.RefreshTime();
   lon.RefreshTime();
   nyc.RefreshTime();

   int sec = CSessionHours::SecRemainingForex();
   datetime forexCloseTime = CSessionHours::ForexCloseTime();

   string Info_00  = "Sydney    :  " + syd.ToString() + " " + (syd.CheckLocalSession() ? "open" : "closed");
   string Info_01  = "Tokyo     :  " + tok.ToString() + " " + (tok.CheckLocalSession() ? "open" : "closed");
   string Info_02  = "Frankfurt :  " + frf.ToString() + " " + (frf.CheckLocalSession() ? "open" : "closed");
   string Info_03  = "London    :  " + lon.ToString() + " " + (lon.CheckLocalSession() ? "open" : "closed");
   string Info_04  = "New York  :  " + nyc.ToString() + " " + (nyc.CheckLocalSession() ? "open" : "closed");
   string Info_06  = "GMT       :  " + CTimeZoneInfo::FormatTimeForPlace(TimeGMT(), ZONE_ID_UTC);
   string Info_07  = "Local     :  " + CTimeZoneInfo::FormatTimeForPlace(TimeLocal(), ZONE_ID_LOCAL);
   string Info_08  = "Broker    :  " + CTimeZoneInfo::FormatTimeForPlace(TimeTradeServer(), ZONE_ID_BROKER);
   string Info_09  = "Fx Close  :  " + CTimeZoneInfo::FormatTimeForPlace(forexCloseTime, ZONE_ID_BROKER);
   string Info_10  = "Closes in :  " + StringFormat("%i sec =  %s", sec, CSessionHours::SecondsToString(sec));

//-- create or update the labels
   int d = 12;
   LabelCreate("time_zone_label_00", X_Dist, Y_Dist + 0*d, Info_00, "Lucida Console", FontSize, syd.CheckLocalSession() ? clrWhiteSmoke : clrRed);
   LabelCreate("time_zone_label_01", X_Dist, Y_Dist + 1*d, Info_01, "Lucida Console", FontSize, tok.CheckLocalSession() ? clrWhiteSmoke : clrRed);
   LabelCreate("time_zone_label_02", X_Dist, Y_Dist + 2*d, Info_02, "Lucida Console", FontSize, frf.CheckLocalSession() ? clrWhiteSmoke : clrRed);
   LabelCreate("time_zone_label_03", X_Dist, Y_Dist + 3*d, Info_03, "Lucida Console", FontSize, lon.CheckLocalSession() ? clrWhiteSmoke : clrRed);
   LabelCreate("time_zone_label_04", X_Dist, Y_Dist + 4*d, Info_04, "Lucida Console", FontSize, nyc.CheckLocalSession() ? clrWhiteSmoke : clrRed);
   LabelCreate("time_zone_label_06", X_Dist, Y_Dist + 6*d, Info_06, "Lucida Console", FontSize, clrWhiteSmoke);
   LabelCreate("time_zone_label_07", X_Dist, Y_Dist + 7*d, Info_07, "Lucida Console", FontSize, clrWhiteSmoke);
   LabelCreate("time_zone_label_08", X_Dist, Y_Dist + 8*d, Info_08, "Lucida Console", FontSize, sec ? clrWhiteSmoke : clrRed);
   LabelCreate("time_zone_label_09", X_Dist, Y_Dist + 9*d, Info_09, "Lucida Console", FontSize, clrWhiteSmoke);
   LabelCreate("time_zone_label_10", X_Dist, Y_Dist + 10*d, Info_10, "Lucida Console", FontSize, clrWhiteSmoke);
   ChartRedraw();
  }
//+------------------------------------------------------------------+
//| Custom indicator iteration function                              |
//+------------------------------------------------------------------+
int OnCalculate(const int rates_total, const int prev_calculated,
                const int begin, const double &price[])
  {
   return(0);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void LabelCreate(string name, int x, int y, string text, string font="Tahoma", int fontsize=9, color clr=clrWhiteSmoke)
  {
   bool b=ObjectCreate(0,name,OBJ_LABEL,0,0,0);
   ObjectSetInteger(0,name,OBJPROP_BACK,false);
   ObjectSetInteger(0,name,OBJPROP_BORDER_TYPE,0);
   ObjectSetInteger(0,name,OBJPROP_COLOR,clr);
   ObjectSetInteger(0,name,OBJPROP_CORNER,0);
   ObjectSetInteger(0,name,OBJPROP_FONTSIZE,fontsize);
   ObjectSetInteger(0,name,OBJPROP_SELECTABLE,false);
   ObjectSetInteger(0,name,OBJPROP_XDISTANCE,x);
   ObjectSetInteger(0,name,OBJPROP_YDISTANCE,y);
   ObjectSetString(0,name,OBJPROP_FONT,font);
   ObjectSetString(0,name,OBJPROP_TEXT,text);
  }
//+------------------------------------------------------------------------------------------+
