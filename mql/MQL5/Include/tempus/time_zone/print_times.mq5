// Uncomment the following line to see debug messages
// #define PRINT_TZ_DETAILS

#include "TimeZoneInfo.mqh"

void OnStart()
  {
   datetime tts = TimeTradeServer();
   datetime loc = TimeLocal();
   datetime gmt = TimeGMT();
   int ServerOffset = CTimeZoneInfo::TimeGMTOffset(ZONE_ID_BROKER);
   int ServerDST    = CTimeZoneInfo::TimeDaylightSavings(ZONE_ID_BROKER);

   PrintFormat("Server            = %s", AccountInfoString(ACCOUNT_SERVER));
   PrintFormat("TimeTradeServer() = %s", string(tts));
   PrintFormat("TimeLocal()       = %s", string(loc));
   PrintFormat("TimeGMT()         = %s", string(gmt));
   PrintFormat("ServerOffset      = %d (GMT%+d)", ServerOffset, ServerOffset/3600);
   PrintFormat("ServerDST         = %d (%s)",     ServerDST, ServerDST ? "DST" : "STD");
   PrintFormat("FormatTimeForPlace(TimeTradeServer()) = %s", CTimeZoneInfo::FormatTimeForPlace(tts,ZONE_ID_BROKER));
   PrintFormat("FormatTimeForPlace(TimeLocal())       = %s", CTimeZoneInfo::FormatTimeForPlace(loc,ZONE_ID_LOCAL));
   PrintFormat("FormatTimeForPlace(TimeGMT())         = %s", CTimeZoneInfo::FormatTimeForPlace(gmt,ZONE_ID_UTC));
  }
//+------------------------------------------------------------------+

/*
 Server            = ICMarketsSC-Demo
 TimeTradeServer() = 2024.11.13 01:35:43
 TimeLocal()       = 2024.11.13 02:35:43
 TimeGMT()         = 2024.11.12 23:35:43
 ServerOffset      = 7200 (GMT+2)
 ServerDST         = 0 (STD)
 FormatTimeForPlace(TimeTradeServer()) = Wed, 2024.11.13 01:35:43 GMT+2 std [ICMarketsSC-Demo]
 FormatTimeForPlace(TimeLocal())       = Wed, 2024.11.13 02:35:43 GMT+3 std [Local-PC]
 FormatTimeForPlace(TimeGMT())         = Tue, 2024.11.12 23:35:43 GMT+0 std [UTC]

*/
