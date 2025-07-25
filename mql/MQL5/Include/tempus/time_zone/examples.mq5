//+------------------------------------------------------------------+
//|                                                     Examples.mq5 |
//|                                      Copyright 2020, CompanyName |
//|                                       http://www.companyname.net |
//+------------------------------------------------------------------+
#property copyright "Copyright © 2018, Amr Ali"
#property link      "https://www.mql5.com/en/users/amrali"
#property version "2.15"
#property description "various examples to show how to use TimeZoneInfo and SessionHours libraries in your project."

// Uncomment the following line to see debug messages
// #define PRINT_TZ_DETAILS

#include "SessionHours.mqh"

#define MAX_ZONE_ID  (ZONE_ID_CUSTOM)

/*
//+------------------------------------------------------------------+
//| Class CTimeZoneInfo.                                             |
//| Purpose: Class to access to the local time for the specified     |
//|          location, as well as time zone information, time        |
//|          changes for the current year.                           |
//|                                                                  |
//| Offset notation used in the library:                             |
//|          Please note, that the library denotes positive time     |
//|          zones by positive offsets, and negative time zones      |
//|          by negative offsets.                                    |
//|          On the contrary, MQL5's built-in TimeGMTOffset()        |
//|          function denotes positive time zones, such as GMT+3,    |
//|          by negative offsets, such as -10800, and vice versa.    |
//+------------------------------------------------------------------+
class CTimeZoneInfo
  {
public:
                     CTimeZoneInfo( ENUM_ZONE_ID placeId, datetime pLocalTime = TIME_NOW );
                    ~CTimeZoneInfo( void );

   string            Name( void );                                                       // Returns the name of time zone
   string            ToString( bool secs = true, bool tzname = true );                   // Returns a string of local time formatted with TZ/DST offset and tzname
   bool              RefreshTime( void );                                                // Refresh the current local time and populate timezone information
   bool              SetLocalTime( datetime pLocalTime = TIME_NOW );                     // Set the local time for this location to the specified time

   datetime          TimeLocal( void );                                                  // Returns the local time in timezone
   datetime          TimeUTC( void );                                                    // Returns the UTC time (the same in all time zones)
   int               TimeGMTOffset( void );                                              // Positive value for positive timezones (eg, GMT+3), otherwise negative. (includes DST)
   int               TimeDaylightSavings( void );                                        // Returns DST correction (in seconds) for timezone, at the set local time.

   datetime          ConvertLocalTime( ENUM_ZONE_ID destinationId );                     // Convert local time in this time zone to a different time zone
   bool              GetDaylightSwitchTimes( datetime &dst_start, datetime &dst_end );   // Get the Daylight Saving Time start and end times for the year
   datetime          GetDaylightNextSwitch( void );                                      // Get the local time of the next Daylight Saving Time switch
   void              PrintObject( void );

   //--- static methods that do not require the creation of an object.
   static datetime   GetCurrentTimeForPlace  ( ENUM_ZONE_ID placeId );                    // Get the current local time for the specified time zone
   static string     FormatTimeForPlace      ( datetime time, ENUM_ZONE_ID placeId, bool secs = true, bool tzname = true );
   static datetime   ConvertTimeForPlace     ( datetime time, ENUM_ZONE_ID placeId, ENUM_ZONE_ID destinationId );
   static int        TimeGMTOffset           ( ENUM_ZONE_ID placeId, datetime time = TIME_NOW );   // Returns total tz offset (UTC+DST) from GMT, for a timezone at given local time
   static int        TimeDaylightSavings     ( ENUM_ZONE_ID placeId, datetime time = TIME_NOW );   // Returns dst correction in seconds, for a timezone at given local time
   static bool       IsDaylightSavingsTime   ( ENUM_ZONE_ID placeId, datetime time = TIME_NOW );   // Checks if a specified time falls in the daylight saving time
   static bool       DaylightSavingsSupported( ENUM_ZONE_ID placeId);                              // Checks if the given timezone supports the Daylight Savings Time policy
   static bool       GetDaylightSwitchTimes  ( ENUM_ZONE_ID placeId, int iYear, datetime &dst_start, datetime &dst_end );
   static bool       GetDaylightSwitchDeltas ( ENUM_ZONE_ID placeId, int iYear, int &delta_start, int &delta_end );

   static bool       SetCustomTimeZone( string name, int baseGMTOffset = 0,              // Defines a time zone that is not found in the library.
                                        ENUM_ZONE_ID dstSchedule = ZONE_ID_UTC );
   static void       SetUsingGoldSymbol( bool enabled = true );                          // Sets the option to use Gold symbol for estimation of server TZ/DST
  };

//+------------------------------------------------------------------+
//| Class CSessionHours.                                             |
//| Purpose: Class to access the local trading session hours for     |
//|          the specified location.                                 |
//|          Derives from class CTimeZoneInfo.                       |
//| Note:    The default session hours are set to 8:00 am - 5:00 pm  |
//|          local time for new CSessionHours objects.               |
//+------------------------------------------------------------------+
class CSessionHours : public CTimeZoneInfo
  {
public:
                     CSessionHours( ENUM_ZONE_ID placeId );
                    ~CSessionHours( void );

   //--- methods to access the local session time
   bool              RefreshTime( void );                            // Refresh the current local time and session hours for the day
   bool              SetLocalTime( datetime pLocalTime );

   //--- methods to override default local session hours
   bool              BeginLocalTime( int pHour, int pMinute );       // Set the local session begin time using a begin hour and minute
   bool              EndLocalTime( int pHour, int pMinute );         // Set the local session end time using an end hour and minute

   //--- methods to access local session hours
   datetime          BeginLocalTime( void );
   datetime          EndLocalTime( void );
   bool              CheckLocalSession( void );                      // Check whether the local trading session is currently active.
   int               SecRemainingSession( void );                    // Time remaining in seconds till local session closes for the day.

   //--- static methods that do not require the creation of an object.
   static datetime   ForexCloseTime( void );                         // The broker time when the Forex market closes for this week.
   static int        SecRemainingForex( void );                      // Time remaining in seconds till Forex market closes for this week.
   static string     SecondsToString( long seconds );                // Format the time in seconds to a string (like 2d 14:58:04)
  };
*/

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void OnStart()
  {
//--- I. Working with Local Timezones

//--- How to get the current time?
     {
      Print("\n========== Get the current time in a timezone ==========");

      CTimeZoneInfo tz(ZONE_ID_TOKYO);
      tz.RefreshTime(); // populate current timezone information

      Print("Name()      : ", tz.Name());
      Print("TimeLocal() : ", tz.TimeLocal());
      Print("ToString()  : ", tz.ToString());
      //
      // ========== Get the current time in a timezone ==========
      // Name()      : Tokyo
      // TimeLocal() : 2024.02.28 19:58:50
      // ToString()  : Wed, 2024.02.28 19:58:50 GMT+9 STD [Tokyo]
     }


//--- Do you need more information?
     {
      Print("\n========== More information about a timezone ==========");

      CTimeZoneInfo tz(ZONE_ID_NEWYORK);
      tz.RefreshTime();

      Print("Name()                  : ", tz.Name());
      Print("TimeUTC()               : ", tz.TimeUTC());
      Print("TimeLocal()             : ", tz.TimeLocal());
      Print("TimeGMTOffset()         : ", tz.TimeGMTOffset());
      Print("TimeDaylightSavings()   : ", tz.TimeDaylightSavings());
      Print("ToString()              : ", tz.ToString());
      datetime dst_start, dst_end;
      tz.GetDaylightSwitchTimes(dst_start, dst_end);
      Print("dst_start               : ", dst_start);
      Print("dst_end                 : ", dst_end);
      Print("GetDaylightNextSwitch() : ", tz.GetDaylightNextSwitch());
      //
      // ========== More information about a timezone ==========
      // Name()                  : New York
      // TimeUTC()               : 2024.03.17 16:50:38
      // TimeLocal()             : 2024.03.17 12:50:38
      // TimeGMTOffset()         : -14400
      // TimeDaylightSavings()   : 3600
      // ToString()              : Sun, 2024.03.17 12:50:38 GMT-4 DST [New York]
      // dst_start               : 2024.03.10 02:00:00
      // dst_end                 : 2024.11.03 02:00:00
      // GetDaylightNextSwitch() : 2024.11.03 02:00:00
     }


//--- How to configure the built-in custom timezone for later use?
     {
      Print("\n========== Configure the built-in custom timezone ==========");

      string        name  = "Custom+3";               // Custom Timezone's name
      int           baseGMTOffset  = 10800;           // Custom Timezone's base GMT offset (in seconds)
      ENUM_ZONE_ID  daylightRuleId = ZONE_ID_LONDON;  // Custom Timezone's DST schedule

      bool success = CTimeZoneInfo::SetCustomTimeZone(name, baseGMTOffset, daylightRuleId);

      Print("Parameter 'name'            : ", name);
      Print("Parameter 'baseGMTOffset'   : ", baseGMTOffset);
      Print("Parameter 'daylightRuleId'  : ", EnumToString(daylightRuleId));
      Print("SetCustomTimeZone() returns : ", success);
      //
      // ========== Configure the built-in custom timezone ==========
      // Parameter 'name'            : Custom+3
      // Parameter 'baseGMTOffset'   : 10800
      // Parameter 'daylightRuleId'  : ZONE_ID_LONDON
      // SetCustomTimeZone() returns : true
     }


//--- Get Current Time in All Timezones
     {
      Print("\n========== Get Current Time in All Timezones ==========");

      for(ENUM_ZONE_ID id=0; id <= MAX_ZONE_ID; id++)
        {
         CTimeZoneInfo tz(id);
         tz.RefreshTime();

         PrintFormat("%-12s:  %s | %s", tz.Name(), TimeToString(tz.TimeLocal()), tz.ToString());
        }
      //
      // ========== Get Current Time in All Timezones ==========
      // Sydney      :  2024.02.28 21:58 | Wed, 2024.02.28 21:58:50 GMT+11 DST [Sydney]
      // Tokyo       :  2024.02.28 19:58 | Wed, 2024.02.28 19:58:50 GMT+9 STD [Tokyo]
      // Frankfurt   :  2024.02.28 11:58 | Wed, 2024.02.28 11:58:50 GMT+1 STD [Frankfurt]
      // London      :  2024.02.28 10:58 | Wed, 2024.02.28 10:58:50 GMT+0 STD [London]
      // New York    :  2024.02.28 05:58 | Wed, 2024.02.28 05:58:50 GMT-5 STD [New York]
      // UTC         :  2024.02.28 10:58 | Wed, 2024.02.28 10:58:50 GMT+0 STD [UTC]
      // Local-PC    :  2024.02.28 12:58 | Wed, 2024.02.28 12:58:50 GMT+2 STD [Local-PC]
      // FXOpen-MT5  :  2024.02.28 12:58 | Wed, 2024.02.28 12:58:50 GMT+2 STD [FXOpen-MT5]
     }


//--- Get the current time for all the time zones.
     {
      Print("\n========== GetCurrentTimeForPlace() ==========");

      for(ENUM_ZONE_ID id=0; id <= MAX_ZONE_ID; id++)
        {
         datetime time = CTimeZoneInfo::GetCurrentTimeForPlace(id);

         PrintFormat("Time        :  %s | %s", TimeToString(time), CTimeZoneInfo::FormatTimeForPlace(time, id));
        }
      //
      // ========== GetCurrentTimeForPlace() ==========
      // Time        :  2024.02.28 21:58 | Wed, 2024.02.28 21:58:50 GMT+11 DST [Sydney]
      // Time        :  2024.02.28 19:58 | Wed, 2024.02.28 19:58:50 GMT+9 STD [Tokyo]
      // Time        :  2024.02.28 11:58 | Wed, 2024.02.28 11:58:50 GMT+1 STD [Frankfurt]
      // Time        :  2024.02.28 10:58 | Wed, 2024.02.28 10:58:50 GMT+0 STD [London]
      // Time        :  2024.02.28 05:58 | Wed, 2024.02.28 05:58:50 GMT-5 STD [New York]
      // Time        :  2024.02.28 10:58 | Wed, 2024.02.28 10:58:50 GMT+0 STD [UTC]
      // Time        :  2024.02.28 12:58 | Wed, 2024.02.28 12:58:50 GMT+2 STD [Local-PC]
      // Time        :  2024.02.28 12:58 | Wed, 2024.02.28 12:58:50 GMT+2 STD [FXOpen-MT5]
     }


//--- How to set the local time for a timezone?
     {
      Print("\n========== Set the local time for a timezone ==========");

      CTimeZoneInfo tz(ZONE_ID_NEWYORK);

      if(tz.SetLocalTime(D'2021.07.15 09:31'))
         PrintFormat("%-12s:  %s | %s", tz.Name(), TimeToString(tz.TimeLocal()), tz.ToString());

      if(tz.SetLocalTime(D'2022.01.23 17:04'))
         PrintFormat("%-12s:  %s | %s", tz.Name(), TimeToString(tz.TimeLocal()), tz.ToString());

      if(tz.SetLocalTime(D'2023.03.12 02:21'))
         PrintFormat("%-12s:  %s | %s", tz.Name(), TimeToString(tz.TimeLocal()), tz.ToString());
      //
      // ========== Set the local time for a timezone ==========
      // New York    :  2021.07.15 09:31 | Thu, 2021.07.15 09:31:00 GMT-4 DST [New York]
      // New York    :  2022.01.23 17:04 | Sun, 2022.01.23 17:04:00 GMT-5 [New York]
      // >>Error: The time 2023.03.12 02:21 does not exist in New York. This is because Daylight Saving Time skipped one hour.
     }


//--- II. Get Timezone Information

//--- 1. List all the time zones, UTC offset and current DST offset
     {
      Print("\n========== UTC offset and current DST offset ==========");

      for(ENUM_ZONE_ID id=0; id <= MAX_ZONE_ID; id++)
        {
         CTimeZoneInfo tz(id);
         tz.RefreshTime(); // populate current timezone information

         PrintFormat("%-12s: GMT%+g | DST%+g", tz.Name(), tz.TimeGMTOffset()/3600., tz.TimeDaylightSavings()/3600.);
        }
      //
      // ========== UTC offset and current DST offset ==========
      // Sydney      : GMT+11 | DST+1
      // Tokyo       : GMT+9 | DST+0
      // Frankfurt   : GMT+1 | DST+0
      // London      : GMT+0 | DST+0
      // New York    : GMT-4 | DST+1
      // UTC         : GMT+0 | DST+0
      // Local-PC    : GMT+2 | DST+0
      // FXOpen-MT5  : GMT+2 | DST+0
     }


// 2. DST switch times for the current year
     {
      Print("\n========== DST switch times for the current year ==========");

      datetime dst_start, dst_end;

      for(ENUM_ZONE_ID id=0; id <= MAX_ZONE_ID; id++)
        {
         CTimeZoneInfo tz(id);
         tz.RefreshTime(); // populate current timezone information

         //--- only for time zones that observe daylight time.
         if(tz.GetDaylightSwitchTimes(dst_start, dst_end))
           {
            PrintFormat("%-12s:  DST starts on %s |  DST ends on %s", tz.Name(), TimeToString(dst_start), TimeToString(dst_end));
           }
        }
      //
      // ========== DST switch times for the current year ==========
      // Sydney      :  DST starts on 2024.04.07 03:00 |  DST ends on 2024.10.06 02:00
      // Frankfurt   :  DST starts on 2024.03.31 02:00 |  DST ends on 2024.10.27 03:00
      // London      :  DST starts on 2024.03.31 01:00 |  DST ends on 2024.10.27 02:00
      // New York    :  DST starts on 2024.03.10 02:00 |  DST ends on 2024.11.03 02:00
     }


// 3. Time of the next DST switch
     {
      Print("\n========== Time of the next DST switch ==========");

      for(ENUM_ZONE_ID id=0; id <= MAX_ZONE_ID; id++)
        {
         CTimeZoneInfo tz(id);
         tz.RefreshTime();

         datetime nxswitch = tz.GetDaylightNextSwitch();

         //--- only for time zones that observe daylight time.
         if(nxswitch)
           {
            PrintFormat("%-12s:  Time: %s |  dstNextSwitch: %s", tz.Name(), TimeToString(tz.TimeLocal()), TimeToString(nxswitch));
           }
        }
      //
      // ========== Time of the next DST switch ==========
      // Sydney      :  Time: 2024.02.28 21:58 |  dstNextSwitch: 2024.04.07 03:00
      // Frankfurt   :  Time: 2024.02.28 11:58 |  dstNextSwitch: 2024.03.31 02:00
      // London      :  Time: 2024.02.28 10:58 |  dstNextSwitch: 2024.03.31 01:00
      // New York    :  Time: 2024.02.28 05:58 |  dstNextSwitch: 2024.03.10 02:00
     }


//--- 4. DST List
     {
      Print("\n========== DST List ==========");

      datetime dst_start, dst_end;
      int delta_start, delta_end;  // clock changes in sec

      for(ENUM_ZONE_ID id=0; id <= MAX_ZONE_ID; id++)
        {
         CTimeZoneInfo timezone(id);

         PrintFormat("========= %s Summer Time (DST) =========", timezone.Name());
         for(int year=2008; year<=2030; year++)
           {
            //--- only for time zones that observe daylight time.
            if(CTimeZoneInfo::GetDaylightSwitchTimes(id, year, dst_start, dst_end))
              {
               CTimeZoneInfo::GetDaylightSwitchDeltas(id, year, delta_start, delta_end);

               PrintFormat("DST starts on %s (%+d) and ends on %s (%+d)",TimeToString(dst_start), delta_start/3600, TimeToString(dst_end), delta_end/3600);
              }
           }

        }
      //
      // ========== DST List ==========
      // ========= Sydney Summer Time (DST) =========
      // DST starts on 2008.04.06 03:00 (-1) and ends on 2008.10.05 02:00 (+1)
      // DST starts on 2009.04.05 03:00 (-1) and ends on 2009.10.04 02:00 (+1)
      // DST starts on 2010.04.04 03:00 (-1) and ends on 2010.10.03 02:00 (+1)
      // DST starts on 2011.04.03 03:00 (-1) and ends on 2011.10.02 02:00 (+1)
      // DST starts on 2012.04.01 03:00 (-1) and ends on 2012.10.07 02:00 (+1)
      // ...
      // ...
      // ...
     }


//--- 5. Broker's GMT Offset
     {
      Print("\n========== Current GMT offset of the broker ========== ");

      CTimeZoneInfo broker(ZONE_ID_BROKER);
      broker.RefreshTime();

      Print("Name()                : ", broker.Name());
      Print("TimeLocal()           : ", broker.TimeLocal());  // broker time
      Print("ToString()            : ", broker.ToString());
      Print("TimeGMTOffset()       : ", broker.TimeGMTOffset());
      Print("TimeDaylightSavings() : ", broker.TimeDaylightSavings());
      //
      // ========== Current GMT offset of the broker ==========
      // Name()                : ICMarketsSC-Demo
      // TimeLocal()           : 2024.03.08 06:33:06
      // ToString()            : Fri, 2024.03.08 06:33:06 GMT+2 STD [ICMarketsSC-Demo]
      // TimeGMTOffset()       : 7200
      // TimeDaylightSavings() : 0
     }

//--- 6. GMT Offset of Chart Candles
     {
      Print("\n========== Past GMT offsets of the broker (chart candles) ==========");

      datetime bartimes[];
      int copied = CopyTime(Symbol(), PERIOD_D1, D'2022.03.18', 9, bartimes);
      if(copied<=0)
         Print("CopyTime() failed.");

      for(int i =0; i < copied; i++)
        {
         datetime t = bartimes[i];

         CTimeZoneInfo broker(ZONE_ID_BROKER);
         broker.SetLocalTime(t);

         PrintFormat("bar #%i  Time: %s |  offset: %5d (GMT%+g) |  dst: %4d |  %s",
                     i+1,
                     TimeToString(broker.TimeLocal()),
                     broker.TimeGMTOffset(),
                     broker.TimeGMTOffset()/3600.0,
                     broker.TimeDaylightSavings(),
                     broker.ToString());
        }
      //
      // ========== Past GMT offsets of the broker (chart candles) ==========
      // bar #1  Time: 2022.03.08 00:00 |  offset:  7200 (GMT+2) |  dst:    0 |  Tue, 2022.03.08 00:00:00 GMT+2 STD [ICMarketsSC-Demo]
      // bar #2  Time: 2022.03.09 00:00 |  offset:  7200 (GMT+2) |  dst:    0 |  Wed, 2022.03.09 00:00:00 GMT+2 STD [ICMarketsSC-Demo]
      // bar #3  Time: 2022.03.10 00:00 |  offset:  7200 (GMT+2) |  dst:    0 |  Thu, 2022.03.10 00:00:00 GMT+2 STD [ICMarketsSC-Demo]
      // bar #4  Time: 2022.03.11 00:00 |  offset:  7200 (GMT+2) |  dst:    0 |  Fri, 2022.03.11 00:00:00 GMT+2 STD [ICMarketsSC-Demo]
      // bar #5  Time: 2022.03.14 00:00 |  offset: 10800 (GMT+3) |  dst: 3600 |  Mon, 2022.03.14 00:00:00 GMT+3 DST [ICMarketsSC-Demo]
      // bar #6  Time: 2022.03.15 00:00 |  offset: 10800 (GMT+3) |  dst: 3600 |  Tue, 2022.03.15 00:00:00 GMT+3 DST [ICMarketsSC-Demo]
      // bar #7  Time: 2022.03.16 00:00 |  offset: 10800 (GMT+3) |  dst: 3600 |  Wed, 2022.03.16 00:00:00 GMT+3 DST [ICMarketsSC-Demo]
      // bar #8  Time: 2022.03.17 00:00 |  offset: 10800 (GMT+3) |  dst: 3600 |  Thu, 2022.03.17 00:00:00 GMT+3 DST [ICMarketsSC-Demo]
      // bar #9  Time: 2022.03.18 00:00 |  offset: 10800 (GMT+3) |  dst: 3600 |  Fri, 2022.03.18 00:00:00 GMT+3 DST [ICMarketsSC-Demo]
     }


//--- III. Converting Between Timezones

//--- Convert current local time to another timezone
     {
      Print("\n========== Convert current local time in Sydney to New York ==========");

      CTimeZoneInfo sydney(ZONE_ID_SYDNEY);
      sydney.RefreshTime();

      datetime localtime = sydney.TimeLocal();
      datetime converted = sydney.ConvertLocalTime(ZONE_ID_NEWYORK);

      PrintFormat("%s | %s", TimeToString(localtime), sydney.ToString());
      PrintFormat("%s | %s", TimeToString(converted), CTimeZoneInfo::FormatTimeForPlace(converted, ZONE_ID_NEWYORK));
      //
      // ========== Convert current local time in Sydney to New York ==========
      // 2024.02.28 21:58 | Wed, 2024.02.28 21:58:50 GMT+11 DST [Sydney]
      // 2024.02.28 05:58 | Wed, 2024.02.28 05:58:50 GMT-5 STD [New York]
     }


//--- Convert a specific local time to another timezone
     {
      Print("\n========== Convert a specific local time in Sydney to New York ==========");

      CTimeZoneInfo sydney(ZONE_ID_SYDNEY);
      sydney.SetLocalTime(D'2016.05.21 14:47:08');

      datetime localtime = sydney.TimeLocal();
      datetime converted = sydney.ConvertLocalTime(ZONE_ID_NEWYORK);

      PrintFormat("%s | %s", TimeToString(localtime), sydney.ToString());
      PrintFormat("%s | %s", TimeToString(converted), CTimeZoneInfo::FormatTimeForPlace(converted, ZONE_ID_NEWYORK));
      //
      // ========== Convert a specific local time in Sydney to New York ==========
      // 2016.05.21 14:47 | Sat, 2016.05.21 14:47:08 GMT+10 STD [Sydney]
      // 2016.05.21 00:47 | Sat, 2016.05.21 00:47:08 GMT-4 DST [New York]
     }


//--- Convert the current local time in all timezones to the broker time
     {
      Print("\n========== Convert the current local time in all timezones to the broker time ==========");

      for(ENUM_ZONE_ID id=0; id <= MAX_ZONE_ID; id++)
        {
         datetime localtime = CTimeZoneInfo::GetCurrentTimeForPlace(id);
         datetime converted = CTimeZoneInfo::ConvertTimeForPlace(localtime, id, ZONE_ID_BROKER);

         PrintFormat("%-49s | %s", CTimeZoneInfo::FormatTimeForPlace(localtime, id), CTimeZoneInfo::FormatTimeForPlace(converted, ZONE_ID_BROKER));
        }
      //
      // ========== Convert the current local time in all timezones to the broker time ==========
      // Wed, 2024.02.28 21:58:50 GMT+11 DST [Sydney]      | Wed, 2024.02.28 12:58:50 GMT+2 STD [FXOpen-MT5]
      // Wed, 2024.02.28 19:58:50 GMT+9 STD [Tokyo]            | Wed, 2024.02.28 12:58:50 GMT+2 STD [FXOpen-MT5]
      // Wed, 2024.02.28 11:58:50 GMT+1 STD [Frankfurt]        | Wed, 2024.02.28 12:58:50 GMT+2 STD [FXOpen-MT5]
      // Wed, 2024.02.28 10:58:50 GMT+0 STD [London]           | Wed, 2024.02.28 12:58:50 GMT+2 STD [FXOpen-MT5]
      // Wed, 2024.02.28 05:58:50 GMT-5 STD [New York]         | Wed, 2024.02.28 12:58:50 GMT+2 STD [FXOpen-MT5]
      // Wed, 2024.02.28 10:58:50 GMT+0 STD [UTC]              | Wed, 2024.02.28 12:58:50 GMT+2 STD [FXOpen-MT5]
      // Wed, 2024.02.28 12:58:50 GMT+2 STD [Local-PC]         | Wed, 2024.02.28 12:58:50 GMT+2 STD [FXOpen-MT5]
      // Wed, 2024.02.28 12:58:50 GMT+2 STD [FXOpen-MT5]       | Wed, 2024.02.28 12:58:50 GMT+2 STD [FXOpen-MT5]
     }


//--- IV. Working with Local Session Hours

//--- A. CTimeZoneInfo Class
     {
      Print("\n======= Local Session Hours (CTimeZoneInfo Class) =======");

      const ENUM_ZONE_ID ids[] = {ZONE_ID_SYDNEY, ZONE_ID_TOKYO, ZONE_ID_FRANKFURT, ZONE_ID_LONDON, ZONE_ID_NEWYORK};

      for(int i = 0; i < ArraySize(ids); i++)
        {
         ENUM_ZONE_ID id = ids[i];

         CTimeZoneInfo tz(id);
         tz.RefreshTime();

         datetime localtime = tz.TimeLocal();

         //--- set session hours to 8:00 am - 5:00 pm local time
         datetime beginlocal = StringToTime(TimeToString(localtime, TIME_DATE) + " " + "08:00");
         datetime endlocal   = StringToTime(TimeToString(localtime, TIME_DATE) + " " + "17:00");

         //--- conversion to broker time
         tz.SetLocalTime(beginlocal);
         datetime beginbroker = tz.ConvertLocalTime(ZONE_ID_BROKER);

         tz.SetLocalTime(endlocal);
         datetime endbroker = tz.ConvertLocalTime(ZONE_ID_BROKER);

         //--- local day of week in timezone
         MqlDateTime st;
         TimeToStruct(localtime, st);
         int dow = st.day_of_week;

         //string state_str = ((dow != SATURDAY && dow != SUNDAY) && (localtime >= beginlocal && localtime < endlocal)) ? "open" : "closed";
         string state_str = ((dow != SATURDAY && dow != SUNDAY) && (TimeTradeServer() >= beginbroker && TimeTradeServer() < endbroker)) ? "open" : "closed";

         PrintFormat("%-12s:  %s |  %s  [session %s]", tz.Name(), CTimeZoneInfo::FormatTimeForPlace(beginbroker, ZONE_ID_BROKER), CTimeZoneInfo::FormatTimeForPlace(endbroker, ZONE_ID_BROKER), state_str);
        }

      Print("-----------------------------------");
      Print("broker time :  ", TimeTradeServer());
      Print("broker time :  ", CTimeZoneInfo::FormatTimeForPlace(TimeTradeServer(), ZONE_ID_BROKER));
      //
      // ======= Local Session Hours (CTimeZoneInfo Class) =======
      // Sydney      :  Wed, 2024.11.13 23:00:00 GMT+2 STD [ICMarketsSC-Demo] |  Thu, 2024.11.14 08:00:00 GMT+2 STD [ICMarketsSC-Demo]  [session open]
      // Tokyo       :  Thu, 2024.11.14 01:00:00 GMT+2 STD [ICMarketsSC-Demo] |  Thu, 2024.11.14 10:00:00 GMT+2 STD [ICMarketsSC-Demo]  [session open]
      // Frankfurt   :  Thu, 2024.11.14 09:00:00 GMT+2 STD [ICMarketsSC-Demo] |  Thu, 2024.11.14 18:00:00 GMT+2 STD [ICMarketsSC-Demo]  [session closed]
      // London      :  Thu, 2024.11.14 10:00:00 GMT+2 STD [ICMarketsSC-Demo] |  Thu, 2024.11.14 19:00:00 GMT+2 STD [ICMarketsSC-Demo]  [session closed]
      // New York    :  Wed, 2024.11.13 15:00:00 GMT+2 STD [ICMarketsSC-Demo] |  Thu, 2024.11.14 00:00:00 GMT+2 STD [ICMarketsSC-Demo]  [session closed]
      // -----------------------------------
      // broker time :  2024.11.14 06:29:16
      // broker time :  Thu, 2024.11.14 06:29:16 GMT+2 STD [ICMarketsSC-Demo]
     }


//--- B. CSessionHours Class

//--- Working with CSessionHours Objects
     {
      Print("\n========== Working with CSessionHours Objects  ==========");

      CSessionHours tz(ZONE_ID_SYDNEY);
      tz.RefreshTime(); // populate timezone and session information

      //--- from parent
      Print("Name()                : ", tz.Name());
      Print("TimeUTC()             : ", tz.TimeUTC());
      Print("TimeLocal()           : ", tz.TimeLocal());
      Print("ToString()            : ", tz.ToString());
      //--- from class
      Print("BeginLocalTime()      : ", tz.BeginLocalTime());
      Print("EndLocalTime()        : ", tz.EndLocalTime());
      Print("CheckLocalSession()   : ", tz.CheckLocalSession());
      Print("SecRemainingSession() : ", tz.SecRemainingSession());
      Print("SecondsToString()     : ", CSessionHours::SecondsToString(tz.SecRemainingSession()));
      //
      // ========== Working with CSessionHours Objects  ==========
      // Name()                : Sydney
      // TimeUTC()             : 2024.11.14 04:29:16
      // TimeLocal()           : 2024.11.14 15:29:16
      // ToString()            : Thu, 2024.11.14 15:29:16 GMT+11 DST [Sydney]
      // BeginLocalTime()      : 2024.11.14 08:00:00
      // EndLocalTime()        : 2024.11.14 17:00:00
      // CheckLocalSession()   : true
      // SecRemainingSession() : 5443
      // SecondsToString()     : 01:30:43
     }


//--- Local Session Hours
     {
      Print("\n======= Local Session Hours (CSessionHours Class) =======");

      const ENUM_ZONE_ID ids[] = {ZONE_ID_SYDNEY, ZONE_ID_TOKYO, ZONE_ID_FRANKFURT, ZONE_ID_LONDON, ZONE_ID_NEWYORK};

      for(int i = 0; i < ArraySize(ids); i++)
        {
         ENUM_ZONE_ID id = ids[i];

         CSessionHours tz(id);
         tz.RefreshTime();

         //--- default session hours are set to 8:00 am - 5:00 pm local time
         datetime beginlocal = tz.BeginLocalTime();
         datetime endlocal   = tz.EndLocalTime();

         //--- conversion to broker time
         datetime beginbroker = CTimeZoneInfo::ConvertTimeForPlace(beginlocal, id, ZONE_ID_BROKER);
         datetime endbroker = CTimeZoneInfo::ConvertTimeForPlace(endlocal, id, ZONE_ID_BROKER);

         string state_str = tz.CheckLocalSession() ? "open, ends in " + CSessionHours::SecondsToString(tz.SecRemainingSession()) : "closed";

         PrintFormat("%-12s:  %s |  %s  [session %s]", tz.Name(), CTimeZoneInfo::FormatTimeForPlace(beginbroker, ZONE_ID_BROKER), CTimeZoneInfo::FormatTimeForPlace(endbroker, ZONE_ID_BROKER), state_str);
        }

      Print("-----------------------------------");
      Print("broker time :  ", TimeTradeServer());
      Print("broker time :  ", CTimeZoneInfo::FormatTimeForPlace(TimeTradeServer(), ZONE_ID_BROKER));
      Print("Fx close    :  ", CTimeZoneInfo::FormatTimeForPlace(CSessionHours::ForexCloseTime(), ZONE_ID_BROKER));
      int sec = CSessionHours::SecRemainingForex();
      Print("closes in   :  ", sec, " sec = ", CSessionHours::SecondsToString(sec));
      //
      // ======= Local Session Hours (CSessionHours Class) =======
      // Sydney      :  Wed, 2024.11.13 23:00:00 GMT+2 STD [ICMarketsSC-Demo] |  Thu, 2024.11.14 08:00:00 GMT+2 STD [ICMarketsSC-Demo]  [session open, ends in 01:30:43]
      // Tokyo       :  Thu, 2024.11.14 01:00:00 GMT+2 STD [ICMarketsSC-Demo] |  Thu, 2024.11.14 10:00:00 GMT+2 STD [ICMarketsSC-Demo]  [session open, ends in 03:30:43]
      // Frankfurt   :  Thu, 2024.11.14 09:00:00 GMT+2 STD [ICMarketsSC-Demo] |  Thu, 2024.11.14 18:00:00 GMT+2 STD [ICMarketsSC-Demo]  [session closed]
      // London      :  Thu, 2024.11.14 10:00:00 GMT+2 STD [ICMarketsSC-Demo] |  Thu, 2024.11.14 19:00:00 GMT+2 STD [ICMarketsSC-Demo]  [session closed]
      // New York    :  Wed, 2024.11.13 15:00:00 GMT+2 STD [ICMarketsSC-Demo] |  Thu, 2024.11.14 00:00:00 GMT+2 STD [ICMarketsSC-Demo]  [session closed]
      // -----------------------------------
      // broker time :  2024.11.14 06:29:16
      // broker time :  Thu, 2024.11.14 06:29:16 GMT+2 STD [ICMarketsSC-Demo]
      // Fx close    :  Fri, 2024.11.15 23:59:59 GMT+2 STD [ICMarketsSC-Demo]
      // closes in   :  149443 sec = 1d 17:30:43
     }


//--- How to override te default sssion hours?
     {
      Print("\n=========== Override the default session hours ===========");

      CSessionHours frankfurt(ZONE_ID_FRANKFURT);

      // change default session times
      frankfurt.BeginLocalTime(9, 0);
      frankfurt.EndLocalTime(19, 0);

      frankfurt.RefreshTime(); // populate new session hours

      datetime beginlocal = frankfurt.BeginLocalTime();
      datetime endlocal   = frankfurt.EndLocalTime();

      PrintFormat("new session hours  :  %s | %s", CTimeZoneInfo::FormatTimeForPlace(beginlocal, ZONE_ID_FRANKFURT), CTimeZoneInfo::FormatTimeForPlace(endlocal, ZONE_ID_FRANKFURT));
      PrintFormat("current local time :  %s", frankfurt.ToString());
      //
      // =========== Override the default session hours ===========
      // new session hours  :  Wed, 2024.02.28 09:00:00 GMT+1 STD [Frankfurt] | Wed, 2024.02.28 19:00:00 GMT+1 STD [Frankfurt]
      // current local time :  Wed, 2024.02.28 11:58:50 GMT+1 STD [Frankfurt]
     }


//--- How to check for closing positions at weekend?
     {
      Print("\n======= Check For Closing Positions at Weekend =======");

      int InpHours   = 2;   // Hours before weekend
      int InpMinutes = 30;  // Minutes before weekend

      int sec = CSessionHours::SecRemainingForex();
      PrintFormat("Time remaining till the weekend : %s", CSessionHours::SecondsToString(sec));
      PrintFormat("Close all if remaining time becomes %s or less.", CSessionHours::SecondsToString(InpHours * 3600 + InpMinutes * 60));

      // check time remaining has reached target
      if(sec <= InpHours * 3600 + InpMinutes * 60)
        {
         // CloseAll();
        }
      //
      // ======= Check For Closing Positions at Weekend =======
      // Time remaining till the weekend : 3d 03:32:27
      // Close all if remaining time becomes 02:30:00 or less.
     }

  }
//+------------------------------------------------------------------+
