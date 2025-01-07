//+------------------------------------------------------------------+
//|                                                   DSTPeriods.mqh |
//|                                           Copyright 2019, Tempus |
//|                                              https://tempus.work |
//+------------------------------------------------------------------+
#property copyright "Copyright 2019, Tempus"
#property link      "https://tempus.work"
#property strict

#include <MqlLog.mqh>
#include <CompatibilityMQL4.mqh>

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class DSTPeriod
{
   
private:
    int               year;
    datetime          start_time;
    datetime          end_time;
   
public:
   
    DSTPeriod(int year_, datetime start_, datetime end_)
    {
        year = year_;
        start_time = start_;
        end_time = end_;
    }
    
    bool isInPeriod(datetime time) const
    {
        return (start_time < time && time < end_time);
    }
};

bool disableDST = true;

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
const DSTPeriod year98(1998, D'29.03.1998 03:00', D'25.10.1998 04:00');
const DSTPeriod year99(1999, D'28.03.1999 03:00', D'31.10.1999 04:00');
const DSTPeriod year00(2000, D'26.03.2000 03:00', D'29.10.2000 04:00');
const DSTPeriod year01(2001, D'25.03.2001 03:00', D'28.10.2001 04:00');
const DSTPeriod year02(2002, D'31.03.2002 03:00', D'27.10.2002 04:00');
const DSTPeriod year03(2003, D'30.03.2003 03:00', D'26.10.2003 04:00');
const DSTPeriod year04(2004, D'28.03.2004 03:00', D'31.10.2004 04:00');
const DSTPeriod year05(2005, D'27.03.2005 03:00', D'30.10.2005 04:00');
const DSTPeriod year06(2006, D'26.03.2006 03:00', D'29.10.2006 04:00');
const DSTPeriod year07(2007, D'25.03.2007 03:00', D'28.10.2007 04:00');
const DSTPeriod year08(2008, D'30.03.2008 03:00', D'26.10.2008 04:00');
const DSTPeriod year09(2009, D'29.03.2009 03:00', D'25.10.2009 04:00');
const DSTPeriod year10(2010, D'28.03.2010 03:00', D'31.10.2010 04:00');
const DSTPeriod year11(2011, D'27.03.2011 03:00', D'30.10.2011 04:00');
const DSTPeriod year12(2012, D'25.03.2012 03:00', D'28.10.2012 04:00');
const DSTPeriod year13(2013, D'31.03.2013 03:00', D'27.10.2013 04:00');
const DSTPeriod year14(2014, D'30.03.2014 03:00', D'26.10.2014 04:00');
const DSTPeriod year15(2015, D'29.03.2015 03:00', D'25.10.2015 04:00');
const DSTPeriod year16(2016, D'27.03.2016 03:00', D'30.10.2016 04:00');
const DSTPeriod year17(2017, D'26.03.2017 03:00', D'29.10.2017 04:00');
const DSTPeriod year18(2018, D'25.03.2018 03:00', D'28.10.2018 04:00');
const DSTPeriod year19(2019, D'31.03.2019 03:00', D'27.10.2019 04:00');
const DSTPeriod year20(2020, D'29.03.2020 03:00', D'25.10.2020 04:00');
const DSTPeriod year21(2021, D'28.03.2021 03:00', D'31.10.2021 04:00');
const DSTPeriod year22(2022, D'27.03.2021 03:00', D'30.10.2021 04:00');
const DSTPeriod year23(2023, D'26.03.2021 03:00', D'29.10.2021 04:00');
const DSTPeriod year24(2024, D'31.03.2021 03:00', D'27.10.2021 04:00');
const DSTPeriod year25(2025, D'30.03.2021 03:00', D'26.10.2021 04:00');
/*
EET = UTC + 2
EEST = EET + DST
EEST = UTC + 2 + 1
*/

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool wasDSTActiveThen(datetime history_moment)
{
    switch(TimeYearMQL4(history_moment))
    {
        case 1998:
            return year98.isInPeriod(history_moment);
        case 1999:
            return year99.isInPeriod(history_moment);
        case 2000:
            return year00.isInPeriod(history_moment);
        case 2001:
            return year01.isInPeriod(history_moment);
        case 2002:
            return year02.isInPeriod(history_moment);
        case 2003:
            return year03.isInPeriod(history_moment);
        case 2004:
            return year04.isInPeriod(history_moment);
        case 2005:
            return year05.isInPeriod(history_moment);
        case 2006:
            return year06.isInPeriod(history_moment);
        case 2007:
            return year07.isInPeriod(history_moment);
        case 2008:
            return year08.isInPeriod(history_moment);
        case 2009:
            return year09.isInPeriod(history_moment);
        case 2010:
            return year10.isInPeriod(history_moment);
        case 2011:
            return year11.isInPeriod(history_moment);
        case 2012:
            return year12.isInPeriod(history_moment);
        case 2013:
            return year13.isInPeriod(history_moment);
        case 2014:
            return year14.isInPeriod(history_moment);
        case 2015:
            return year15.isInPeriod(history_moment);
        case 2016:
            return year16.isInPeriod(history_moment);
        case 2017:
            return year17.isInPeriod(history_moment);
        case 2018:
            return year18.isInPeriod(history_moment);
        case 2019:
            return year19.isInPeriod(history_moment);
        case 2020:
            return year20.isInPeriod(history_moment);
        case 2021:
            return year21.isInPeriod(history_moment);
        default:
            LOG_ERROR("Request date " + string(history_moment) + " does not have specified DST period ");
            return false;
    }
}

//
// if then and now we aren't in DST OR then and now we are in DST
//    we should not move the time period with DST offset
// if we were in DST and now we are not
//    we should add an hour to the offset
// if we weren't in DST and now we are
//    we should remove one hour

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int getDSTOffsetFromNow(datetime history_moment)
{
    if (disableDST) return 0;
    //int SECONDS_IN_HOUR = 3600;
    int DST_now = TimeDaylightSavings();
    bool wasDST = wasDSTActiveThen(history_moment);

    if((wasDST && (DST_now != 0)) || (!wasDST && (DST_now == 0)))
        return 0;
    else if(wasDST && (DST_now == 0))
        return SECONDS_IN_HOUR;
    else
        return -SECONDS_IN_HOUR;
}
//+------------------------------------------------------------------+
