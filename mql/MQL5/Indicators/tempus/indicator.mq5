//+------------------------------------------------------------------+
//|                                                 Tempus indicator |
//|                                                         Papakaya |
//+------------------------------------------------------------------+
#property copyright "Papakaya LTD"
#property link      "https://www.papakaya.com"
//#property version   "000.900"
#property strict
#property indicator_chart_window
#property indicator_buffers 4
#property indicator_plots   4
//--- plot Open
#property indicator_label1  "Sensored"
#property indicator_type1   DRAW_HISTOGRAM
#property indicator_color1  clrRed
#property indicator_style1  STYLE_SOLID
#property indicator_width1  1
//--- plot High
#property indicator_label2  "Sensored"
#property indicator_type2   DRAW_HISTOGRAM
#property indicator_color2  clrRed
#property indicator_style2  STYLE_SOLID
#property indicator_width2  1
//--- plot Low
#property indicator_label3  "Sensored"
#property indicator_type3   DRAW_HISTOGRAM
#property indicator_color3  clrRed
#property indicator_style3  STYLE_SOLID
#property indicator_width3  1
//--- plot Close
#property indicator_label4  "Sensored"
#property indicator_type4   DRAW_HISTOGRAM
#property indicator_color4  clrRed
#property indicator_style4  STYLE_SOLID
#property indicator_width4  1


#include <TempusMvc.mqh>
#include <AveragePrice.mqh>
#include <TempusGMT.mqh>


//================================================

input uint     BarNumber = 15;
input string   Username   = "svrwave";
input string   Password   = "svrwave";
input string   ServerUrl  = "10.4.8.5:8080";
input string   DataSet = "2023";
input bool     RequestHigh = false;
input bool     RequestLow = false;
input bool     RequestOpen = false;
input bool     RequestClose = false;
input int      PauseBeforePolling = 1;
input int      MaxPendingRequests = 3;
input bool     KeepHistoryBars = true;
input bool     DemoMode = false;
input bool     Average = true;
input int      TargetPeriod   = 240;
input int      TimeOffset     = 120;
input bool     DisableDST     = true;

int            FBWidth                 = 1;
color          FBColorUp,FBColorDown;
long           ChartType;
datetime       Poll_time;
int            CurrentRequestedBarsNumber = 0;
int            CurrentRecievedBarsNumber = 0;

MqlNet         net;
TempusController controller;
TempusGraph    view;
datetime pending_requests[];


const datetime C_zero_time = datetime(0);

int file_handle;
int real_handle;
string file_name = "predictions.csv";
string real_data_filename = "real.csv";

#define LOG_PREDICTIONS

//================================================

int OnInit()
{
   disableDST = DisableDST;
   //IndicatorShortName("tempus_indicator");
   GlobalVariableSet(C_chart_predictions_identifier, 0.);
   
   ChartSetInteger(0,CHART_SHOW_GRID,0);
   ChartSetInteger(0,CHART_SHOW_PERIOD_SEP,1);
   ChartSetInteger(0,CHART_AUTOSCROLL,1);
   ChartSetInteger(0,CHART_SHIFT,1);
   ChartSetDouble(0,CHART_SHIFT_SIZE,20);
   
   
   TempusGMTInit(TargetPeriod, TimeOffset, DisableDST);
   
   if(!net.Open(ServerUrl)) {
      Print("Incorrect server address");
      return(INIT_PARAMETERS_INCORRECT);
   }
   long StatusCode;
   net.Post("mt4/login", "username="+Username+"&password="+Password, StatusCode);
   if(StatusCode!=200) {
      Print("Incorrect USERNAME or PASSWORD", StatusCode);
      return(INIT_PARAMETERS_INCORRECT);
   }
   
   view.init(BarNumber, PeriodSeconds(), KeepHistoryBars, DemoMode, Average);
   controller.init(PeriodSeconds(), RequestHigh, RequestLow, RequestOpen, RequestClose, Average);
   const datetime current_bar_time = DemoMode == false ? GetTargetTime(0) : GetTargetTime(BarNumber);
   controller.do_request(net, current_bar_time, BarNumber, DataSet);


   EventSetTimer(30);
   
   ArrayResize(pending_requests, MaxPendingRequests, MaxPendingRequests);
   ArrayFill(pending_requests, 0, MaxPendingRequests, C_zero_time);
      
   #ifdef LOG_PREDICTIONS
      file_handle=FileOpen(file_name,FILE_CSV|FILE_READ|FILE_WRITE,',');
      real_handle=FileOpen(real_data_filename,FILE_CSV|FILE_READ|FILE_WRITE,',');
      if(file_handle!=INVALID_HANDLE)
      {
         FileWrite(file_handle, "REAL_TIME", "Time", "REAL_CLOSE", "Close", "REAL_HIGH", "High", "REAL_LOW", "Low", "REAL_OPEN", "Open");           FileWrite(real_handle, "Time", "Close", "Open", "High","Low");
        }
      else{
         PrintFormat("Failed to open %s file, Error code = %d",file_handle,GetLastError());
     }
      FileClose(file_handle);
   #endif

   return(INIT_SUCCEEDED);
}
  
//================================================

void OnDeinit(const int reason) {
   EventKillTimer();
   #ifdef LOG_PREDICTIONS
      FileClose(file_handle);
      FileClose(real_handle);
   #endif
   view.close();   
}

//================================================

AveragePrice *prepareAverage(const int time_index)
{
   MqlRates rates[];
   ArraySetAsSeries(rates, true);
   // int copied = CopyRates(Symbol(), PERIOD_M1, Time[time_index] - PeriodSeconds() + 1, Time[time_index], rates);
   int copied_ct = CopyRates(Symbol(), PERIOD_M1, GetUTCServerTime(time_index), GetUTCServerTime(time_index) + PeriodSeconds(), rates);
   if (copied_ct > 0) {
      AveragePrice *p_current = new AveragePrice(rates, copied_ct);
      return p_current;
      //svr_client.send_bar(InputQueueFinal, StrPeriod, true, current);
      //LOG_INFO("Sent Tick for Time: " + string(current.tm) + " Value: " + string(current.value));
   } else {
      LOG_ERROR("Failed to get history data for the symbol " + Symbol() + " at position " + string(time_index));
      return NULL;
   }
}

//================================================

int doCalculate(const int rates_total)
{
   static datetime lastRequestTime = C_zero_time;
   static datetime nearestResponseTime = C_zero_time;
   const datetime current_bar_time = DemoMode == false ? GetTargetTime(0) : GetTargetTime(BarNumber);
   const datetime nextFigure = current_bar_time;

   if (lastRequestTime != nextFigure && GlobalVariableGet(C_chart_identifier) >= double(lastRequestTime))
   {
      int request_ix = 0;
      while ( request_ix < MaxPendingRequests && pending_requests[request_ix] != C_zero_time )
         ++request_ix;
         
      if(request_ix == MaxPendingRequests)
      {
         request_ix = 0;
         for(int ir = 0; ir < MaxPendingRequests; ++ir)
            if(pending_requests[ir] < pending_requests[request_ix])
               request_ix = ir;
      }
      
      lastRequestTime = nextFigure; 
      nearestResponseTime = TimeCurrent() + PauseBeforePolling;
            
      controller.do_request(net, nextFigure, BarNumber, DataSet);
      pending_requests[request_ix] = nextFigure;
      
      LOG_INFO("Requested " + string(BarNumber) + " bar(s) starting from " + 
               string(nextFigure) + " request #" + string(request_ix));
            
      view.fadeFigure();
      
      return rates_total;
   }
   
   bool redrawn = false;
   
   for (int request_ix = 0; request_ix < MaxPendingRequests; ++request_ix) 
   {
      if ( TimeCurrent() >= nearestResponseTime && pending_requests[request_ix] != C_zero_time) 
      { 
         LOG_DEBUG("Retrieving response " + string(pending_requests[request_ix]) + 
                  " request #" + string(request_ix));
         TempusFigure figs[];
         if(controller.get_results(net, pending_requests[request_ix], BarNumber, DataSet, figs))
         {
            static datetime lastBarTime = C_zero_time;
            const uint ai = ArraySize(figs) - 1;
		
            LOG_DEBUG("Received bars from " + string(figs[0].tm) + " to " + string(figs[ai].tm));
	         if (floor(figs[0].tm / PeriodSeconds()) == floor(current_bar_time/( PeriodSeconds() ))) {
	            GlobalVariableSet(C_chart_predictions_identifier, figs[0].cl);
               view.redraw(figs, pending_requests[request_ix], Average);
               redrawn = true;
            } else {               
		         LOG_ERROR("Received old bar with time value " +  TimeToStr(figs[0].tm,TIME_DATE| TIME_SECONDS) + ". Current bar is: " + TimeToStr(current_bar_time,TIME_DATE| TIME_SECONDS) + ". Skipping it");
            }
            
            if(figs[ai].tm > lastBarTime)
            {            
               lastBarTime = figs[ai].tm;
               LOG_INFO("Received a bar dated " + string(lastBarTime) + " high: " + StringFormat("%.5f", figs[ai].hi) + 
                        " low: " + StringFormat("%.5f", figs[ai].lo) + " of request #"+string(request_ix) + " at " +TimeToStr(pending_requests[request_ix]));
            }
            pending_requests[request_ix] = C_zero_time;
            
#ifdef LOG_PREDICTIONS
            if (Average == true) {
               AveragePrice *p_current_average_price = prepareAverage(0);
               FileWrite(real_handle, TimeToStr( GetTargetTime(BarNumber), TIME_DATE | TIME_SECONDS ), p_current_average_price.value);
               file_handle = FileOpen(file_name,FILE_CSV|FILE_READ|FILE_WRITE,',');
               FileSeek(file_handle, 0, SEEK_END);
               
               for(int res = 0; res < ArraySize(figs); ++res)
               {
                  FileWrite(
                     file_handle, 
                     TimeToStr(GetTargetTime(BarNumber - res - 1), TIME_DATE | TIME_SECONDS ), string(figs[res].tm), 
   			         GetTargetClose(BarNumber - res - 1), StringFormat("%.5f", figs[res].cl), 
   			         GetTargetHigh(BarNumber - res - 1), StringFormat("%.5f", figs[res].hi),
   			         GetTargetLow(BarNumber - res - 1), StringFormat("%.5f", figs[res].lo),
   			         GetTargetOpen(BarNumber - res - 1), StringFormat("%.5f", figs[res].op)
   			         );
               }
               delete p_current_average_price;
               FileClose(file_handle);
            } else {               
               FileWrite(real_handle, TimeToStr(GetTargetTime(BarNumber), TIME_DATE | TIME_SECONDS ), GetTargetOpen(0), GetTargetHigh(0), GetTargetLow(0), GetTargetClose(0));
               file_handle = FileOpen(file_name,FILE_CSV|FILE_READ|FILE_WRITE,',');
               FileSeek(file_handle, 0, SEEK_END);
               for(int res = 0; res < ArraySize(figs); ++res )
               {
                  FileWrite(
                     file_handle, TimeToStr(GetTargetTime(BarNumber), TIME_DATE | TIME_SECONDS ), string(figs[res].tm), 
   			         GetTargetClose(BarNumber), StringFormat("%.5f", figs[res].cl), 
   			         GetTargetHigh(BarNumber), StringFormat("%.5f", figs[res].hi),
   			         GetTargetLow(BarNumber), StringFormat("%.5f", figs[res].lo),
   			         GetTargetOpen(BarNumber), StringFormat("%.5f", figs[res].op)
   			         );
               }
               FileClose(file_handle);
            }
#endif
         } else {
            LOG_DEBUG("Response for request #" + string(request_ix) + " " + string(pending_requests[request_ix]) + " not ready yet.");
         }
      }
   }
      
   if(!redrawn) {
      view.fadeFigure();
   } else {
      nearestResponseTime = TimeCurrent() + PauseBeforePolling;
   }
      
   return rates_total;
}

//+------------------------------------------------------------------+
//| Custom indicator iteration function                              |
//+------------------------------------------------------------------+
int OnCalculate(const int rates_total,
                const int prev_calculated,
                const datetime &time[],
                const double &open[],
                const double &high[],
                const double &low[],
                const double &close[],
                const long &tick_volume[],
                const long &volume[],
                const int &spread[])
{
   return doCalculate(rates_total);
}

void OnTimer()
{
   doCalculate(0);
}

//=========================================================================================================
