//+------------------------------------------------------------------+
//|                                                 send_pending.mq4 |
//|                      Copyright © 2004, MetaQuotes Software Corp. |
//|                                       http://www.metaquotes.net/ |
//+------------------------------------------------------------------+
#property copyright "Copyright © 2004, MetaQuotes Software Corp."
#property link      "http://www.metaquotes.net/"
#property show_confirm

//+------------------------------------------------------------------+
//| script "send pending order with expiration data"                 |
//+------------------------------------------------------------------+
int start()
  {
   int    ticket,expiration;
   double point;
//----
   point=MarketInfo(Symbol(),MODE_POINT);
   expiration=CurTime()+PERIOD_D1*60;
//----
   while(true)
     {
      ticket=OrderSend(Symbol(),OP_SELLSTOP,1.0,Bid-100*point,0,0,0,"some comment",16384,expiration,Green);
      if(ticket<=0) Print("Error = ",GetLastError());
      else { Print("ticket = ",ticket); break; }
      //---- 10 seconds wait
      Sleep(10000);
     }
//----
   return(0);
  }
//+------------------------------------------------------------------+