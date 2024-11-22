//+------------------------------------------------------------------+
//|                                                 send_history.mq4 |
//|                        Copyright 2012, MetaQuotes Software Corp. |
//|                                        http://www.metaquotes.net |
//+------------------------------------------------------------------+
#property copyright "Copyright 2012, MetaQuotes Software Corp."
#property link      "http://www.metaquotes.net"

#include <Utilities.mqh>

extern string sAddr = "localhost";// "172.16.240.1";
extern string sProto = "http";



/*
int send_history()
{   
   if(!negotiate_period_to_send(first_times, last_times)) return(-1);
   
   string lines[]; // uncompressed chunked data
   
   
   string header = "Timestamp;Open;High;Low;Close;Volume;";
   string metadata = StringConcatenate(
      "header=", header,
      "&timeframe=", Period(),
      "&queue=", Symbol(), "_", Period(),
      "&delimiter=", StringGetChar(delimiter,0));
      
   
   
   int lineno = 0;
   int chunkno = 0;
  
/* +-----------------------------------------------------------------------------+
   | for each bar generate one line of data following the header:                |
   | [Timestamp;Open;High;Low;Close;Volume;]                                     |
   | Parameters sent with each request (uncompressed) (Queue-wise):              |
   | header=&timeframe=&queue&delimiter=ASCII#&zdatalen                          |
   | @header - the data mapping of every row (used to parse at server side)      |
   | @timeframe - e.g. 60, 240 (in minutes)                                      |
   | @first_timestamp - the timestamp of the oldest data sent                    |
   | @last_timestamp - the timestamp of the newest data sent                     |
   | @queue - e.g. EURUSD, Custom_data etc.                                      |
   | @delimiter - ASCII code of the char used as delimiter (default ";")         |
   | @zdatalen - length (in bytes) of the compressed data sent                   |
   | @zdata - binary encoded zlib-compressed chunk of data                       |
   +-----------------------------------------------------------------------------+ * /
 for(int bar = 0; bar < Bars; bar++){
   
      // if current Bar data is already present at the server, skip
      if(Time[bar] > last_time) continue;
      
      // if max chunk size or end of period to send is reached, send
      if(TotalValidChars(lines) >= zchunksize || Time[bar] < first_time){
         // compress, send, clear local lines buffer, reset line counter
                  
         int zbuf[]; // zipped data buffer
         int plain_buf[]; // plain data buffer
         int pbytes = StrsToArr(plain_buf, lines);
         int zbytes = i_compress(plain_buf, pbytes, zbuf);
         string post_data[2]; // metadata and zipped data        
         
         
           
         for(int i = 0; i < ArraySize(zbuf); i++){
            string hex = IntegerToHexString(zbuf[i]);
            
            post_data[1] = StringConcatenate(post_data[1], StringSubstr(hex, 6, 2), 
               StringSubstr(hex, 4, 2), StringSubstr(hex, 2, 2),StringSubstr(hex, 0, 2));
                   
         }
         
         post_data[0] = StringConcatenate(metadata, "&zdatalen=", StringLen(post_data[1]), "&zdata=");
         
        
         string response;
         
         ssHttpPOST(
            StringConcatenate(sProto, "://", sAddr, "/", sPostH), 
            post_data, response);
            
         Print("Response: ", response);
         break;
         
       
         
         if(Time[bar] < first_time) break;
            
         lineno = 0; // reset line counter
         ArrayResize(lines, 1); // clear lines buffer
         lines[lineno] = "";
         chunkno++;
      }
   
      string line = "";   
      line = StringConcatenate(
         TimeToStr(Time[bar], TIME_DATE | TIME_SECONDS), delimiter,
         Open[bar], delimiter,
         High[bar], delimiter,
         Low[bar], delimiter,
         Close[bar], delimiter,
         Volume[bar], delimiter);
         
     
      
      if(ArraySize(lines) <= lineno) 
         ArrayResize(lines, ArraySize(lines)+1);
         
      lines[lineno] = line;
      lineno++;
   }
   
   //FileClose(hFile);   
   //FileClose(hBin);  
    return(0);
  }*/
//+------------------------------------------------------------------+
//| script program start function                                    |
//+------------------------------------------------------------------+
int start()
{
    string target_url;
    StringConcatenate(target_url, sProto, "://", sAddr);
    return(send_history(target_url));
}
//+------------------------------------------------------------------+