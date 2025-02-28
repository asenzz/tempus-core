//+------------------------------------------------------------------+
//|                                                     lzw_test.mq4 |
//|                        Copyright 2012, MetaQuotes Software Corp. |
//|                                        http://www.metaquotes.net |
//+------------------------------------------------------------------+
#property copyright "Copyright 2012, MetaQuotes Software Corp."
#property link      "http://www.metaquotes.net"

//#import "liblzw.dll"

//int compress(string a, int &arr[], int buf_siz);
//int fnlzw();

//#import
#include <StringUtils.mqh>
//+------------------------------------------------------------------+
//| script program start function                                    |
//+------------------------------------------------------------------+
int start()
  {
  
//----   
   int buf[100];
   string plain = "TOBEORNOTTOBEORTOBEORNOTTOBEORNOTTOBEORTOBEORNOTTOBEORNOTTOBEORTOBEOR";
   
   int sz = s_compress(plain, buf);
   
   Print("Returned compressed size: ", sz);   
   
   string compressed_bytes;
   
   for(int i = 0; i < MathCeil(sz / 4.0); i++) compressed_bytes = IntegerToHexString(buf[i]) + " ";
      
   Print("Compressed bytes: ", compressed_bytes);   
   
   string uncompressed = i_uncompress(buf, sz);
   
   Print("Uncompressed: \"", uncompressed, "\"");
   
//----
   return(0);
  }
//+------------------------------------------------------------------+