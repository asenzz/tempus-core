//+------------------------------------------------------------------+
//|                                                 unsigned_int.mq4 |
//|                        Copyright 2012, MetaQuotes Software Corp. |
//|                                        http://www.metaquotes.net |
//+------------------------------------------------------------------+
#property copyright "Copyright 2012, MetaQuotes Software Corp."
#property link      "http://www.metaquotes.net"
/*
#import "zlib1.dll"
//int compress2(Bytef * dest, uLongf * destLen, const Bytef * source, uLong sourceLen, int level);
int compress(datetime &dest[], int &destLen[], string source, int sourceLen);
int uncompress (datetime &dest[], int &destLen[], datetime source[], int sourceLen);
#import*/



#include <StringUtils.mqh>
//#include <stdlib.mqh>
//+------------------------------------------------------------------+
//| script program start function                                    |
//+------------------------------------------------------------------+
int start()
   {
   
   string test[3];
   test[0] = "abcdef";
   test[1] = "1234567";
   test[2] = "ABCDEFGH";
   
   Print("Total valid chars: " , TotalValidChars(test));
   
   for(int i = 0; i < ArraySize(test); i++){
      string out = "";//StringConcatenate(IntegerToHexString(buf[i]), ": [");
      
      for(int j = 0; j < StringLen(test[i]); j++)
         out = StringConcatenate(out, "\"", StringSubstr(test[i], j, 1), "\"", StringGetChar(test[i], j), " [", IntegerToHexString(StringGetChar(test[i], j)), "], ");
        
     // out = StringConcatenate(out, GetCharAt(buf[i], 3), "]");
      Print(out);
   }
   
   int buf[];
   
   int sz = StrToArr(buf, test);
   
   Print("Buffer size: ", ArraySize(buf), " total chars: ", sz);
   
   for(i = 0; i < ArraySize(buf); i++){
      out = StringConcatenate(IntegerToHexString(buf[i]), ": [");
      
      for(j = 0; j < 3; j++)
         out = StringConcatenate(out, GetCharAt(buf[i], j), ", ");
        
      out = StringConcatenate(out, GetCharAt(buf[i], 3), "]");
      Print(out);
   }
   
   
   
   int compressed1[];
   
   int c_size1 = i_compress(buf, sz, compressed1);
   
   Print("Compressed1 size: ", c_size1, " Array size: ", ArraySize(compressed1));
   
   for(i = 0; i < ArraySize(compressed1); i++){
      out = StringConcatenate(IntegerToHexString(compressed1[i]), ": [");
      
      for(j = 0; j < 3; j++)
         out = StringConcatenate(out, GetCharAt(compressed1[i], j), ", ");
        
      out = StringConcatenate(out, GetCharAt(compressed1[i], 3), "]");
      Print(out);
   }
   
   int compressed2[];
   
   int c_size2 = s_compress(test, compressed2);
   
   Print("Compressed2 size: ", c_size2, " Array size: ", ArraySize(compressed2));
   
   for(i = 0; i < ArraySize(compressed2); i++){
      out = StringConcatenate(IntegerToHexString(compressed2[i]), ": [");
      
      for(j = 0; j < 3; j++)
         out = StringConcatenate(out, GetCharAt(compressed2[i], j), ", ");
        
      out = StringConcatenate(out, GetCharAt(compressed2[i], 3), "]");
      Print(out);
   }
   
   int uncompressed1[];
   
   int u_size1 = i_uncompress(compressed1, c_size1, uncompressed1);
   
   Print("Uncompressed1 size: ", u_size1, " array size: ", ArraySize(uncompressed1));
   
   for(i = 0; i < ArraySize(uncompressed1); i++){
      out = StringConcatenate(IntegerToHexString(uncompressed1[i]), ": [");
      
      for(j = 0; j < 3; j++)
         out = StringConcatenate(out, GetCharAt(uncompressed1[i], j), ", ");
        
      out = StringConcatenate(out, GetCharAt(uncompressed1[i], 3), "]");
      Print(out);
   }
   
   int uncompressed2[];
   
   int u_size2 = i_uncompress(compressed2, c_size2, uncompressed2);
   
   Print("Uncompressed2 size: ", u_size2, " array size: ", ArraySize(uncompressed2));
   
   for(i = 0; i < ArraySize(uncompressed2); i++){
      out = StringConcatenate(IntegerToHexString(uncompressed2[i]), ": [");
      
      for(j = 0; j < 3; j++)
         out = StringConcatenate(out, GetCharAt(uncompressed2[i], j), ", ");
        
      out = StringConcatenate(out, GetCharAt(uncompressed2[i], 3), "]");
      Print(out);
   }
   
   /*int i = 0x00000000;
   
   string str = "abcd";
   
   AddCharAt(i, StringGetChar(str,0),0);
   
   Print(IntegerToHexString(i));
   
   AddCharAt(i, StringGetChar(str,1),1);
   
   Print(IntegerToHexString(i));
   
   AddCharAt(i, StringGetChar(str,2),2);
   
   Print(IntegerToHexString(i));
   
   AddCharAt(i, StringGetChar(str,3),3);
   
   Print(IntegerToHexString(i));*/
  
  /*
      datetime dest[100];
      int destLen[1];
      destLen[0] = 100;
      
      string to_compress = "TOBEORNOTTOBEORTOBEORNOT";
   
      compress(dest, destLen, to_compress, StringLen(to_compress));
      
      Print("Compressed size: ", destLen[0]);
      string output = "";
      for(int i = 0; i < MathCeil(destLen[0]/4.0); i++)
         output = StringConcatenate(output, " ", IntegerToHexString(dest[i]));
         
      Print(output);
         
      int uarrLen[1];   
      
      datetime uncompressed[100];
      uarrLen[0] = ArraySize(uncompressed);
   
      uncompress(uncompressed, uarrLen, dest, destLen[0]);
      string uncmprsd = ArrToStr(uncompressed, uarrLen[0]);
      Print("Uncompressed: ", uncmprsd, " size: ", StringLen(uncmprsd));
   */
      return(0);
   }
//+------------------------------------------------------------------+