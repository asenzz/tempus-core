//+------------------------------------------------------------------+
//|                                                  StringToArr.mq4 |
//|                        Copyright 2012, MetaQuotes Software Corp. |
//|                                        http://www.metaquotes.net |
//+------------------------------------------------------------------+
#property copyright "Copyright 2012, MetaQuotes Software Corp."
#property link      "http://www.metaquotes.net"


int StrToArr(string str, int &arr[]){

   int integers = MathCeil(StringLen(str) / 4.0); // bytes needed to store the string

   Print("[StrToArr] String len: ", StringLen(str), " Array size: ", ArraySize(arr), " integers: ", integers);

   // ArrayInitialize(arr, integers);
   // if(ArraySize(arr) == 0) return(0);
   
   if(ArraySize(arr) < integers){
      Print("Buffer too small!");
      return(-1);
   }
   
   int arr_iter = 0;
   int i = 0;
   // increments charsPerInt per iteration
   for(;i < StringLen(str);)
   {
      // NOTE: MQL represents int type as four-byte data internally
      int integer = 0 & 0x00000000;      
      
      integer = StringGetChar(str, i) & 0x000000FF; 
      if(i == StringLen(str)-1)
      {
         arr[arr_iter] = integer;    
         break;
      }     
      i++; 
      
      integer = 0x0000FFFF & ((StringGetChar(str, i) << 8) | integer); 
      if(i == StringLen(str)-1)
      {
         arr[arr_iter] = integer;    
         break;
      }
      i++;
      
      integer = 0x00FFFFFF & ((StringGetChar(str, i) << 16) | integer);
      if(i == StringLen(str)-1)
      {
         arr[arr_iter] = integer;    
         break;
      }
      i++;
      
      integer = 0xFFFFFFFF & ((StringGetChar(str, i) << 24) | integer); 
      if(i == StringLen(str)-1)
      {
         arr[arr_iter] = integer;    
         break;
      }      
      i++;
      
      arr[arr_iter] = integer;
      arr_iter++;      
   }

   return(i);
}


string ArrToStr(int cBuffer[], int len = 0)
{
   string text = "";
   int i = 0;
   
   if(len <= 0)
      len = ArraySize(cBuffer);
      
   if(len == 0) return("");
   //Print("Buffer:");
   for(; i < len; i++){   
   
      //Print(cBuffer[i]);
      
      text = StringConcatenate(text, CharToStr(cBuffer[i] & 0x000000FF));
      //if(StringLen(text) == len)
      //   break;
         
      text = StringConcatenate(text, CharToStr((cBuffer[i] >> 8) & 0x000000FF));
      
      //if(StringLen(text) == len)
      //   break;
         
      text = StringConcatenate(text, CharToStr((cBuffer[i] >> 16) & 0x000000FF));
      
      //if(StringLen(text) == len)
      //   break;
         
      text = StringConcatenate(text, CharToStr((cBuffer[i] >> 24) & 0x000000FF));
   }
   
   if(i != len) Print("Possibly incomplete data returned from ArrToStr! i:", i, " len:", len);
   //Print("ArrayToStr: ", text);
   return(text);
}

//+------------------------------------------------------------------+
//| script program start function                                    |
//+------------------------------------------------------------------+
int start()
  {
//----

   int arr[6];
   string str = "This is example string.";
   
   if(StrToArr(str, arr) == -1){
      Alert("Error in StrToArr()");
      return(-1);
   }
   
   Print("[start] Returned: ", sz, " Array size: ", ArraySize(arr));
   
   for(int i = 0; i < ArraySize(arr); i++)
      Print(arr[i]);
      
   string res = ArrToStr(arr);
   
   Print("ArrToStr: \"", res, "\"");
   
//----
   return(0);
  }
//+------------------------------------------------------------------+