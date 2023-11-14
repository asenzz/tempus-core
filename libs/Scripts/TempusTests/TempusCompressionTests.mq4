//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+

#include <tempus-compression.mqh>

#define ASSERT(expr, message, file, line) if(!(expr)) {Alert("Assertion failed: " + message + " in " + file + ": " +string (line)); return 1;}
#define assert(expr) ASSERT(expr, #expr, __FILE__, __LINE__)

void OnStart()
{
   Print("LZ4 version number: " + string(LZ4_versionNumber()));
   
   string what[]; ArrayResize (what, 4);
   what[0] = "These are functions for working with timeseries and indicators. A timeseries differs from the usual data array by its reverse ordering - elements of timeseries are indexed from the end of an array to its begin (from the most recent data to the oldest ones). To copy the time-series values and indicator data, it's recommended to use dynamic arrays only, because copying functions are designed to allocate the necessary size of arrays that receive values."
      "There is an important exception to this rule: if timeseries and indicator values need to be copied often, for example at each call of OnTick() in Expert Advisors or at each call of OnCalculate() in indicators, in this case one should better use statically distributed arrays, because operations of memory allocation for dynamic arrays require additional time, and this will have effect during testing and optimization."
      "When using functions accessing timeseries and indicator values, indexing direction should be taken into account. This is described in the Indexing Direction in Arrays, Buffers and Timeseries section."
      "Access to indicator and timeseries data is implemented irrespective of the fact whether the requested data are ready (the so called asynchronous access). This is critically important for the calculation of custom indicator, so if there are no data, functions of Copy...() type immediately return an error. However, when accessing form Expert Advisors and scripts, several attempts to receive data are made in a small pause, which is aimed at providing some time necessary to download required timeseries or to calculate indicator values."
      "If data (symbol name and/or timeframe differ from the current ones) are requested from another chart, the situation is possible that the corresponding chart was not opened in the client terminal and the necessary data must be requested from the server. In this case, error ERR_HISTORY_WILL_UPDATED (4066 - the requested history data are under updating) will be placed in the last_error variable, and one will has to re-request (see example of ArrayCopySeries())."
      "The Organizing Data Access section describes details of receiving, storing and requesting price data in the MetaTrader 4 client terminal.";
    what[1] = "Array is an arranged set of values of one-type variables that have a common name. Arrays can be one-dimensional and multidimensional. The maximum admissible amount of dimensions in an array is four. Arrays of any data types are allowed."
      "Array element is a part of an array; it is an indexed variable having the same name and some value." ;
    what[2] = "Array-timeseries is an array with a predefined name (Open, Close, High, Low, Volume or Time), the elements of which contain values of corresponding characteristics of historic bars."
      "Data contained in arrays-timseries carry a very important information and are widely used in programming in MQL4. Each array-timeseries is a one-dimensional array and contains historic data about one certain bar characteristic. Each bar is characterized by an opening price Open[], closing price Close[], maximal price High[], minimal price Low[], volume Volume[] and opening time Time[]. For example, array-timeseries Open[] carries information about opening prices of all bars present in a security window: the value of array element Open[1] is the opening price of the first bar, Open[2] is the opening price of the second bar, etc. The same is true for other timeseries."
      "Zero bar is a current bar that has not fully formed yet. In a chart window the zero bar is the last right one."
      "Bars (and corresponding indexes of arrays-timeseries) counting is started from the zero bar. The values of array-timeseries elements with the index [0] are values that characterize the zero bar. For example, the value of Open[0] is the opening price of a zero bar. Fig. 61 shows numeration of bars and bar characteristics (reflected in a chart window when a mouse cursor is moved to a picture).";
    what[3] = "A large part of information processed by application programs is contained in arrays.";
   
   CompStream c;
   DecompStream d;
   
   for(uint i = 0; i < 1000; ++i)
   {
      uint const ind = i % ArraySize(what);
      char buffer[];
      c.compress(what[ind], buffer);
      
      string decompressed;
      d.decompress(buffer, decompressed, StringLen(what[ind]));
      
      assert (StringCompare(decompressed, what[ind]) == 0);
   }
      
}
//+------------------------------------------------------------------+
