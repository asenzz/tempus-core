//+------------------------------------------------------------------+
//|                                               TempusMVCTests.mq4 |
//|                        Copyright 2016, MetaQuotes Software Corp. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+

#include <TempusMVC.mqh>

void OnStart()
{
   TestTempusGraph();
   
}
//+------------------------------------------------------------------+

#define ASSERT(expr, message, file, line) if(!(expr)) {Print("Assertion failed: " + message + " in " + file + ": " +string (line)); int p[1]; p[2] = 0;}

#define assert(expr) ASSERT(expr, #expr, __FILE__, __LINE__)

void TestTempusGraph()
{
   TempusGraph gr;
   gr.init(5);
   
   datetime strt = StrToTime("1980.01.10 10:00:00");
   
   TempusFigure f; f.op = 0; f.hi = 1; f.lo = 2; f.cl = 3; f.tm = strt;
   gr.redraw(f, strt);
   
   assert (gr.getFirstTime() == strt);
   
   assert (gr.getFigure(4, f));
   assert (gr.getFigure(5, f) == false);
   
   assert (gr.getFigure(0, f));
   assert (f.op == 0 && f.hi == 1 && f.lo == 2 && f.cl == 3 && f.tm == strt);

   TempusFigure f1; f1.op = 0.1; f1.hi = 1.1; f1.lo = 2.1; f1.cl = 3.1; ; f1.tm = strt + 60 *1;
   gr.redraw(f1, strt);
   
   assert (gr.getFigure(0, f));
   assert (f.op == 0 && f.hi == 1 && f.lo == 2 && f.cl == 3 && f.tm == strt);
   assert (gr.getFigure(1, f));
   assert (f.op == 0.1 && f.hi == 1.1 && f.lo == 2.1 && f.cl == 3.1 && f.tm == strt+60*1);

   TempusFigure f2; f2.op = 0.2; f2.hi = 1.2; f2.lo = 2.2; f2.cl = 3.2; ; f2.tm = strt + 60 *2;
   TempusFigure f3; f3.op = 0.3; f3.hi = 1.3; f3.lo = 2.3; f3.cl = 3.3; ; f3.tm = strt + 60 *3;
   TempusFigure f4; f4.op = 0.4; f4.hi = 1.4; f4.lo = 2.4; f4.cl = 3.4; ; f4.tm = strt + 60 *4;
   
   gr.redraw(f2, strt);
   gr.redraw(f3, strt);
   gr.redraw(f4, strt);

   assert (gr.getFigure(2, f));
   assert (f.op == 0.2 && f.hi == 1.2 && f.lo == 2.2 && f.cl == 3.2 && f.tm == strt+60*2);

   assert (gr.getFigure(3, f));
   assert (f.op == 0.3 && f.hi == 1.3 && f.lo == 2.3 && f.cl == 3.3 && f.tm == strt+60*3);

   assert (gr.getFigure(4, f));
   assert (f.op == 0.4 && f.hi == 1.4 && f.lo == 2.4 && f.cl == 3.4 && f.tm == strt+60*4);
   
   strt += 60;
   
   gr.redraw(strt);   

   assert (gr.getFigure(0, f));
   assert (f.op == 0.1 && f.hi == 1.1 && f.lo == 2.1 && f.cl == 3.1 && f.tm == strt+60*0);
   
   assert (gr.getFigure(1, f));
   assert (f.op == 0.2 && f.hi == 1.2 && f.lo == 2.2 && f.cl == 3.2 && f.tm == strt+60*1);

   assert (gr.getFigure(2, f));
   assert (f.op == 0.3 && f.hi == 1.3 && f.lo == 2.3 && f.cl == 3.3 && f.tm == strt+60*2);

   assert (gr.getFigure(3, f));
   assert (f.op == 0.4 && f.hi == 1.4 && f.lo == 2.4 && f.cl == 3.4 && f.tm == strt+60*3);
   
   assert (gr.getFigure(4, f));
   assert (f.tm == 0);
   
}