

int start(){

   Print("Current timestamp: ", TimeToStr(Time[0], TIME_DATE | TIME_SECONDS), 
   " Previous timestamp: ", TimeToStr(Time[1], TIME_DATE | TIME_SECONDS),
   " Current Volume: ", Volume[0],
   " Previous Volume: ", Volume[1]);
   
   return(0);
}