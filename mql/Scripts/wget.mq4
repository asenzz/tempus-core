//+------------------------------------------------------------------+
//|                                                         http.mq4 |
//|                                Copyright © 2008, Berkers Trading |
//|                                                http://quotar.com |
//+------------------------------------------------------------------+
#property copyright "Copyright © 2008, Berkers Trading"
#property link      "http://quotar.com"
 
//+------------------------------------------------------------------+
//| DLL imports                                                      |
//+------------------------------------------------------------------+

#import "wininet.dll"
int InternetAttemptConnect (int x);

int InternetOpenA (
   string lpszAgent,
   int dwAccessType,
   string lpszProxyName,
   string lpszProxyBypass,
   int dwFlags
);
 
int InternetConnectA(
   int hInternet,
   string lpszServerName,
   int nServerPort,
   string lpszUsername,
   string lpszPassword,
   int dwService,
   int dwFlags,
   int dwContext
);

int InternetOpenUrlA(
   int hInternetSession, 
   string sUrl, 
   string sHeaders = "", 
   int lHeadersLength = 0,
   int lFlags = 0, 
   int lContext = 0
);
 
int HttpOpenRequestA(
   int hConnect,
   string lpszVerb,
   string lpszObjectName,
   string lpszVersion,
   string lpszReferer,
   string lplpszAcceptTypes,
   int dwFlags,
   int dwContext
);
 
bool HttpSendRequestA(
   int hRequest,
   string lpszHeaders,
   int dwHeadersLength,
   string lpOptional,
   int dwOptionalLength
);
 
bool InternetReadFile(
   int hFile,
   string lpBuffer,
   int dwNumberOfBytesToRead,
   int &lpdwNumberOfBytesRead[]
);

bool HttpAddRequestHeadersA(
   int hRequest,
   string lpszHeaders,
   int dwHeadersLength,
   int dwModifiers
);
 
bool InternetCloseHandle(
   int hInternet
);
 
#import
 
//#include <stdfunctions.mqh>
/*
InternetOpen 
 
dwAccessType
INTERNET_OPEN_TYPE_DIRECT
INTERNET_OPEN_TYPE_PRECONFIG
INTERNET_OPEN_TYPE_PRECONFIG_WITH_NO_AUTOPROXY
INTERNET_OPEN_TYPE_PROXY
 
dwFlags
INTERNET_FLAG_ASYNC
INTERNET_FLAG_FROM_CACHE
INTERNET_FLAG_OFFLINE
 
*/
 
//InternetConnect - nServerPort
#define INTERNET_DEFAULT_FTP_PORT      21
#define INTERNET_DEFAULT_GOPHER_PORT   70
#define INTERNET_DEFAULT_HTTP_PORT     80
#define INTERNET_DEFAULT_HTTPS_PORT    443
#define INTERNET_DEFAULT_SOCKS_PORT    1080
#define INTERNET_INVALID_PORT_NUMBER   0
 
//InternetConnect - dwService
#define INTERNET_SERVICE_FTP     1
#define INTERNET_SERVICE_GOPHER  2
#define INTERNET_SERVICE_HTTP    3
 
//HttpOpenRequest - dwFlags
/*#define INTERNET_FLAG_CACHE_IF_NET_FAIL
#define INTERNET_FLAG_HYPERLINK
#define INTERNET_FLAG_IGNORE_CERT_CN_INVALID
#define INTERNET_FLAG_IGNORE_CERT_DATE_INVALID
#define INTERNET_FLAG_IGNORE_REDIRECT_TO_HTTP
#define INTERNET_FLAG_IGNORE_REDIRECT_TO_HTTPS
#define INTERNET_FLAG_KEEP_CONNECTION
#define INTERNET_FLAG_NEED_FILE
#define INTERNET_FLAG_NO_AUTH
#define INTERNET_FLAG_NO_AUTO_REDIRECT*/
#define INTERNET_FLAG_NO_CACHE_WRITE            0x04000000
/*#define INTERNET_FLAG_NO_COOKIES
#define INTERNET_FLAG_NO_UI*/
#define INTERNET_FLAG_PRAGMA_NOCACHE            0x00000100
#define INTERNET_FLAG_RELOAD                    0x80000000
/*#define INTERNET_FLAG_RESYNCHRONIZE
#define INTERNET_FLAG_SECURE*/
 
void StringExplode(string sDelimiter, string sExplode, string &sReturn[]){
   
   int ilBegin = -1,ilEnd = 0;
   int ilElement=0;
   while (ilEnd != -1){
      ilEnd = StringFind(sExplode, sDelimiter, ilBegin+1);
      ArrayResize(sReturn,ilElement+1);
      sReturn[ilElement] = "";     
      if (ilEnd == -1){
         if (ilBegin+1 != StringLen(sExplode)){
            sReturn[ilElement] = StringSubstr(sExplode, ilBegin+1, StringLen(sExplode));
         }
      } else { 
         if (ilBegin+1 != ilEnd){
            sReturn[ilElement] = StringSubstr(sExplode, ilBegin+1, ilEnd-ilBegin-1);
         }
      }      
      ilBegin = StringFind(sExplode, sDelimiter,ilEnd);  
      ilElement++;    
   }
}

string wpost(string sURL, string headers)
{
   string sURLexplode[];
   StringExplode("/",sURL, sURLexplode);
   int hInternetOpen = InternetOpenA ("mql4",1,"","",0);
   int hInternetConnect = InternetConnectA(hInternetOpen, sURLexplode[2], INTERNET_DEFAULT_HTTP_PORT, "", "", INTERNET_SERVICE_HTTP, 0, 0);
   
   int hHttpOpenRequest = HttpOpenRequestA(hInternetConnect, "POST", sURLexplode[3], "", 0, 0, 0, 0);
              
   
   if(! HttpAddRequestHeadersA(hHttpOpenRequest, "Content-Type: application/x-www-form-urlencoded", -1, 0))
      Alert("Cannot add request headers!");           
   
   
   bool bHttpSendRequestA = HttpSendRequestA(hHttpOpenRequest, "", 0, headers, StringLen(headers));
   string mContent = "";
   int lpdwNumberOfBytesRead[1];
   string sResult;
   while (InternetReadFile(hHttpOpenRequest, mContent, 255, lpdwNumberOfBytesRead) != FALSE) {
        if (lpdwNumberOfBytesRead[0] == 0) break;
        sResult = StringConcatenate(sResult , StringSubstr(mContent, 0, lpdwNumberOfBytesRead[0]));
   } 
   bool bRes = InternetCloseHandle(hInternetOpen); 
   return(sResult);
}
 
string wget(string sURL, string args = ""){
   
   string sURLexplode[];
   StringExplode("/",sURL, sURLexplode);
   int rv = InternetAttemptConnect(0);
   if(rv != 0){ Alert("InternetAttemptConnect() failed!");  return(0);  }
     
   int hInternetOpen = InternetOpenA ("mql4",0,"","",0);
   int hInternetConnect = InternetConnectA(hInternetOpen, sURL, INTERNET_DEFAULT_HTTP_PORT, "", "", INTERNET_SERVICE_HTTP, 0, 0);
   if(hInternetConnect <= 0) { Alert("InternetConnect failed!"); return("");}
   
   int hHttpOpenRequest = HttpOpenRequestA(hInternetConnect, "GET", "", 0, 0, 0, 
              INTERNET_FLAG_NO_CACHE_WRITE | INTERNET_FLAG_PRAGMA_NOCACHE | INTERNET_FLAG_RELOAD , 0);
              
   if(hHttpOpenRequest <= 0) { Alert("HttpOpenRequestA failed!"); return("");}
   
   Print("HttpRequest: ", hHttpOpenRequest);
   
   if(HttpSendRequestA(hHttpOpenRequest, "", 0, 0, 0)){
      Print("Request sent!");
   }
   else {      
      Alert("Request NOT sent! Error: ", GetLastError());
      return("");
   }
   
   //int hInternetSession = InternetOpenA("Microsoft Internet Explorer", 0, "", "", 0);
   
   /*if(hInternetOpen <= 0){
       Alert("InternetOpenA() failed!");
       return(0);         
     }
   int hURL = InternetOpenUrlA(hInternetOpen, sURL, headers, StringLen(headers), 0, 0);
   if(hURL <= 0)
   {
      Alert("InternetOpenUrlA() failed");
      InternetCloseHandle(hInternetOpen);
      return(0);         
   }  */
   
   string mContent = "";
   int lpdwNumberOfBytesRead[1];
   string sResult;
   while (InternetReadFile(hHttpOpenRequest, mContent, 255, lpdwNumberOfBytesRead) != FALSE) {
        if (lpdwNumberOfBytesRead[0] == 0) break;
        sResult = StringConcatenate(sResult , StringSubstr(mContent, 0, lpdwNumberOfBytesRead[0]));
   } 
   InternetCloseHandle(hInternetOpen); 
   InternetCloseHandle(hHttpOpenRequest);
   return(sResult);
}

int start(){
   //string headers = "test=true&something=value";
   //int HttpOpen = InternetOpenA("HTTP_Client_Sample", 0, "", "", 0); 
   //int HttpConnect = InternetConnectA(HttpOpen, "", 80, "", "", 3, 0, 1); 
   //int HttpRequest = InternetOpenUrlA(HttpOpen, "http://vg-pc/mql_query", headers, StringLen(headers), 0, 0);
   Print(wget("http://localhost/"));
   //Alert("Get complete!");
   //Print(wpost("http://vg-pc/mql_query?test=2", "test=1"));
   //Alert("POST complete!");
   
   
   
   return(1);
}