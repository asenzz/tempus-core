//+------------------------------------------------------------------+
//|                                                  InternetLib.mqh |
//|                                 Copyright � 2010 www.fxmaster.de |
//|                                         Coding by Sergeev Alexey |
//+------------------------------------------------------------------+
#property copyright   "www.fxmaster.de  2010"
#property link        "www.fxmaster.de"
#property version     "1.00"
#property description "Library for work with wininet.dll"
#property library


//#include <stderror.mqh>
#include <stdlib.mqh>
#include <Hash.mqh>
#include <MqlLog.mqh>


#define RECEIVE_TIMEOUT 600000


#define FALSE 0

#define HINTERNET uint
#define BOOL int
#define INTERNET_PORT int
#define LPINTERNET_BUFFERS int
#define DWORD uint
#define DWORD_PTR uint
#define LPDWORD int&
#define LPVOID uchar&
#define LPSTR string&
#define LPCWSTR const string&
#define LPCTSTR const string&
#define LPTSTR string&
//LPCTSTR *  int
//LPVOID   uchar& +_[]

#import "Kernel32.dll"
DWORD GetLastError(const int);
#import

#import "wininet.dll"
DWORD InternetAttemptConnect(DWORD dwReserved);
HINTERNET InternetOpenW(LPCTSTR lpszAgent, DWORD dwAccessType, LPCTSTR lpszProxyName, LPCTSTR lpszProxyBypass, DWORD dwFlags);
HINTERNET InternetConnectW(const HINTERNET hInternet, LPCTSTR lpszServerName, INTERNET_PORT nServerPort, LPCTSTR lpszUsername, LPCTSTR lpszPassword, DWORD dwService, DWORD dwFlags, DWORD_PTR dwContext);
//HINTERNET HttpOpenRequestW(HINTERNET hConnect, LPCTSTR lpszVerb, LPCTSTR lpszObjectName, LPCTSTR lpszVersion, LPCTSTR lpszReferer, LPCTSTR lplpszAcceptTypes, uint/*DWORD*/ dwFlags, DWORD_PTR dwContext);
int HttpOpenRequestW(int hConnect, LPCTSTR lpszVerb, LPCTSTR lpszObjectName, string &lpszVersion, string &lpszReferer, ulong lplpszAcceptTypes, uint dwFlags, ulong dwContext);
BOOL HttpSendRequestW(HINTERNET hRequest, LPCTSTR lpszHeaders, DWORD dwHeadersLength, LPVOID lpOptional[], DWORD dwOptionalLength);
BOOL HttpQueryInfoW(HINTERNET hRequest, DWORD dwInfoLevel, LPVOID lpvBuffer[], LPDWORD lpdwBufferLength, LPDWORD lpdwIndex);
HINTERNET InternetOpenUrlW(HINTERNET hInternet, LPCTSTR lpszUrl, LPCTSTR lpszHeaders, DWORD dwHeadersLength, uint/*DWORD*/ dwFlags, DWORD_PTR dwContext);
BOOL InternetReadFile(HINTERNET hFile, LPVOID lpBuffer[], DWORD dwNumberOfBytesToRead, LPDWORD lpdwNumberOfBytesRead);
BOOL InternetCloseHandle(HINTERNET hInternet);
BOOL InternetSetOptionW(HINTERNET hInternet, DWORD dwOption, LPDWORD lpBuffer, DWORD dwBufferLength);
BOOL InternetQueryOptionW(HINTERNET hInternet, DWORD dwOption, LPDWORD lpBuffer, LPDWORD lpdwBufferLength);
BOOL InternetSetCookieW(LPCTSTR lpszUrl, LPCTSTR lpszCookieName, LPCTSTR lpszCookieData);
BOOL InternetGetCookieW(LPCTSTR lpszUrl, LPCTSTR lpszCookieName, LPVOID lpszCookieData[], LPDWORD lpdwSize);
#import

/*
#define OPEN_TYPE_PRECONFIG           1  // use confuguration by default
#define OPEN_TYPE_DIRECT              0
#define FLAG_KEEP_CONNECTION 0x00400000  // keep connection
#define FLAG_PRAGMA_NOCACHE  0x00000100  // no cache
#define FLAG_RELOAD          0x80000000  // reload page when request
#define FLAG_NO_CACHE_WRITE  0x04000000
#define SERVICE_HTTP                  3  // Http service
#define HTTP_QUERY_CONTENT_LENGTH     5
#define HTTP_QUERY_COOKIE             44
#define INTERNET_FLAG_SECURE          0x00800000
#define INTERNET_FLAG_IGNORE_CERT_CN_INVALID   0x00001000
#define INTERNET_FLAG_IGNORE_CERT_DATE_INVALID 0x00002000
*/

#define HTTP_QUERY_SET_COOKIE         43
#define HTTP_QUERY_STATUS_CODE        19
#define OPEN_TYPE_DIRECT              0
#define INTERNET_DEFAULT_HTTPS_PORT   443
#define INTERNET_FLAG_NO_CACHE_WRITE  0x04000000

#define OPEN_TYPE_PRECONFIG              0   // use default configuration
#define INTERNET_SERVICE_FTP    1 // Ftp service
#define INTERNET_SERVICE_HTTP    3 // Http service 
#define HTTP_QUERY_CONTENT_LENGTH    5

#define INTERNET_FLAG_PRAGMA_NOCACHE  0x00000100  // no caching of page
#define INTERNET_FLAG_KEEP_CONNECTION  0x00400000  // keep connection
#define INTERNET_FLAG_SECURE            0x00800000
#define INTERNET_FLAG_RELOAD    0x80000000  // get page from server when calling it
#define INTERNET_OPTION_SECURITY_FLAGS     31
#define INTERNET_OPTION_RECEIVE_TIMEOUT     6

#define ERROR_INTERNET_INVALID_CA        12045
#define INTERNET_FLAG_IGNORE_CERT_DATE_INVALID  0x00002000
#define INTERNET_FLAG_IGNORE_CERT_CN_INVALID    0x00001000
#define SECURITY_FLAG_IGNORE_CERT_CN_INVALID    INTERNET_FLAG_IGNORE_CERT_CN_INVALID
#define SECURITY_FLAG_IGNORE_CERT_DATE_INVALID  INTERNET_FLAG_IGNORE_CERT_DATE_INVALID
#define SECURITY_FLAG_IGNORE_UNKNOWN_CA         0x00000100
#define SECURITY_FLAG_IGNORE_WRONG_USAGE        0x00000200

//+------------------------------------------------------------------+

const uint RegularConnectionFlags = INTERNET_FLAG_KEEP_CONNECTION | INTERNET_FLAG_PRAGMA_NOCACHE;

//+------------------------------------------------------------------+

struct ParsedUrl
{
    string           Host;
    int              Port;
    bool             isSecure;
};

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool ParseUrl(string Url, ParsedUrl &purl)
{
    const string sep = ":";
    const ushort u_sep = StringGetCharacter(sep, 0);
    string result[];

    StringToLower(Url);
    StringTrimRight(Url);
    StringTrimLeft(Url);

    const int k = StringSplit(Url, u_sep, result);

    if( k == 0) {
        LOG_ERROR("ParseFqdn", "An empty Url passed");
        return false;
    }

    if( k < 0 ) {
        LOG_ERROR("ParseFqdn:", ErrorDescription(GetLastError()));
        return false;
    }

    if( k > 3 ) {
        LOG_ERROR("ParseFqdn:", "only two columns are allowed in the URL" + Url);
        return false;
    }

    long port = 80;
    string host = "";
    bool secure = false;
    int hostIndex = 0;

    if(StringCompare(result[0], "https") == 0) {
        port = INTERNET_DEFAULT_HTTPS_PORT;
        secure = true;

        hostIndex = 1;
        result[hostIndex] = StringSubstr(result[hostIndex], 2, StringLen(result[hostIndex]) - 2);

    } else if (StringCompare(result[0], "https") == 0) {
        hostIndex = 1;
        result[hostIndex] = StringSubstr(result[hostIndex], 2, StringLen(result[hostIndex]) - 2);
    }

    host = result[hostIndex];

    if( k > hostIndex + 1 )
        port = StringToInteger(result[hostIndex + 1]);

    purl.Host = host;
    purl.Port = int(port);
    purl.isSecure = secure;

    return true;
}


//+-------------------------------------------------------------------+
//|                                                                   |
//|                     M I S C E L L A N E O U S                     |
//|                                                                   |
//+-------------------------------------------------------------------+

#ifdef WITH_PERF_TIMERS
class PerfTimer
{
    uint             number;
    ulong            totalTime;
    ulong            start_time;
    string           tmrName;
    ulong            prnEvery;

    void             print();

public:
                     PerfTimer();
                    ~PerfTimer();
                     PerfTimer(const string &timerName, const ulong printEvery = 10);

    void             on();
    void             off();
    ulong            getAvg();
};
#else
class PerfTimer
{
public:
                     PerfTimer();
                    ~PerfTimer();
                     PerfTimer(const string &timerName, const ulong printEvery = 10);

    void             on();
    void             off();
    ulong            getAvg();
};
#endif

//+------------------------------------------------------------------+


//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class MqlNet
{

    int              Session;    // session descriptor
    int              Connect;    // connection descriptor
    string           Cookie;     // session cookie
    string           Host;       // host name
    int              Port;       // port
    bool             isSecure;   // https

public:
                     MqlNet();   // class constructor
                    ~MqlNet();   // destructor
    bool             Open(const string &aHost, const int aPort, const bool secure = false); // create session and open connection
    bool             Open(const string &Fqdn); // create session and open connection from FQDN like https://host:port
    void             Close();    // close session and connection
    bool             Request(const string &Verb, const string &Request, string &Out, const bool toFile = false, const string addData = "", const bool fromFile = false); // send request
    string           Post(const string &Path, const string &Body, long &StatusCode);

    string           ReadResponse(const int hRequest);

    bool             RpcCall(const string &Object, const string &Method, const Hash &Params, string &Response);
    bool             RpcCall(const string &Object, const string &Method, const Hash &Params[], string &Response);

    bool             OpenURL(const string &URL, string &Out, const bool toFile); // open page
    void             ReadPage(const int hRequest, string &Out, const bool toFile); // read page

    string           HttpQueryInfo(const int hRequest, const int HTTP_QUERY);

    string           GetServerCookieValue(const int hRequest); // get cookie from response
    long             GetStatusCode(const int hRequest); // get response status code
    long             GetContentSize(const int hURL); //get content size
    int              FileToArray(const string &FileName, uchar &data[]); // copying file to the array
    string           ToJSON(const Hash &params);
};


//------------------------------------------------------------------ MqlNet
void MqlNet::MqlNet()
{
// default values
    Session = -1;
    Connect = -1;
    Host = "";
    isSecure = false;
}

//------------------------------------------------------------------ ~MqlNet
void MqlNet::~MqlNet()
{
// close all descriptors
    Close();
}

//------------------------------------------------------------------ Open
bool MqlNet::Open(const string &aHost, const int aPort, const bool secure)
{
    if (aHost == "") {
        LOG_ERROR("MqlNet::Open", "Host is not specified");
        return (false);
    }

    if (!TerminalInfoInteger(TERMINAL_DLLS_ALLOWED)) {
        LOG_ERROR("MqlNet::Open", "DLL is not allowed");
        return (false);
    }
    if (Session > 0 || Connect > 0) Close();

    if (InternetAttemptConnect(0) != 0) {
        LOG_SYS_ERR("MqlNet::Open", "InternetAttemptConnect failed");
        return (false);
    }
    string UserAgent = "Mozilla";
    string nill = "";

    Session = (int) InternetOpenW(UserAgent, OPEN_TYPE_PRECONFIG, nill, nill, 0); // open session

    if (Session <= 0) {
        LOG_SYS_ERR("MqlNet::Open", "InternetOpenW failed");
        Close();
        return (false);
    }
    Connect = (int) InternetConnectW(Session, aHost, aPort, nill, nill, INTERNET_SERVICE_HTTP, 0, 0);
    if (Connect <= 0) {
        LOG_SYS_ERR("MqlNet::Open", "InternetConnectW failed");
        Close();
        return (false);
    }
    Host = aHost;
    Port = aPort;
    isSecure = secure;

    LOG_INFO("MqlNet::Open", (isSecure ? "Securely" : "Unsecurely") + " connected to " + string(Host) + ":" + string(Port));

    return (true);
}

//------------------------------------------------------------------ Open
bool MqlNet::Open(const string &Url)
{

    ParsedUrl purl;
    if (!ParseUrl(Url, purl)) return false;

    return Open(purl.Host, purl.Port, purl.isSecure);
}

//------------------------------------------------------------------ Close
void MqlNet::Close()
{
    if (Session > 0) InternetCloseHandle(Session);
    Session = -1;
    if (Connect > 0) InternetCloseHandle(Connect);
    Connect = -1;
    LOG_INFO ("MqlNet::Close", "Connection closed");
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
string MqlNet::Post(const string &Path, const string &Body, long &StatusCode)
{
    int res, hRequest, hSend;     // To receive the operation execution result
    char postData[];  // Data array to send POST requests
    char responseData[];
    string responseHeaders;
    string cookieHeader = "cppcms_session=";
    string Vers = "HTTP/1.1";
    string nill = "";
    string Verb = "POST";
    string head = "Content-Type: application/x-www-form-urlencoded";
    string acceptTypes = "";
    StatusCode = -1;

    string auth;

    ArrayResize(postData, StringLen(Body) - 1);
    StringToCharArray(Body, postData, 0, WHOLE_ARRAY, CP_UTF8);

    ResetLastError();

    if (Session <= 0 || Connect <= 0) {
        Close();
        if (!Open(Host, Port, isSecure)) {
            LOG_SYS_ERR("MqlNet::Post", "Open failed");
            Close();
            return (nill);
        }
    }

    uint recv_timeout = RECEIVE_TIMEOUT;
    InternetSetOptionW(Connect, INTERNET_OPTION_RECEIVE_TIMEOUT, recv_timeout, sizeof(recv_timeout));
    LOG_DEBUG("", "Opening request...");
    hRequest = HttpOpenRequestW(Connect, Verb, Path, Vers, nill, 0, RegularConnectionFlags | ( INTERNET_FLAG_SECURE * isSecure ), 0);

    if (hRequest <= 0) {
        LOG_SYS_ERR("MqlNet::Post", "HttpOpenRequestW failed");
        InternetCloseHandle(Connect);
        return (nill);
    }
    LOG_DEBUG("", "Sending request...");
// send request
    hSend = HttpSendRequestW(hRequest, head, StringLen(head), postData, ArraySize(postData));
    if (!hSend) {
        LOG_SYS_ERR("MqlNet::Post", "HttpSendRequestW failed " + string(hSend));
        InternetCloseHandle(hRequest);
        Close();
        return (nill);
    }

    StatusCode = GetStatusCode(hRequest);

    if (StatusCode == 401) {
        LOG_ERROR("MqlNet::Post", "Authorization failed");
        Close();
        return (nill);
    }

    if (StatusCode != 200) {
        LOG_ERROR("MqlNet::Post", "Server error: HTTP Code " + (string) StatusCode);
        Close();
        return (nill);
    }
    LOG_DEBUG("", "Setting cookie...");
    responseHeaders = GetServerCookieValue(hRequest);
    res = StringFind(responseHeaders, cookieHeader);

    if (res >= 0) { // SetCookie header received
        auth = StringSubstr(responseHeaders, res + StringLen(cookieHeader));
        LOG_DEBUG("MqlNet::Post", "SetCookie auth header received: " + auth);
        Cookie = cookieHeader + StringSubstr(auth, 0, StringFind(auth, ";"));
        string cookieURL = "http://" + Host + "/";
        string cookieName = "svrwave";
        if (!InternetSetCookieW(cookieURL, cookieName, Cookie)) {
            LOG_SYS_ERR("", "Failed to set cookie");
            Close();
            return (nill);
        }

    } else {
        LOG_DEBUG("MqlNet::Post", "Cookie not received!");
    }

    LOG_DEBUG("", "Reading response...");
    string Response = ReadResponse(hRequest);

    InternetCloseHandle(hSend);
    InternetCloseHandle(hRequest);

    LOG_DEBUG("", "Done...");
    return (Response);
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
string MqlNet::ToJSON(const Hash &params)
{

    string json = "{";

    int i = 0;
    for (HashLoop loop(GetPointer(params)); loop.hasNext(); loop.next()) {
        if (i++ > 0) {
            json += ",";
        }
        json += "\"" + loop.key() + "\":\"" + loop.val().toString() + "\"";
    }

    json += "}";
    LOG_VERBOSE("MqlNet::ToJSON", json);
    return (json);
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool MqlNet::RpcCall(const string &Object, const string &Method, const Hash &Params, string &Response)
{
#ifdef DEBUG_CONNECTOR
    static const string call_name = "RpcCall.1";
    static PerfTimer t1(call_name, 100);
    t1.on();
#endif

    static const string Headers = "Content-Type: application/json\r\n";
    static const string Verb = "POST";
    static string Vers = "HTTP/1.1";
    static string nill = "";
    const string Path = "web/" + Object + "/ajax";

    if (Session <= 0 || Connect <= 0) {
        Close();
        if (!Open(Host, Port, isSecure)) {
            Close();
            return (false);
        }
    }

    uint recv_timeout = RECEIVE_TIMEOUT;
    InternetSetOptionW(Connect, INTERNET_OPTION_RECEIVE_TIMEOUT, recv_timeout, sizeof(recv_timeout));

    int flags = INTERNET_FLAG_NO_CACHE_WRITE | INTERNET_FLAG_PRAGMA_NOCACHE | INTERNET_FLAG_KEEP_CONNECTION | ( INTERNET_FLAG_SECURE * isSecure ) ;
    string acceptTypes = "Accept: text/*\r\n";
#ifdef DEBUG_CONNECTOR
    t1.off();
    static const string call_name2 = "RpcCall.2";
    static PerfTimer t2(call_name, 100);
    t2.on();
#endif

// hRequest = HttpOpenRequestW(Connect, Verb, Object, Vers, nill, "", flags, NULL);
// hRequest = HttpOpenRequestW(Connect, Verb, Object, Vers, nill, 0,  RegularConnectionFlags | ( INTERNET_FLAG_SECURE * isSecure ), 0);
    const int hRequest = HttpOpenRequestW(Connect, Verb, Path, Vers, nill, 0, flags, 0);

    if (hRequest <= 0) {
        LOG_SYS_ERR("", "HttpOpenRequestW failed");
        Close();
        return false;
    }

    const string json = "{\"id\":1,\"method\":\"" + Method + "\",\"params\":[" + ToJSON(Params) + "]}";
    uchar jsonArray[];
    LOG_DEBUG("", "Sending " + json);
    StringToCharArray(json, jsonArray);
    
#ifdef DEBUG_CONNECTOR
    t2.off();
    static const string call_name3 = "RpcCall.3";
    static PerfTimer t3(call_name3, 100);
    t3.on();
#endif

    const int hSend = HttpSendRequestW(hRequest, Headers, StringLen(Headers), jsonArray, ArraySize(jsonArray) - 1);
    if (hSend <= 0) {
        LOG_SYS_ERR("MqlNet::RpcCall", "HttpSendRequestW failed " + string(hSend));
        Close();
        return false;
    }
    
#ifdef DEBUG_CONNECTOR
    t3.off();
    const string call_name4 = "RpcCall.4";
    static PerfTimer t4(call_name4, 100);
    t4.on();
#endif

    Response = ReadResponse(hRequest);
    LOG_DEBUG("", "Received response:" + Response);
    
#ifdef DEBUG_CONNECTOR
    t4.off();
    static const call_name5 = "RpcCall.5";
    static PerfTimer t5(call_name5, 100);
    t5.on();
#endif
    
    InternetCloseHandle(hSend);
    InternetCloseHandle(hRequest);

#ifdef DEBUG_CONNECTOR
    t5.off();
#endif

    return true;
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool MqlNet::RpcCall(const string &Object, const string &Method, const Hash &Params[], string &Response)
{
#ifdef DEBUG_CONNECTOR
    static const string call_name = "RpcCall.1";
    static PerfTimer t1(call_name, 100);
    t1.on();
#endif

    static const string Headers = "Content-Type: application/json\r\n";
    static const string Verb = "POST";
    static string Vers = "HTTP/1.1";
    static string nill = "";
    const string Path = "web/" + Object + "/ajax";

    if (Session <= 0 || Connect <= 0) {
        Close();
        if (!Open(Host, Port, isSecure)) {
            Close();
            return (false);
        }
    }

    uint recv_timeout = RECEIVE_TIMEOUT;
    InternetSetOptionW(Connect, INTERNET_OPTION_RECEIVE_TIMEOUT, recv_timeout, sizeof(recv_timeout));

    int flags = INTERNET_FLAG_NO_CACHE_WRITE | INTERNET_FLAG_PRAGMA_NOCACHE | INTERNET_FLAG_KEEP_CONNECTION | ( INTERNET_FLAG_SECURE * isSecure ) ;
    string acceptTypes = "Accept: text/*\r\n";
#ifdef DEBUG_CONNECTOR
    t1.off();
    static const string call_name2 = "RpcCall.2";
    static PerfTimer t2(call_name, 100);
    t2.on();
#endif

// hRequest = HttpOpenRequestW(Connect, Verb, Object, Vers, nill, "", flags, NULL);
// hRequest = HttpOpenRequestW(Connect, Verb, Object, Vers, nill, 0,  RegularConnectionFlags | ( INTERNET_FLAG_SECURE * isSecure ), 0);
    const int hRequest = HttpOpenRequestW(Connect, Verb, Path, Vers, nill, 0, flags, 0);

    if (hRequest <= 0) {
        LOG_SYS_ERR("", "HttpOpenRequestW failed");
        Close();
        return false;
    }

    string json;
    const int param_ct = ArraySize(Params);
    for (int i = 0; i < param_ct; ++i)
        json += "{\"id\":" + IntegerToString(i + 1) + ",\"method\":\"" + Method + "\",\"params\":[" + ToJSON(Params[i]) + "]} ";
    uchar jsonArray[];
    LOG_DEBUG("", "Sending " + json);
    StringToCharArray(json, jsonArray);
    
#ifdef DEBUG_CONNECTOR
    t2.off();
    static const string call_name3 = "RpcCall.3";
    static PerfTimer t3(call_name3, 100);
    t3.on();
#endif

    const int hSend = HttpSendRequestW(hRequest, Headers, StringLen(Headers), jsonArray, ArraySize(jsonArray) - 1);
    if (hSend <= 0) {
        LOG_SYS_ERR("MqlNet::RpcCall", "HttpSendRequestW failed " + string(hSend));
        Close();
        return false;
    }
    
#ifdef DEBUG_CONNECTOR
    t3.off();
    const string call_name4 = "RpcCall.4";
    static PerfTimer t4(call_name4, 100);
    t4.on();
#endif

    Response = ReadResponse(hRequest);
    LOG_DEBUG("", "Received response:" + Response);
    
#ifdef DEBUG_CONNECTOR
    t4.off();
    static const call_name5 = "RpcCall.5";
    static PerfTimer t5(call_name5, 100);
    t5.on();
#endif
    
    InternetCloseHandle(hSend);
    InternetCloseHandle(hRequest);

#ifdef DEBUG_CONNECTOR
    t5.off();
#endif

    return true;
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
string MqlNet::ReadResponse(const int hRequest)
{
    int BUFSIZ = 1024;
    int lReturn;
    uchar sBuffer[];
    ArrayResize(sBuffer, BUFSIZ, BUFSIZ);

    int totalData = 0;

    string Response = "";

    while (InternetReadFile(hRequest, sBuffer, BUFSIZ, lReturn) != false) {
        if (lReturn == 0) {
            break;
        }

        string buf = CharArrayToString(sBuffer);
        StringAdd(Response, buf);

        // update total # of elts in sData
        totalData += lReturn;
    }

    return Response;
}

//------------------------------------------------------------------ Request
bool MqlNet::Request(const string &Verb, const string &Object, string &Out, const bool toFile = false, const string addData = "", const bool fromFile = false)
{
    if (toFile && Out == "") {
        LOG_ERROR("MqlNet::Request", "File is not specified");
        return (false);
    }
    uchar data[];
    int hRequest, hSend;
    static string Vers = "HTTP/1.1";
    static string nill = "";
    if (fromFile) {
        if (FileToArray(addData, data) < 0) {
            LOG_SYS_ERR("MqlNet::Request", "FileToArray failed");
            return (false);
        }
    } // file loaded to the array
    else StringToCharArray(addData, data);

    if (Session <= 0 || Connect <= 0) {
        Close();
        if (!Open(Host, Port, isSecure)) {
            Close();
            return (false);
        }
    }
// create descriptor for the request
    string acceptTypes = "";
    hRequest = HttpOpenRequestW(Connect, Verb, Object, Vers, nill, 0, RegularConnectionFlags | ( INTERNET_FLAG_SECURE * isSecure ), 0);
    if (hRequest <= 0) {
        LOG_SYS_ERR("", "HttpOpenRequestW failed");
        InternetCloseHandle(Connect);
        return (false);
    }
// send request
// request headed
    static const string head = "Content-Type: application/x-www-form-urlencoded";
// send request
    hSend = HttpSendRequestW(hRequest, head, StringLen(head), data, ArraySize(data) - 1);
    if (hSend <= 0) {
        LOG_SYS_ERR("", "HttpSendRequestW failed");
        InternetCloseHandle(hRequest);
        Close();
    }
// read page
    ReadPage(hRequest, Out, toFile);
// close all descriptors
    InternetCloseHandle(hRequest);
    InternetCloseHandle(hSend);
    return (true);
}

//------------------------------------------------------------------ OpenURL
bool MqlNet::OpenURL(const string &URL, string &Out, const bool toFile)
{
    static const string nill = "";
    if (Session <= 0 || Connect <= 0) {
        Close();
        if (!Open(Host, Port, isSecure)) {
            Close();
            return (false);
        }
    }
    uint hURL = InternetOpenUrlW(Session, URL, nill, 0, INTERNET_FLAG_RELOAD | INTERNET_FLAG_PRAGMA_NOCACHE, 0);
    if (hURL < 1) {
        LOG_SYS_ERR("MqlNet::OpenURL", "InternetOpenUrlW failed");
        return (false);
    }
// read to Out
    ReadPage(hURL, Out, toFile);
// close
    InternetCloseHandle(hURL);
    return (true);
}

//------------------------------------------------------------------ ReadPage
void MqlNet::ReadPage(const int hRequest, string &Out, const bool toFile)
{
// read page
    uchar ch[100];
    string toStr = "";
    int dwBytes, h;
    while (InternetReadFile(hRequest, ch, 100, dwBytes)) {
        if (dwBytes <= 0) break;
        toStr = toStr + CharArrayToString(ch, 0, dwBytes);
    }
    if (toFile) {
        h = FileOpen(Out, FILE_BIN | FILE_WRITE);
        FileWriteString(h, toStr);
        FileClose(h);
    } else Out = toStr;
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
string MqlNet::HttpQueryInfo(const int hRequest, const int HTTP_QUERY)
{
    int len = 2048, ind = 0;
    uchar buf[2048];
    int Res = HttpQueryInfoW(hRequest, HTTP_QUERY, buf, len, ind);
    if (Res == 0) {
//        LOG_SYS_ERR("MqlNet::HttpQueryInfo", "HttpQueryInfoW failed");
        return ("");
    }
    string s;
    for (int i = 0; i < len; i++) {
        uchar uc = (uchar) buf[i];
        if (uc != 0)
            StringAdd(s, CharToString(uc));
    }
    if (StringLen(s) <= 0) return ("");
    return (s);
}

//------------------------------------------------------------------ GetServerCookieValue
string MqlNet::GetServerCookieValue(const int hRequest)
{
    return (HttpQueryInfo(hRequest, HTTP_QUERY_SET_COOKIE));
}

//------------------------------------------------------------------ GetServerCookieValue
long MqlNet::GetStatusCode(const int hRequest)
{
    return (StringToInteger(HttpQueryInfo(hRequest, HTTP_QUERY_STATUS_CODE)));
}

//------------------------------------------------------------------ GetContentSize
long MqlNet::GetContentSize(const int hRequest)
{
    return (StringToInteger(HttpQueryInfo(hRequest, HTTP_QUERY_CONTENT_LENGTH)));
}

//----------------------------------------------------- FileToArray
int MqlNet::FileToArray(const string &FileName, uchar &data[])
{
    int h, i, size;
    h = FileOpen(FileName, FILE_BIN | FILE_READ);
    if (h < 0) return (-1);
    FileSeek(h, 0, SEEK_SET);
    size = (int) FileSize(h);
    ArrayResize(data, (int) size);
    for (i = 0; i < size; i++) {
        data[i] = (uchar) FileReadInteger(h, CHAR_VALUE);
    }
    FileClose(h);
    return (size);
}
//+------------------------------------------------------------------+


/**/

#ifdef WITH_PERF_TIMERS
PerfTimer::PerfTimer(): number(0), totalTime(0), start_time(0), prnEvery(0)
{
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
PerfTimer::PerfTimer(const string &timerName, const ulong printEvery): number(0), totalTime(0), start_time(0), tmrName(timerName), prnEvery(printEvery)
{
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
PerfTimer::~PerfTimer()
{
    print();
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void PerfTimer::on()
{
    start_time = GetMicrosecondCount();
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void PerfTimer::off()
{
    totalTime += GetMicrosecondCount() - start_time;
    ++number;

    if(prnEvery && number % prnEvery == 0)
        print();
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
ulong PerfTimer::getAvg(void)
{
    if(number == 0)
        return 0;
    return totalTime / number;
}


//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void PerfTimer::print()
{
    if(StringLen(tmrName) > 0)
        LOG_PERF ("Timer: " + tmrName, "Number: " + string(number) + " Avg: " + string(getAvg()));
}
#else
PerfTimer::PerfTimer() {}
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
PerfTimer::PerfTimer(const string &timerName, const ulong printEvery) {}
PerfTimer::~PerfTimer() {}
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void PerfTimer::on() {}
void PerfTimer::off() {}
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
ulong PerfTimer::getAvg(void)
{
    return 0;
}
#endif;


//+------------------------------------------------------------------+
