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

#define WININET_DLL

#define FALSE 0

#define HINTERNET uint
#define BOOL int
#define BOOLEAN uchar
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
HINTERNET InternetConnectW(HINTERNET hInternet, LPCTSTR lpszServerName, INTERNET_PORT nServerPort, LPCTSTR lpszUsername, LPCTSTR lpszPassword, DWORD dwService, DWORD dwFlags, DWORD_PTR dwContext);
//HINTERNET HttpOpenRequestW(HINTERNET hConnect, LPCTSTR lpszVerb, LPCTSTR lpszObjectName, LPCTSTR lpszVersion, LPCTSTR lpszReferer, LPCTSTR lplpszAcceptTypes, uint/*DWORD*/ dwFlags, DWORD_PTR dwContext);
int HttpOpenRequestW(int hConnect, LPCTSTR lpszVerb, LPCTSTR lpszObjectName, string &lpszVersion, ulong lpszReferer, ulong lplpszAcceptTypes, uint dwFlags, ulong dwContext);
BOOL HttpSendRequestW(HINTERNET hRequest, LPCTSTR lpszHeaders, DWORD dwHeadersLength, LPVOID lpOptional[], DWORD dwOptionalLength);
BOOL HttpQueryInfoW(HINTERNET hRequest, DWORD dwInfoLevel, LPVOID lpvBuffer[], LPDWORD lpdwBufferLength, LPDWORD lpdwIndex);
HINTERNET InternetOpenUrlW(HINTERNET hInternet, LPCTSTR lpszUrl, LPCTSTR lpszHeaders, DWORD dwHeadersLength, uint dwFlags, DWORD_PTR dwContext);
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

struct ParsedUrl {
    string           session_host;
    int              session_port;
    bool             session_secure;
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

    purl.session_host = host;
    purl.session_port = int(port);
    purl.session_secure = secure;

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
static const string C_headers_json = "Content-Type: application/json\r\n";
static const int C_len_headers_json = StringLen(C_headers_json);
static const string C_post_verb = "POST";
static string C_http_ver = "HTTP/1.1";
static string C_empty_string = "";
static const int C_web_request_timeout = 120;
static string cookieHeader = "cppcms_session=";
static string head_www_form = "Content-Type: application/x-www-form-urlencoded";
static const int len_head = StringLen(head_www_form);
static string acceptTypes[] = { "Accept: text/*\r\n" };
static string UserAgent = "Mozilla";

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class MqlNet
{
    int              Session;    // session descriptor
    int              Connect;    // connection descriptor
    string           session_cookie;     // session cookie
    string           session_host;       // host name
    int              session_port;       // port
    bool             session_secure;   // https
    string           session_url_prefix;

public:
                     MqlNet();   // class constructor
                    ~MqlNet();   // destructor
    bool             Open(const string &host, const int port, const bool secure = false); // create session and open connection
    bool             Open(const string &Fqdn); // create session and open connection from FQDN like https://host:port
    void             Close();    // close session and connection
    bool             Request(const string &this_verb, const string &request_body, string &response, const bool toFile = false, const string addData = "", const bool fromFile = false); // send request
    string           Post(const string &this_path, const string &this_body, long &http_status_code);

    string           ReadResponse(const int hRequest);

    bool             RpcCall_(string this_path, const string &Method, const Hash &Params, string &response);
    bool             RpcCall(const string &this_path, const string &Method, const Hash &Params, string &response);
    bool             RpcCall(const string &this_path, const string &Method, const Hash &Params[], string &response);

    bool             OpenURL(const string &URL, string &response, const bool toFile); // open page
    void             ReadPage(const int hRequest, string &response, const bool toFile); // read page

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
    session_host = "";
    session_secure = false;
}

//------------------------------------------------------------------ ~MqlNet
void MqlNet::~MqlNet()
{
// close all descriptors
    Close();
}

//------------------------------------------------------------------ Open
bool MqlNet::Open(const string &host, const int port, const bool secure)
{
    if (host == "") {
        LOG_ERROR("", "session_host is not specified");
        return false;
    }
    
#ifdef WININET_DLL
    if (!TerminalInfoInteger(TERMINAL_DLLS_ALLOWED)) {
        LOG_ERROR("", "DLL is not allowed");
        return false;
    }
    if (Session > 0 || Connect > 0) Close();

    if (InternetAttemptConnect(0) != 0) {
        LOG_SYS_ERR("", "InternetAttemptConnect failed");
        return false;
    }

    Session = (int) InternetOpenW(UserAgent, OPEN_TYPE_PRECONFIG, C_empty_string, C_empty_string, 0); // open session

    if (Session <= 0) {
        LOG_SYS_ERR("", "InternetOpenW failed");
        Close();
        return false;
    }
    Connect = (int) InternetConnectW(Session, host, port, C_empty_string, C_empty_string, INTERNET_SERVICE_HTTP, 0, 0);
    if (Connect <= 0) {
        LOG_SYS_ERR("", "InternetConnectW failed");
        Close();
        return false;
    }
#endif
    session_host = host;
    session_port = port;
    session_secure = secure;
    session_url_prefix = (secure ? "https://" : "http://") + session_host + ":" + IntegerToString(session_port) + "/";
    LOG_INFO("", (session_secure ? "Securely" : "Unsecurely") + " connected to " + session_host + ", port " + IntegerToString(session_port));
    return true;
}

//------------------------------------------------------------------ Open
bool MqlNet::Open(const string &Url)
{

    ParsedUrl purl;
    if (!ParseUrl(Url, purl)) return false;
    return Open(purl.session_host, purl.session_port, purl.session_secure);
}

//------------------------------------------------------------------ Close
void MqlNet::Close()
{
#ifdef WININET_DLL
    if (Session > 0) InternetCloseHandle(Session);
    Session = -1;
    if (Connect > 0) InternetCloseHandle(Connect);
    Connect = -1;
    LOG_INFO ("", "Connection closed");
#endif
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
string MqlNet::Post(const string &this_path, const string &this_body, long &http_status_code)
{
#ifdef WININET_DLL
    int res, hRequest, hSend;       // To receive the operation execution result
    char postData[];                // Data array to send POST requests
    char responseData[];
    string responseHeaders;

    http_status_code = -1;

    string auth;

    ArrayResize(postData, StringLen(this_body) - 1);
    StringToCharArray(this_body, postData, 0, WHOLE_ARRAY, CP_UTF8);
    postData[ArraySize(postData) - 1] = ' ';

    ResetLastError();

    if (Session <= 0 || Connect <= 0) {
        Close();
        if (!Open(session_host, session_port, session_secure)) {
            LOG_SYS_ERR("MqlNet::Post", "Open failed");
            Close();
            return C_empty_string;
        }
    }

    uint recv_timeout = RECEIVE_TIMEOUT;
    InternetSetOptionW(Connect, INTERNET_OPTION_RECEIVE_TIMEOUT, recv_timeout, sizeof(recv_timeout));
    LOG_DEBUG("", "Opening request...");
    hRequest = HttpOpenRequestW(Connect, C_post_verb, this_path, C_http_ver, 0, 0, RegularConnectionFlags | ( INTERNET_FLAG_SECURE * session_secure ), 0);

    if (hRequest <= 0) {
        LOG_SYS_ERR("MqlNet::Post", "HttpOpenRequestW failed");
        InternetCloseHandle(Connect);
        return C_empty_string;
    }
    LOG_DEBUG("", "Sending request...");
// send request
    hSend = HttpSendRequestW(hRequest, head_www_form, len_head, postData, ArraySize(postData));
    if (!hSend) {
        LOG_SYS_ERR("", "HttpSendRequestW failed " + string(hSend));
        InternetCloseHandle(hRequest);
        Close();
        return C_empty_string;
    }

    http_status_code = GetStatusCode(hRequest);

    if (http_status_code == 401) {
        LOG_ERROR("", "Authorization failed");
        Close();
        return C_empty_string;
    }

    if (http_status_code != 200) {
        LOG_ERROR("", "Server error: HTTP Code " + (string) http_status_code);
        Close();
        return C_empty_string;
    }
    LOG_DEBUG("", "Setting cookie...");
    responseHeaders = GetServerCookieValue(hRequest);
    res = StringFind(responseHeaders, cookieHeader);

    if (res >= 0) { // SetCookie header received
        auth = StringSubstr(responseHeaders, res + StringLen(cookieHeader));
        LOG_DEBUG("", "SetCookie auth header received: " + auth);
        session_cookie = cookieHeader + StringSubstr(auth, 0, StringFind(auth, ";"));
        const string cookieURL = "http://" + session_host + "/";
        static const string cookieName = "svrwave";
        if (!InternetSetCookieW(cookieURL, cookieName, session_cookie)) {
            LOG_SYS_ERR("", "Failed to set cookie");
            Close();
            return C_empty_string;
        }

    } else
        LOG_DEBUG("", "session_cookie not received!");

    LOG_DEBUG("", "Reading response...");
    const string response = ReadResponse(hRequest);

    InternetCloseHandle(hSend);
    InternetCloseHandle(hRequest);

    LOG_DEBUG("", "Done...");
#else
    string response_headers, cookie_value;
    char response_chr[], body_chr[];
    StringToCharArray(this_body, body_chr);
    body_chr[ArraySize(body_chr) - 1] = ' ';
    // http_status_code = WebRequest("POST", session_url_prefix + this_path, cookie_value, C_empty_string, C_web_request_timeout, body_chr, ArraySize(body_chr), response_chr, response_headers);
    http_status_code = WebRequest("POST", session_url_prefix + this_path, head_www_form, C_web_request_timeout, body_chr, response_chr, response_headers);
    const string response = CharArrayToString(response_chr);
    LOG_VERBOSE("", "Received " + response + ", headers " + response_headers + ", code " + IntegerToString(http_status_code) + ", path " + this_path + ", body " + this_body + ", cookie " + cookie_value);
#endif
    return response;
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
    LOG_VERBOSE("", json);
    return json;
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool MqlNet::RpcCall_(string this_path, const string &Method, const Hash &Params, string &response)
{
    const string json = "{\"id\":1,\"method\":\"" + Method + "\",\"params\":[" + ToJSON(Params) + "]}";
    this_path = "web/" + this_path + "/ajax";
    
#ifdef WININET_DLL
    static int flags = INTERNET_FLAG_NO_CACHE_WRITE | INTERNET_FLAG_PRAGMA_NOCACHE | INTERNET_FLAG_KEEP_CONNECTION | ( INTERNET_FLAG_SECURE * session_secure ) ;

    uint retries = 0;
// hRequest = HttpOpenRequestW(Connect, C_post_verb, this_path, C_http_ver, C_empty_string, "", flags, NULL);
// hRequest = HttpOpenRequestW(Connect, C_post_verb, this_path, C_http_ver, C_empty_string, 0,  RegularConnectionFlags | ( INTERNET_FLAG_SECURE * session_secure ), 0);
    static const uint C_max_retries = 5;
    static uint recv_timeout = RECEIVE_TIMEOUT;
    static const int sizeof_recv_timeout = sizeof(recv_timeout);

#ifdef DEBUG_CONNECTOR
    static const string call_name = "RpcCall.1";
    static PerfTimer t1(call_name, 100);
    t1.on();
#endif

    if (Session <= 0 || Connect <= 0) {
        Close();
        if (!Open(session_host, session_port, session_secure)) {
            Close();
            return false;
        }
    }

    InternetSetOptionW(Connect, INTERNET_OPTION_RECEIVE_TIMEOUT, recv_timeout, sizeof_recv_timeout);

#ifdef DEBUG_CONNECTOR
    t1.off();
    static const string call_name2 = "RpcCall.2";
    static PerfTimer t2(call_name, 100);
    t2.on();
#endif

    const uint hRequest = HttpOpenRequestW(Connect, C_post_verb, this_path, C_http_ver, 0, 0, flags, 0);
    if (!hRequest) {
        LOG_SYS_ERR("", "HttpOpenRequestW failed");
        Close();
        return false;
    }

    uchar jsonArray[];
    LOG_DEBUG("", "Sending " + json);
    StringToCharArray(json, jsonArray);
    jsonArray[ArraySize(jsonArray) - 1] = ' ';
    
#ifdef DEBUG_CONNECTOR
    t2.off();
    static const string call_name3 = "RpcCall.3";
    static PerfTimer t3(call_name3, 100);
    t3.on();
#endif

    retries = 0;
    const bool r_send = HttpSendRequestW(hRequest, C_headers_json, C_len_headers_json, jsonArray, ArraySize(jsonArray));
    if (r_send == false) {
        LOG_SYS_ERR("", "HttpSendRequestW failed " + string(r_send));
        Close();
        return false;
    }

#ifdef DEBUG_CONNECTOR
    t3.off();
    const string call_name4 = "RpcCall.4";
    static PerfTimer t4(call_name4, 100);
    t4.on();
#endif

    response = ReadResponse(hRequest);
    LOG_DEBUG("", "Received response:" + response);

#ifdef DEBUG_CONNECTOR
    t4.off();
    static const call_name5 = "RpcCall.5";
    static PerfTimer t5(call_name5, 100);
    t5.on();
#endif

    // InternetCloseHandle(hSend);
    InternetCloseHandle(hRequest);
    return true;
#ifdef DEBUG_CONNECTOR
    t5.off();
#endif
#else
    string result_headers;
    uchar response_chr[], json_chr[];
    StringToCharArray(json, json_chr);
    json_chr[ArraySize(json_chr) - 1] = ' ';
    // const int http_status_code = WebRequest(C_post_verb, session_url_prefix + this_path, C_empty_string, C_empty_string, C_web_request_timeout, 
    //     json_chr, ArraySize(json_chr), response_chr, result_headers);
    const int http_status_code = WebRequest(C_post_verb, session_url_prefix + this_path, C_headers_json, C_web_request_timeout, json_chr, response_chr, result_headers);
    response = CharArrayToString(response_chr);
    LOG_VERBOSE("", "Path " + this_path + ", received " + response + ", headers " + result_headers + ", code " + IntegerToString(http_status_code));
    return http_status_code != -1;
#endif
}


bool MqlNet::RpcCall(const string &this_path, const string &Method, const Hash &Params, string &response)
{
    bool res;
    uint retries = 0;
    static const uint C_max_retries = 5;
    do { 
        res = RpcCall_(this_path, Method, Params, response);
    } while (res == false && ++retries < C_max_retries);
    if (res == false) LOG_ERROR("", "RpcCall " + this_path + " " + Method + " failed");
    return res;
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
string MqlNet::ReadResponse(const int hRequest)
{
#define BUFSIZ 1024
    int lReturn;
    uchar sBuffer[BUFSIZ];
    string response;
    while (InternetReadFile(hRequest, sBuffer, BUFSIZ, lReturn) != false && lReturn > 0)
        StringAdd(response, CharArrayToString(sBuffer, 0, lReturn));
    return response;
}

//------------------------------------------------------------------ Request
bool MqlNet::Request(const string &this_verb, const string &this_path, string &response, const bool toFile = false, const string addData = "", const bool fromFile = false)
{
    if (toFile && response == "") {
        LOG_ERROR("", "File is not specified");
        return false;
    }
    
    uchar data[];
    if (fromFile) {
        if (FileToArray(addData, data) < 0) {
            LOG_SYS_ERR("", "FileToArray failed");
            return (false);
        }
    } // file loaded to the array
    else {
        StringToCharArray(addData, data);
        data[ArraySize(data) - 1] = ' ';
    }
    
#ifdef WININET_DLL
    int hRequest, hSend;
    if (Session <= 0 || Connect <= 0) {
        Close();
        if (!Open(session_host, session_port, session_secure)) {
            Close();
            return (false);
        }
    }
// create descriptor for the request
    hRequest = HttpOpenRequestW(Connect, this_verb, this_path, C_http_ver, 0, 0, RegularConnectionFlags | ( INTERNET_FLAG_SECURE * session_secure ), 0);
    if (hRequest <= 0) {
        LOG_SYS_ERR("", "HttpOpenRequestW failed");
        InternetCloseHandle(Connect);
        return (false);
    }
// send request
    hSend = HttpSendRequestW(hRequest, head_www_form, len_head, data, ArraySize(data) - 1);
    if (hSend <= 0) {
        LOG_SYS_ERR("", "HttpSendRequestW failed");
        InternetCloseHandle(hRequest);
        Close();
    }
// read page
    ReadPage(hRequest, response, toFile);
// close all descriptors
    InternetCloseHandle(hRequest);
    InternetCloseHandle(hSend);
    return true;
    
#else

    char response_chr[];
    string response_header;
    const int http_status_code = WebRequest(this_verb, session_url_prefix + this_path, C_empty_string, C_empty_string, C_web_request_timeout, data, 
        ArraySize(data), response_chr, response_header);
    response = CharArrayToString(response_chr);
    LOG_VERBOSE("", "Received " + response + ", headers " + response_header + ", code " + IntegerToString(http_status_code));    
    return http_status_code != -1;
    
#endif
}

//------------------------------------------------------------------ OpenURL
bool MqlNet::OpenURL(const string &URL, string &response, const bool toFile)
{
    if (Session <= 0 || Connect <= 0) {
        Close();
        if (!Open(session_host, session_port, session_secure)) {
            Close();
            return false;
        }
    }
    uint hURL = InternetOpenUrlW(Session, URL, C_empty_string, 0, INTERNET_FLAG_RELOAD | INTERNET_FLAG_PRAGMA_NOCACHE, 0);
    if (hURL < 1) {
        LOG_SYS_ERR("", "InternetOpenUrlW failed");
        return false;
    }
// read to response
    ReadPage(hURL, response, toFile);
// close
    InternetCloseHandle(hURL);
    return true;
}

//------------------------------------------------------------------ ReadPage
void MqlNet::ReadPage(const int hRequest, string &response, const bool toFile)
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
        h = FileOpen(response, FILE_BIN | FILE_WRITE);
        FileWriteString(h, toStr);
        FileClose(h);
    } else response = toStr;
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
string MqlNet::HttpQueryInfo(const int hRequest, const int HTTP_QUERY)
{
    int len = 2048, ind = 0;
    uchar buf[2048];
    BOOL Res = HttpQueryInfoW(hRequest, HTTP_QUERY, buf, len, ind);
    if (Res == 0) {
        LOG_SYS_ERR("", "Failed");
        return C_empty_string;
    }
    string s;
    for (int i = 0; i < len; i++) {
        uchar uc = (uchar) buf[i];
        if (uc != 0)
            StringAdd(s, CharToString(uc));
    }
    if (StringLen(s) <= 0) return C_empty_string;
    return (s);
}

//------------------------------------------------------------------ GetServerCookieValue
string MqlNet::GetServerCookieValue(const int hRequest)
{
    return HttpQueryInfo(hRequest, HTTP_QUERY_SET_COOKIE);
}

//------------------------------------------------------------------ GetServerCookieValue
long MqlNet::GetStatusCode(const int hRequest)
{
    return StringToInteger(HttpQueryInfo(hRequest, HTTP_QUERY_STATUS_CODE));
}

//------------------------------------------------------------------ GetContentSize
long MqlNet::GetContentSize(const int hRequest)
{
    return StringToInteger(HttpQueryInfo(hRequest, HTTP_QUERY_CONTENT_LENGTH));
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
    for (i = 0; i < size; i++)
        data[i] = (uchar) FileReadInteger(h, CHAR_VALUE);
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
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
PerfTimer::~PerfTimer() {}
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void PerfTimer::on() {}
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
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
