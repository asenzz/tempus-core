#property link      "http://www.mql5.com"
#property version   "1.00"
#property strict
#property script_show_inputs
#property description "Sample script posting a user message "
#property description "on the wall on mql5.com"

input string Username = "test";
input string Password = "password";
input string BaseUrl = "http://192.168.16.145/";
input string LoginPath = "mt4/login/";

string CookieHeader;

bool doLogin(string username, string password) {
    int res;     // To receive the operation execution result
    char post_data[];  // Data array to send POST requests
    char response_data[];
    string response_headers;
    string cookieHeader = "Set-Cookie: cppcms_session=";

    string str = "username=" + username + "&password=" + password;
    string auth;

    ArrayResize(post_data, StringToCharArray(str, post_data, 0, WHOLE_ARRAY, CP_UTF8) - 1);

    ResetLastError();

    res = WebRequest(
            "POST", // http method
            BaseUrl + LoginPath, // url
            NULL, // cookie
            NULL, // referer
            5000, // timeout
            post_data, // POST data
            ArraySize(post_data), // POST data size
            response_data, // response data
            response_headers // response headers
    );

    if (res != 200) {
        Print("Authorization error #" + (string) res + ", LastError=" + (string) GetLastError());
        return (false);
    }

    string response_data_str = CharArrayToString(response_data);
    res = StringFind(response_headers, cookieHeader);

    if (res < 0) {
        Print("Error, authorization data not found in the server response (check login/password)");
        return (false);
    }

    auth = StringSubstr(response_headers, res + StringLen(cookieHeader));
    CookieHeader = "cppcms_session=" + StringSubstr(auth, 0, StringFind(auth, ";")) + "\r\n";

    return (true);
}

struct RPCResponse {

public:
    string headers;
    string response;

    string toString(void) {
        string str = "Headers: " + headers;
        str = StringConcatenate(str, " Response: ", response);

        return str;
    }
};

int doRPCRequest(string domain, string method, string &params[], RPCResponse &rpcResponse) {

    string url = BaseUrl + "web/" + domain + "/ajax";
    string json = "{\"id\":0, \"method\":\":" + method + "\",\"params\":[";
    char result[];
    string result_headers;

    ResetLastError();

    for (int i = 0; i < ArraySize(params); i++) {
        json += params[i];
        if (i < ArraySize(params) - 1) {
            json += ",";
        }
    }

    json += "]}";
    char json_array[];
    ArrayResize(json_array, StringLen(json));
    StringToCharArray(json, json_array);

    int response = WebRequest(
            "POST",
            url,
            CookieHeader,
            NULL,
            5000,
            json_array,
            StringLen(json),
            result,
            result_headers
    );

    if (response == 200) {
        rpcResponse.headers = result_headers;
        rpcResponse.response = CharArrayToString(result);
        Print(rpcResponse.toString());
    } else {
        Print("Request error #" + (string) response + ", LastError=" + (string) GetLastError());
    }

    return response;
}

//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart() {
    if (doLogin(Username, Password)) {
        RPCResponse response;
        string params[];
        int res = doRPCRequest("queue", "getAllInputQueues", params, response);
    }
}
//+------------------------------------------------------------------+