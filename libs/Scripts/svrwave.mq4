#define SERVER_OK "OK"

#include <Utilities.mqh>

extern string sAddr = "localhost";
//"172.16.240.1";
extern string sProto = "http";
extern string sQueryH = "mql_query";
extern string sPostH = "mql_receive";
extern string sLogin = "user=vgfeit&password=vgfeit";


int start() {

    string
            GETResponse = "",
            POSTResponse = "";
    int args[];
    string s[2];
    s[0] = sLogin;
    s[1] = "&input_queue=EURUSD";
    int bytes = StrToArr(args, s);
    //if(HttpGET(StringConcatenate(sProto, "://", sAddr, "/", sQueryH), sLogin, GETResponse))
    //   Print(GETResponse);

    if (HttpPOST(StringConcatenate(sProto, "://", sAddr, "/", sPostH), args, bytes, POSTResponse))
        Print(POSTResponse);

    //send_all_bars();

    return (0);
}

bool get_server_ready() {
    string args = StringConcatenate(sLogin, "&symbol=", Symbol(), "&request=write");
    string response = "";
    string url = StringConcatenate(sProto, "://", sAddr, "/", sQueryH);

    if (!HttpGET(url, args, response))
        return (false);

    if (StrCmpCase(response, SERVER_OK, true))
        return (true);

    else Print("[get_server_ready] ", response);
    return (false);
}

// TODO: use gzip or other compression to send bulk data by range
// TODO: negotiate missing (server-side) / offered (client-side) values before send
bool send_all_bars() {
    if (!get_server_ready()) {
        Print("[send_all_bars] Server not ready to receive data!");
        return (false);
    }

    // iterates back in time and send data synchronously
    //for(
    int i = 0;
    //i<Bars; i++){
    string args, response;
    string url = StringConcatenate(sProto, "://", sAddr, "/", sPostH);

    StringConcatenate(args,
            sLogin, // username and password
            "&symbol=", Symbol(), // the name of the current financial instrument (the input table queue)
            "&time=", TimeToStr(Time[i], TIME_DATE | TIME_SECONDS), // open time of each bar of the current chart. TimeToStr: "yyyy.mm.dd hh:mi"
            "&open=", Open[i], // open prices of each bar of the current chart
            "&high=", High[i], // highest prices of each bar of the current chart
            "&low=", Low[i], // lowest prices of each bar of the current chart
            "&close=", Close[i], // close prices for each bar of the current chart
            "&volume=", Volume[i], // tick volumes of each bar of the current chart
            "&timeframe=", Period(), // Timeframe of the chart (chart period)
            "&client_time=", TimeToStr(TimeCurrent(), TIME_DATE | TIME_SECONDS) // current client datetime
    );

    /*if(!HttpPOST(url, args, response)){
       Print("[send_all_bars] ", response);
       return(false);
    } */// if
    //} // for

    return (true);
} // send_all_bars


