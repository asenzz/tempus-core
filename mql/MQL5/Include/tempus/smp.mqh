//+------------------------------------------------------------------+
//|                            Tempus Streams Messaging Protocol
//|                                                       Jarko Asen |
//|                                                www.zarkoasen.com |
//+------------------------------------------------------------------+
#property copyright "Jarko Asen"
#property link      "www.zarkoasen.com"
// TODO Unify headers across CPP and MQL code

#include <tempus/types.mqh>
#include <tempus/components.mqh>

#define DECLARE_STREAM(x) \
    string file_path_##x; \
    string mutex_path_##x; \
    int file_handle_##x;

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class streams_messaging
{
    DECLARE_STREAM(req)

    DECLARE_STREAM(resp)

    DECLARE_STREAM(offer)

    DECLARE_STREAM(range)

    DECLARE_STREAM(datain)

    void close_message_stream(const string &file_path, const int file_handle);

    bool init_message_stream(const string &file_path, int &file_handle);

    void trim_stream(const string &file_path, int &file_handle);

    bool file_exists(const string &file_path);

    bool create_empty(const string &file_path);

    void lock(const string &file_path);

    void unlock(const string &file_path);

    bool initialize_message_streams();

    void close_message_streams();

public:
    static const int C_msg_cycle;
    static const int C_data_cycle;
    static const string C_queue_prefix;
    static const string C_folder_name;
    static const string C_mutex_suffix;

    streams_messaging(const string &streams_root_path);

    ~streams_messaging();

    static streams_messaging *get();

    bool offer_range(const string &table_name, datetime &from, datetime &to);

    void send_queue(
        const string &table_name,
        const string &columns[],
        const datetime &times[],
        const uint32_t &volumes[],
        const aprice &prices[],
        const uint32_t start_row,
        const uint32_t end_row);

    void get_results(const uint32_t dataset_id, const datetime from, const uint count, const datetime resolution, price_row &results[]);

};

//+------------------------------------------------------------------+

#include <tempus/smp_impl.mqh>