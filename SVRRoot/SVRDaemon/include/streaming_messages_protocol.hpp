//
// Created by zarko on 28/01/2025.
//

#ifndef SVR_STREAMING_MESSAGES_PROTOCOL_HPP
#define SVR_STREAMING_MESSAGES_PROTOCOL_HPP

#include <deque>
#include <string>
#include <boost/unordered/unordered_flat_map.hpp>
#include "model/DataRow.hpp"
#include "DatasetService.hpp"

namespace svr {
namespace daemon {

typedef struct _msg_table {
    std::deque<std::string> column_names;
    std::deque<datamodel::DataRow_ptr> rows;
} t_msg_table, *t_msg_table_ptr;

using t_msg_tables = boost::unordered_flat_map<std::string, t_msg_table>;

template<typename T> concept queue_binary_type =
std::same_as<T, double> || std::same_as<T, float> ||
std::same_as<T, uint64_t> || std::same_as<T, uint32_t> || std::same_as<T, uint16_t> || std::same_as<T, uint8_t> ||
std::same_as<T, uint64_t> || std::same_as<T, int32_t> || std::same_as<T, int16_t> || std::same_as<T, int8_t>;

class stream_message_queue : public std::basic_fstream<char> {
    const std::string stream_path;

public:
    using std::basic_fstream<char>::write, std::basic_fstream<char>::read;

    stream_message_queue(const char file_path[]);

    template<queue_binary_type T> inline void write(const T value)
    {
        (void) write(reinterpret_cast<const char *>(&value), sizeof(value));
    }

    template<queue_binary_type T> inline T read()
    {
        T value;
        (void) read(reinterpret_cast<char *>(&value), sizeof(value));
        return value;
    }

    // Writes Pascal-type string to the stream
    void write(const std::string &value);

    void write(const bpt::ptime &value);

    template<typename T> T read()
    {
        THROW_EX_FS(std::invalid_argument, "Unsupported type " << typeid(T).name());
    }

    void reset();
};

template<> std::string stream_message_queue::read();

template<> bpt::ptime stream_message_queue::read();

class streaming_messages_protocol {
    static constexpr char C_req_file_path[] = "/dev/shm/tempus/req";
    static constexpr char C_resp_file_path[] = "/dev/shm/tempus/resp";
    static const char *C_resp_mutex_path;

    static constexpr char C_offer_file_path[] = "/dev/shm/tempus/offer";
    static constexpr char C_range_file_path[] = "/dev/shm/tempus/range";

    static constexpr char C_datain_file_path[] = "/dev/shm/tempus/datain";
    static const char *C_datain_mutex_path;

    std::chrono::milliseconds sleep_interval = std::chrono::milliseconds(10);
    stream_message_queue datain_fs, response_fs;
    std::thread worker;

    streaming_messages_protocol();

    void lock(const std::string &mutex_path);

public:
    t_msg_tables receive_queues_data(const boost::posix_time::ptime &timenau);

    static streaming_messages_protocol &get();

    static bool create_mutex(const std::string &filename);

    void write_response(const bigint dataset_id, const business::t_stream_results &stream_results);
};

}
}

/* Description of input data stream message queue */
struct data_msg_queue {
    uint8_t written_msg_count; // Written by broker connector
    uint8_t read_msg_count; // Written by daemon

    /* messages content */
    uint8_t table_count;

    /* for every table */
    uint8_t table_name_len;
    char *table_name;
    uint8_t column_count;

    /* for every column */
    uint8_t column_name_len;
    char *column_name;
    uint32_t value_count; // row count

    /* for every row */
    uint32_t row_time;
    double row_volume;
    double value_bid; // column values
    double value_ask;
    /* ... */
};

/* Description of prediction response message queue */
struct response_msg_queue {
    uint8_t written_msg_count; // Written by daemon
    uint8_t read_msg_count; // Written by broker connector

    /* messages content */
    uint32_t dataset_id;
    uint16_t row_count;
    struct response_row {
        uint32_t row_time;
        double row_value_bid;
        double row_value_ask;
    } *response_rows;

    /* ... */
};


#endif //SVR_STREAMING_MESSAGES_PROTOCOL_HPP
