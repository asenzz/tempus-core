//+------------------------------------------------------------------+
//|                                              tempus_smp_impl.mqh |
//|                                                       Jarko Asen |
//|                                                www.zarkoasen.com |
//+------------------------------------------------------------------+
#include <tempus/smp.mqh>
#include <tempus/errordescription.mqh>
   

const uint8_t C_price_columns = 2;

#define DEFINE_STREAM(x) \
    file_path_##x = streams_root_path + "\\" + #x; \
    mutex_path_##x = streams_root_path + "\\" + #x + C_mutex_suffix; \
    init_message_stream(file_path_##x, file_handle_##x);


const int streams_messaging::C_msg_cycle = 100; // ms
const int streams_messaging::C_data_cycle = 10; // ms
const string streams_messaging::C_queue_prefix = "q_svrwave_";
const string streams_messaging::C_mutex_suffix = "m";
const string streams_messaging::C_folder_name = "tempus";

streams_messaging::streams_messaging(const string &streams_root_path)
{
    DEFINE_STREAM(req);
    DEFINE_STREAM(resp);
    DEFINE_STREAM(offer);
    DEFINE_STREAM(range);
    DEFINE_STREAM(datain);
}

streams_messaging::~streams_messaging()
{
    close_message_streams();
}

void streams_messaging::close_message_stream(const string &file_path, const int file_handle)
{
    if (file_handle == INVALID_HANDLE) {
        LOG_ERROR("Invalid handle for file " + file_path + " supplied.");
        return;
    }

    FileFlush(file_handle);
    FileClose(file_handle);
    LOG_VERBOSE("File " + file_path + " closed succesfully.");
}

bool streams_messaging::init_message_stream(const string &file_path, int &file_handle)
{
    static const int file_stream_default_flags = FILE_BIN | FILE_SHARE_WRITE | FILE_SHARE_READ;
    file_handle = FileOpen(file_path, file_stream_default_flags);
    if (file_handle == INVALID_HANDLE) {
        LOG_ERROR("Operation file open " + file_path + " failed, error " + ErrorDescription(GetLastError()));
        return false;
    } else {
        if (FileSize(file_handle) < 2) FileWriteInteger(file_handle, 0, sizeof(uint16_t));
        LOG_VERBOSE("File " + file_path + " created successfully.");
        return true;
    }
}

void streams_messaging::trim_stream(const string &file_path, int &file_handle)
{
    if (FileSize(file_handle) <= 2) return;
    FileSeek(file_handle, 0, SEEK_SET);
    const uint8_t written_ct = (uint8_t) FileReadInteger(file_handle, sizeof(uint8_t));
    const uint8_t read_ct = (uint8_t) FileReadInteger(file_handle, sizeof(uint8_t));
    if (read_ct == 0 || read_ct < written_ct) return;

    FileClose(file_handle);
    do {
        LOG_DEBUG("Truncating file " + file_path);
        file_handle = FileOpen(file_path, FILE_WRITE);
    } while (file_handle == INVALID_HANDLE);
    FileClose(file_handle);

    init_message_stream(file_path, file_handle);
}


bool streams_messaging::file_exists(const string &file_path)
{
    return FileOpen(file_path, FILE_READ | FILE_COMMON) == INVALID_HANDLE;
}

bool streams_messaging::create_empty(const string &file_path)
{
    const int handle = FileOpen(file_path, FILE_WRITE);
    if (handle == INVALID_HANDLE) return false;
    FileClose(handle);
    return true;
}


void streams_messaging::lock(const string &file_path)
{
    while (file_exists(file_path)) Sleep(C_data_cycle);
    create_empty(file_path);
}


void streams_messaging::unlock(const string &file_path)
{
    FileDelete(file_path);
}


bool streams_messaging::initialize_message_streams()
{
    return 
       init_message_stream(file_path_req, file_handle_req) &&
       init_message_stream(file_path_resp, file_handle_resp) &&

       init_message_stream(file_path_offer, file_handle_offer) &&
       init_message_stream(file_path_range, file_handle_range) &&

       init_message_stream(file_path_datain, file_handle_datain);
}


void streams_messaging::close_message_streams()
{
    close_message_stream(file_path_req, file_handle_req);
    close_message_stream(file_path_resp, file_handle_resp);

    close_message_stream(file_path_offer, file_handle_offer);
    close_message_stream(file_path_range, file_handle_range);

    close_message_stream(file_path_datain, file_handle_datain);
}


bool streams_messaging::offer_range(const string &table_name, datetime &from, datetime &to)
{
    const string queue_name = C_queue_prefix + table_name;
    lock(mutex_path_offer);
    trim_stream(file_path_offer, file_handle_offer);

    FileSeek(file_handle_range, 0, SEEK_SET);
    const uint8_t written_range = (uint8_t) FileReadInteger(file_handle_range, sizeof(uint8_t));

    FileSeek(file_handle_offer, 0, SEEK_END);
    FileWriteInteger(file_handle_offer, (uint8_t) StringLen(queue_name), sizeof(uint8_t));
    FileWriteString(file_handle_offer, queue_name);
    FileWriteInteger(file_handle_offer, uint32_t(from), sizeof(uint32_t));
    FileWriteInteger(file_handle_offer, uint32_t(to), sizeof(uint32_t));

    FileSeek(file_handle_offer, 0, SEEK_SET);
    uint8_t written_offer = (uint8_t) FileReadInteger(file_handle_offer, 1) + 1;        
    FileSeek(file_handle_offer, 0, SEEK_SET);
    FileWriteInteger(file_handle_offer, written_offer, sizeof(uint8_t));
    FileFlush(file_handle_offer);
    FileDelete(mutex_path_offer);
    
    lock(mutex_path_range);
    uint8_t new_written_range;
    do {
        Sleep(C_msg_cycle);
        FileSeek(file_handle_range, 0, SEEK_SET);
        new_written_range = (uint8_t) FileReadInteger(file_handle_range, sizeof(uint8_t));
    } while (new_written_range <= written_range);

    Sleep(C_msg_cycle);
    FileSeek(file_handle_range, 2, SEEK_SET);
    bool answered = false;
    uint8_t read_range_msg = 0;
    do {
        const uint8_t queue_name_len = (uint8_t) FileReadInteger(file_handle_range, sizeof(uint8_t));
        if (read_range_msg == written_range && FileReadString(file_handle_range, queue_name_len) == queue_name) {
            from = FileReadInteger(file_handle_range, sizeof(uint32_t));
            to = FileReadInteger(file_handle_range, sizeof(uint32_t));
            answered = true;
        } else {
            FileSeek(file_handle_range, queue_name_len + 2 * sizeof(uint32_t), SEEK_CUR);
        }
        ++read_range_msg;
    } while (!FileIsEnding(file_handle_range) && read_range_msg < new_written_range);
    unlock(mutex_path_range);
    
    return answered;
}


void streams_messaging::send_queue(
    const string &table_name, 
    const string &columns[], 
    const datetime &times[], 
    const uint32_t &volumes[], 
    const aprice &prices[], 
    const uint32_t start_row, 
    const uint32_t end_row)
{
    const string queue_name = C_queue_prefix + table_name;
    lock(mutex_path_datain);
        
    trim_stream(file_path_datain, file_handle_datain);

    FileSeek(file_handle_datain, 0, SEEK_END);
    FileWriteInteger(file_handle_datain, (uint8_t) StringLen(queue_name), sizeof(uint8_t));
    FileWriteString(file_handle_datain, queue_name);
    
    FileWriteInteger(file_handle_datain, uint8_t(C_price_columns), sizeof(uint8_t));
    for (uint8_t c = 0; c < C_price_columns; ++c) {
        FileWriteInteger(file_handle_datain, (uint8_t) StringLen(columns[c]), sizeof(uint8_t));
        FileWriteString(file_handle_datain, columns[c]);
    }
    const uint32_t row_ct = end_row - start_row;
    FileWriteInteger(file_handle_datain, row_ct, sizeof(uint32_t));
    for (uint32_t row_ix = start_row; row_ix < end_row; ++row_ix) {
        FileWriteInteger(file_handle_datain, (uint32_t) times[row_ix], sizeof(uint32_t));
        FileWriteDouble(file_handle_datain, volumes[row_ix]);
        FileWriteDouble(file_handle_datain, prices[row_ix].bid);
        FileWriteDouble(file_handle_datain, prices[row_ix].ask);
    }

    FileSeek(file_handle_datain, 0, SEEK_SET);
    uint8_t written_packs = (uint8_t) FileReadInteger(file_handle_datain, 1) + 1;
    while (written_packs >= UCHAR_MAX) {
        trim_stream(file_path_datain, file_handle_datain);
        FileSeek(file_handle_datain, 0, SEEK_SET);
        written_packs = (uint8_t) FileReadInteger(file_handle_datain, 1) + 1;
    }

    FileSeek(file_handle_datain, 0, SEEK_SET);
    FileWriteInteger(file_handle_datain, written_packs, sizeof(uint8_t));
    FileFlush(file_handle_datain);
    
    unlock(mutex_path_datain);
}


bool in(const datetime time, const datetime &times[])
{
    for (int i = 0; i < ArraySize(times); ++i)
        if (time == times[i]) return true;
    return false;
}


void streams_messaging::get_results(const uint32_t dataset_id, const datetime from, const uint count, const datetime resolution, price_row &results[])
{
    datetime result_times[];
    ArrayResize(result_times, count);
    for (uint i = 0; i < count; ++i)
        result_times[i] = from + i * resolution;
        
    lock(mutex_path_resp);
    
    FileSeek(file_handle_resp, 0, SEEK_SET);   
    const uint8_t written_ct = (uint8_t) FileReadInteger(file_handle_resp, sizeof(uint8_t));
    const uint8_t read_ct = (uint8_t) FileReadInteger(file_handle_resp, sizeof(uint8_t));
    for (uint8_t msg_i = 0; msg_i < written_ct; ++msg_i)
        if (msg_i >= read_ct) {
            const uint16_t rows_ct = (uint16_t) FileReadInteger(file_handle_resp, sizeof(uint16_t));
            if (dataset_id == FileReadInteger(file_handle_resp, sizeof(uint32_t)))
                for (uint16_t i = 0; i < rows_ct; ++i) {
                    const datetime row_time = FileReadInteger(file_handle_resp, sizeof(uint32_t));
                    if (!in(row_time, result_times)) continue;
                    const int results_ct = ArraySize(results);
                    ArrayResize(results, results_ct + 1);
                    results[results_ct].tm = row_time;
                    results[results_ct].cl.bid = FileReadDouble(file_handle_resp);
                    results[results_ct].cl.ask = FileReadDouble(file_handle_resp);
                }
            else
                FileSeek(file_handle_resp, rows_ct * (sizeof(uint32_t) + 2 * sizeof(double)), SEEK_CUR);
        } else {
            FileSeek(file_handle_resp, sizeof(uint32_t), SEEK_CUR);
            const uint16_t rows_ct = (uint16_t) FileReadInteger(file_handle_resp, sizeof(uint16_t));
            FileSeek(file_handle_resp, rows_ct * (sizeof(uint32_t) + 2 * sizeof(double)), SEEK_CUR);
        }
    
    LOG_DEBUG("Received " + IntegerToString(ArraySize(results)) + " results for " + IntegerToString(dataset_id) + ", from " + 
        TimeToString(from, C_time_mode) + ", count " + IntegerToString(count));
    
    trim_stream(file_path_resp, file_handle_resp);
    FileFlush(file_handle_resp);
    
    unlock(mutex_path_resp);
}