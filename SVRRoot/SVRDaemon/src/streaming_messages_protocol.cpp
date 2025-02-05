//
// Created by zarko on 28/01/2025.
//

#include <filesystem>
#include "streaming_messages_protocol.hpp"
#include "appcontext.hpp"

namespace svr {
namespace daemon {

constexpr char C_mutex_suffix[] = "m";

stream_message_queue::stream_message_queue(const char file_path[]) : std::basic_fstream<char>(file_path, std::ios::in | std::ios::out | std::ios::binary), stream_path(file_path)
{
    if (!is_open()) LOG4_THROW("Failed opening file " << file_path);
    seekg(0, std::ios::end);
    if (tellg() == 0) {
        seekp(0, std::ios::beg);
        write<uint16_t>(0);
        flush();
    }
}

void stream_message_queue::reset()
{
    std::filesystem::resize_file(stream_path, 0);
    seekp(0, std::ios::beg);
    write<uint16_t>(0);
    flush();
}

// Writes Pascal-type string to the stream
void stream_message_queue::write(const std::string &value)
{
    if (value.size() > UINT8_MAX) LOG4_THROW("String too long " << value.size() << ", should be less or equal to " << UINT8_MAX);
    stream_message_queue::write < uint8_t > (value.size());
    write(value.data(), value.size());
}

void stream_message_queue::write(const bpt::ptime &value)
{
    stream_message_queue::write<uint32_t>(bpt::to_time_t(value));
}

template<> std::string stream_message_queue::read()
{
    const auto len = read < uint8_t > ();
    std::string value(len, '\0');
    read(value.data(), len);
    return value;
}

template<> bpt::ptime stream_message_queue::read()
{
    uint32_t value;
    read(reinterpret_cast<char *>(&value), sizeof(uint32_t));
    return bpt::from_time_t(std::time_t(value));
}


const char *streaming_messages_protocol::C_resp_mutex_path = common::concat(streaming_messages_protocol::C_resp_file_path, C_mutex_suffix);
const char *streaming_messages_protocol::C_datain_mutex_path = common::concat(streaming_messages_protocol::C_datain_file_path, C_mutex_suffix);

streaming_messages_protocol::streaming_messages_protocol() :
        sleep_interval(PROPS.get_stream_loop_interval()),
        datain_fs(C_datain_file_path),
        response_fs(C_resp_file_path)
{
}

streaming_messages_protocol &streaming_messages_protocol::get()
{
    static streaming_messages_protocol instance;
    return instance;
}

bool streaming_messages_protocol::create_mutex(const std::string &filename)
{
    // Create the empty file using std::ofstream
    std::ofstream ofs(filename);
    if (!ofs) {
        LOG4_ERROR("Failed to create the file " << filename);
        return false;
    }
    ofs.close(); // Close the file
    return true;
}

void streaming_messages_protocol::lock(const std::string &mutex_path)
{
    while (std::filesystem::exists(mutex_path)) std::this_thread::sleep_for(sleep_interval);
    if (!create_mutex(mutex_path)) LOG4_THROW("Failed creating mutex " << mutex_path);
}

void streaming_messages_protocol::write_response(const bigint dataset_id, const business::t_stream_results &stream_results)
{
    response_fs.seekg(0, std::ios::beg);
    lock(C_resp_mutex_path);

    const auto res_wrote_ct = response_fs.read<uint8_t>();
    const auto res_read_ct = response_fs.read<uint8_t>();
    if (res_read_ct >= res_wrote_ct) response_fs.reset();

    response_fs.seekp(0, std::ios::end);
    response_fs.write<uint32_t>(dataset_id);

    // Merge rows from different columns
    const auto &first_column_values = *stream_results.cbegin();
    const auto do_merge = stream_results.size() > 1;
    assert(!do_merge || std::all_of(C_default_exec_policy, std::next(stream_results.cbegin()), stream_results.cend(),
                                    [&first_column_values](const auto &p) { return p.second.size() == first_column_values.second.size() &&
                                    p.second.front()->size() == first_column_values.second.front()->size(); }));

    response_fs.write<uint8_t>(stream_results.size() * first_column_values.second.front()->size()); // Column count
    response_fs.write<uint16_t>(first_column_values.second.size()); // Values count
    for (const auto &p_result_row: first_column_values.second) {
        response_fs.write(p_result_row->get_value_time());
        response_fs.write(p_result_row->get_tick_volume());
        for (const auto &v: p_result_row->get_values()) response_fs.write(v);
        if (do_merge)
            for (auto it_stream_results = std::next(stream_results.cbegin()); it_stream_results != stream_results.cend(); ++it_stream_results) {
                const auto &column_vals_merge = *it_stream_results;
                const auto it_first_time =
                        std::find_if(C_default_exec_policy, column_vals_merge.second.cbegin(), column_vals_merge.second.cend(),
                                     [&p_result_row](const auto &p) { return p->get_value_time() == p_result_row->get_value_time(); });
                if (it_first_time == column_vals_merge.second.cend()) LOG4_THROW("Failed merging rows, response is inconsistent.");
                for (const auto &v: (**it_first_time).get_values()) response_fs.write(v);
            }
    }

    response_fs.seekp(0, std::ios::beg);
    response_fs.write<uint8_t>(res_wrote_ct + 1);
    response_fs.flush();

    std::filesystem::remove(C_resp_mutex_path);
}

t_msg_tables streaming_messages_protocol::receive_queues_data(const bpt::ptime &timenau)
{
    datain_fs.seekg(0, std::ios::beg);
    lock(C_datain_mutex_path);
    t_msg_tables tables;
    const auto dat_wrote_ct = datain_fs.read<uint8_t>();
    const auto dat_read_ct = datain_fs.read<uint8_t>();
    if (dat_wrote_ct > dat_read_ct) {
        uint8_t msg_i = 0;
        while (msg_i < dat_wrote_ct) {
            const auto tables_ct = datain_fs.read<uint8_t>();
            for (DTYPE(tables_ct) t = 0; t < tables_ct; ++t) {
                const auto table_name_len = datain_fs.read<uint8_t>();
                t_msg_table_ptr p_table;
                if (msg_i >= dat_read_ct) {
                    const auto table_name = datain_fs.read<std::string>();
                    auto tables_it = tables.find(table_name);
                    if (tables_it != tables.cend()) {
                        LOG4_TRACE("Table " << table_name << " already exists");
                        p_table = &tables_it->second;
                    } else {
                        bool rc;
                        std::tie(tables_it, rc) = tables.emplace(table_name, t_msg_table{});
                        if (rc) p_table = &tables_it->second;
                        else {
                            LOG4_ERROR("Failed creating message for table " << table_name);
                            p_table = nullptr;
                        }
                    }
                } else {
                    datain_fs.ignore(table_name_len);
                    p_table = nullptr;
                }

                const auto columns_ct = datain_fs.read<uint8_t>();
                for (DTYPE(columns_ct) c = 0; c < columns_ct; ++c) {
                    const auto column_name_len = datain_fs.read<uint8_t>();
                    if (p_table) {
                        auto &column_name = p_table->column_names.emplace_back(std::string());
                        column_name.resize(column_name_len);
                        datain_fs.read(column_name.data(), column_name_len);
                    } else datain_fs.ignore(column_name_len);
                }

                const auto row_ct = datain_fs.read<uint32_t>();
                if (p_table) {
                    p_table->rows.resize(row_ct);
                    for (DTYPE(row_ct) v = 0; v < row_ct; ++v) {
                        p_table->rows[v] = otr<datamodel::DataRow>(datain_fs.read<bpt::ptime>(), timenau, datain_fs.read<double>(), columns_ct);
                        auto &values = p_table->rows[v]->get_values();
                        for (uint8_t c = 0; c < columns_ct; ++c) values[c] = datain_fs.read<double>();
                    }
                } else
                    datain_fs.ignore(row_ct * ((columns_ct + 1) * sizeof(double) + sizeof(uint32_t)));
            }
            ++msg_i;
        }
        datain_fs.seekp(1, std::ios::beg);
        datain_fs.write(dat_read_ct);
        datain_fs.flush();
    }
    datain_fs.reset();
    std::filesystem::remove(C_datain_mutex_path);

    return tables;
}

}
}