//
// Created by zarko on 27/07/2025.
//

#ifndef STREAMING_MESSAGES_PROTOCOL_TPP
#define STREAMING_MESSAGES_PROTOCOL_TPP

#include "streaming_messages_protocol.hpp"

namespace svr {
namespace daemon {

template<queue_binary_type T> void stream_message_queue::write(const T value)
{
    (void) write(reinterpret_cast<const char *>(&value), sizeof(value));
}

template<queue_binary_type T> T stream_message_queue::read()
{
    T value;
    (void) read(reinterpret_cast<char *>(&value), sizeof(value));
    return value;
}

template<typename T> T stream_message_queue::read()
{
    THROW_EX_FS(std::invalid_argument, "Unsupported type " << typeid(T).name());
}

}
}

#endif //STREAMING_MESSAGES_PROTOCOL_TPP
