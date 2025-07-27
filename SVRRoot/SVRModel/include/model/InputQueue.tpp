//
// Created by zarko on 27/07/2025.
//

#ifndef INPUTQUEUE_TPP
#define INPUTQUEUE_TPP

namespace svr {
namespace datamodel {

template<typename T> std::basic_ostream<T> &
operator<<(std::basic_ostream<T> &s, const InputQueue &iq)
{
    return s << iq.to_string();
}

template<typename T> std::basic_ostream<T> &
operator<<(const InputQueue &iq, std::basic_ostream<T> &s)
{
    return s << iq.to_string();
}

}

}

#endif //INPUTQUEUE_TPP
