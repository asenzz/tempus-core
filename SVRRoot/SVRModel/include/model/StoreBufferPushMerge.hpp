#ifndef STOREBUFFERPUSHMERGE_HPP
#define STOREBUFFERPUSHMERGE_HPP

namespace svr {

template<class T>
void store_buffer_push_merge(T &dest, T const &src)
{
    dest = src;
}

}

#endif /* STOREBUFFERPUSHMERGE_HPP */

