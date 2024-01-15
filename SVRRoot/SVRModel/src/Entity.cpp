//
// Created by zarko on 2023-06-27.
//
#include "model/Entity.hpp"

namespace svr {
namespace datamodel {

std::ostream &operator <<(std::ostream &os, Entity &e)
{
    os << e.to_string();
    return os;
}

void Entity::set_id(const bigint _id)
{
    id = _id;
    on_set_id();
}

std::basic_ostream<char> &Entity::operator <<(std::basic_ostream<char> &os) const
{
    os << to_string();
    return os;
}

std::basic_ostream<char> &operator <<(std::basic_ostream<char> &os, const Entity &e)
{
    os << e.to_string();
    return os;
}


}
}
