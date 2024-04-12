//
// Created by zarko on 2023-06-27.
//
#include "model/Entity.hpp"

namespace svr {
namespace datamodel {

void Entity::set_id(const bigint _id)
{
    id = _id;
    on_set_id();
}

}

}