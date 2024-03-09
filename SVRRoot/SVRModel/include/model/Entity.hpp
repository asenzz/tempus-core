#pragma once

#include <memory>
#include "common/types.hpp"

// #define ENTITY_INIT_ID

namespace svr {
namespace datamodel {

class Entity
{
    virtual void init_id() = 0;

public:
    Entity() : id(0)
    {}

    Entity(const bigint id) : id(id)
    {}

    virtual ~Entity() = default;

    bigint get_id() const
    { return id; }

    void set_id(const bigint _id);

    virtual std::string to_string() const = 0;

    virtual std::basic_ostream<char> &operator<<(std::basic_ostream<char> &os) const;

protected:
    bigint id = 0;

private:
    virtual void on_set_id()
    {}
};

using Entity_ptr = std::shared_ptr<Entity>;

std::basic_ostream<char> &operator<<(std::basic_ostream<char> &os, Entity &e);

}

}

#include "dbcache.tpp"
