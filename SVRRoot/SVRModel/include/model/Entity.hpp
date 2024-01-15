#pragma once

#include "common/types.hpp"

namespace svr {
namespace datamodel {

class Entity
{
public:
    Entity() : id(0) {}
    Entity(const bigint id) : id(id) {}
    virtual ~Entity() = default;
    bigint get_id() const { return id; }

    void set_id(const bigint _id);
    virtual std::string to_string() const = 0;

    virtual std::basic_ostream<char> &operator <<(std::basic_ostream<char> &os) const;
protected:
    bigint id;

private:
    virtual void on_set_id() {}
};

std::basic_ostream<char> &operator <<(std::basic_ostream<char> &os, Entity &e);

}
}
