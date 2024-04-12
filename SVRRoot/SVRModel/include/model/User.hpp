#pragma once

#include <common/types.hpp>
#include <model/Entity.hpp>
#include <model/Priority.hpp>


namespace svr {
namespace datamodel {

enum class ROLE
{
    ADMIN, USER
};

class User : public Entity
{

private:
    std::string user_name;
    std::string email;
    std::string name;
    std::string password;
    ROLE role = ROLE::USER;
    Priority priority = Priority::Normal;

public:
    bool operator==(User const &other) const;

    User() = default;

    User(const bigint user_id,
         const std::string &username,
         const std::string &email,
         const std::string &password,
         const std::string &name = "",
         const ROLE &_role = ROLE::USER,
         Priority _priority = Priority::Normal
    );

    ~User() = default;

    virtual void init_id() override;

    void set_email(const std::string &email);

    void set_name(const std::string &name);

    void set_role(ROLE role);

    void set_password(const std::string &password);

    void set_user_name(const std::string &userName);

    std::string get_name() const;

    std::string get_email() const;

    std::string get_user_name() const;

    ROLE get_role() const;

    std::string get_password() const;

    Priority const &get_priority() const;

    void set_priority(Priority const &priority);

    virtual std::string to_string() const override;

    virtual std::string to_json_string() const;
};

}
}

using User_ptr = std::shared_ptr<svr::datamodel::User>;
