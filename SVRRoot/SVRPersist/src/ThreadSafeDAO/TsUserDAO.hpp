#pragma once

#include "TsDaoBase.hpp"
#include <DAO/UserDAO.hpp>

namespace svr {
namespace dao {

THREADSAFE_DAO_CLASS_DECLARATION_HEADER (TsUserDAO, UserDAO)

    virtual User_ptr get_by_user_name(const std::string& user_name);
    virtual std::vector<User_ptr> get_all_users();
    virtual std::vector<User_ptr> get_all_users_by_priority();
    virtual bigint get_next_id();

    virtual bool exists(std::string const & userName);

    virtual int save(const User_ptr&);
    virtual int update(const User_ptr&);
    virtual int remove(const User_ptr&);

    virtual bool login(const std::string& user_name, const std::string& enc_password);
};

}
}
