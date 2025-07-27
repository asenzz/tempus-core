#pragma once

#include <memory>
#include <vector>

namespace svr { namespace dao { class UserDAO; } }
namespace svr { namespace datamodel {
class User;
using User_ptr = std::shared_ptr<User>;
} }

namespace svr {
namespace business {

class UserService {

	svr::dao::UserDAO& user_dao;

public:

    UserService(svr::dao::UserDAO& userDao) : user_dao(userDao) {}

    datamodel::User_ptr get_user_by_user_name(const std::string& user_name);
    int save(const datamodel::User_ptr&);
    bool exists(const std::string& user_name);
    int remove(const datamodel::User_ptr&);
    bool login(const std::string& user_name, const std::string& password);

    std::vector<datamodel::User_ptr> get_all_users();

    std::vector<datamodel::User_ptr> get_all_users_by_priority();
};

} /* namespace business */
} /* namespace svr */
