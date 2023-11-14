#pragma once

#include <memory>
#include <vector>

namespace svr { namespace dao { class UserDAO; } }
namespace svr { namespace datamodel { class User; } }

using User_ptr=std::shared_ptr<svr::datamodel::User>;

namespace svr {
namespace business {

class UserService {

	svr::dao::UserDAO& userDao;

public:

    UserService(svr::dao::UserDAO& userDao) : userDao(userDao) {}

    User_ptr get_user_by_user_name(const std::string& user_name);
    int save(const User_ptr&);
    bool exists(const std::string& user_name);
    int remove(const User_ptr&);
    bool login(const std::string& user_name, const std::string& password);

    std::vector<User_ptr> get_all_users();

    std::vector<User_ptr> get_all_users_by_priority();
};

} /* namespace business */
} /* namespace svr */
