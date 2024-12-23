#include <util/validation_utils.hpp>
#include "UserService.hpp"
#include <DAO/UserDAO.hpp>
#include <model/User.hpp>
#include <common/logging.hpp>

using namespace svr::common;


namespace svr {
namespace business {

int UserService::save(const User_ptr &user)
{
    REJECT_NULLPTR(user);

    // if the password is longer than 16 characters then it is already hashed
    if (user->get_password().length() > 0 && user->get_password().length() < 16) {
        user->set_password(make_md5_hash(user->get_password()));
    }

    return user_dao.save(user);
}

int UserService::remove(const User_ptr &user)
{
    REJECT_NULLPTR(user);
    if (!exists(user->get_user_name())) {
        LOG4_ERROR("Cannot remove user " << user->to_string() << " because doesn't exist!");
        return false;
    }
    return user_dao.remove(user);
}

User_ptr UserService::get_user_by_user_name(const std::string &user_name)
{
    return user_dao.get_by_user_name(user_name);
}

std::vector<User_ptr> UserService::get_all_users()
{
    return user_dao.get_all_users();
}

bool UserService::exists(const std::string &user_name)
{
    return user_dao.exists(user_name);
}

bool UserService::login(const std::string &user_name, const std::string &password)
{
    return user_dao.login(user_name, password);
}

std::vector<User_ptr> UserService::get_all_users_by_priority()
{
    return user_dao.get_all_users_by_priority();
}

} /* namespace business */
} /* namespace svr */


