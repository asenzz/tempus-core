#include "TsUserDAO.hpp"

namespace svr {
namespace dao {

DEFINE_THREADSAFE_DAO_CONSTRUCTOR (TsUserDAO, UserDAO)
{}

User_ptr TsUserDAO::get_by_user_name(const std::string& user_name)
{
    return ts_call<User_ptr>(&UserDAO::get_by_user_name, user_name);
}


std::vector<User_ptr> TsUserDAO::get_all_users()
{
    return ts_call<std::vector<User_ptr>>(&UserDAO::get_all_users);
}


std::vector<User_ptr> TsUserDAO::get_all_users_by_priority()
{
    return ts_call<std::vector<User_ptr>>(&UserDAO::get_all_users_by_priority);
}


bigint TsUserDAO::get_next_id()
{
    return ts_call<bigint>(&UserDAO::get_next_id);
}


bool TsUserDAO::exists(std::string const & user_name)
{
    return ts_call<bool>(&UserDAO::exists, user_name);
}


int TsUserDAO::save(const User_ptr& user)
{
    return ts_call<int>(&UserDAO::save, user);
}


int TsUserDAO::update(const User_ptr& user)
{
    return ts_call<int>(&UserDAO::update, user);
}


int TsUserDAO::remove(const User_ptr& user)
{
    return ts_call<int>(&UserDAO::remove, user);
}


bool TsUserDAO::login(const std::string& user_name, const std::string& enc_password)
{
    return ts_call<bool>(&UserDAO::login, user_name, enc_password);
}


}}
