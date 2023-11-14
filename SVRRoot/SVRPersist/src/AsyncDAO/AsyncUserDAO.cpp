#include "AsyncUserDAO.hpp"

#include "../PgDAO/PgUserDAO.hpp"
#include "AsyncImplBase.hpp"

#include "model/User.hpp"
#include <common/Logging.hpp>
#include "common/ScopeExit.hpp"

namespace svr { namespace dao {

namespace {
    static bool cmp_primary_key(User_ptr const & lhs, User_ptr const & rhs)
    {
        return reinterpret_cast<unsigned long>(lhs.get()) && reinterpret_cast<unsigned long>(rhs.get())
                && lhs->get_user_name() == rhs->get_user_name();
    }
    static bool cmp_whole_value(User_ptr const & lhs, User_ptr const & rhs)
    {
        return reinterpret_cast<unsigned long>(lhs.get()) && reinterpret_cast<unsigned long>(rhs.get())
                && *lhs == *rhs;
    }
}

struct AsyncUserDAO::AsyncImpl
    : AsyncImplBase<User_ptr, decltype(std::ptr_fun(cmp_primary_key)), decltype(std::ptr_fun(cmp_whole_value)), PgUserDAO>
{
    AsyncImpl(svr::common::PropertiesFileReader& sql_properties, svr::dao::DataSource& data_source)
    :AsyncImplBase(sql_properties, data_source, std::ptr_fun(cmp_primary_key), std::ptr_fun(cmp_whole_value), 10, 10)
    {}

    void store(User_ptr user)
    {
        std::scoped_lock lg(pgMutex);
        if(pgDao.exists(user->get_user_name()))
            pgDao.update(user);
        else
        {
            user->set_id(pgDao.get_next_id());
            pgDao.save(user);
        }
    }
};

AsyncUserDAO::AsyncUserDAO(svr::common::PropertiesFileReader &sql_properties, svr::dao::DataSource &data_source)
: UserDAO(sql_properties, data_source), pImpl(*new AsyncImpl(sql_properties, data_source))
{}

AsyncUserDAO::~AsyncUserDAO()
{
    delete & pImpl;
}

User_ptr AsyncUserDAO::get_by_user_name(const std::string& user_name)
{
    User_ptr user{ std::make_shared<svr::datamodel::User>() };
    user->set_user_name(user_name);

    pImpl.seekAndCache(user, &PgUserDAO::get_by_user_name, user_name);
    return user;
}

std::vector<User_ptr> AsyncUserDAO::get_all_users()
{
    std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.get_all_users();
}

std::vector<User_ptr> AsyncUserDAO::get_all_users_by_priority()
{
    std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.get_all_users_by_priority();
}

bigint AsyncUserDAO::get_next_id()
{
    std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.get_next_id();
}

bool AsyncUserDAO::exists(std::string const &user_name)
{
    User_ptr user{ std::make_shared<svr::datamodel::User>() };
    user->set_user_name(user_name);

    if(pImpl.cached(user))
        return true;

    std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.exists(user_name);
}

int AsyncUserDAO::save(const User_ptr& user)
{
    pImpl.cache(user);
    return 1;
}

int AsyncUserDAO::update(const User_ptr& user)
{
    pImpl.cache(user);
    return 1;
}

int AsyncUserDAO::remove(const User_ptr& user)
{
    return pImpl.remove(user);
}

bool AsyncUserDAO::login(const std::string& user_name, const std::string& enc_password)
{
    std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.login(user_name, enc_password);
}

} }
