#include "AsyncUserDAO.hpp"
#include "../PgDAO/PgUserDAO.hpp"
#include "AsyncImplBase.hpp"
#include "model/User.hpp"
#include <common/logging.hpp>
#include "common/ScopeExit.hpp"


namespace svr {
namespace dao {

namespace {
static const auto cmp_primary_key = [](User_ptr const &lhs, User_ptr const &rhs) -> bool {
    return reinterpret_cast<uint64_t>(lhs.get()) && reinterpret_cast<uint64_t>(rhs.get())
           && lhs->get_user_name() == rhs->get_user_name();
};
static const auto cmp_whole_value = [](User_ptr const &lhs, User_ptr const &rhs) -> bool {
    return reinterpret_cast<uint64_t>(lhs.get()) && reinterpret_cast<uint64_t>(rhs.get())
           && *lhs == *rhs;
};
}

struct AsyncUserDAO::AsyncImpl
        : AsyncImplBase<User_ptr, DTYPE(cmp_primary_key), DTYPE(cmp_whole_value), PgUserDAO> {
    AsyncImpl(common::PropertiesReader &sql_properties, dao::DataSource &data_source)
            : AsyncImplBase(sql_properties, data_source, cmp_primary_key, cmp_whole_value, 10, 10)
    {}

    void store(User_ptr user)
    {
        const std::scoped_lock lg(pgMutex);
        if (pgDao.exists(user->get_user_name()))
            pgDao.update(user);
        else {
            user->set_id(pgDao.get_next_id());
            pgDao.save(user);
        }
    }
};

AsyncUserDAO::AsyncUserDAO(common::PropertiesReader &sql_properties, dao::DataSource &data_source)
        : UserDAO(sql_properties, data_source), pImpl(*new AsyncImpl(sql_properties, data_source))
{}

AsyncUserDAO::~AsyncUserDAO()
{
    delete &pImpl;
}

User_ptr AsyncUserDAO::get_by_user_name(const std::string &user_name)
{
    auto user = ptr<datamodel::User>(0, user_name, "", "");
    pImpl.seekAndCache(user, &PgUserDAO::get_by_user_name, user_name);
    return user;
}

std::vector<User_ptr> AsyncUserDAO::get_all_users()
{
    const std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.get_all_users();
}

std::vector<User_ptr> AsyncUserDAO::get_all_users_by_priority()
{
    const std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.get_all_users_by_priority();
}

bigint AsyncUserDAO::get_next_id()
{
    const std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.get_next_id();
}

bool AsyncUserDAO::exists(std::string const &user_name)
{
    auto user = ptr<datamodel::User>(0, user_name, "", "");
    if (pImpl.cached(user)) return true;

    const std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.exists(user_name);
}

int AsyncUserDAO::save(const User_ptr &user)
{
    pImpl.cache(user);
    return 1;
}

int AsyncUserDAO::update(const User_ptr &user)
{
    pImpl.cache(user);
    return 1;
}

int AsyncUserDAO::remove(const User_ptr &user)
{
    return pImpl.remove(user);
}

bool AsyncUserDAO::login(const std::string &user_name, const std::string &enc_password)
{
    const std::scoped_lock lg(pImpl.pgMutex);
    return pImpl.pgDao.login(user_name, enc_password);
}

}
}
