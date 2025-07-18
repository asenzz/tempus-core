#include "PgUserDAO.hpp"
#include "common/logging.hpp"
#include "DAO/UserRowMapper.hpp"
#include "DAO/DataSource.hpp"


namespace svr {
namespace dao {

PgUserDAO::PgUserDAO(common::PropertiesReader &tempus_config, dao::DataSource &data_source) : UserDAO(tempus_config, data_source)
{}

bigint PgUserDAO::get_next_id()
{
    LOG4_DEBUG("Getting next user id");
    return bigint(data_source.query_for_type<long>(AbstractDAO::get_sql("get_next_id")));
}

int PgUserDAO::save(const User_ptr &user)
{

    if (exists(user->get_user_name())) return update(user);
    user->set_id(get_next_id());
    return data_source.update(AbstractDAO::get_sql("save"),
                              user->get_id(),
                              user->get_user_name(),
                              user->get_password(),
                              user->get_email(),
                              user->get_name(),
                              user->get_role(),
                              user->get_priority());
}

int PgUserDAO::update(const User_ptr &user)
{
    LOG4_DEBUG("Updating user: " << user->to_string());
    return data_source.update(AbstractDAO::get_sql("update"),
                              user->get_password(),
                              user->get_email(),
                              user->get_name(),
                              user->get_role(),
                              user->get_priority(),
                              user->get_user_name());
}

int PgUserDAO::remove(const User_ptr &user)
{
    LOG4_DEBUG("Removing user: " << user->to_string());
    return data_source.update(AbstractDAO::get_sql("remove"), user->get_user_name());
}

bool PgUserDAO::exists(std::string const &user_name)
{
    return 1 == data_source.query_for_type<long>(AbstractDAO::get_sql("existsByUsername"), user_name);
}

User_ptr PgUserDAO::get_by_user_name(const std::string &user_name)
{
    UserRowMapper row_mapper;
    return data_source.query_for_object(&row_mapper, AbstractDAO::get_sql("get_by_user_name"), user_name);
}

std::vector<User_ptr> PgUserDAO::get_all_users()
{
    UserRowMapper row_mapper;
    return data_source.query_for_array(row_mapper, AbstractDAO::get_sql("get_all_users"));
}


bool PgUserDAO::login(const std::string &user_name, const std::string &enc_password)
{
    return data_source.query_for_type<bool>(get_sql("login"), user_name, enc_password);
}

std::vector<User_ptr> PgUserDAO::get_all_users_by_priority()
{
    UserRowMapper row_mapper;
    return data_source.query_for_array(row_mapper, get_sql("get_all_users_by_priority"));
}

}
}
