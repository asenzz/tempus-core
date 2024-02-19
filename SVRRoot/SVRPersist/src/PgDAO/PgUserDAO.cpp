#include "PgUserDAO.hpp"
#include <common/Logging.hpp>
#include <DAO/UserRowMapper.hpp>
#include <DAO/DataSource.hpp>


namespace svr { namespace dao {

PgUserDAO::PgUserDAO(svr::common::PropertiesFileReader& tempus_config, svr::dao::DataSource& data_source)
:UserDAO(tempus_config, data_source)
{}

bigint PgUserDAO::get_next_id() {
    LOG4_DEBUG("Getting next user id");
    return bigint ( data_source.query_for_type<long>(AbstractDAO::get_sql("get_next_id")) );
}

int PgUserDAO::save(const User_ptr& user) {

    if(exists(user->get_user_name()))
        return update(user);
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

int PgUserDAO::update(const User_ptr& user) {
    LOG4_DEBUG("Updating user: " << user->to_string());
    return data_source.update(AbstractDAO::get_sql("update"),
            user->get_password(),
            user->get_email(),
            user->get_name(),
            user->get_role(),
            user->get_priority(),
            user->get_user_name());
}

int PgUserDAO::remove(const User_ptr& user) {
    LOG4_DEBUG("Removing user: " << user->to_string());
    return data_source.update(AbstractDAO::get_sql("remove"), user->get_user_name());
}

bool PgUserDAO::exists(std::string const & userName) {
    return 1 == data_source.query_for_type<long>(AbstractDAO::get_sql("existsByUsername"), userName);
}

User_ptr PgUserDAO::get_by_user_name(const std::string &user_name) {
    UserRowMapper rowMapper;
    return data_source.query_for_object(&rowMapper, AbstractDAO::get_sql("get_by_user_name"), user_name);
}

std::vector<User_ptr> PgUserDAO::get_all_users() {
    UserRowMapper rowMapper;
    return data_source.query_for_array(rowMapper, AbstractDAO::get_sql("get_all_users"));
}


bool PgUserDAO::login(const std::string &user_name, const std::string &enc_password) {
    return data_source.query_for_type<bool>(get_sql("login"), user_name, enc_password);
}

std::vector<User_ptr> PgUserDAO::get_all_users_by_priority()
{
    UserRowMapper rowMapper;
    return data_source.query_for_array(rowMapper, get_sql("get_all_users_by_priority"));
}

}}
