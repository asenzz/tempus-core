#pragma once

#include  <DAO/AbstractDAO.hpp>

namespace svr { namespace datamodel { class User; } }
using User_ptr = std::shared_ptr<svr::datamodel::User>;

namespace svr {
namespace dao {

class UserDAO : public AbstractDAO {
public:
    static UserDAO * build(svr::common::PropertiesFileReader& sql_properties, svr::dao::DataSource& data_source, svr::common::ConcreteDaoType dao_type, bool use_threadsafe_dao);

    UserDAO(svr::common::PropertiesFileReader& sql_properties, svr::dao::DataSource& data_source);

    virtual User_ptr get_by_user_name(const std::string& user_name) = 0;
    virtual std::vector<User_ptr> get_all_users() = 0;
    virtual std::vector<User_ptr> get_all_users_by_priority() = 0;
    virtual bigint get_next_id() = 0;

    virtual bool exists(std::string const & user_name) = 0;

    virtual int save(const User_ptr&) = 0;
    virtual int update(const User_ptr&) = 0;
    virtual int remove(const User_ptr&) = 0;

    virtual bool login(const std::string& user_name, const std::string& enc_password) = 0;
};

}
}

using UserDAO_ptr = std::shared_ptr<svr::dao::UserDAO>;
