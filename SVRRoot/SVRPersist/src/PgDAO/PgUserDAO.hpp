#ifndef PGUSERDAO_H
#define PGUSERDAO_H

#include "DAO/UserDAO.hpp"

namespace svr {
namespace dao {

class PgUserDAO : public UserDAO {
public:
    PgUserDAO(common::PropertiesReader &sql_properties, dao::DataSource &data_source);

    datamodel::User_ptr get_by_user_name(const std::string &user_name);

    std::vector<datamodel::User_ptr> get_all_users();

    std::vector<datamodel::User_ptr> get_all_users_by_priority();

    bigint get_next_id();

    bool exists(std::string const &user_name);

    int save(const datamodel::User_ptr &);

    int update(const datamodel::User_ptr &);

    int remove(const datamodel::User_ptr &);

    bool login(const std::string &user_name, const std::string &enc_password);
};

}
}

#endif /* PGUSERDAO_H */

