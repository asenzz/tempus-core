#include <DAO/UserDAO.hpp>
#include "PgDAO/PgUserDAO.hpp"
#include "AsyncDAO/AsyncUserDAO.hpp"
#include "ThreadSafeDAO/TsUserDAO.hpp"

namespace svr {
namespace dao {

UserDAO * UserDAO::build(svr::common::PropertiesFileReader& tempus_config, svr::dao::DataSource& data_source, common::ConcreteDaoType dao_type, bool use_threadsafe_dao)
{
    return AbstractDAO::build<UserDAO, PgUserDAO, AsyncUserDAO, TsUserDAO>(tempus_config, data_source, dao_type, use_threadsafe_dao);
}

UserDAO::UserDAO(svr::common::PropertiesFileReader& tempus_config, svr::dao::DataSource& data_source)
:AbstractDAO(tempus_config, data_source, "UserDAO.properties")
{}

}
}
