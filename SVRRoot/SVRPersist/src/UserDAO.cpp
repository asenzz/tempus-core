#include <DAO/UserDAO.hpp>
#include "PgDAO/PgUserDAO.hpp"
#include "AsyncDAO/AsyncUserDAO.hpp"
#include "ThreadSafeDAO/TsUserDAO.hpp"

namespace svr {
namespace dao {

UserDAO * UserDAO::build(svr::common::PropertiesFileReader& sqlProperties, svr::dao::DataSource& dataSource, common::ConcreteDaoType daoType, bool use_threadsafe_dao)
{
    return AbstractDAO::build<UserDAO, PgUserDAO, AsyncUserDAO, TsUserDAO>(sqlProperties, dataSource, daoType, use_threadsafe_dao);
}

UserDAO::UserDAO(svr::common::PropertiesFileReader& sqlProperties, svr::dao::DataSource& dataSource)
:AbstractDAO(sqlProperties, dataSource, "UserDAO.properties")
{}

}
}
