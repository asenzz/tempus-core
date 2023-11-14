#include <DAO/RequestDAO.hpp>
#include <DAO/DataSource.hpp>
#include "PgDAO/PgRequestDAO.hpp"
#include "AsyncDAO/AsyncRequestDAO.hpp"
#include "ThreadSafeDAO/TsRequestDAO.hpp"

namespace svr {
namespace dao {

RequestDAO::RequestDAO(svr::common::PropertiesFileReader& sqlProperties, svr::dao::DataSource& dataSource)
: AbstractDAO(sqlProperties, dataSource, "RequestDAO.properties")
{
}

RequestDAO * RequestDAO::build(svr::common::PropertiesFileReader& sqlProperties, svr::dao::DataSource& dataSource, svr::common::ConcreteDaoType daoType, bool use_threadsafe_dao)
{
    return AbstractDAO::build<RequestDAO, PgRequestDAO, AsyncRequestDAO, TsRequestDAO>(sqlProperties, dataSource, daoType, use_threadsafe_dao);
}

}
}

