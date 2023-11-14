#include <DAO/SVRParametersDAO.hpp>
#include "PgDAO/PgSVRParametersDAO.hpp"
#include "AsyncDAO/AsyncSVRParametersDAO.hpp"
#include "ThreadSafeDAO/TsSVRParametersDAO.hpp"

namespace svr{
namespace dao{

SVRParametersDAO * SVRParametersDAO::build(svr::common::PropertiesFileReader& sqlProperties, svr::dao::DataSource& dataSource, svr::common::ConcreteDaoType daoType, bool use_threadsafe_dao)
{
    return AbstractDAO::build<SVRParametersDAO, PgSVRParametersDAO, AsyncSVRParametersDAO, TsSVRParametersDAO>(sqlProperties, dataSource, daoType, use_threadsafe_dao);
}

SVRParametersDAO::SVRParametersDAO(svr::common::PropertiesFileReader& sqlProperties, svr::dao::DataSource& dataSource)
: AbstractDAO(sqlProperties, dataSource, "SVRParametersDAO.properties")
{}

}
}
