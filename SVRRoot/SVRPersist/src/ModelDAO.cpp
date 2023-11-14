#include <DAO/ModelDAO.hpp>
#include "PgDAO/PgModelDAO.hpp"
#include "AsyncDAO/AsyncModelDAO.hpp"
#include "ThreadSafeDAO/TsModelDAO.hpp"

namespace svr {
namespace dao {

ModelDAO * ModelDAO::build(svr::common::PropertiesFileReader& sqlProperties, svr::dao::DataSource& dataSource, svr::common::ConcreteDaoType daoType, bool use_threadsafe_dao)
{
    return AbstractDAO::build<ModelDAO, PgModelDAO, AsyncModelDAO, TsModelDAO>(sqlProperties, dataSource, daoType, use_threadsafe_dao);
}

ModelDAO::ModelDAO(svr::common::PropertiesFileReader& sqlProperties, svr::dao::DataSource& dataSource)
: AbstractDAO(sqlProperties, dataSource, "ModelDAO.properties")
{}


}
}
