#include <DAO/EnsembleDAO.hpp>
#include "PgDAO/PgEnsembleDAO.hpp"
#include "AsyncDAO/AsyncEnsembleDAO.hpp"
#include "ThreadSafeDAO/TsEnsembleDAO.hpp"

namespace svr {
namespace dao {

EnsembleDAO * EnsembleDAO::build(svr::common::PropertiesFileReader& sqlProperties, svr::dao::DataSource& dataSource, svr::common::ConcreteDaoType daoType, bool use_threadsafe_dao)
{
    return AbstractDAO::build<EnsembleDAO, PgEnsembleDAO, AsyncEnsembleDAO, TsEnsembleDAO>(sqlProperties, dataSource, daoType, use_threadsafe_dao);
}

EnsembleDAO::EnsembleDAO(svr::common::PropertiesFileReader& sqlProperties, svr::dao::DataSource& dataSource)
:AbstractDAO(sqlProperties, dataSource, "EnsembleDAO.properties")
{}

}
}
