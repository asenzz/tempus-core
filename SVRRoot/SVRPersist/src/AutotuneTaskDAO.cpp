#include <DAO/AutotuneTaskDAO.hpp>
#include "PgDAO/PgAutotuneTaskDAO.hpp"
#include "AsyncDAO/AsyncAutotuneTaskDAO.hpp"
#include "ThreadSafeDAO/TsAutotuneTaskDAO.hpp"

namespace svr {
namespace dao {

AutotuneTaskDAO * AutotuneTaskDAO::build(svr::common::PropertiesFileReader& sqlProperties, svr::dao::DataSource& dataSource, svr::common::ConcreteDaoType daoType, bool use_threadsafe_dao)
{
    return AbstractDAO::build<AutotuneTaskDAO, PgAutotuneTaskDAO, AsyncAutotuneTaskDAO, TsAutotuneTaskDAO>(sqlProperties, dataSource, daoType, use_threadsafe_dao);
}


AutotuneTaskDAO::AutotuneTaskDAO(svr::common::PropertiesFileReader& sqlProperties, svr::dao::DataSource& dataSource)
:AbstractDAO(sqlProperties, dataSource, "AutotuneTaskDAO.properties")
{}


} /* namespace dao */
} /* namespace svr */
