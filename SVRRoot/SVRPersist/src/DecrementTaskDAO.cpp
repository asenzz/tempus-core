#include <DAO/DecrementTaskDAO.hpp>
#include "PgDAO/PgDecrementTaskDAO.hpp"
#include "AsyncDAO/AsyncDecrementTaskDAO.hpp"
#include "ThreadSafeDAO/TsDecrementTaskDAO.hpp"

namespace svr {
namespace dao {

DecrementTaskDAO * DecrementTaskDAO::build(svr::common::PropertiesFileReader& sqlProperties, svr::dao::DataSource& dataSource, svr::common::ConcreteDaoType daoType, bool use_threadsafe_dao)
{
    return AbstractDAO::build<DecrementTaskDAO, PgDecrementTaskDAO, AsyncDecrementTaskDAO, TsDecrementTaskDAO>(sqlProperties, dataSource, daoType, use_threadsafe_dao);
}


DecrementTaskDAO::DecrementTaskDAO(svr::common::PropertiesFileReader& sqlProperties, svr::dao::DataSource& dataSource)
: AbstractDAO(sqlProperties, dataSource, "DecrementTaskDAO.properties")
{}

} // namespace dao
} // namespace svr
