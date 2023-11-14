#include "PgDAO/PgDeconQueueDAO.hpp"
#include "AsyncDAO/AsyncDeconQueueDAO.hpp"
#include "ThreadSafeDAO/TsDeconQueueDAO.hpp"

using svr::common::ConcreteDaoType;

namespace svr{
namespace dao{

DeconQueueDAO * DeconQueueDAO::build(svr::common::PropertiesFileReader& sqlProperties, svr::dao::DataSource& dataSource, ConcreteDaoType daoType, bool use_threadsafe_dao)
{
    return AbstractDAO::build<DeconQueueDAO, PgDeconQueueDAO, AsyncDeconQueueDAO, TsDeconQueueDAO>(sqlProperties, dataSource, daoType, use_threadsafe_dao);
}

DeconQueueDAO::DeconQueueDAO(svr::common::PropertiesFileReader& sqlProperties, svr::dao::DataSource& dataSource)
: AbstractDAO(sqlProperties, dataSource, "DeconQueueDAO.properties")
{}

}
}
